use std::collections::HashMap;
use std::io::Write;

use memmap2::Mmap;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

pub mod data;
pub mod error;
pub mod forward;
pub mod math;
pub mod reader;
pub mod state;
pub mod tokenizer;
pub mod kernels;
pub mod tests;

use forward::forward;
use reader::parse_general;
use state::Header;
use std::time::Instant;


use crate::{
    data::{Tensor, Transformer, Mode, Cache}, error::{ForwardPassError, GenerateError, MetaDataError, ModelLoadError, ReaderError}, math::rope_angles, reader::{parse_tensor_data, parse_tensors}, state::{
        GgufMetadataValue::*, GgufMetadataValueType::GGUF_METADATA_VALUE_TYPE_ARRAY, MetaDataField,
    }, tokenizer::{decode, embed, encode, sample},
};

pub fn load_model<'a>(
    map: &'a Mmap,
) -> Result<
    (
        Transformer,
        Vec<MetaDataField>,
        Vec<Tensor>,
        HashMap<String, &'a [u8]>,
    ),
    ModelLoadError,
> {
    // We enforce that the header values are valid here
    if map.len() < 24 {
        return Err(ReaderError::UnexpectedEof("gguf header", map.len()).into());
    }

    let head = Header {
        m_number: &map[..4].try_into().map_err(|_| ReaderError::BadMagic)?,
        version: &map[4..8]
            .try_into()
            .map_err(|_| ReaderError::UnexpectedEof("version", 4))?,
        tensor_count: &map[8..16]
            .try_into()
            .map_err(|_| ReaderError::UnexpectedEof("tensor count", 8))?,
        md_k_v_pairs: &map[16..24]
            .try_into()
            .map_err(|_| ReaderError::UnexpectedEof("metadata count", 16))?,
    };

    // Magic is literally "GGUF"
    if head.m_number != b"GGUF" {
        return Err(ReaderError::BadMagic.into());
    }

    let version = u32::from_le_bytes(*head.version);
    if !(2..=3).contains(&version) {
        return Err(ReaderError::BadVersion(version).into());
    }

    let (gguf_info, offset) = parse_general(map, 24, u64::from_le_bytes(*head.md_k_v_pairs))?;

    let mut layers = None;
    let mut dim = None;
    let mut embed_dim = None;
    let mut q_heads = None;
    let mut k_v_heads = None;
    let mut base_freq = None;
    let mut alignment = 32;

    for field in &gguf_info {
        match (field.name.as_str(), &field.value) {
            ("llama.block_count", GGUF_METADATA_VALUE_U32(value)) => layers = Some(*value),
            ("llama.rope.dimension_count", GGUF_METADATA_VALUE_U32(value)) => {
                dim = Some(*value as usize)
            }
            ("llama.embedding_length", GGUF_METADATA_VALUE_U32(value)) => {
                embed_dim = Some(*value as usize)
            }
            ("llama.attention.head_count", GGUF_METADATA_VALUE_U32(value)) => {
                q_heads = Some(*value as usize)
            }
            ("llama.attention.head_count_kv", GGUF_METADATA_VALUE_U32(value)) => {
                k_v_heads = Some(*value as usize)
            }
            ("llama.rope.freq_base", GGUF_METADATA_VALUE_F32(value)) => base_freq = Some(*value),
            ("general.alignment", GGUF_METADATA_VALUE_U32(value)) => alignment = *value as usize,
            _ => {}
        }
    }

    let mut mode = Mode::default();
    #[cfg(feature = "q_act")]
    {
        mode = Mode::Quantize
    }



    let transformer_info = Transformer {
        layers: layers.ok_or(MetaDataError::MissingField("Layers"))?,
        dim: dim.ok_or(MetaDataError::MissingField("Dimension"))?,
        embed_dim: embed_dim.ok_or(MetaDataError::MissingField("Embedding Length"))?,
        q_heads: q_heads.ok_or(MetaDataError::MissingField("Query Attention Heads"))?,
        k_v_heads: k_v_heads.ok_or(MetaDataError::MissingField("K/V Attention Heads"))?,
        base_freq: base_freq.ok_or(MetaDataError::MissingField("Rope Base Frequency"))?,
        mode
    };

    let (read_tensors, offset) =
        parse_tensors(map, offset, u64::from_le_bytes(*head.tensor_count))?;

    // Tensor data starts at the next aligned boundary after the tensor infos,
    // per the spec that's general.alignment with a default of 32
    let data_start = offset.div_ceil(alignment) * alignment;

    let tensor_map = read_tensors
        .par_iter()
        .map(|tensor| {
            parse_tensor_data(
                map,
                data_start + usize::from_le_bytes(tensor.offset),
                tensor,
            )
            .map(|data| (tensor.name.clone(), data))
            .map_err(|e| ModelLoadError::TensorDataError(format!("{}: {e}", tensor.name)))
        })
        .collect::<Result<HashMap<_, _>, _>>()?;

    Ok((transformer_info, gguf_info, read_tensors, tensor_map))
}

pub fn generate(
    input: &str,
    transformer: &Transformer,
    metadata_info: &[MetaDataField],
    tensor_data: &HashMap<String, &[u8]>,
    embed_tensor: &[u8],
) -> Result<String, GenerateError> {
    let now = Instant::now();

    // Fetch tokenizer arrays from metadata
    let mut vocab = None;
    let mut merges = None;

    for field in metadata_info {
        match field.name.as_str() {
            "tokenizer.ggml.tokens" => vocab = Some(field),
            "tokenizer.ggml.merges" => merges = Some(field),
            _ => {}
        }
    }

    let vocab = vocab.ok_or(MetaDataError::MissingField("ggml.tokens"))?;
    let merges = merges.ok_or(MetaDataError::MissingField("ggml.merges"))?;

    // Tokenize the input sentence
    let encoded = encode(vocab, merges, input)?;

    let mut kv_cache = Cache::new();
    let mut logits = vec![];

    let mut pos = 0;

    // Prompt pass, run every prompt token through to fill the kv cache
    for token in encoded {
        let embedded = embed(token, embed_tensor, transformer.embed_dim);
        let angles = rope_angles(transformer.base_freq, pos, transformer.dim);
        logits = forward(
            embedded,
            pos,
            transformer,
            tensor_data,
            &mut kv_cache,
            angles
        )?;
        pos += 1
    }

    let mut tokens = vec![];

    // break loop when we hit EOS
    loop {
        let token_id = sample(&logits);

        /*  The correct way to do this is load the EOS from the metadata, and the BOS too:
            "tokenizer.ggml.bos_token_id": GGUF_METADATA_VALUE_U32(1)
            "tokenizer.ggml.eos_token_id": GGUF_METADATA_VALUE_U32(2)
            For now we'll leave this hardcoded in.

            We also add '13', which is a newline token.
            Without this, the model has a tendency to spiral into making lists with
            questions like "The capital of France is"
            In a true production runtime, we would use the chat template:
            "tokenizer.chat_template":
            For now we will break the generate loop when we hit a newline.
        */
        if token_id == 13 || token_id == 2 {
            break;
        }

        tokens.push(token_id);

        // Stream each token as it lands, generation is slow enough to watch ;)
        print!("{}", decode(vocab, &[token_id])?);
        let _ = std::io::stdout().flush();

        let embedded = embed(token_id, embed_tensor, transformer.embed_dim);
        let angles = rope_angles(transformer.base_freq, pos, transformer.dim);

        logits = forward(
            embedded,
            pos,
            transformer,
            tensor_data,
            &mut kv_cache,
            angles
        )?;

        pos += 1
    }

    let output = decode(vocab, &tokens)?;

    println!("");
    println!("Total: {}", now.elapsed().as_secs_f32());

    Ok(output)
}

// Generate up to a capped token count, return raw ids
pub fn generate_capped(
    input: &str,
    cap: usize,
    transformer: &Transformer,
    metadata_info: &[MetaDataField],
    tensor_data: &HashMap<String, &[u8]>,
    embed_tensor: &[u8],
) -> Result<Vec<usize>, GenerateError> {
    // Fetch tokenizer arrays from metadata
    let mut vocab = None;
    let mut merges = None;

    for field in metadata_info {
        match field.name.as_str() {
            "tokenizer.ggml.tokens" => vocab = Some(field),
            "tokenizer.ggml.merges" => merges = Some(field),
            _ => {}
        }
    }

    let vocab = vocab.ok_or(MetaDataError::MissingField("ggml.tokens"))?;
    let merges = merges.ok_or(MetaDataError::MissingField("ggml.merges"))?;

    // Tokenize the input sentence
    let encoded = encode(vocab, merges, input)?;

    let mut kv_cache = Cache::new();
    let mut logits = vec![];

    let mut pos = 0;

    // Prompt pass, run every prompt token through to fill the kv cache
    for token in encoded {
        let embedded = embed(token, embed_tensor, transformer.embed_dim);
        let angles = rope_angles(transformer.base_freq, pos, transformer.dim);

        logits = forward(
            embedded,
            pos,
            transformer,
            tensor_data,
            &mut kv_cache,
            angles
        )?;
        pos += 1
    }

    let mut tokens = vec![];

    // break loop when we hit cap
    loop {
        
        if tokens.len() >= cap {
            break;
        }

        let token_id = sample(&logits);
        tokens.push(token_id);

        let embedded = embed(token_id, embed_tensor, transformer.embed_dim);
        let angles = rope_angles(transformer.base_freq, pos, transformer.dim);

        logits = forward(
            embedded,
            pos,
            transformer,
            tensor_data,
            &mut kv_cache,
            angles
        )?;

        pos += 1
    }

    Ok(tokens)
}

pub fn test_forward(
    transformer: &Transformer,
    tensor_data: &HashMap<String, &[u8]>,
) -> Result<String, ForwardPassError> {
    // Mock token embedded
    let mock_token = vec![1.0; transformer.embed_dim];

    let angles = rope_angles(transformer.base_freq, 0, transformer.dim);
    let mut kv_cache = Cache::new();
    let logits = forward(mock_token, 0, transformer, tensor_data, &mut kv_cache, angles)?;
    let token_id = sample(&logits);
    Ok(token_id.to_string())
}

pub fn print_metadata(info: Vec<MetaDataField>) {
    for field in info {
        if field.vtype == GGUF_METADATA_VALUE_TYPE_ARRAY {
            println!("{:?}: {:?} (Not Listed)", field.name, field.vtype);
        } else {
            println!("{:?}: {:?}", field.name, field.value);
        }
    }
}

pub fn print_tensors(info: Vec<Tensor>) {
    for tensor in info {
        let scalar: u64 = tensor.shape.iter().product();
        let data_type = match tensor.dtype {
            0 => "F32",
            8 => "Q8_0",
            _ => "Unsupported",
        };
        println!(
            "{} {:?} {} {}",
            tensor.name, tensor.shape, scalar, data_type
        );
    }
}
