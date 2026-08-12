use std::collections::HashMap;

use crate::{
    error::TokenizerError,
    state::{GgufMetadataValue::*, MetaDataField},
};

use half::f16;

pub fn embed(id: usize, embed_tensor: &[u8], dim: usize) -> Vec<f32> {
    // Each Q8 block is 34 bytes, 32 i8 elements & 2 bytes for the scalar per block:
    let row_bytes = dim + dim / 32 * 2;
    let index = row_bytes * id;

    let embed_vector_bytes = &embed_tensor[index..index + row_bytes];

    let mut embed_vector = vec![];

    // Dequantize
    let blocks = embed_vector_bytes.chunks(34);

    for block in blocks {
        let scale = f16::from_le_bytes([block[0], block[1]]);
        embed_vector.extend(
            block[2..34]
                .iter()
                .map(|weight| i8::from_le_bytes([*weight]) as f32 * f32::from(scale)),
        );
    }

    embed_vector
}

// BPE in three phases: normalize, split into symbols, then merge
// The old version tried to do the split and merge in one pass over chunked pairs,
// which only ever merged pairs that landed on even offsets
pub fn encode(
    vocab: &MetaDataField,
    merges: &MetaDataField,
    text: &str,
) -> Result<Vec<usize>, TokenizerError> {
    let GGUF_METADATA_VALUE_ARRAY(vocab_array) = &vocab.value else {
        return Err(TokenizerError::WrongType("array of tokens"));
    };

    let GGUF_METADATA_VALUE_ARRAY(merges_array) = &merges.value else {
        return Err(TokenizerError::WrongType("array of merges"));
    };

    // Build lookup maps once, linear scans over a 32k vocab per token are brutal
    let mut vocab_map = HashMap::new();
    for (id, token) in vocab_array.iter().enumerate() {
        if let GGUF_METADATA_VALUE_STRING(token) = token {
            vocab_map.insert(token.as_str(), id);
        }
    }

    // A merge's position in the list is its priority, earlier = merge first
    let mut merge_ranks = HashMap::new();
    for (rank, merge) in merges_array.iter().enumerate() {
        if let GGUF_METADATA_VALUE_STRING(merge) = merge {
            merge_ranks.insert(merge.as_str(), rank);
        }
    }

    // Phase 1, normalize:
    // "▁" is 'LOWER_ONE_EIGHT_BLOCK' and is not the same as "_"
    // Sentencepiece prepends a space, then every space becomes "▁"
    // Only real spaces, newlines and tabs fall through to byte fallback below
    let normalized = format!(" {text}").replace(' ', "▁");

    // Phase 2, split into symbols, one per char, over the whole text
    // No word splitting here, so runs of whitespace survive intact
    let mut symbols: Vec<String> = normalized.chars().map(String::from).collect();

    // Phase 3, merge: find the best ranked adjacent pair anywhere in the
    // sequence, merge it, repeat until nothing merges anymore
    loop {
        let mut best: Option<(usize, usize)> = None;

        for i in 0..symbols.len().saturating_sub(1) {
            let pair = format!("{} {}", symbols[i], symbols[i + 1]);
            if let Some(&rank) = merge_ranks.get(pair.as_str()) {
                if best.is_none_or(|(best_rank, _)| rank < best_rank) {
                    best = Some((rank, i));
                }
            }
        }

        let Some((_, i)) = best else { break };

        let merged = symbols[i].clone() + &symbols[i + 1];
        symbols[i] = merged;
        symbols.remove(i + 1);
    }

    /*  The correct way to do this is load the BOS from the metadata:
        "tokenizer.ggml.bos_token_id": GGUF_METADATA_VALUE_U32(1)
        For now we'll leave this hardcoded in.
    */
    let mut token_ids = vec![1];

    for symbol in symbols {
        if let Some(&id) = vocab_map.get(symbol.as_str()) {
            token_ids.push(id);
        } else {
            // Byte fallback, anything not in the vocab becomes its raw utf8
            // bytes as <0xXX> tokens, unk (0) only if even those are missing
            for byte in symbol.bytes() {
                let byte_token = format!("<0x{byte:02X}>");
                token_ids.push(vocab_map.get(byte_token.as_str()).copied().unwrap_or(0));
            }
        }
    }

    Ok(token_ids)
}

// Pure inverse of encode
pub fn decode(vocab: &MetaDataField, ids: &[usize]) -> Result<String, TokenizerError> {
    let GGUF_METADATA_VALUE_ARRAY(vocab_array) = &vocab.value else {
        return Err(TokenizerError::WrongType("array of tokens"));
    };

    // Collect bytes not chars, byte fallback tokens can split
    // a single utf8 char across several tokens
    let mut bytes: Vec<u8> = vec![];

    for id in ids {
        let Some(GGUF_METADATA_VALUE_STRING(token)) = vocab_array.get(*id) else {
            return Err(TokenizerError::UnknownId(*id));
        };

        if let Some(byte) = parse_byte_token(token) {
            bytes.push(byte);
        } else {
            bytes.extend(token.replace('▁', " ").bytes());
        }
    }

    Ok(String::from_utf8_lossy(&bytes).to_string())
}

// Byte tokens look like <0x0A> in the vocab and carry one raw byte
fn parse_byte_token(token: &str) -> Option<u8> {
    let hex = token.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(hex, 16).ok()
}

// Sampler, logits -> token
// Greedy is argmax over the raw logits
pub fn sample(logits: &[f32]) -> usize {
    #[cfg(debug_assertions)]
    {
        let mut top: Vec<_> = logits.iter().enumerate().collect();
        top.sort_by(|a, b| b.1.total_cmp(a.1));
        println!("Top 5 {:?}: ", &top[0..5.min(top.len())]);
    }

    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(id, _)| id)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::GgufMetadataValueType::GGUF_METADATA_VALUE_TYPE_ARRAY;

    // Tiny hand built vocab, enough to exercise every path
    fn mock_field(values: &[&str]) -> MetaDataField {
        MetaDataField {
            name: "mock".to_string(),
            vtype: GGUF_METADATA_VALUE_TYPE_ARRAY,
            value: GGUF_METADATA_VALUE_ARRAY(
                values
                    .iter()
                    .map(|v| GGUF_METADATA_VALUE_STRING(v.to_string()))
                    .collect(),
            ),
        }
    }

    fn mock_vocab() -> MetaDataField {
        mock_field(&[
            "<unk>", "<s>", "</s>", "<0x21>", "▁", "a", "b", "▁a", "ab", "▁ab", "▁b",
        ])
    }

    fn mock_merges() -> MetaDataField {
        mock_field(&["▁ a", "a b", "▁a b", "▁ b"])
    }

    #[test]
    fn encode_merge_order_test() {
        // "▁ a" outranks "a b", so "ab" must become ▁a+b then ▁ab, not ▁+ab
        let ids = encode(&mock_vocab(), &mock_merges(), "ab").unwrap();

        assert_eq!(ids, vec![1, 9], "expected [bos, ▁ab]");
    }

    #[test]
    fn encode_whitespace_test() {
        // Double space has to survive as its own ▁, the old encoder dropped it
        let ids = encode(&mock_vocab(), &mock_merges(), "a  b").unwrap();

        assert_eq!(ids, vec![1, 7, 4, 10], "expected [bos, ▁a, ▁, ▁b]");
    }

    #[test]
    fn encode_byte_fallback_test() {
        // "!" is not in the mock vocab as a char, only as the byte token <0x21>
        let ids = encode(&mock_vocab(), &mock_merges(), "a b!").unwrap();

        assert_eq!(ids, vec![1, 7, 10, 3], "expected [bos, ▁a, ▁b, <0x21>]");
    }

    #[test]
    fn round_trip_test() {
        // Skip BOS, then the only artifact left is the sentencepiece prefix space
        let ids = encode(&mock_vocab(), &mock_merges(), "a b!").unwrap();
        let text = decode(&mock_vocab(), &ids[1..]).unwrap();

        assert_eq!(text, " a b!");
    }

    #[test]
    fn decode_unknown_id_test() {
        let result = decode(&mock_vocab(), &[999]);

        assert!(matches!(result, Err(TokenizerError::UnknownId(999))));
    }

    #[test]
    fn sample_test() {
        assert_eq!(sample(&[-1.0, 4.0, 2.0]), 1);
    }
}
