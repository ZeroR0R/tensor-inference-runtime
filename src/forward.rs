// Forward pass through the model
use crate::{
    data::{Cache, Mode, Transformer},
    error::ForwardPassError,
    kernels::activated_q8,
    math::{dot_prod_f32, q8_act_matmul, q8_matmul, rmsnorm, rope, softmax, swiglu},
    reader::convert_f32,
};

use rayon::prelude::*;
use std::{collections::HashMap};

pub fn forward(
    mut token: Vec<f32>,
    pos: usize,
    transformer_info: &Transformer,
    tensor_data_map: &HashMap<String, &[u8]>,
    kv_cache: &mut Cache,
    angles: Vec<f32>
) -> Result<(Vec<f32>), ForwardPassError> {
    // Define values from GGUF
    let dim = transformer_info.dim;
    let layers = transformer_info.layers;
    let mut layer_count = 0;

    // Token embed check
    #[cfg(debug_assertions)]
    save_intermediate(&token, "embed");

    while layer_count != layers {
        let rms_norm_attn_w = get_tensor(
            tensor_data_map,
            &format!("blk.{layer_count}.attn_norm.weight"),
        )?;

        let o = rmsnorm(&token, &convert_f32(&rms_norm_attn_w));

        #[cfg(debug_assertions)]
        save_intermediate(&o, "Object");

        let query_weight =
            get_tensor(tensor_data_map, &format!("blk.{layer_count}.attn_q.weight"))?;

        // We use o's length here, as llama is mostly vec * matrix vs
        // matrix * matrix. Though this is a limitation.
        let mut query = [0.0; 2048];

        matmul(&o, &query_weight, &mut query, o.len(), &transformer_info);

        let key_weight = get_tensor(tensor_data_map, &format!("blk.{layer_count}.attn_k.weight"))?;
        let value_weight =
            get_tensor(tensor_data_map, &format!("blk.{layer_count}.attn_v.weight"))?;

        let mut key = [0.0; 256];
        matmul(&o, &key_weight, &mut key, o.len(), &transformer_info);

        let mut value = [0.0; 256];
        matmul(&o, &value_weight, &mut value, o.len(), transformer_info);

        // Query is now [2048], key and value are [256]

        // Prints
        #[cfg(debug_assertions)]
        {
            save_intermediate(&query, "Query");
            save_intermediate(&key, "key");
            save_intermediate(&value, "value");
        }

        // ROPE
        rope(&mut query, &mut key, &angles, dim);

        #[cfg(debug_assertions)]
        {
            save_intermediate(&query, "Query ROPE");
            save_intermediate(&key, "key ROPE");
        }

        // Store k_v len for future use (they are pairs, k_len == v_len here)
        let k_v_len = key.len();

        // Add the current key/value to their respective caches
        kv_cache.key_cache.extend(key);
        kv_cache.value_cache.extend(value);

        // Grouped Query Attention
        // 2048 / 32 is 32 chunks of 64 values
        let attention_query: Vec<_> = query
            .chunks(query.len() / transformer_info.q_heads)
            .collect();

        // Weighted values
        let mut weighted_values = vec![];

        // 4 chunks with 8 vectors, each with 64 elements
        let query_chunks =
            attention_query.chunks(transformer_info.q_heads / transformer_info.k_v_heads);


        let mut keys:Vec<_> = vec![];
        let mut offset = pos + 1;
        while offset != 0 {
            offset -= 1;

            let pos_start = offset as u32 * transformer_info.layers * k_v_len as u32;
            let range_start = (layer_count * 256 + pos_start) as usize;

            keys.push(&kv_cache.key_cache[range_start..range_start+256]);
        }

        let mut values:Vec<_> = vec![];
        let mut offset = pos + 1;
        while offset != 0 {
            offset -= 1;

            let pos_start = offset as u32 * transformer_info.layers * k_v_len as u32;
            let range_start = (layer_count * 256 + pos_start) as usize;

            values.push(&kv_cache.value_cache[range_start..range_start+256]);
        }


        let mut key_heads: Vec<Vec<f32>> = vec![];
        let mut value_heads: Vec<Vec<f32>> = vec![];

        for key in keys {
            let heads = key.chunks(key.len() / transformer_info.k_v_heads);

            for head in heads.enumerate() {
                if key_heads.get(head.0).is_some() {
                    key_heads[head.0].extend(head.1);
                } else {
                    key_heads.push(head.1.to_vec());
                }
            }
        }

        for value in values {
            let heads = value.chunks(value.len() / transformer_info.k_v_heads);

            for head in heads.enumerate() {
                if value_heads.get(head.0).is_some() {
                    value_heads[head.0].extend(head.1)
                } else {
                    value_heads.push(head.1.to_vec())
                }
            }
        }

        for head in query_chunks.enumerate() {
            let current_key_head = key_heads[head.0].clone();
            let current_value_head = value_heads[head.0].clone();

            for vector in head.1 {
                let mut scores = vec![];
                for khead in current_key_head.chunks(dim) {
                    let current = dot_prod_f32(vector, khead) / f32::sqrt(dim as f32);
                    scores.push(current);
                }

                scores = softmax(&scores);
                let mut weighted: Vec<f32> = vec![0.0; dim];

                for vhead in current_value_head.chunks(dim).enumerate() {
                    weighted = weighted
                        .iter()
                        .enumerate()
                        .map(|v| v.1 + vhead.1[v.0] * scores[vhead.0])
                        .collect();
                }

                weighted_values.extend(weighted);
            }
        }

        // Final matmul to get output of attention
        let attn_out_weight = get_tensor(
            tensor_data_map,
            &format!("blk.{layer_count}.attn_output.weight"),
        )?;

        #[cfg(debug_assertions)]
        save_intermediate(&weighted_values, "weighted values");

        let mut f_out = [0.0; 2048];

        matmul(&weighted_values, &attn_out_weight, &mut f_out, weighted_values.len(), &transformer_info);

        #[cfg(debug_assertions)]
        save_intermediate(&f_out, "Fout");

        token = token.iter().enumerate().map(|x| x.1 + f_out[x.0]).collect();

        // RMSNorm FFN
        let ffn_rmsnorm_weight = get_tensor(
            tensor_data_map,
            &format!("blk.{layer_count}.ffn_norm.weight"),
        )?;
        let ffn_normalized = rmsnorm(&token, &convert_f32(ffn_rmsnorm_weight));

        #[cfg(debug_assertions)]
        save_intermediate(&ffn_normalized, "ffn norm");

        // FFN
        let ffn_gate_weight = get_tensor(
            tensor_data_map,
            &format!("blk.{layer_count}.ffn_gate.weight"),
        )?;
        let ffn_up_weight =
            get_tensor(tensor_data_map, &format!("blk.{layer_count}.ffn_up.weight"))?;
        let ffn_down_weight = get_tensor(
            tensor_data_map,
            &format!("blk.{layer_count}.ffn_down.weight"),
        )?;

        // SwiGLU
        let mut activated = [0.0;2048];
        
        swiglu(
            &ffn_normalized,
            &ffn_gate_weight,
            &ffn_up_weight,
            &ffn_down_weight,
            &mut activated,
            o.len(),
        );

        #[cfg(debug_assertions)]
        save_intermediate(&activated, "activated");

        // Residual Connection
        token = token
            .par_iter()
            .enumerate()
            .map(|x| x.1 + activated[x.0])
            .collect();

        layer_count += 1
    }

    // Final RMSNorm output_norm.weight
    let final_output_norm = get_tensor(tensor_data_map, "output_norm.weight")?;
    token = rmsnorm(&token, &convert_f32(final_output_norm));

    #[cfg(debug_assertions)]
    save_intermediate(&token, "final");

    // Logit Classifier:
    let output_weight = get_tensor(tensor_data_map, "output.weight")?;
    let mut logits = [0.0; 32000];

    matmul(&token, &output_weight, &mut logits, token.len(), &transformer_info);
    
    #[cfg(debug_assertions)]
    save_intermediate(&logits, "final");

    Ok((logits.to_vec()))
}

// Pull this out so its easier to handle multiple matmul flags
fn matmul(activation: &[f32], weight: &[u8], mut out: &mut [f32], len: usize, transformer_info: &Transformer) {
    match &transformer_info.mode {
        Mode::Normal => {
            q8_matmul(activation, weight, out, len);
        },
        Mode::Quantize => {
            let mut quantized = [0;2176];

            // Quantize activation
            activated_q8(&activation, &mut quantized);

            q8_act_matmul(&quantized, &weight, &mut out, len);
        }
    }
}

fn get_tensor<'a>(
    data_map: &'a HashMap<String, &[u8]>,
    block: &str,
) -> Result<&'a [u8], ForwardPassError> {
    data_map
        .get(block)
        .copied()
        .ok_or_else(|| ForwardPassError::TensorLoadError(block.to_owned()))
}

// Debug only, print summary stats per step to diff against llama.cpp's intermediates
#[allow(dead_code)]
fn save_intermediate(tensor: &[f32], step: &str) {
    println!("Tensor: {step}");
    let sum: f32 = tensor.iter().sum();
    let max = tensor.iter().max_by(|a, b| a.total_cmp(b));
    let min = tensor.iter().max_by(|a, b| b.total_cmp(a));

    let abs_sum: f32 = tensor.iter().map(|t| t.abs()).sum();
    println!(
        "Size: {:?}, Total: {:}, ABS_SUM: {}, Avg: {}, Min: {}, Max: {}",
        tensor.len(),
        sum,
        abs_sum,
        sum / tensor.len() as f32,
        min.unwrap(),
        max.unwrap()
    );
    let mut first_vals = tensor.chunks(5);
    println!("First 5: ");
    println!("{:?}", first_vals.next())
}

// Same but dumps the whole tensor, kept around for hunting layer level bugs
#[allow(dead_code)]
fn save_intermediate_full(tensor: Vec<f32>, step: &str) {
    println!("Tensor: {step}");
    let sum: f32 = tensor.iter().sum();
    let max = tensor.iter().max_by(|a, b| a.total_cmp(b));
    let min = tensor.iter().max_by(|a, b| b.total_cmp(a));
    println!(
        "Size: {:?}, Total: {:}, Avg: {}, Min: {}, Max: {}",
        tensor.len(),
        sum,
        sum / tensor.len() as f32,
        min.unwrap(),
        max.unwrap()
    );
    println!("Tensor total: {:?}", &tensor)
}
