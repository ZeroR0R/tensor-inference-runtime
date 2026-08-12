// Forward pass through the model
use crate::{
    data::Transformer,
    error::ForwardPassError,
    math::{dot_prod_f32, q8_matmul, rmsnorm, rope, softmax, swiglu},
    reader::convert_f32,
};

use rayon::prelude::*;
use std::collections::HashMap;

pub fn forward(
    mut token: Vec<f32>,
    pos: usize,
    transformer_info: &Transformer,
    tensor_data_map: &HashMap<String, &[u8]>,
    mut key_cache: Vec<Vec<f32>>,
    mut value_cache: Vec<Vec<f32>>,
) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>), ForwardPassError> {
    // Define values from GGUF
    let dim = transformer_info.dim;
    let base_freq: f32 = transformer_info.base_freq;
    let layers = transformer_info.layers;
    let mut layer_count = 0;

    // Token embed check
    #[cfg(debug_assertions)]
    save_intermediate(token.clone(), "embed");

    while layer_count != layers {
        let rms_norm_attn_w = get_tensor(
            tensor_data_map,
            &format!("blk.{layer_count}.attn_norm.weight"),
        )?;

        let o = rmsnorm(&token, &convert_f32(&rms_norm_attn_w));

        #[cfg(debug_assertions)]
        save_intermediate(o.clone(), "Object");

        let query_weight =
            get_tensor(tensor_data_map, &format!("blk.{layer_count}.attn_q.weight"))?;

        // We use o's length here, as llama is mostly vec * matrix vs
        // matrix * matrix. Though this is a limitation.
        let mut query = q8_matmul(&o, &query_weight, o.len());

        let key_weight = get_tensor(tensor_data_map, &format!("blk.{layer_count}.attn_k.weight"))?;
        let value_weight =
            get_tensor(tensor_data_map, &format!("blk.{layer_count}.attn_v.weight"))?;

        let mut key = q8_matmul(&o, &key_weight, o.len());
        let value = q8_matmul(&o, &value_weight, o.len());

        // Query is now [2048], key and value are [256]

        // Prints
        #[cfg(debug_assertions)]
        {
            save_intermediate(query.clone(), "Query");
            save_intermediate(key.clone(), "key");
            save_intermediate(value.clone(), "value");
        }

        // ROPE
        (query, key) = rope(&query, &key, base_freq, pos, dim);

        #[cfg(debug_assertions)]
        {
            save_intermediate(query.clone(), "Query ROPE");
            save_intermediate(key.clone(), "key ROPE");
        }

        // Store k_v len for future use (they are pairs, k_len == v_len here)
        let k_v_len = key.len();

        // Add the current key/value to their respective caches
        if key_cache.get(layer_count as usize).is_some() {
            key_cache[layer_count as usize].extend(key);
        } else {
            key_cache.push(key)
        }

        if value_cache.get(layer_count as usize).is_some() {
            value_cache[layer_count as usize].extend(value);
        } else {
            value_cache.push(value);
        }

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

        let keys = key_cache[layer_count as usize].chunks(k_v_len);
        let values = value_cache[layer_count as usize].chunks(k_v_len);

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
                for key in current_key_head.chunks(dim) {
                    let current = dot_prod_f32(vector, key) / f32::sqrt(dim as f32);
                    scores.push(current);
                }

                scores = softmax(&scores);
                let mut weighted: Vec<f32> = vec![0.0; dim];

                for value in current_value_head.chunks(dim).enumerate() {
                    weighted = weighted
                        .iter()
                        .enumerate()
                        .map(|v| v.1 + value.1[v.0] * scores[value.0])
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
        save_intermediate(weighted_values.to_vec().clone(), "weighted values");

        let f_out = q8_matmul(&weighted_values, &attn_out_weight, weighted_values.len());

        #[cfg(debug_assertions)]
        save_intermediate(f_out.clone(), "Fout");

        token = token.iter().enumerate().map(|x| x.1 + f_out[x.0]).collect();

        // RMSNorm FFN
        let ffn_rmsnorm_weight = get_tensor(
            tensor_data_map,
            &format!("blk.{layer_count}.ffn_norm.weight"),
        )?;
        let ffn_normalized = rmsnorm(&token, &convert_f32(ffn_rmsnorm_weight));

        #[cfg(debug_assertions)]
        save_intermediate(ffn_normalized.clone(), "ffn norm");

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
        let activated = swiglu(
            &ffn_normalized,
            &ffn_gate_weight,
            &ffn_up_weight,
            &ffn_down_weight,
            o.len(),
        );

        #[cfg(debug_assertions)]
        save_intermediate(activated.clone(), "activated");

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
    save_intermediate(token.clone(), "final");

    // Logit Classifier:
    let output_weight = get_tensor(tensor_data_map, "output.weight")?;
    let logits = q8_matmul(&token, &output_weight, token.len());

    Ok((logits, key_cache, value_cache))
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
fn save_intermediate(tensor: Vec<f32>, step: &str) {
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
