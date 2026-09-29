use half::f16;
use rayon::prelude::*;

use crate::reader::convert_q8;

// Dot Product
pub fn dot_prod_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;

    for x in a.iter().enumerate() {
        sum += x.1 * b[x.0]
    }

    sum
}

// Dot Product Q8
pub fn dot_prod_q8(a: &[f32], b: &[u8]) -> f32 {
    let mut sum = 0.0;

    // We walk 32 f32s values per block of b
    let zipper = a.chunks(32).zip(b.chunks(34));

    for (block_a, block_b) in zipper {
        let scale = f16::from_le_bytes([block_b[0], block_b[1]]);

        let block_sum: f32 = block_a
            .iter()
            .zip(&block_b[2..])
            .map(|(a, &b)| a * (b as i8) as f32)
            .sum();
        sum += block_sum * f32::from(scale)
    }

    sum
}

// Dot Product Activation Q8
pub fn dot_prod_act_q8(a: &[u8], b: &[u8]) -> f32 {
    let zipper = a.chunks_exact(34).zip(b.chunks_exact(34)); 

    zipper.map(|(block_a, block_b)| {
        let scale_a = f32::from(f16::from_le_bytes([block_a[0], block_a[1]]));
        let scale_b = f32::from(f16::from_le_bytes([block_b[0], block_b[1]]));

        let block_sum: f32 = block_a[2..].iter().zip(&block_b[2..])
        .map(|(a, &b)| *a as i8 as f32 * b as i8 as f32).sum();
        block_sum * scale_a * scale_b
    }).sum()
}

// True parallel? No. par_chunks is unneeded overhead and flat map iter
// is not acutally parallel
pub fn par_f32_matmul_flat(a: &[f32], b: &[f32], shape: &[u64; 2]) -> Vec<f32> {
    let size = shape[1] as usize;

    a.par_chunks(size)
        .flat_map_iter(|vec_a| b.chunks(size).map(move |vec_b| dot_prod_f32(vec_a, vec_b)))
        .collect()
}

// Vector * Mat, really the only operation here so make it explicit
pub fn v_matmul(a: &[f32], b: &[f32], out: &mut [f32]) {
    let b_chunks = b.par_chunks(64 * a.len());
    let c_chunks = out.par_chunks_mut(64);

    let zipped = b_chunks.zip(c_chunks);
    
    let _ = zipped.into_par_iter().for_each(|(b, c)| {
        c[0] = dot_prod_f32(&a, b);
    });

}

// Q8 Matrix Mul
// Old Q8 Mat mul
pub fn old_q8_matmul(a: &[f32], b: &[u8], k: usize) -> Vec<f32> {
    a.par_chunks(k)
        .flat_map_iter(|vec_a| {
            b.chunks(k / 32 * 34)
                .map(move |vec_b| dot_prod_q8(vec_a, vec_b))
        })
        .collect()
}

// Proper par Q8 v_mul
pub fn q8_matmul(a: &[f32], b: &[u8], out: &mut [f32], k: usize) {
    let b_chunks = b.par_chunks(k / 32 * 34 * 64);
    let c_chunks = out.par_chunks_mut(64);

    let zipped = b_chunks.zip(c_chunks);

    zipped.for_each(|(b, c)| {
        b.par_chunks(k / 32 * 34).zip(c).for_each(|(b, c)| {
            *c = dot_prod_q8(a, b)
        });
    });
}

// Q8 Activation matmul
pub fn q8_act_matmul(a: &[u8], b: &[u8], out: &mut [f32], k: usize) {
    let b_chunks = b.par_chunks(k / 32 * 34 * 64);
    let c_chunks = out.par_chunks_mut(64);

    let zipped = b_chunks.zip(c_chunks);

    zipped.for_each(|(b, c)| {
        b.par_chunks(k / 32 * 34).zip(c).for_each(|(b, c)| {
            *c = dot_prod_act_q8(a, b)
        });
    });
}

// Q8 Dequant
// Given a Q8 block and a scale, dequant into
// a block of f32 values
pub fn q8_dequant_block(quantized: Vec<i8>, scale: f32) -> Vec<f32> {
    let dequant = quantized.iter().map(|x| *x as f32 * scale).collect();

    dequant
}

// Q8 Byte to Dequant
// Given raw Q8 bytes, dequantize all blocks
// Returns a flat vector
pub fn q8_full_dequant(raw_bytes: Vec<u8>) -> Vec<f32> {
    let mut full_dequant = Vec::new();

    let blocks = raw_bytes.chunks(34);

    for block in blocks {
        let (weights, scale) = convert_q8(block.to_vec());
        full_dequant.extend(q8_dequant_block(weights, scale))
    }

    full_dequant
}

// RMSNorm
// Root Mean Square Layer Normalization
pub fn rmsnorm(a: &[f32], b: &[f32]) -> Vec<f32> {
    let total: f32 = a.iter().map(|x| x * x).sum();

    let rms_value = ((total / a.len() as f32) + 1e-5).sqrt();

    a.iter()
        .enumerate()
        .map(|x| (x.1 * b[x.0]) / rms_value)
        .collect()
}

// Softmax
pub fn softmax(vector: &[f32]) -> Vec<f32> {
    let max = vector.iter().max_by(|a, b| a.total_cmp(b)).unwrap();

    let exp: Vec<f32> = vector.iter().map(|f| f32::exp(f - max)).collect();

    let total: f32 = exp.iter().sum();

    exp.iter().map(|e| e / total).collect()
}

// SwiGlu
pub fn swiglu(x: &[f32], w_gate: &[u8], w_up: &[u8], w_down: &[u8], out: &mut [f32], k: usize) {
    let mut gate= [0.0; 5632];
    q8_matmul(x, w_gate, &mut gate, k);

    let mut up= [0.0; 5632];
    q8_matmul(x, w_up, &mut up, k);

    let hidden = gate
        .par_iter()
        .enumerate()
        .map(|g| (g.1 / (1.0 + f32::exp(-g.1))) * up[g.0])
        .collect::<Vec<f32>>();

    q8_matmul(&hidden, w_down, out, hidden.len())
}

// Angles precompute
pub fn rope_angles(base: f32, pos: usize, dim: usize) -> Vec<f32> {
    let mut thetas: Vec<f32> = vec![];

    let mut head_dim: usize = 0;

    while head_dim != dim / 2 {
        let frequency = base.powf((-2.0 * head_dim as f32) / dim as f32);
        thetas.push(pos as f32 * frequency);
        head_dim += 1
    }

    thetas
}

// Rope
pub fn rope(query: &mut [f32], key: &mut [f32], thetas: &[f32], dim: usize) {
    // Chunk per attention head, rope's dim from the gguf is the head size
    let query_chunks = query.chunks_mut(dim);
    let key_chunks = key.chunks_mut(dim);

    for chunk in query_chunks {
        let mut pairs = chunk.chunks_mut(2);

        for angle in thetas.clone() {
            let pair_vec = &mut pairs.next().unwrap();

            let encode_0 = angle.cos() * pair_vec[0] + -angle.sin() * pair_vec[1];
            let encode_1 = angle.sin() * pair_vec[0] + angle.cos() * pair_vec[1];
            
            pair_vec[0] = encode_0;
            pair_vec[1] = encode_1;
        }
    }

    for chunk in key_chunks {
        let mut pairs = chunk.chunks_mut(2);

        for angle in thetas.clone() {
            let pair_vec = &mut pairs.next().unwrap();

            let encode_0 = angle.cos() * pair_vec[0] + -angle.sin() * pair_vec[1];
            let encode_1 = angle.sin() * pair_vec[0] + angle.cos() * pair_vec[1];
            
            pair_vec[0] = encode_0;
            pair_vec[1] = encode_1;
        }
    }
}

// Tests

#[cfg(test)]
fn assert_close(out: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(out.len(), expected.len(), "length mismatch");

    for (i, (o, e)) in out.iter().zip(expected).enumerate() {
        assert!((o - e).abs() < tol, "index {i}: {o} vs {e}");
    }
}

#[cfg(test)]
fn q8_block(scale: f32, weights: &[i8; 32]) -> Vec<u8> {
    let mut block = f16::from_f32(scale).to_le_bytes().to_vec();
    block.extend(weights.iter().map(|w| *w as u8));
    block
}

#[test]
fn dot_product_test() {
    let a: Vec<f32> = vec![1.0, 2.0, 3.0];
    let b: Vec<f32> = vec![2.0, 4.0, 6.0];

    let product: f32 = 28.0;
    assert_eq!(dot_prod_f32(&a, &b), product, "Basic dot product");
}

#[test]
fn par_matrix_mul_flat_test() {
    let a = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];

    let b = vec![2.0, 2.0, 2.0];

    let c = vec![12.0, 12.0];

    assert_eq!(par_f32_matmul_flat(&a, &b, &[2, 3]), c, "Simple Matrix");

    // 10x3
    let a_comp = vec![
        1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0,
        1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0,
    ];

    // 3x1
    let b_comp = vec![2.0, 2.0, 2.0];

    // 10x1
    let c_comp = vec![12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0];

    assert_eq!(
        par_f32_matmul_flat(&a_comp, &b_comp, &[10, 3]),
        c_comp,
        "Bigger Matrix Matrix"
    );
}

#[test]
fn q8_dequant_test() {
    // First output.weights from Tiny Llama
    let _raw_byte_block = [
        71, 11, 56, 154, 145, 253, 194, 245, 46, 247, 37, 39, 219, 21, 71, 189, 38, 242, 236, 129,
        4, 69, 239, 47, 18, 230, 203, 25, 234, 250, 220, 240, 29, 241,
    ];

    // Scalar:
    let scale = 0.0002220869;

    // Quantized values
    let quantized_int8_vals = [
        56, -102, -111, -3, -62, -11, 46, -9, 37, 39, -37, 21, 71, -67, 38, -14, -20, -127, 4, 69,
        -17, 47, 18, -26, -53, 25, -22, -6, -36, -16, 29, -15,
    ];

    // Dequantized f32 values
    let dequantized_vals = [
        0.01243687,
        -0.02265286,
        -0.02465165,
        -0.00066626,
        -0.01376939,
        -0.00244296,
        0.010216,
        -0.00199878,
        0.00821722,
        0.00866139,
        -0.00821722,
        0.00466383,
        0.01576817,
        -0.01487982,
        0.0084393,
        -0.00310922,
        -0.00444174,
        -0.02820504,
        0.00088835,
        0.015324,
        -0.00377548,
        0.01043808,
        0.00399756,
        -0.00577426,
        -0.01177061,
        0.00555217,
        -0.00488591,
        -0.00133252,
        -0.00799513,
        -0.00355339,
        0.00644052,
        -0.0033313,
    ];

    // Expected values were captured at 8 decimals, so compare with tolerance
    assert_close(
        &q8_dequant_block(quantized_int8_vals.to_vec(), scale),
        &dequantized_vals,
        1e-6,
    );
}

#[test]
fn dot_prod_q8_test() {
    // Scale 1.0 is exact in f16, weights are the row index
    let weights: [i8; 32] = core::array::from_fn(|i| i as i8);
    let block = q8_block(1.0, &weights);

    let a = vec![2.0; 32];

    // 2 * (0 + 1 + ... + 31) = 992
    assert_eq!(dot_prod_q8(&a, &block), 992.0);
}

#[test]
fn q8_matmul_test() {
    // Two rows of 64, so two Q8 blocks per row, scales exact in f16
    let w1: [i8; 32] = core::array::from_fn(|i| i as i8 - 16);
    let w2: [i8; 32] = core::array::from_fn(|i| 8 - i as i8);

    let mut b = vec![];
    b.extend(q8_block(0.5, &w1));
    b.extend(q8_block(0.25, &w2));
    b.extend(q8_block(1.5, &w2));
    b.extend(q8_block(2.0, &w1));

    let a: Vec<f32> = (0..64).map(|i| i as f32 / 16.0 - 2.0).collect();

    let b_f32 = q8_full_dequant(b.clone());
    let expected = par_f32_matmul_flat(&a, &b_f32, &[2, 64]);
    let mut out = [0.0;2];
    q8_matmul(&a, &b, &mut out, 64);

    assert_close(&out, &expected, 1e-4);
}

#[test]
fn rms_norm_test() {
    let vector_a = vec![3.0, 5.0, 8.0];
    let vector_b = vec![2.0, 4.0, 6.0];

    let expected = [1.049_781_318, 3.499_271_061, 8.398_250_546];

    assert_close(&rmsnorm(&vector_a, &vector_b), &expected, 1e-5)
}

#[test]
fn softmax_test() {
    let u_flat = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];

    let expected = vec![
        0.04501528658519023,
        0.12236423552739882,
        0.3326204778874109,
        0.04501528658519023,
        0.12236423552739882,
        0.3326204778874109,
    ];

    let out = softmax(&u_flat);

    assert_close(&out, &expected, 1e-6);

    // The one property softmax must always hold
    let total: f32 = out.iter().sum();
    assert!((total - 1.0).abs() < 1e-6, "softmax must sum to 1.0")
}

#[test]
fn rope_test() {
    // One head of size 4 at pos 1, small enough to check by hand
    let mut query = vec![1.0, 0.0, 0.0, 1.0];
    let mut key: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    let pos = 1;
    let dim = 4;
    let base: f32 = 10000.0;

    let theta = rope_angles(base, pos, dim);

    rope(&mut query, &mut key, &theta, dim);
    let expected = [0.5403023, 0.841471, -0.0099998, 0.99995];

    assert_close(&query, &expected, 1e-4);
    // Same input and head size, so key must rotate identically
    assert_close(&key, &expected, 1e-4)
}

/* 
#[test]
fn swiglu_test() {
    let x: Vec<f32> = (0..32).map(|i| i as f32 / 16.0 - 1.0).collect();

    let mut w_gate = vec![];
    let mut w_up = vec![];
    let mut w_down = vec![];

    for row in 0..32i8 {
        w_gate.extend(q8_block(
            1.0,
            &core::array::from_fn(|i| (row + i as i8) % 5 - 2),
        ));
        w_up.extend(q8_block(
            1.0,
            &core::array::from_fn(|i| (row + i as i8) % 7 - 3),
        ));
        w_down.extend(q8_block(
            1.0,
            &core::array::from_fn(|i| (row + i as i8) % 3 - 1),
        ));
    }

    // Reference is the same formula down the f32 path
    let gate = par_f32_matmul_flat(&x, &q8_full_dequant(w_gate.clone()), &[32, 32]);
    let up = par_f32_matmul_flat(&x, &q8_full_dequant(w_up.clone()), &[32, 32]);

    let hidden: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(g, u)| (g / (1.0 + f32::exp(-g))) * u)
        .collect();

    let expected = par_f32_matmul_flat(&hidden, &q8_full_dequant(w_down.clone()), &[32, 32]);
    let mut out = [0.0; 32];
    swiglu(&x, &w_gate, &w_up, &w_down, &mut out, 32);
    println!("Expected: {:?}", out);

    assert_close(&out, &expected, 1e-3)
}
*/