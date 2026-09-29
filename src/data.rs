// Tensor Data
#[derive(Debug, PartialEq, Clone)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub n_elements: u32,
    pub dtype: u32,
    pub offset: [u8; 8],
}

// Forward Pass Modes
// Temporary, a way to toggle between quantizing the activation vs running normally
// Quantizing the activation vector is actually slower on normal, but will be faster with SIMD
// This will eventually evolve into a true selector for Q8, Q6, etc
#[derive(Default, Debug, PartialEq, Clone)]
pub enum Mode {
    #[default]
    Normal,
    Quantize
}

// Transformer Data
// dim is the rope/head dimension, embed_dim is the model width
#[derive(Debug, PartialEq, Clone)]
pub struct Transformer {
    pub layers: u32,
    pub dim: usize,
    pub embed_dim: usize,
    pub q_heads: usize,
    pub k_v_heads: usize,
    pub base_freq: f32,
    pub mode: Mode
}

#[derive(Debug, PartialEq, Clone)]
pub struct TransformerWeights {
    pub token_embed: f32,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Cache {
    pub key_cache: Vec<f32>,
    pub value_cache: Vec<f32>
}


impl Cache {
    pub fn new() -> Self {
        Cache { key_cache: vec![], value_cache: vec![] }
    }
}
