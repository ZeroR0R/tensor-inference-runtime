// Tensor Data
#[derive(Debug, PartialEq, Clone)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub n_elements: u32,
    pub dtype: u32,
    pub offset: [u8; 8],
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
}

#[derive(Debug, PartialEq, Clone)]
pub struct TransformerWeights {
    pub token_embed: f32,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Cache {
    pub layer_id: u32,
    pub head_count: u32,
    pub heads: Vec<Vec<f32>>,
}
