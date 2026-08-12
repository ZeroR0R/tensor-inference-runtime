// GGUF Header
#[derive(Debug)]
pub struct Header<'a> {
    pub m_number: &'a [u8; 4],
    pub version: &'a [u8; 4],
    pub tensor_count: &'a [u8; 8],
    pub md_k_v_pairs: &'a [u8; 8],
}

#[derive(Debug, PartialEq, Clone)]
pub struct MetaDataField {
    pub name: String,
    pub vtype: GgufMetadataValueType,
    pub value: GgufMetadataValue,
}

// Names mirror the gguf spec on purpose, so allow the shouting
#[allow(non_camel_case_types)]
#[derive(Debug, PartialEq, PartialOrd, Clone)]
pub enum GgufMetadataValue {
    GGUF_METADATA_VALUE_U8(u8),
    GGUF_METADATA_VALUE_I8(i8),
    GGUF_METADATA_VALUE_U16(u16),
    GGUF_METADATA_VALUE_I16(i16),
    GGUF_METADATA_VALUE_U32(u32),
    GGUF_METADATA_VALUE_I32(i32),
    GGUF_METADATA_VALUE_F32(f32),

    GGUF_METADATA_VALUE_BOOL(bool),
    GGUF_METADATA_VALUE_STRING(String),
    GGUF_METADATA_VALUE_ARRAY(Vec<GgufMetadataValue>),

    GGUF_METADATA_VALUE_U64(u64),
    GGUF_METADATA_VALUE_I64(i64),
    GGUF_METADATA_VALUE_F64(f64),
}

#[allow(non_camel_case_types)]
#[derive(Debug, PartialEq, Clone)]
pub enum GgufMetadataValueType {
    GGUF_METADATA_VALUE_TYPE_U8 = 0,
    GGUF_METADATA_VALUE_TYPE_I8 = 1,
    GGUF_METADATA_VALUE_TYPE_U16 = 2,
    GGUF_METADATA_VALUE_TYPE_I16 = 3,
    GGUF_METADATA_VALUE_TYPE_U32 = 4,
    GGUF_METADATA_VALUE_TYPE_I32 = 5,
    GGUF_METADATA_VALUE_TYPE_F32 = 6,

    GGUF_METADATA_VALUE_TYPE_BOOL = 7,
    GGUF_METADATA_VALUE_TYPE_STRING = 8,
    GGUF_METADATA_VALUE_TYPE_ARRAY = 9,

    GGUF_METADATA_VALUE_TYPE_U64 = 10,
    GGUF_METADATA_VALUE_TYPE_I64 = 11,
    GGUF_METADATA_VALUE_TYPE_F64 = 12,
}
