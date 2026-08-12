use thiserror;

// Low level byte reading errors, anything that means the file itself is bad
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ReaderError {
    #[error("unexpected end of file reading {0} at offset {1}")]
    UnexpectedEof(&'static str, usize),

    #[error("invalid utf8 in string at offset {0}")]
    InvalidUtf8(usize),

    #[error("bad magic number, not a gguf file")]
    BadMagic,

    #[error("unsupported gguf version: {0}")]
    BadVersion(u32),

    #[error("unsupported metadata value type: {0}")]
    UnsupportedType(u32),

    #[error("unsupported tensor dtype: {0}")]
    UnsupportedDtype(u32),

    #[error("q8_0 tensor length {0} is not a multiple of 32")]
    BadQuantBlock(u64),
}

#[derive(Debug, thiserror::Error)]
pub enum MetaDataError {
    #[error("missing required field: {0}")]
    MissingField(&'static str),

    #[error("field has wrong type; expected {expec}")]
    WrongType {
        field: &'static str,
        expec: &'static str,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("tokenizer field has wrong type; expected {0}")]
    WrongType(&'static str),

    #[error("token id {0} is not in the vocab")]
    UnknownId(usize),
}

#[derive(Debug, thiserror::Error)]
pub enum ModelLoadError {
    #[error("Metadata Load error: {0}")]
    MetaDataError(#[from] MetaDataError),

    #[error("Tensor Weight Data Load Error: {0}")]
    TensorDataError(String),

    #[error("Reader Error: {0}")]
    ReaderError(#[from] ReaderError),
}

#[derive(Debug, thiserror::Error)]
pub enum ForwardPassError {
    #[error("unable to load tensor: {0}")]
    TensorLoadError(String),
}

#[derive(Debug, thiserror::Error)]
pub enum GenerateError {
    #[error("Forward Pass Error: {0}")]
    ForwardError(#[from] ForwardPassError),

    #[error("Tokenizer Error: {0}")]
    TokenizerError(#[from] TokenizerError),

    #[error("Metadata Load error: {0}")]
    MetaDataError(#[from] MetaDataError),
}
