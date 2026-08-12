use std::{fs::File, io::Error};

use half::f16;

use crate::{
    data::Tensor,
    error::ReaderError,
    state::{
        GgufMetadataValue, GgufMetadataValue::*, GgufMetadataValueType, GgufMetadataValueType::*,
        MetaDataField,
    },
};
use memmap2::Mmap;

pub fn read_gguf(path: &str) -> Result<Mmap, Error> {
    let file = File::open(path)?;

    let mmap = unsafe { Mmap::map(&file) };

    Ok(mmap?)
}

// Bounds checked slice, every read in this file goes through here
// A truncated or lying file becomes an error instead of a panic
fn take<'a>(
    map: &'a [u8],
    offset: usize,
    len: usize,
    what: &'static str,
) -> Result<&'a [u8], ReaderError> {
    let end = offset
        .checked_add(len)
        .ok_or(ReaderError::UnexpectedEof(what, offset))?;

    map.get(offset..end)
        .ok_or(ReaderError::UnexpectedEof(what, offset))
}

fn read_u32(map: &[u8], offset: usize, what: &'static str) -> Result<(u32, usize), ReaderError> {
    let bytes = take(map, offset, 4, what)?;
    Ok((u32::from_le_bytes(bytes.try_into().unwrap()), offset + 4))
}

fn read_u64(map: &[u8], offset: usize, what: &'static str) -> Result<(u64, usize), ReaderError> {
    let bytes = take(map, offset, 8, what)?;
    Ok((u64::from_le_bytes(bytes.try_into().unwrap()), offset + 8))
}

// One scalar value, read at its real width from the GGUF spec
fn read_scalar(
    map: &[u8],
    offset: usize,
    vtype: u32,
) -> Result<(GgufMetadataValue, usize), ReaderError> {
    let value = match vtype {
        0 => GGUF_METADATA_VALUE_U8(take(map, offset, 1, "u8")?[0]),
        1 => GGUF_METADATA_VALUE_I8(take(map, offset, 1, "i8")?[0] as i8),
        2 => GGUF_METADATA_VALUE_U16(u16::from_le_bytes(
            take(map, offset, 2, "u16")?.try_into().unwrap(),
        )),
        3 => GGUF_METADATA_VALUE_I16(i16::from_le_bytes(
            take(map, offset, 2, "i16")?.try_into().unwrap(),
        )),
        4 => GGUF_METADATA_VALUE_U32(u32::from_le_bytes(
            take(map, offset, 4, "u32")?.try_into().unwrap(),
        )),
        5 => GGUF_METADATA_VALUE_I32(i32::from_le_bytes(
            take(map, offset, 4, "i32")?.try_into().unwrap(),
        )),
        6 => GGUF_METADATA_VALUE_F32(f32::from_le_bytes(
            take(map, offset, 4, "f32")?.try_into().unwrap(),
        )),
        7 => GGUF_METADATA_VALUE_BOOL(take(map, offset, 1, "bool")?[0] != 0),
        10 => GGUF_METADATA_VALUE_U64(u64::from_le_bytes(
            take(map, offset, 8, "u64")?.try_into().unwrap(),
        )),
        11 => GGUF_METADATA_VALUE_I64(i64::from_le_bytes(
            take(map, offset, 8, "i64")?.try_into().unwrap(),
        )),
        12 => GGUF_METADATA_VALUE_F64(f64::from_le_bytes(
            take(map, offset, 8, "f64")?.try_into().unwrap(),
        )),
        other => return Err(ReaderError::UnsupportedType(other)),
    };

    Ok((value, offset + scalar_width(vtype)))
}

fn scalar_width(vtype: u32) -> usize {
    match vtype {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4 | 5 | 6 => 4,
        _ => 8,
    }
}

pub fn parse_general(
    map: &[u8],
    mut offset: usize,
    mut count: u64,
) -> Result<(Vec<MetaDataField>, usize), ReaderError> {
    let mut info = Vec::new();

    while count != 0 {
        let (name, _offset) = parse_string(map, offset)?;
        offset = _offset;

        let (field_type, _offset) = read_u32(map, offset, "field type")?;
        offset = _offset;

        match field_type {
            8 => {
                let (value, _offset) = parse_string(map, offset)?;
                offset = _offset;

                info.push(MetaDataField {
                    name,
                    vtype: GGUF_METADATA_VALUE_TYPE_STRING,
                    value: GGUF_METADATA_VALUE_STRING(value),
                });
            }
            9 => {
                // Array means there is another value type in it we have to read first,
                // then a u64 count, then that many values back to back
                let (elem_type, _offset) = read_u32(map, offset, "array element type")?;
                offset = _offset;

                let (elem_count, _offset) = read_u64(map, offset, "array length")?;
                offset = _offset;

                let mut values = Vec::with_capacity(elem_count as usize);

                for _ in 0..elem_count {
                    if elem_type == 8 {
                        let (value, _offset) = parse_string(map, offset)?;
                        offset = _offset;
                        values.push(GGUF_METADATA_VALUE_STRING(value));
                    } else {
                        // Nested arrays land here too and come back as UnsupportedType(9)
                        let (value, _offset) = read_scalar(map, offset, elem_type)?;
                        offset = _offset;
                        values.push(value);
                    }
                }

                info.push(MetaDataField {
                    name,
                    vtype: GGUF_METADATA_VALUE_TYPE_ARRAY,
                    value: GGUF_METADATA_VALUE_ARRAY(values),
                });
            }
            scalar => {
                let (value, _offset) = read_scalar(map, offset, scalar)?;
                offset = _offset;

                info.push(MetaDataField {
                    name,
                    vtype: value_type(scalar),
                    value,
                });
            }
        }

        count -= 1
    }

    Ok((info, offset))
}

// Maps the wire value back to our enum, read_scalar already rejected anything unknown
fn value_type(vtype: u32) -> GgufMetadataValueType {
    match vtype {
        0 => GGUF_METADATA_VALUE_TYPE_U8,
        1 => GGUF_METADATA_VALUE_TYPE_I8,
        2 => GGUF_METADATA_VALUE_TYPE_U16,
        3 => GGUF_METADATA_VALUE_TYPE_I16,
        4 => GGUF_METADATA_VALUE_TYPE_U32,
        5 => GGUF_METADATA_VALUE_TYPE_I32,
        6 => GGUF_METADATA_VALUE_TYPE_F32,
        7 => GGUF_METADATA_VALUE_TYPE_BOOL,
        8 => GGUF_METADATA_VALUE_TYPE_STRING,
        9 => GGUF_METADATA_VALUE_TYPE_ARRAY,
        10 => GGUF_METADATA_VALUE_TYPE_U64,
        11 => GGUF_METADATA_VALUE_TYPE_I64,
        _ => GGUF_METADATA_VALUE_TYPE_F64,
    }
}

// A gguf string is a u64 length then that many bytes of utf8
fn parse_string(map: &[u8], offset: usize) -> Result<(String, usize), ReaderError> {
    let (length, offset) = read_u64(map, offset, "string length")?;

    let bytes = take(map, offset, length as usize, "string bytes")?;

    let value = str::from_utf8(bytes).map_err(|_| ReaderError::InvalidUtf8(offset))?;

    Ok((value.to_string(), offset + length as usize))
}

pub fn parse_tensors(
    map: &[u8],
    mut offset: usize,
    mut count: u64,
) -> Result<(Vec<Tensor>, usize), ReaderError> {
    let mut tensors = vec![];

    while count != 0 {
        let (name, _offset) = parse_string(map, offset)?;
        offset = _offset;

        let (dim, _offset) = read_u32(map, offset, "tensor dimension count")?;
        offset = _offset;

        let mut values = vec![];

        for _ in 0..dim {
            let (value, _offset) = read_u64(map, offset, "tensor dimension")?;
            offset = _offset;
            values.push(value);
        }

        let (dtype, _offset) = read_u32(map, offset, "tensor dtype")?;
        offset = _offset;

        tensors.push(Tensor {
            name,
            shape: values,
            n_elements: dim,
            dtype,
            offset: take(map, offset, 8, "tensor offset")?.try_into().unwrap(),
        });
        offset += 8;
        count -= 1
    }

    Ok((tensors, offset))
}

pub fn parse_tensor_data<'a>(
    map: &'a [u8],
    offset: usize,
    tensor: &Tensor,
) -> Result<&'a [u8], ReaderError> {
    let weight_count: u64 = tensor.shape.iter().product();

    let weight_bytes = match tensor.dtype {
        // F32
        0 => parse_f32(weight_count, 4, map, offset)?,
        // Q8_0
        8 => parse_q8(weight_count, 34, map, offset)?,
        other => return Err(ReaderError::UnsupportedDtype(other)),
    };

    Ok(weight_bytes)
}

pub fn parse_q8<'a>(
    count: u64,
    byte_count: usize,
    map: &'a [u8],
    offset: usize,
) -> Result<&'a [u8], ReaderError> {
    // Q8_0 only comes in blocks of 32 weights, anything else means we misread the shape
    if count % 32 != 0 {
        return Err(ReaderError::BadQuantBlock(count));
    }

    let byte_total = (count as usize / 32) * byte_count;

    take(map, offset, byte_total, "q8_0 tensor data")
}

fn parse_f32<'a>(
    count: u64,
    byte_count: usize,
    map: &'a [u8],
    offset: usize,
) -> Result<&'a [u8], ReaderError> {
    let byte_total = count as usize * byte_count;

    take(map, offset, byte_total, "f32 tensor data")
}

pub fn convert_q8(raw_bytes: Vec<u8>) -> (Vec<i8>, f32) {
    let scale_bytes = Vec::from(&raw_bytes[0..2]);

    let weight_bytes = &raw_bytes[2..34];

    let weights = weight_bytes
        .iter()
        .map(|x| i8::from_le_bytes([*x]))
        .collect();

    (
        weights,
        f16::from_le_bytes(scale_bytes.try_into().unwrap()).into(),
    )
}

pub fn convert_f32(raw_bytes: &[u8]) -> Vec<f32> {
    raw_bytes
        .chunks(4)
        .map(|f| f32::from_le_bytes(f.try_into().unwrap()))
        .collect()
}

#[test]
fn parse_general_test() {
    let mut bytes = vec![];
    bytes.extend(1u64.to_le_bytes());
    bytes.extend(b"n");
    bytes.extend(4u32.to_le_bytes());
    bytes.extend(22u32.to_le_bytes());

    let (fields, offset) = parse_general(&bytes, 0, 1).unwrap();

    assert_eq!(fields[0].name, "n");
    assert_eq!(fields[0].value, GGUF_METADATA_VALUE_U32(22));
    assert_eq!(offset, bytes.len(), "should land exactly at the end");
}

#[test]
fn parse_general_bool_test() {
    // Bool is 1 byte on the wire, not 4
    let mut bytes = vec![];
    bytes.extend(1u64.to_le_bytes());
    bytes.extend(b"b");
    bytes.extend(7u32.to_le_bytes());
    bytes.push(1);

    let (fields, offset) = parse_general(&bytes, 0, 1).unwrap();

    assert_eq!(fields[0].value, GGUF_METADATA_VALUE_BOOL(true));
    assert_eq!(offset, bytes.len());
}

#[test]
fn parse_general_truncated_test() {
    // Claims a 100 byte string but the file ends, must be an error not a panic
    let mut bytes = vec![];
    bytes.extend(100u64.to_le_bytes());
    bytes.extend(b"n");

    let result = parse_general(&bytes, 0, 1);

    assert_eq!(result, Err(ReaderError::UnexpectedEof("string bytes", 8)));
}

#[test]
fn parse_q8_block_test() {
    // 33 weights can't be Q8_0, blocks are exactly 32
    let bytes = vec![0u8; 1024];

    assert_eq!(
        parse_q8(33, 34, &bytes, 0),
        Err(ReaderError::BadQuantBlock(33))
    );
}
