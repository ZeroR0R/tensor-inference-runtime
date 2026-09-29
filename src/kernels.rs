use half::f16;
use rayon::{iter::{IndexedParallelIterator, ParallelIterator}, slice::{ParallelSlice, ParallelSliceMut}};

// Activation Quantization

// Q8 is fairly simple, scale is == to the largest absolute value / 127

pub fn activated_q8(activation: &[f32], out: &mut [u8]) {
    let vec = activation.par_chunks(32);
    let activated = out.par_chunks_mut(34);

    // Tie the pre quantized values to their post quantized locations
    let zipped = vec.zip(activated);

    zipped.for_each(|(pre, post)| {
        match pre.iter().max_by(|a, b| a.abs().total_cmp(&b.abs())) {
            Some(max) => {
                let scale = max / 127.0;
                post[0..=1].copy_from_slice(&f16::to_le_bytes(f16::from_f32(scale)));
                pre.iter().enumerate().for_each(|(idx, x)| {
                    post[idx+2] = i8::to_le_bytes((x / scale).round() as i8)[0];
                });
            }
            // If it is none, that means the largest value is 0
            None => {
                post[0..34].copy_from_slice(&[0u8;34]);
            }
        }

    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_first_block() {
        let pre_activation = vec![22.0;2048];
        let mut out = vec![0;2176];


        // First Block
        let expected = [
            139, 49, 127, 127, 127, 127,
            127, 127, 127, 127, 127, 127, 127,
            127, 127, 127, 127, 127, 127, 127, 127,
            127, 127, 127, 127, 127, 127, 127, 
            127, 127, 127, 127, 127, 127
            ];

        activated_q8(&pre_activation, &mut out);
        
        assert_eq!(&out[0..34], expected)
    }

}