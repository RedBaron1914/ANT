#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Spiking linear layer accumulation using AVX2.
/// Accumulates INT8 weights for active binary (u8) spikes into INT32 outputs.
#[inline(always)]
pub unsafe fn spiking_linear_layer_avx2(
    weights: &[i8], 
    spikes_x: &[u8], 
    out: &mut [i32], 
    cols: usize
) {
    unsafe {
        for i in 0..spikes_x.len() {
            if spikes_x[i] == 1 { 
                let row_offset = i * cols;
                let row = &weights[row_offset .. row_offset + cols];
                
                let avx_end = (cols / 8) * 8;
                
                for j in (0..avx_end).step_by(8) {
                    let w = _mm256_cvtepi8_epi32(_mm_loadl_epi64(row.as_ptr().add(j) as *const __m128i));
                    let o = _mm256_loadu_si256(out.as_ptr().add(j) as *const __m256i);
                    
                    let sum = _mm256_add_epi32(w, o);
                    _mm256_storeu_si256(out.as_mut_ptr().add(j) as *mut __m256i, sum);
                }
                
                // Scalar tail fallback
                for j in avx_end..cols {
                    out[j] += row[j] as i32;
                }
            }
        }
    }
}
