//! Block dequantization — converts GGML quantized formats to f32.
//!
//! Each function takes raw bytes and element count, returning `Vec<f32>`.
//! These are pure functions with no dependencies on layer types.
//!
//! ## Supported formats
//!
//! - **Q8_0**: 8-bit quantized, 32-element blocks. Simplest format.
//! - **Q4_0**: 4-bit quantized, 32-element blocks. Basic nibble packing.
//! - **Q4_K**: 4-bit K-quant, 256-element super-blocks. Most popular on HuggingFace.
//! - **Q6_K**: 6-bit K-quant, 256-element super-blocks. Higher quality.
//! - **Q5_K**: 5-bit K-quant, 256-element super-blocks.
//! - **Q2_K**: 2-bit K-quant, 256-element super-blocks. Aggressive compression.
//! - **Q3_K**: 3-bit K-quant, 256-element super-blocks.

use crate::gguf::f16_to_f32;

// ===========================================================================
// Q8_0: 32-element blocks, 34 bytes each
// Layout: f16 scale (2 bytes) + 32 × i8 values (32 bytes)
// Dequant: val[i] = quant[i] * scale
// ===========================================================================

/// Block size for Q8_0.
pub const Q8_0_BLOCK_SIZE: usize = 32;
/// Bytes per Q8_0 block: 2 (f16 scale) + 32 (i8 values) = 34.
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Dequantize Q8_0 data to f32.
pub fn dequant_q8_0(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements.div_ceil(Q8_0_BLOCK_SIZE);
    assert!(
        data.len() >= n_blocks * Q8_0_BLOCK_BYTES,
        "Q8_0 data too short: need {} bytes for {} elements, got {}",
        n_blocks * Q8_0_BLOCK_BYTES, n_elements, data.len()
    );

    let mut output = Vec::with_capacity(n_elements);

    for block_idx in 0..n_blocks {
        let block = &data[block_idx * Q8_0_BLOCK_BYTES..];

        // f16 scale
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));

        // 32 × i8 values
        let values_in_block = Q8_0_BLOCK_SIZE.min(n_elements - block_idx * Q8_0_BLOCK_SIZE);
        for i in 0..values_in_block {
            let q = block[2 + i] as i8;
            output.push(q as f32 * scale);
        }
    }

    output
}

// ===========================================================================
// Q4_0: 32-element blocks, 18 bytes each
// Layout: f16 scale (2 bytes) + 16 × u8 nibble pairs (16 bytes)
// Each byte holds 2 values: low nibble = first, high nibble = second
// Values are unsigned 0..15, centered at 8: val = (nibble - 8) * scale
// ===========================================================================

/// Block size for Q4_0.
pub const Q4_0_BLOCK_SIZE: usize = 32;
/// Bytes per Q4_0 block: 2 (f16 scale) + 16 (nibble pairs) = 18.
pub const Q4_0_BLOCK_BYTES: usize = 18;

/// Dequantize Q4_0 data to f32.
/// Dequantize Q4_0 data to f32.
///
/// Block layout (18 bytes per block of 32 values):
/// - 2 bytes: f16 scale (delta)
/// - 16 bytes: nibble pairs (qs)
///
/// llama.cpp convention: first 16 values from low nibbles, next 16 from high nibbles.
/// For j = 0..15:
///   element[j]    = ((qs[j] & 0x0F) - 8) * scale
///   element[j+16] = ((qs[j] >> 4)   - 8) * scale
pub fn dequant_q4_0(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements.div_ceil(Q4_0_BLOCK_SIZE);
    assert!(
        data.len() >= n_blocks * Q4_0_BLOCK_BYTES,
        "Q4_0 data too short: need {} bytes for {} elements, got {}",
        n_blocks * Q4_0_BLOCK_BYTES, n_elements, data.len()
    );

    let mut output = vec![0.0f32; n_elements];

    for block_idx in 0..n_blocks {
        let block = &data[block_idx * Q4_0_BLOCK_BYTES..];
        let base = block_idx * Q4_0_BLOCK_SIZE;

        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));

        let remaining = n_elements - base;
        let half = 16.min(remaining);
        for j in 0..half {
            let nibble = block[2 + j] & 0x0F;
            output[base + j] = (nibble as i32 - 8) as f32 * scale;
        }
        let second_half = 16.min(remaining.saturating_sub(16));
        for j in 0..second_half {
            let nibble = block[2 + j] >> 4;
            output[base + 16 + j] = (nibble as i32 - 8) as f32 * scale;
        }
    }

    output
}

// ===========================================================================
// Q4_K: 256-element super-blocks (K-quant)
// Layout (144 bytes per super-block):
//   - f16 d (2 bytes): super-block scale
//   - f16 dmin (2 bytes): super-block minimum
//   - 12 bytes: packed scales and mins for 8 sub-blocks (6-bit each)
//   - 128 bytes: 256 × 4-bit quantized values (nibble pairs)
//
// Each super-block has 8 sub-blocks of 32 elements.
// Sub-block i has its own 6-bit scale (sc) and 6-bit min (m).
// Dequant: val = d * sc * nibble - dmin * m
// ===========================================================================

/// Block size for Q4_K.
pub const Q4_K_BLOCK_SIZE: usize = 256;
/// Bytes per Q4_K super-block.
pub const Q4_K_BLOCK_BYTES: usize = 144;

/// Dequantize Q4_K data to f32.
///
/// llama.cpp layout: 4 chunks of 64 elements per super-block.
/// Each chunk uses 32 bytes of quants: first 32 values from low nibbles,
/// next 32 from high nibbles. Each half has its own scale and min.
pub fn dequant_q4_k(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements.div_ceil(Q4_K_BLOCK_SIZE);
    assert!(
        data.len() >= n_blocks * Q4_K_BLOCK_BYTES,
        "Q4_K data too short: need {} bytes for {} elements, got {}",
        n_blocks * Q4_K_BLOCK_BYTES, n_elements, data.len()
    );

    let mut output = vec![0.0f32; n_elements];

    for block_idx in 0..n_blocks {
        let block = &data[block_idx * Q4_K_BLOCK_BYTES..];
        let base = block_idx * Q4_K_BLOCK_SIZE;

        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));

        let (scales, mins) = unpack_q4_k_scales_mins(&block[4..16]);
        let quants = &block[16..144];

        let values_in_block = Q4_K_BLOCK_SIZE.min(n_elements - base);

        // 4 chunks of 64 elements, each using 32 bytes of quants
        let mut is = 0;
        for chunk in 0..4 {
            let q = &quants[chunk * 32..];
            let d1 = d * scales[is] as f32;
            let m1 = dmin * mins[is] as f32;
            let d2 = d * scales[is + 1] as f32;
            let m2 = dmin * mins[is + 1] as f32;

            for (l, &qb) in q[..32].iter().enumerate() {
                let el = chunk * 64 + l;
                if el < values_in_block {
                    output[base + el] = d1 * (qb & 0x0F) as f32 - m1;
                }
            }
            for (l, &qb) in q[..32].iter().enumerate() {
                let el = chunk * 64 + 32 + l;
                if el < values_in_block {
                    output[base + el] = d2 * (qb >> 4) as f32 - m2;
                }
            }
            is += 2;
        }
    }

    output
}

/// Unpack 6-bit scales and mins for 8 sub-blocks from 12 bytes.
///
/// Matches llama.cpp's `get_scale_min_k4` helper exactly:
///   - Sub-blocks 0..3: 6 bits from bytes 0..3 (scales) and 4..7 (mins)
///   - Sub-blocks 4..7: 4 bits from bytes 8..11, 2 high bits from bits 6-7
///     of bytes 0..3 (scales) and 4..7 (mins)
fn unpack_q4_k_scales_mins(sm: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];

    // Sub-blocks 0-3: low 6 bits directly
    for j in 0..4 {
        scales[j] = sm[j] & 0x3F;
        mins[j] = sm[j + 4] & 0x3F;
    }
    // Sub-blocks 4-7: 4 bits from high bytes + 2 high bits from low bytes
    for j in 0..4 {
        scales[j + 4] = (sm[8 + j] & 0x0F) | ((sm[j] >> 6) << 4);
        mins[j + 4] = (sm[8 + j] >> 4) | ((sm[j + 4] >> 6) << 4);
    }

    (scales, mins)
}

// ===========================================================================
// Q6_K: 256-element super-blocks (K-quant, 6-bit)
// Layout (210 bytes per super-block):
//   - 128 bytes: low 4 bits of quantized values (nibble packed)
//   - 64 bytes: high 2 bits of quantized values (2-bit packed, 4 per byte)
//   - 16 bytes: 16 × i8 scales for 16 sub-blocks of 16
//   - f16 d (2 bytes): super-block scale
// Dequant: val = d * scale[sub] * (q - 32)
// ===========================================================================

/// Block size for Q6_K.
pub const Q6_K_BLOCK_SIZE: usize = 256;
/// Bytes per Q6_K super-block.
pub const Q6_K_BLOCK_BYTES: usize = 210;

/// Dequantize Q6_K data to f32.
/// Dequantize Q6_K data to f32.
///
/// llama.cpp Q6_K layout (210 bytes per 256-element super-block):
/// - ql[128]: low 4 bits, interleaved in groups of 32
/// - qh[64]:  high 2 bits, interleaved in groups of 32
/// - sc[16]:  i8 scales per 16-element sub-block
/// - d:       f16 super-block scale
///
/// Decoded in two halves of 128 elements. For each half:
///   ql[l+0]:  low nibble → elements [l], high nibble → elements [l+64]
///   ql[l+32]: low nibble → elements [l+32], high nibble → elements [l+96]
///   qh[l]:    2 bits for each of the 4 groups
pub fn dequant_q6_k(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements.div_ceil(Q6_K_BLOCK_SIZE);
    assert!(
        data.len() >= n_blocks * Q6_K_BLOCK_BYTES,
        "Q6_K data too short: need {} bytes for {} elements, got {}",
        n_blocks * Q6_K_BLOCK_BYTES, n_elements, data.len()
    );

    let mut output = vec![0.0f32; n_elements];

    for block_idx in 0..n_blocks {
        let block = &data[block_idx * Q6_K_BLOCK_BYTES..];
        let base = block_idx * Q6_K_BLOCK_SIZE;

        // f16 super-block scale at offset 208
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

        // Process in two halves of 128 elements each
        for half in 0..2u32 {
            let ql = &block[(half as usize * 64)..];
            let qh = &block[128 + (half as usize * 32)..];
            let sc = &block[192 + (half as usize * 8)..];
            let y_base = base + half as usize * 128;

            for l in 0..32usize {
                let remaining = n_elements.saturating_sub(y_base);
                if l >= remaining { break; }

                let is0 = 0; // scale index for elements [0..15] and [16..31]
                let is2 = 2;
                let is4 = 4;
                let is6 = 6;

                let q1 = ((ql[l] & 0x0F) | ((qh[l] & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;

                let sc0 = sc[is0 + l / 16] as i8 as f32;
                let sc2 = sc[is2 + l / 16] as i8 as f32;
                let sc4 = sc[is4 + l / 16] as i8 as f32;
                let sc6 = sc[is6 + l / 16] as i8 as f32;

                if l < remaining { output[y_base + l] = d * sc0 * q1 as f32; }
                if l + 32 < remaining { output[y_base + l + 32] = d * sc2 * q2 as f32; }
                if l + 64 < remaining { output[y_base + l + 64] = d * sc4 * q3 as f32; }
                if l + 96 < remaining { output[y_base + l + 96] = d * sc6 * q4 as f32; }
            }
        }
    }

    output
}

// ===========================================================================
// Q5_K: 256-element super-blocks (K-quant, 5-bit)
// Layout (176 bytes per super-block):
//   - f16 d (2 bytes)
//   - f16 dmin (2 bytes)
//   - 12 bytes: packed scales and mins (same as Q4_K)
//   - 32 bytes: high bits (1 bit per value, packed)
//   - 128 bytes: low 4 bits (nibble packed)
// Dequant: val = d * sc * q - dmin * m, where q = lo | (hi << 4) is 5-bit
// ===========================================================================

/// Block size for Q5_K.
pub const Q5_K_BLOCK_SIZE: usize = 256;
/// Bytes per Q5_K super-block.
pub const Q5_K_BLOCK_BYTES: usize = 176;

/// Dequantize Q5_K data to f32.
///
/// llama.cpp layout: 4 chunks of 64 elements per super-block.
/// Same nibble interleaving as Q4_K, plus a high-bit array (qh).
/// qh has 32 bytes (one per position within a chunk); each byte provides
/// 8 bits across the 4 chunks × 2 halves.
pub fn dequant_q5_k(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements.div_ceil(Q5_K_BLOCK_SIZE);
    assert!(
        data.len() >= n_blocks * Q5_K_BLOCK_BYTES,
        "Q5_K data too short: need {} bytes for {} elements, got {}",
        n_blocks * Q5_K_BLOCK_BYTES, n_elements, data.len()
    );

    let mut output = vec![0.0f32; n_elements];

    for block_idx in 0..n_blocks {
        let block = &data[block_idx * Q5_K_BLOCK_BYTES..];
        let base = block_idx * Q5_K_BLOCK_SIZE;

        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));

        let (scales, mins) = unpack_q4_k_scales_mins(&block[4..16]);
        let qh = &block[16..48]; // 32 bytes — high bits
        let ql = &block[48..176]; // 128 bytes — low nibbles

        let values_in_block = Q5_K_BLOCK_SIZE.min(n_elements - base);

        // 4 chunks of 64 elements
        let mut is = 0;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for chunk in 0..4 {
            let q = &ql[chunk * 32..];
            let d1 = d * scales[is] as f32;
            let m1 = dmin * mins[is] as f32;
            let d2 = d * scales[is + 1] as f32;
            let m2 = dmin * mins[is + 1] as f32;

            for l in 0..32 {
                let el = chunk * 64 + l;
                if el < values_in_block {
                    let lo = q[l] & 0x0F;
                    let hi: u8 = if qh[l] & u1 != 0 { 16 } else { 0 };
                    output[base + el] = d1 * (lo + hi) as f32 - m1;
                }
            }
            for l in 0..32 {
                let el = chunk * 64 + 32 + l;
                if el < values_in_block {
                    let lo = q[l] >> 4;
                    let hi: u8 = if qh[l] & u2 != 0 { 16 } else { 0 };
                    output[base + el] = d2 * (lo + hi) as f32 - m2;
                }
            }
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    output
}

// ===========================================================================
// Q2_K: 256-element super-blocks (K-quant, 2-bit)
// Layout (84 bytes per super-block):
//   - 16 bytes: scales and mins for 16 sub-blocks (4-bit packed, 2 per byte)
//   - 64 bytes: 256 × 2-bit values (4 per byte)
//   - f16 d (2 bytes)
//   - f16 dmin (2 bytes)
// ===========================================================================

/// Block size for Q2_K.
pub const Q2_K_BLOCK_SIZE: usize = 256;
/// Bytes per Q2_K super-block.
pub const Q2_K_BLOCK_BYTES: usize = 84;

/// Dequantize Q2_K data to f32.
///
/// llama.cpp layout: 2 groups of 128 elements, each using 32 q-bytes.
/// Within each group, 4 shift iterations extract 2 bits at different positions.
/// Each shift produces 2 sub-blocks of 16 elements from q[0..15] and q[16..31].
pub fn dequant_q2_k(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements.div_ceil(Q2_K_BLOCK_SIZE);
    assert!(
        data.len() >= n_blocks * Q2_K_BLOCK_BYTES,
        "Q2_K data too short: need {} bytes for {} elements, got {}",
        n_blocks * Q2_K_BLOCK_BYTES, n_elements, data.len()
    );

    let mut output = vec![0.0f32; n_elements];

    for block_idx in 0..n_blocks {
        let block = &data[block_idx * Q2_K_BLOCK_BYTES..];
        let base = block_idx * Q2_K_BLOCK_SIZE;

        let scales_mins = &block[0..16];
        let quants = &block[16..80];
        let d = f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[82], block[83]]));

        let values_in_block = Q2_K_BLOCK_SIZE.min(n_elements - base);

        let mut is = 0;
        let mut q_offset = 0;
        for _group in 0..2 {
            let q = &quants[q_offset..];
            let mut shift = 0u32;
            for _j in 0..4 {
                // First sub-block: 16 elements from q[0..15]
                let sc_byte = scales_mins[is];
                let dl = d * (sc_byte & 0x0F) as f32;
                let ml = dmin * (sc_byte >> 4) as f32;
                is += 1;

                let el_base = base + _group * 128 + _j * 32;
                for l in 0..16 {
                    let el = _group * 128 + _j * 32 + l;
                    if el < values_in_block {
                        let qval = ((q[l] >> shift) & 3) as f32;
                        output[el_base + l] = dl * qval - ml;
                    }
                }
                // Second sub-block: 16 elements from q[16..31]
                let sc_byte = scales_mins[is];
                let dl = d * (sc_byte & 0x0F) as f32;
                let ml = dmin * (sc_byte >> 4) as f32;
                is += 1;

                for l in 0..16 {
                    let el = _group * 128 + _j * 32 + 16 + l;
                    if el < values_in_block {
                        let qval = ((q[l + 16] >> shift) & 3) as f32;
                        output[el_base + 16 + l] = dl * qval - ml;
                    }
                }
                shift += 2;
            }
            q_offset += 32;
        }
    }

    output
}

// ===========================================================================
// Q3_K: 256-element super-blocks (K-quant, 3-bit)
// Layout (110 bytes per super-block):
//   - 32 bytes: high bit of each value (256 bits)
//   - 64 bytes: low 2 bits of each value (4 per byte)
//   - 12 bytes: packed scales (same as Q4_K format but different interpretation)
//   - f16 d (2 bytes)
// Dequant: val = d * scale * (q - 4), where q is 3-bit (0..7)
// ===========================================================================

/// Block size for Q3_K.
pub const Q3_K_BLOCK_SIZE: usize = 256;
/// Bytes per Q3_K super-block.
pub const Q3_K_BLOCK_BYTES: usize = 110;

/// Dequantize Q3_K data to f32.
///
/// llama.cpp layout: 2 groups of 128 elements, each using 32 q-bytes.
/// Same shift-based iteration as Q2_K. Scale unpacking uses 32-bit
/// word manipulation to extract 16 × 6-bit values from 12 bytes.
/// High-bit mask (hmask) uses a rotating bit per sub-block pair.
pub fn dequant_q3_k(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements.div_ceil(Q3_K_BLOCK_SIZE);
    assert!(
        data.len() >= n_blocks * Q3_K_BLOCK_BYTES,
        "Q3_K data too short: need {} bytes for {} elements, got {}",
        n_blocks * Q3_K_BLOCK_BYTES, n_elements, data.len()
    );

    let mut output = vec![0.0f32; n_elements];

    for block_idx in 0..n_blocks {
        let block = &data[block_idx * Q3_K_BLOCK_BYTES..];
        let base = block_idx * Q3_K_BLOCK_SIZE;

        let hmask = &block[0..32];
        let ql = &block[32..96];
        let scales_raw = &block[96..108];
        let d_all = f16_to_f32(u16::from_le_bytes([block[108], block[109]]));

        // Unpack 16 × 6-bit scales using llama.cpp's u32 bit manipulation
        let scales = unpack_q3_k_scales(scales_raw);

        let values_in_block = Q3_K_BLOCK_SIZE.min(n_elements - base);

        // 2 groups of 128 elements, each using 32 q-bytes
        let mut is = 0;
        let mut q_offset = 0;
        let mut hm: u8 = 1;

        for _group in 0..2 {
            let q = &ql[q_offset..];
            let mut shift = 0u32;

            for _j in 0..4 {
                // First sub-block: 16 elements from q[0..15]
                let dl = d_all * (scales[is] as f32 - 32.0);
                is += 1;
                let el_base = _group * 128 + _j * 32;

                for l in 0..16 {
                    let el = el_base + l;
                    if el < values_in_block {
                        let lo = ((q[l] >> shift) & 3) as i32;
                        let hi = if hmask[l] & hm != 0 { 0 } else { 4i32 };
                        output[base + el] = dl * (lo - hi) as f32;
                    }
                }

                // Second sub-block: 16 elements from q[16..31]
                let dl = d_all * (scales[is] as f32 - 32.0);
                is += 1;

                for l in 0..16 {
                    let el = el_base + 16 + l;
                    if el < values_in_block {
                        let lo = ((q[l + 16] >> shift) & 3) as i32;
                        let hi = if hmask[l + 16] & hm != 0 { 0 } else { 4i32 };
                        output[base + el] = dl * (lo - hi) as f32;
                    }
                }

                shift += 2;
                hm <<= 1;
            }
            q_offset += 32;
        }
    }

    output
}

/// Unpack 16 × 6-bit Q3_K scales from 12 bytes.
///
/// Replicates llama.cpp's u32 bit manipulation exactly:
///   aux[0..1] = 12 raw bytes as three u32s
///   tmp = aux[2] provides the high 2 bits for each scale
///   Result: 16 unsigned bytes (used as `scale - 32` in dequant)
fn unpack_q3_k_scales(raw: &[u8]) -> [u8; 16] {
    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;

    let mut aux = [0u32; 4];
    aux[0] = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
    aux[1] = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
    aux[2] = u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]);

    let tmp = aux[2];
    // Order matters: aux[2] and aux[3] read original aux[0]/aux[1]
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);

    let mut scales = [0u8; 16];
    for i in 0..4 {
        let bytes = aux[i].to_le_bytes();
        scales[i * 4] = bytes[0];
        scales[i * 4 + 1] = bytes[1];
        scales[i * 4 + 2] = bytes[2];
        scales[i * 4 + 3] = bytes[3];
    }
    scales
}

// ===========================================================================
// Q5_0: 32-element blocks, 22 bytes each
// Layout: f16 scale (2 bytes) + 4 bytes high bits + 16 bytes nibbles
// Each value is 5 bits: 4 low bits from nibble + 1 high bit
// Values are unsigned 0..31, centered at 16: val = (q - 16) * scale
// ===========================================================================

/// Block size for Q5_0.
pub const Q5_0_BLOCK_SIZE: usize = 32;
/// Bytes per Q5_0 block: 2 (f16 scale) + 4 (high bits) + 16 (nibbles) = 22.
pub const Q5_0_BLOCK_BYTES: usize = 22;

/// Dequantize Q5_0 data to f32.
///
/// Block layout (22 bytes per block of 32 values):
/// - 2 bytes: f16 scale (delta)
/// - 4 bytes: high bits (qh), 1 bit per value
/// - 16 bytes: low nibbles (qs), packed as pairs
///
/// Value reconstruction (llama.cpp convention):
/// For j = 0..15:
///   element[j]    = ((qs[j] & 0x0F) | ((qh >> j)      & 1) << 4) - 16) * scale
///   element[j+16] = ((qs[j] >> 4)   | ((qh >> (j+16)) & 1) << 4) - 16) * scale
pub fn dequant_q5_0(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements.div_ceil(Q5_0_BLOCK_SIZE);
    assert!(
        data.len() >= n_blocks * Q5_0_BLOCK_BYTES,
        "Q5_0 data too short: need {} bytes for {} elements, got {}",
        n_blocks * Q5_0_BLOCK_BYTES, n_elements, data.len()
    );

    let mut output = vec![0.0f32; n_elements];

    for block_idx in 0..n_blocks {
        let block = &data[block_idx * Q5_0_BLOCK_BYTES..];
        let base = block_idx * Q5_0_BLOCK_SIZE;

        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));

        // High bits: 4 bytes = 32 bits (1 per value)
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);

        // Low 4 bits: 16 bytes (qs), packed as pairs
        let qs = &block[6..22];

        // llama.cpp layout: first 16 from low nibbles, next 16 from high nibbles
        let remaining = n_elements - base;
        let half = 16.min(remaining);
        for j in 0..half {
            let lo = qs[j] & 0x0F;
            let hi_bit = ((qh >> j) & 1) as u8;
            let q = (lo | (hi_bit << 4)) as i32 - 16;
            output[base + j] = q as f32 * scale;
        }
        let second_half = 16.min(remaining.saturating_sub(16));
        for j in 0..second_half {
            let lo = qs[j] >> 4;
            let hi_bit = ((qh >> (j + 16)) & 1) as u8;
            let q = (lo | (hi_bit << 4)) as i32 - 16;
            output[base + 16 + j] = q as f32 * scale;
        }
    }

    output
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_basic() {
        // One block: scale=1.0 (f16), values=[1,2,3,...,32]
        let mut block = vec![0u8; Q8_0_BLOCK_BYTES];
        // f16 1.0 = 0x3C00
        block[0] = 0x00;
        block[1] = 0x3C;
        for i in 0..32 {
            block[2 + i] = (i as i8 + 1) as u8;
        }
        let result = dequant_q8_0(&block, 32);
        for i in 0..32 {
            assert!(
                (result[i] - (i as f32 + 1.0)).abs() < 0.01,
                "q8_0[{i}]: expected {}, got {}",
                i + 1, result[i]
            );
        }
    }

    #[test]
    fn q8_0_negative() {
        let mut block = vec![0u8; Q8_0_BLOCK_BYTES];
        // f16 0.5 = 0x3800
        block[0] = 0x00;
        block[1] = 0x38;
        // -1 as u8 = 255
        block[2] = 255; // -1
        block[3] = 254; // -2
        let result = dequant_q8_0(&block, 2);
        assert!((result[0] - (-0.5)).abs() < 0.01, "got {}", result[0]);
        assert!((result[1] - (-1.0)).abs() < 0.01, "got {}", result[1]);
    }

    #[test]
    fn q8_0_zero_scale() {
        let block = vec![0u8; Q8_0_BLOCK_BYTES];
        let result = dequant_q8_0(&block, 32);
        for &v in &result {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn q4_0_basic() {
        // One block: scale=1.0, all nibbles = 8 (center) → 0.0
        let mut block = vec![0u8; Q4_0_BLOCK_BYTES];
        block[0] = 0x00;
        block[1] = 0x3C; // f16 1.0
        // 0x88 = nibbles (8, 8) → both zero
        for i in 0..16 {
            block[2 + i] = 0x88;
        }
        let result = dequant_q4_0(&block, 32);
        for (i, &v) in result.iter().enumerate() {
            assert!(v.abs() < 0.01, "q4_0[{i}]: expected 0, got {v}");
        }
    }

    #[test]
    fn q4_0_extremes() {
        let mut block = vec![0u8; Q4_0_BLOCK_BYTES];
        block[0] = 0x00;
        block[1] = 0x3C; // f16 1.0
        // Byte[2] = 0xF0: low nibble = 0 → element[0], high nibble = 0xF → element[16]
        // llama.cpp split layout: elements [0..15] from low nibbles, [16..31] from high nibbles
        block[2] = 0xF0;
        let result = dequant_q4_0(&block, 32);
        assert!((result[0] - (-8.0)).abs() < 0.01, "element 0 (low nibble 0): {}", result[0]);
        assert!((result[16] - 7.0).abs() < 0.01, "element 16 (high nibble 0xF): {}", result[16]);
    }

    #[test]
    fn q8_0_multi_block() {
        // Two blocks
        let mut data = vec![0u8; Q8_0_BLOCK_BYTES * 2];
        // Block 0: scale=1.0, values=1..32
        data[0] = 0x00;
        data[1] = 0x3C;
        for i in 0..32 {
            data[2 + i] = (i as i8 + 1) as u8;
        }
        // Block 1: scale=2.0 (f16 0x4000), values=1..32
        data[Q8_0_BLOCK_BYTES] = 0x00;
        data[Q8_0_BLOCK_BYTES + 1] = 0x40;
        for i in 0..32 {
            data[Q8_0_BLOCK_BYTES + 2 + i] = (i as i8 + 1) as u8;
        }
        let result = dequant_q8_0(&data, 64);
        assert_eq!(result.len(), 64);
        // Block 0: 1*1, 2*1, ...
        assert!((result[0] - 1.0).abs() < 0.01);
        // Block 1: 1*2, 2*2, ...
        assert!((result[32] - 2.0).abs() < 0.01);
        assert!((result[33] - 4.0).abs() < 0.01);
    }
}
