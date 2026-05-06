//! GPU primitives for PolarQuant-compressed KV.
//!
//! Phase 2a: just `attn_score_polar`. Given a *rotated* query and a packed
//! compressed K cache (angles + radius), dispatches the
//! `attn_score_polar` shader to produce per-(head, position) scores in
//! the rotated/compressed domain. Mirrors `attn_score.wgsl` numerically
//! (rotation preserves dot products) but reads ~4x less K data per token.
//!
//! This module is intentionally a primitive: it allocates GPU buffers
//! per call, dispatches, and reads back. Production wiring into the full
//! attention pipeline (with the matching value path and resident GPU
//! buffers) comes later. For now it serves as a unit-testable shim that
//! we can compare directly against the CPU `QuantizedKvCache::dot_key`.

use std::f32::consts::PI;
use std::sync::Arc;

use crate::compute::wgpu_backend::GpuDevice;

/// Number of angle buckets — must match the CPU `polar` module's value.
const NUM_BUCKETS: usize = 8;

/// Pack a flat `u8` angle stream into `u32` words, 4 buckets per word
/// (low byte first). Pads with zero buckets so the result has length
/// `ceil(angles.len() / 4)`. The shader unpacks via
/// `(word >> ((i & 3) * 8)) & 0xFFu`.
pub fn pack_angles_to_u32(angles: &[u8]) -> Vec<u32> {
    let n_words = (angles.len() + 3) / 4;
    let mut out = vec![0u32; n_words];
    for (i, &b) in angles.iter().enumerate() {
        let word = i / 4;
        let shift = (i % 4) * 8;
        out[word] |= (b as u32) << shift;
    }
    out
}

/// Build the angle LUT in the `vec4<f32>[8]` layout the shader expects.
/// `(cos, sin, 0, 0)` per bucket; matches the CPU `AngleLUT` constants.
/// Returned as a flat `[f32; 32]` ready for upload.
pub fn polar_lut_vec4() -> [f32; NUM_BUCKETS * 4] {
    let mut out = [0.0f32; NUM_BUCKETS * 4];
    for i in 0..NUM_BUCKETS {
        let theta = -PI + (2.0 * PI * i as f32) / NUM_BUCKETS as f32;
        out[i * 4] = theta.cos();
        out[i * 4 + 1] = theta.sin();
        // out[i*4 + 2..4] stays 0 (alignment padding for vec4<f32>)
    }
    out
}

/// Params struct matching `attn_score_polar.wgsl`. 32 bytes, std140-friendly.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnScorePolarParams {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_seq: u32,
    heads_per_kv: u32,
    n_pairs: u32,
    scale: f32,
}

/// Run `attn_score_polar` end-to-end on CPU-resident inputs and read
/// back the resulting score tensor.
///
/// `rq`: rotated query, length `n_heads * head_dim`.
/// `k_angles`: bucket-per-byte stream, length `seq_len * n_kv_heads * (head_dim/2)`.
/// `k_radius`: per-(pos, head) radius, length `seq_len * n_kv_heads`.
///
/// Returns `scores[head * max_seq + t]`. `max_seq == seq_len` here — the
/// shader handles separate `max_seq >= seq_len` for masked decode but
/// this primitive is the unit-test path so we set them equal.
#[allow(clippy::too_many_arguments)]
pub fn attn_score_polar_oneshot(
    gpu: &Arc<GpuDevice>,
    rq: &[f32],
    k_angles: &[u8],
    k_radius: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Vec<f32> {
    assert!(head_dim % 2 == 0, "head_dim must be even");
    assert!(n_heads % n_kv_heads == 0, "n_heads must divide by n_kv_heads");
    assert_eq!(rq.len(), n_heads * head_dim);
    let n_pairs = head_dim / 2;
    assert_eq!(k_angles.len(), seq_len * n_kv_heads * n_pairs);
    assert_eq!(k_radius.len(), seq_len * n_kv_heads);

    let heads_per_kv = (n_heads / n_kv_heads) as u32;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let packed_angles = pack_angles_to_u32(k_angles);
    let lut = polar_lut_vec4();

    // Upload buffers.
    let rq_buf = gpu.create_storage_buffer(bytemuck::cast_slice(rq), "polar.rq");
    let angles_buf = gpu.create_storage_buffer(bytemuck::cast_slice(&packed_angles), "polar.k_angles");
    let radius_buf = gpu.create_storage_buffer(bytemuck::cast_slice(k_radius), "polar.k_radius");

    let scores_len = n_heads * seq_len;
    let scores_bytes = (scores_len * std::mem::size_of::<f32>()) as u64;
    let scores_buf = gpu.create_empty_buffer(scores_bytes, "polar.scores");

    let params = AttnScorePolarParams {
        n_heads: n_heads as u32,
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        seq_len: seq_len as u32,
        max_seq: seq_len as u32,
        heads_per_kv,
        n_pairs: n_pairs as u32,
        scale,
    };
    let params_buf = gpu.create_params_buffer(&params);

    // LUT uniform: 8 vec4<f32> = 128 bytes.
    let lut_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("polar.angle_lut"),
        size: std::mem::size_of_val(&lut) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&lut_buf, 0, bytemuck::cast_slice(&lut));

    let pipeline = &gpu.pipelines.attn_score_polar;
    let bind = gpu.make_bind_group(
        pipeline,
        &[&rq_buf, &angles_buf, &radius_buf, &scores_buf, &params_buf, &lut_buf],
    );

    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("attn_score_polar.oneshot"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("attn_score_polar.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        let groups = ((scores_len as u32) + 255) / 256;
        pass.dispatch_workgroups(groups, 1, 1);
    }

    // Readback.
    let staging = gpu.create_staging_buffer(scores_bytes);
    encoder.copy_buffer_to_buffer(&scores_buf, 0, &staging, 0, scores_bytes);
    gpu.queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    gpu.device.poll(wgpu::Maintain::Wait);
    receiver.recv().unwrap().unwrap();

    let mapped = slice.get_mapped_range();
    let scores: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    staging.unmap();

    scores
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::quantized_kv_cache::QuantizedKvCache;
    use crate::ops::polar;

    #[test]
    fn pack_angles_packs_four_per_word() {
        let angles = [1u8, 2, 3, 4, 5, 6, 7, 0];
        let packed = pack_angles_to_u32(&angles);
        assert_eq!(packed.len(), 2);
        // word 0: 0x04030201, word 1: 0x00070605
        assert_eq!(packed[0], 0x04030201);
        assert_eq!(packed[1], 0x00070605);
    }

    #[test]
    fn pack_angles_pads_partial_word() {
        let angles = [9u8, 10, 11];
        let packed = pack_angles_to_u32(&angles);
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0], 0x000B0A09);
    }

    #[test]
    fn polar_lut_matches_cpu_lut() {
        let cpu = polar::AngleLUT::new();
        let gpu = polar_lut_vec4();
        for i in 0..NUM_BUCKETS {
            assert!((gpu[i * 4]     - cpu.cos[i]).abs() < 1e-7);
            assert!((gpu[i * 4 + 1] - cpu.sin[i]).abs() < 1e-7);
        }
    }

    /// End-to-end: build a small `QuantizedKvCache`, score a known query
    /// on both CPU (`dot_key`) and GPU (`attn_score_polar` shader), assert
    /// the scores match within float tolerance.
    ///
    /// This is the load-bearing test for phase 2a — it proves the shader
    /// is computing the same thing as the verified CPU primitive.
    #[test]
    fn shader_matches_cpu_dot_key() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        // Small but realistic shape: 4 query heads, 2 KV heads (GQA),
        // head_dim 8 (4 polar pairs), 6 cached positions.
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 8;
        let seq_len = 6;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut cache = QuantizedKvCache::new(n_kv_heads, head_dim, seq_len, /*seed*/ 42);
        // Fill with varied K and V (ignored here — we test scoring only).
        let kv_dim = n_kv_heads * head_dim;
        for t in 0..seq_len {
            let k: Vec<f32> = (0..kv_dim)
                .map(|i| ((t * 13 + i * 7) as f32 * 0.05).sin())
                .collect();
            let v = vec![0.0f32; kv_dim];
            cache.append_one(&k, &v);
        }

        // A varied query, one vector per Q head.
        let q: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| ((i * 11) as f32 * 0.07).cos())
            .collect();

        // CPU expected: scores[head, t] = dot_key(t, kv_h, q[head]) * scale,
        // where kv_h = head / heads_per_kv.
        let heads_per_kv = n_heads / n_kv_heads;
        let mut expected = vec![0.0f32; n_heads * seq_len];
        for head in 0..n_heads {
            let kv_h = head / heads_per_kv;
            let q_slice = &q[head * head_dim..(head + 1) * head_dim];
            for t in 0..seq_len {
                expected[head * seq_len + t] = cache.dot_key(t, kv_h, q_slice) * scale;
            }
        }

        // Rotate Q on CPU before shader (the shader expects rotated-domain Q).
        let rotation = polar::generate_rotation_matrix(head_dim, /*seed*/ 42);
        let mut rq = vec![0.0f32; n_heads * head_dim];
        for head in 0..n_heads {
            let qs = &q[head * head_dim..(head + 1) * head_dim];
            polar::rotate(&rotation, qs, &mut rq[head * head_dim..(head + 1) * head_dim]);
        }

        // Pull raw compressed slices off the cache for upload.
        let n_pairs = head_dim / 2;
        let k_angles: Vec<u8> = (0..seq_len * n_kv_heads * n_pairs)
            .map(|i| {
                // Reconstruct from CPU dequantize: easier to just grab via
                // a public slice helper. We expose `k_angles_slice` for GPU
                // upload; reuse it here.
                cache.k_angles_slice()[i]
            })
            .collect();
        let k_radius: Vec<f32> = cache.k_radius_slice().to_vec();

        let got = attn_score_polar_oneshot(
            &gpu, &rq, &k_angles, &k_radius,
            n_heads, n_kv_heads, head_dim, seq_len,
        );

        assert_eq!(got.len(), expected.len());
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            // Both paths apply the same polar dequantization to K and the
            // same rotation to Q, so the only error is float-order
            // differences. Tolerance 1e-4 is generous.
            assert!(
                (g - e).abs() < 1e-4,
                "score[{i}] differs: gpu={g}, cpu={e}, |Δ|={}", (g - e).abs(),
            );
        }
    }
}
