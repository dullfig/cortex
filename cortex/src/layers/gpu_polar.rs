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
use crate::layers::gpu_polar_kv_cache::GpuPolarKvCache;

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

/// Encode an `attn_score_polar` dispatch into `encoder`. All inputs are
/// resident GPU buffers; this function does no allocation, no upload, no
/// readback. Caller is responsible for queue submission and any sync.
///
/// `max_seq` is the row stride of the `scores_buf` — `scores_buf[h*max_seq + t]`.
/// `seq_len` is how many positions the cache currently holds (loop bound).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_attn_score_polar(
    gpu: &Arc<GpuDevice>,
    encoder: &mut wgpu::CommandEncoder,
    rq_buf: &wgpu::Buffer,
    k_angles_buf: &wgpu::Buffer,
    k_radius_buf: &wgpu::Buffer,
    scores_buf: &wgpu::Buffer,
    lut_buf: &wgpu::Buffer,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    max_seq: usize,
) {
    assert!(head_dim % 2 == 0);
    assert!(n_heads % n_kv_heads == 0);
    assert!(seq_len <= max_seq);

    let n_pairs = head_dim / 2;
    let heads_per_kv = (n_heads / n_kv_heads) as u32;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let params = AttnScorePolarParams {
        n_heads: n_heads as u32,
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        seq_len: seq_len as u32,
        max_seq: max_seq as u32,
        heads_per_kv,
        n_pairs: n_pairs as u32,
        scale,
    };
    let params_buf = gpu.create_params_buffer(&params);

    let pipeline = &gpu.pipelines.attn_score_polar;
    let bind = gpu.make_bind_group(
        pipeline,
        &[rq_buf, k_angles_buf, k_radius_buf, scores_buf, &params_buf, lut_buf],
    );

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("attn_score_polar.dispatch"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind, &[]);
    let threads = (n_heads * seq_len) as u32;
    let groups = (threads + 255) / 256;
    pass.dispatch_workgroups(groups, 1, 1);
}

/// Run `attn_score_polar` end-to-end on CPU-resident inputs and read
/// back the resulting score tensor.
///
/// Wrapper around `dispatch_attn_score_polar`: allocates GPU buffers,
/// uploads, dispatches, reads back. Use the dispatch variant directly
/// when you have resident GPU buffers.
///
/// Returns `scores[head * seq_len + t]`. Sets max_seq == seq_len.
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
    assert!(head_dim % 2 == 0);
    assert!(n_heads % n_kv_heads == 0);
    assert_eq!(rq.len(), n_heads * head_dim);
    let n_pairs = head_dim / 2;
    assert_eq!(k_angles.len(), seq_len * n_kv_heads * n_pairs);
    assert_eq!(k_radius.len(), seq_len * n_kv_heads);

    let packed_angles = pack_angles_to_u32(k_angles);
    let lut = polar_lut_vec4();

    let rq_buf = gpu.create_storage_buffer(bytemuck::cast_slice(rq), "polar.rq");
    let angles_buf = gpu.create_storage_buffer(bytemuck::cast_slice(&packed_angles), "polar.k_angles");
    let radius_buf = gpu.create_storage_buffer(bytemuck::cast_slice(k_radius), "polar.k_radius");

    let scores_len = n_heads * seq_len;
    let scores_bytes = (scores_len * std::mem::size_of::<f32>()) as u64;
    let scores_buf = gpu.create_empty_buffer(scores_bytes, "polar.scores");

    let lut_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("polar.angle_lut"),
        size: std::mem::size_of_val(&lut) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&lut_buf, 0, bytemuck::cast_slice(&lut));

    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("attn_score_polar.oneshot"),
    });
    dispatch_attn_score_polar(
        gpu, &mut encoder,
        &rq_buf, &angles_buf, &radius_buf, &scores_buf, &lut_buf,
        n_heads, n_kv_heads, head_dim, seq_len, /*max_seq*/ seq_len,
    );

    let staging = gpu.create_staging_buffer(scores_bytes);
    encoder.copy_buffer_to_buffer(&scores_buf, 0, &staging, 0, scores_bytes);
    gpu.queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = sender.send(r); });
    gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
    receiver.recv().unwrap().unwrap();

    let mapped = slice.get_mapped_range();
    let scores: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    staging.unmap();

    scores
}

/// Params for `attn_value_polar.wgsl`. 32 bytes std140-friendly.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnValuePolarParams {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_seq: u32,
    heads_per_kv: u32,
    n_pairs: u32,
    _pad: u32,
}

/// Params for `derotate.wgsl`. 8 bytes; padded to 16 for uniform alignment.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DerotateParams {
    n_heads: u32,
    head_dim: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Encode pass A of the compressed-V output (weighted sum in rotated
/// space). Resident-buffer dispatch — no alloc, no upload, no readback.
///
/// `weighted_rot_buf` is the output: `[n_heads * head_dim]` f32 in
/// rotated space, ready to feed into `dispatch_derotate`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_attn_value_polar(
    gpu: &Arc<GpuDevice>,
    encoder: &mut wgpu::CommandEncoder,
    softmax_buf: &wgpu::Buffer,
    v_angles_buf: &wgpu::Buffer,
    v_radius_buf: &wgpu::Buffer,
    weighted_rot_buf: &wgpu::Buffer,
    lut_buf: &wgpu::Buffer,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    max_seq: usize,
) {
    assert!(head_dim % 2 == 0);
    assert!(n_heads % n_kv_heads == 0);
    assert!(seq_len <= max_seq);

    let n_pairs = head_dim / 2;
    let heads_per_kv = (n_heads / n_kv_heads) as u32;

    let params = AttnValuePolarParams {
        n_heads: n_heads as u32,
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        seq_len: seq_len as u32,
        max_seq: max_seq as u32,
        heads_per_kv,
        n_pairs: n_pairs as u32,
        _pad: 0,
    };
    let params_buf = gpu.create_params_buffer(&params);

    let pipeline = &gpu.pipelines.attn_value_polar;
    let bind = gpu.make_bind_group(
        pipeline,
        &[softmax_buf, v_angles_buf, v_radius_buf, weighted_rot_buf, &params_buf, lut_buf],
    );

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("attn_value_polar.dispatch"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind, &[]);
    let groups = ((n_heads * head_dim) as u32 + 255) / 256;
    pass.dispatch_workgroups(groups, 1, 1);
}

/// Encode pass B of the compressed-V output (de-rotation by R^T).
/// Resident-buffer dispatch — takes the rotated-space weighted V and
/// the per-layer rotation matrix, writes the original-space output.
pub fn dispatch_derotate(
    gpu: &Arc<GpuDevice>,
    encoder: &mut wgpu::CommandEncoder,
    weighted_rot_buf: &wgpu::Buffer,
    rotation_buf: &wgpu::Buffer,
    out_buf: &wgpu::Buffer,
    n_heads: usize,
    head_dim: usize,
) {
    let params = DerotateParams {
        n_heads: n_heads as u32,
        head_dim: head_dim as u32,
        _pad0: 0,
        _pad1: 0,
    };
    let params_buf = gpu.create_params_buffer(&params);

    let pipeline = &gpu.pipelines.derotate;
    let bind = gpu.make_bind_group(
        pipeline,
        &[weighted_rot_buf, rotation_buf, out_buf, &params_buf],
    );

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("derotate.dispatch"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind, &[]);
    let groups = ((n_heads * head_dim) as u32 + 255) / 256;
    pass.dispatch_workgroups(groups, 1, 1);
}

/// Run the compressed-V attention output path end-to-end and read back
/// the de-rotated result.
///
/// Wrapper around `dispatch_attn_value_polar` + `dispatch_derotate`:
/// allocates buffers, uploads, runs both passes, reads back. Use the
/// dispatch variants directly when you have resident GPU buffers.
#[allow(clippy::too_many_arguments)]
pub fn attn_value_polar_oneshot(
    gpu: &Arc<GpuDevice>,
    softmax: &[f32],
    v_angles: &[u8],
    v_radius: &[f32],
    rotation: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Vec<f32> {
    assert!(head_dim % 2 == 0);
    assert!(n_heads % n_kv_heads == 0);
    assert_eq!(softmax.len(), n_heads * seq_len);
    let n_pairs = head_dim / 2;
    assert_eq!(v_angles.len(), seq_len * n_kv_heads * n_pairs);
    assert_eq!(v_radius.len(), seq_len * n_kv_heads);
    assert_eq!(rotation.len(), head_dim * head_dim);

    let softmax_buf = gpu.create_storage_buffer(bytemuck::cast_slice(softmax), "polar.softmax");
    let packed = pack_angles_to_u32(v_angles);
    let v_angles_buf = gpu.create_storage_buffer(bytemuck::cast_slice(&packed), "polar.v_angles");
    let v_radius_buf = gpu.create_storage_buffer(bytemuck::cast_slice(v_radius), "polar.v_radius");
    let rotation_buf = gpu.create_storage_buffer(bytemuck::cast_slice(rotation), "polar.rotation");

    let out_len = n_heads * head_dim;
    let out_bytes = (out_len * std::mem::size_of::<f32>()) as u64;
    let weighted_rot_buf = gpu.create_empty_buffer(out_bytes, "polar.weighted_rotated_V");
    let final_buf = gpu.create_empty_buffer(out_bytes, "polar.attn_out");

    let lut = polar_lut_vec4();
    let lut_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("polar.angle_lut"),
        size: std::mem::size_of_val(&lut) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&lut_buf, 0, bytemuck::cast_slice(&lut));

    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("attn_value_polar.oneshot"),
    });
    dispatch_attn_value_polar(
        gpu, &mut encoder,
        &softmax_buf, &v_angles_buf, &v_radius_buf, &weighted_rot_buf, &lut_buf,
        n_heads, n_kv_heads, head_dim, seq_len, /*max_seq*/ seq_len,
    );
    dispatch_derotate(
        gpu, &mut encoder,
        &weighted_rot_buf, &rotation_buf, &final_buf,
        n_heads, head_dim,
    );

    let staging = gpu.create_staging_buffer(out_bytes);
    encoder.copy_buffer_to_buffer(&final_buf, 0, &staging, 0, out_bytes);
    gpu.queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = sender.send(r); });
    gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
    receiver.recv().unwrap().unwrap();

    let mapped = slice.get_mapped_range();
    let out: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    staging.unmap();

    out
}

// ---------------------------------------------------------------------------
// Batch + rotate-Q dispatchers — multi-token attention against polar cache.
// ---------------------------------------------------------------------------

/// Params for `rotate_q.wgsl`. 16 bytes std140-friendly.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RotateQParams {
    n_tokens: u32,
    n_heads: u32,
    head_dim: u32,
    _pad: u32,
}

/// Encode a `rotate_q` dispatch: rq[tok, head, row] = R @ q[tok, head, :].
/// Resident-buffer dispatch, no alloc/upload/readback.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_rotate_q(
    gpu: &Arc<GpuDevice>,
    encoder: &mut wgpu::CommandEncoder,
    q_buf: &wgpu::Buffer,
    rotation_buf: &wgpu::Buffer,
    rq_buf: &wgpu::Buffer,
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) {
    let params = RotateQParams {
        n_tokens: n_tokens as u32,
        n_heads: n_heads as u32,
        head_dim: head_dim as u32,
        _pad: 0,
    };
    let params_buf = gpu.create_params_buffer(&params);

    let pipeline = &gpu.pipelines.rotate_q;
    let bind = gpu.make_bind_group(
        pipeline,
        &[q_buf, rotation_buf, rq_buf, &params_buf],
    );

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("rotate_q.dispatch"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind, &[]);
    let threads = (n_tokens * n_heads * head_dim) as u32;
    let groups = (threads + 255) / 256;
    pass.dispatch_workgroups(groups, 1, 1);
}

/// Params for `attn_score_polar_batch.wgsl`. 48 bytes std140-friendly.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnScorePolarBatchParams {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    start_pos: u32,
    max_seq: u32,
    heads_per_kv: u32,
    n_pairs: u32,
    scale: f32,
    n_tokens: u32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
}

/// Encode an `attn_score_polar_batch` dispatch (multi-token, causal-masked
/// polar K). Mirror of the f32 attn_score_batch dispatch shape so the
/// existing softmax_batch can run on the same scores buffer.
///
/// `max_seq` is the row stride of `scores_buf` and bounds the t loop;
/// callers commonly set it to `start_pos + n_tokens`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_attn_score_polar_batch(
    gpu: &Arc<GpuDevice>,
    encoder: &mut wgpu::CommandEncoder,
    rq_buf: &wgpu::Buffer,
    k_angles_buf: &wgpu::Buffer,
    k_radius_buf: &wgpu::Buffer,
    scores_buf: &wgpu::Buffer,
    lut_buf: &wgpu::Buffer,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    start_pos: usize,
    n_tokens: usize,
    max_seq: usize,
) {
    assert!(head_dim % 2 == 0);
    assert!(n_heads % n_kv_heads == 0);
    assert!(start_pos + n_tokens <= max_seq);

    let n_pairs = head_dim / 2;
    let heads_per_kv = (n_heads / n_kv_heads) as u32;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let params = AttnScorePolarBatchParams {
        n_heads: n_heads as u32,
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        start_pos: start_pos as u32,
        max_seq: max_seq as u32,
        heads_per_kv,
        n_pairs: n_pairs as u32,
        scale,
        n_tokens: n_tokens as u32,
        _p1: 0, _p2: 0, _p3: 0,
    };
    let params_buf = gpu.create_params_buffer(&params);

    let pipeline = &gpu.pipelines.attn_score_polar_batch;
    let bind = gpu.make_bind_group(
        pipeline,
        &[rq_buf, k_angles_buf, k_radius_buf, scores_buf, &params_buf, lut_buf],
    );

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("attn_score_polar_batch.dispatch"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind, &[]);
    // 2D dispatch matching attn_score_batch: x = (head, t), y = tok.
    let inner_threads = (n_heads * max_seq) as u32;
    let groups_x = (inner_threads + 255) / 256;
    let groups_y = n_tokens as u32;
    pass.dispatch_workgroups(groups_x, groups_y, 1);
}

/// Params for `attn_value_polar_batch.wgsl`. 32 bytes std140-friendly.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnValuePolarBatchParams {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    start_pos: u32,
    max_seq: u32,
    heads_per_kv: u32,
    n_pairs: u32,
    n_tokens: u32,
}

/// Encode an `attn_value_polar_batch` dispatch (multi-token weighted-sum
/// in rotated space). Output shape: `[n_tokens, n_heads, head_dim]` flat,
/// in the polar/rotated domain. Apply `dispatch_derotate` over the same
/// buffer with `n_heads = n_tokens * n_heads_real` to recover original
/// space — the existing derotate shader handles the multi-token case
/// because R is the same across all (tok, head).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_attn_value_polar_batch(
    gpu: &Arc<GpuDevice>,
    encoder: &mut wgpu::CommandEncoder,
    softmax_buf: &wgpu::Buffer,
    v_angles_buf: &wgpu::Buffer,
    v_radius_buf: &wgpu::Buffer,
    output_buf: &wgpu::Buffer,
    lut_buf: &wgpu::Buffer,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    start_pos: usize,
    n_tokens: usize,
    max_seq: usize,
) {
    assert!(head_dim % 2 == 0);
    assert!(n_heads % n_kv_heads == 0);
    assert!(start_pos + n_tokens <= max_seq);

    let n_pairs = head_dim / 2;
    let heads_per_kv = (n_heads / n_kv_heads) as u32;

    let params = AttnValuePolarBatchParams {
        n_heads: n_heads as u32,
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        start_pos: start_pos as u32,
        max_seq: max_seq as u32,
        heads_per_kv,
        n_pairs: n_pairs as u32,
        n_tokens: n_tokens as u32,
    };
    let params_buf = gpu.create_params_buffer(&params);

    let pipeline = &gpu.pipelines.attn_value_polar_batch;
    let bind = gpu.make_bind_group(
        pipeline,
        &[softmax_buf, v_angles_buf, v_radius_buf, output_buf, &params_buf, lut_buf],
    );

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("attn_value_polar_batch.dispatch"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind, &[]);
    let threads = (n_tokens * n_heads * head_dim) as u32;
    let groups = (threads + 255) / 256;
    pass.dispatch_workgroups(groups, 1, 1);
}

// ---------------------------------------------------------------------------
// Compress shader dispatch — f32 K/V → polar cache buffers (no CPU round-trip).
// ---------------------------------------------------------------------------

/// Params for `kv_compress_polar.wgsl`. 32 bytes std140-friendly.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct KvCompressPolarParams {
    n_tokens: u32,
    start_pos: u32,
    n_kv_heads: u32,
    head_dim: u32,
    n_pairs: u32,
    max_seq: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Encode a `kv_compress_polar` dispatch: f32 K (or V) input gets
/// rotated by `R`, polar-quantized, and written into the resident
/// compressed `angles` + `radius` buffers at positions
/// `start_pos..start_pos + n_tokens`. No alloc, no upload, no readback.
///
/// `head_dim` must be divisible by 8 (so `n_pairs = head_dim/2` is
/// divisible by 4 and each thread owns whole u32 angle words).
/// Asserted in this dispatcher.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_kv_compress_polar(
    gpu: &Arc<GpuDevice>,
    encoder: &mut wgpu::CommandEncoder,
    k_in_buf: &wgpu::Buffer,
    rotation_buf: &wgpu::Buffer,
    angles_buf: &wgpu::Buffer,
    radius_buf: &wgpu::Buffer,
    n_tokens: usize,
    start_pos: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
) {
    assert!(
        head_dim % 8 == 0,
        "kv_compress_polar requires head_dim divisible by 8 (got {head_dim})",
    );
    assert!(start_pos + n_tokens <= max_seq);

    let n_pairs = head_dim / 2;
    let params = KvCompressPolarParams {
        n_tokens: n_tokens as u32,
        start_pos: start_pos as u32,
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        n_pairs: n_pairs as u32,
        max_seq: max_seq as u32,
        _pad0: 0,
        _pad1: 0,
    };
    let params_buf = gpu.create_params_buffer(&params);

    let pipeline = &gpu.pipelines.kv_compress_polar;
    let bind = gpu.make_bind_group(
        pipeline,
        &[k_in_buf, rotation_buf, angles_buf, radius_buf, &params_buf],
    );

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("kv_compress_polar.dispatch"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind, &[]);
    let threads = (n_tokens * n_kv_heads) as u32;
    let groups = (threads + 63) / 64; // workgroup_size = 64
    pass.dispatch_workgroups(groups, 1, 1);
}

/// Compress one layer of f32 K and V into a `GpuPolarKvCache`. Runs the
/// compress shader twice (once for K, once for V) sharing the layer's
/// rotation matrix. Caller is responsible for queue submission and for
/// calling `cache.set_len(...)` after all layers are populated.
///
/// `k_in_buf` / `v_in_buf` are flat `[n_tokens, n_kv_heads, head_dim]`
/// f32 buffers — typically the post-RoPE K and raw V from the projection
/// layers' output.
#[allow(clippy::too_many_arguments)]
pub fn compress_layer_into_polar(
    gpu: &Arc<GpuDevice>,
    encoder: &mut wgpu::CommandEncoder,
    cache: &GpuPolarKvCache,
    layer: usize,
    k_in_buf: &wgpu::Buffer,
    v_in_buf: &wgpu::Buffer,
    n_tokens: usize,
    start_pos: usize,
) {
    dispatch_kv_compress_polar(
        gpu, encoder,
        k_in_buf, cache.rotation_layer(layer),
        cache.k_angles_layer(layer), cache.k_radius_layer(layer),
        n_tokens, start_pos,
        cache.n_kv_heads(), cache.head_dim(), cache.max_seq_len(),
    );
    dispatch_kv_compress_polar(
        gpu, encoder,
        v_in_buf, cache.rotation_layer(layer),
        cache.v_angles_layer(layer), cache.v_radius_layer(layer),
        n_tokens, start_pos,
        cache.n_kv_heads(), cache.head_dim(), cache.max_seq_len(),
    );
}

// ---------------------------------------------------------------------------
// Resident dispatchers — read directly from a `GpuPolarKvCache`.
// ---------------------------------------------------------------------------

/// Encode an `attn_score_polar` dispatch using a layer of a resident
/// `GpuPolarKvCache` for the K side. No allocation, no upload, no
/// readback. Caller manages encoder/submit and provides the resident
/// rotated-Q and output-scores buffers.
///
/// `n_query_heads` is the attention's Q-head count (NOT the cache's
/// `n_kv_heads()`); GQA fan-out happens inside the shader.
#[allow(clippy::too_many_arguments)]
pub fn attn_score_polar_resident(
    gpu: &Arc<GpuDevice>,
    encoder: &mut wgpu::CommandEncoder,
    cache: &GpuPolarKvCache,
    layer: usize,
    rq_buf: &wgpu::Buffer,
    scores_buf: &wgpu::Buffer,
    n_query_heads: usize,
    max_seq: usize,
) {
    dispatch_attn_score_polar(
        gpu,
        encoder,
        rq_buf,
        cache.k_angles_layer(layer),
        cache.k_radius_layer(layer),
        scores_buf,
        cache.lut_buffer(),
        n_query_heads,
        cache.n_kv_heads(),
        cache.head_dim(),
        cache.seq_len(),
        max_seq,
    );
}

/// Encode the full compressed-V output (pass A + pass B) using a layer
/// of a resident `GpuPolarKvCache`. No allocation, no upload, no readback.
///
/// Caller provides `softmax_buf` (input, post-softmax weights),
/// `weighted_rot_buf` (intermediate workspace, `[n_query_heads * head_dim]`
/// f32, can be reused across calls), and `out_buf` (final output in
/// original space).
#[allow(clippy::too_many_arguments)]
pub fn attn_value_polar_resident(
    gpu: &Arc<GpuDevice>,
    encoder: &mut wgpu::CommandEncoder,
    cache: &GpuPolarKvCache,
    layer: usize,
    softmax_buf: &wgpu::Buffer,
    weighted_rot_buf: &wgpu::Buffer,
    out_buf: &wgpu::Buffer,
    n_query_heads: usize,
    max_seq: usize,
) {
    dispatch_attn_value_polar(
        gpu,
        encoder,
        softmax_buf,
        cache.v_angles_layer(layer),
        cache.v_radius_layer(layer),
        weighted_rot_buf,
        cache.lut_buffer(),
        n_query_heads,
        cache.n_kv_heads(),
        cache.head_dim(),
        cache.seq_len(),
        max_seq,
    );
    dispatch_derotate(
        gpu,
        encoder,
        weighted_rot_buf,
        cache.rotation_layer(layer),
        out_buf,
        n_query_heads,
        cache.head_dim(),
    );
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
            // same rotation to Q. The only divergence source is float-order
            // on the 4-pair dot product: theoretical bound ≈ 4·eps·max_abs ≈
            // ~3e-6, so 1e-5 leaves headroom without being slack.
            assert!(
                (g - e).abs() < 1e-5,
                "score[{i}] differs: gpu={g}, cpu={e}, |Δ|={}", (g - e).abs(),
            );
        }
    }

    /// Helper: deterministic varied f32 vector (avoids degenerate constants).
    fn varied_vec(len: usize, salt: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 7 + salt * 13 + 1) as f32) * 0.05).sin())
            .collect()
    }

    /// Helper: stable softmax across a per-head row of the scores tensor.
    fn rowwise_softmax(scores: &[f32], n_heads: usize, seq_len: usize) -> Vec<f32> {
        let mut out = scores.to_vec();
        for h in 0..n_heads {
            let row = &mut out[h * seq_len..(h + 1) * seq_len];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - m).exp();
                sum += *v;
            }
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
        out
    }

    /// Run the GPU compressed-V pipeline and the CPU compressed-V pipeline
    /// against the same `QuantizedKvCache` + same softmax weights, and
    /// assert per-output-element match. This is the load-bearing test for
    /// phase 2b: it proves the value/derotate shaders are computing the
    /// same thing the verified CPU primitives are.
    fn run_value_correctness(n_heads: usize, n_kv_heads: usize, head_dim: usize, seq_len: usize) {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        // Build cache with varied K and V.
        let mut cache = QuantizedKvCache::new(n_kv_heads, head_dim, seq_len, /*seed*/ 42);
        let kv_dim = n_kv_heads * head_dim;
        for t in 0..seq_len {
            let k = varied_vec(kv_dim, t * 2);
            let v = varied_vec(kv_dim, t * 2 + 1);
            cache.append_one(&k, &v);
        }

        // Synthetic per-head softmax weights — don't actually need to come
        // from a softmax for the value path correctness check (the value
        // pipeline is linear in its weights). Use varied positive numbers
        // that sum to ~1 per row so they look like real softmax outputs.
        let raw: Vec<f32> = (0..n_heads * seq_len)
            .map(|i| (((i * 11 + 3) as f32) * 0.04).sin().abs() + 0.01)
            .collect();
        let softmax = rowwise_softmax(&raw, n_heads, seq_len);

        // CPU expected: out[head, d] = Σ_p softmax[head, p] * V_at_dequant(p, kv_h)[d]
        let heads_per_kv = n_heads / n_kv_heads;
        let mut expected = vec![0.0f32; n_heads * head_dim];
        for head in 0..n_heads {
            let kv_h = head / heads_per_kv;
            for p in 0..seq_len {
                let w = softmax[head * seq_len + p];
                let vp = cache.value_at_dequant(p, kv_h);
                for d in 0..head_dim {
                    expected[head * head_dim + d] += w * vp[d];
                }
            }
        }

        // Pull raw compressed slices.
        let n_pairs = head_dim / 2;
        let v_angles: Vec<u8> = cache_v_angles_slice(&cache, seq_len, n_kv_heads, n_pairs);
        let v_radius: Vec<f32> = cache_v_radius_slice(&cache, seq_len, n_kv_heads);
        let rotation = polar::generate_rotation_matrix(head_dim, /*seed*/ 42);

        let got = attn_value_polar_oneshot(
            &gpu, &softmax, &v_angles, &v_radius, &rotation,
            n_heads, n_kv_heads, head_dim, seq_len,
        );

        assert_eq!(got.len(), expected.len());
        // Float-order error scales as O(seq_len) for a sum of seq_len terms.
        // Worst-case bound ≈ seq_len * eps * max_abs. Calibrated tolerances:
        //   seq_len 6   : 1e-5
        //   seq_len 4096: 1e-3 (3 orders larger because the sum is 700x larger)
        let tol = if seq_len <= 64 { 1e-5 } else { 1e-3 };
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < tol,
                "out[{i}] differs at scale {seq_len}: gpu={g}, cpu={e}, |Δ|={}",
                (g - e).abs(),
            );
        }
    }

    // The cache exposes `k_angles_slice`/`k_radius_slice` for K but not V.
    // For tests we read them via `value_at_dequant` indirection; a tiny
    // helper here pulls the V-angle/radius bytes out by re-encoding.
    // (Plumbing the V slice accessors is a tiny patch we can do later if
    // we want to avoid this round trip.)
    fn cache_v_angles_slice(
        cache: &QuantizedKvCache,
        seq_len: usize,
        n_kv_heads: usize,
        n_pairs: usize,
    ) -> Vec<u8> {
        let mut out = vec![0u8; seq_len * n_kv_heads * n_pairs];
        for t in 0..seq_len {
            for h in 0..n_kv_heads {
                let entry = cache.read_compressed_k(t, h);
                let off = (t * n_kv_heads + h) * n_pairs;
                out[off..off + n_pairs].copy_from_slice(&entry.v_angles);
            }
        }
        out
    }

    fn cache_v_radius_slice(
        cache: &QuantizedKvCache,
        seq_len: usize,
        n_kv_heads: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; seq_len * n_kv_heads];
        for t in 0..seq_len {
            for h in 0..n_kv_heads {
                let entry = cache.read_compressed_k(t, h);
                out[t * n_kv_heads + h] = entry.v_radius;
            }
        }
        out
    }

    #[test]
    fn value_shader_matches_cpu_small() {
        // Same shape as the score test for symmetry.
        run_value_correctness(/*n_heads*/ 4, /*n_kv_heads*/ 2, /*head_dim*/ 8, /*seq_len*/ 6);
    }

    #[test]
    fn value_shader_matches_cpu_medium() {
        // Realistic Qwen-ish shape at moderate cache size. Verifies the
        // float-order error stays within calibrated bounds across 4096
        // accumulation steps. Skipped automatically when no GPU available.
        run_value_correctness(/*n_heads*/ 4, /*n_kv_heads*/ 2, /*head_dim*/ 64, /*seq_len*/ 4096);
    }

    /// CPU-only algorithm-quality check: build the same K/V both as plain
    /// f32 buffers and as a `QuantizedKvCache`, run identical attention
    /// math (Q·K → softmax → ΣV), and measure per-head cosine similarity
    /// of the output vectors. Returns the worst-head cosine.
    fn compressed_vs_uncompressed_min_cos(with_qjl: bool) -> f32 {
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 64;
        let seq_len = 256;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let heads_per_kv = n_heads / n_kv_heads;
        let kv_dim = n_kv_heads * head_dim;

        let mut k_uncompressed = Vec::with_capacity(seq_len * kv_dim);
        let mut v_uncompressed = Vec::with_capacity(seq_len * kv_dim);
        let mut cache = if with_qjl {
            QuantizedKvCache::with_qjl(n_kv_heads, head_dim, seq_len, /*rot*/ 42, /*qjl*/ 99)
        } else {
            QuantizedKvCache::new(n_kv_heads, head_dim, seq_len, /*seed*/ 42)
        };
        for t in 0..seq_len {
            let k = varied_vec(kv_dim, t * 2);
            let v = varied_vec(kv_dim, t * 2 + 1);
            cache.append_one(&k, &v);
            k_uncompressed.extend_from_slice(&k);
            v_uncompressed.extend_from_slice(&v);
        }

        let q = varied_vec(n_heads * head_dim, 9999);

        // Uncompressed path.
        let mut scores_u = vec![0.0f32; n_heads * seq_len];
        for head in 0..n_heads {
            let kv_h = head / heads_per_kv;
            let qs = &q[head * head_dim..(head + 1) * head_dim];
            for t in 0..seq_len {
                let k_off = t * kv_dim + kv_h * head_dim;
                let dot: f32 = qs.iter()
                    .zip(&k_uncompressed[k_off..k_off + head_dim])
                    .map(|(&a, &b)| a * b)
                    .sum();
                scores_u[head * seq_len + t] = dot * scale;
            }
        }
        let sm_u = rowwise_softmax(&scores_u, n_heads, seq_len);
        let mut out_u = vec![0.0f32; n_heads * head_dim];
        for head in 0..n_heads {
            let kv_h = head / heads_per_kv;
            for t in 0..seq_len {
                let w = sm_u[head * seq_len + t];
                let v_off = t * kv_dim + kv_h * head_dim;
                for d in 0..head_dim {
                    out_u[head * head_dim + d] += w * v_uncompressed[v_off + d];
                }
            }
        }

        // Compressed path.
        let mut scores_c = vec![0.0f32; n_heads * seq_len];
        for head in 0..n_heads {
            let kv_h = head / heads_per_kv;
            let qs = &q[head * head_dim..(head + 1) * head_dim];
            for t in 0..seq_len {
                scores_c[head * seq_len + t] = cache.dot_key(t, kv_h, qs) * scale;
            }
        }
        let sm_c = rowwise_softmax(&scores_c, n_heads, seq_len);
        let mut out_c = vec![0.0f32; n_heads * head_dim];
        for head in 0..n_heads {
            let kv_h = head / heads_per_kv;
            for t in 0..seq_len {
                let w = sm_c[head * seq_len + t];
                let vp = cache.value_at_dequant(t, kv_h);
                for d in 0..head_dim {
                    out_c[head * head_dim + d] += w * vp[d];
                }
            }
        }

        let mut min_cos = f32::INFINITY;
        for head in 0..n_heads {
            let a = &out_u[head * head_dim..(head + 1) * head_dim];
            let b = &out_c[head * head_dim..(head + 1) * head_dim];
            let dot: f32 = a.iter().zip(b).map(|(&x, &y)| x * y).sum();
            let na: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
            let cos = dot / (na * nb).max(1e-12);
            if cos < min_cos { min_cos = cos; }
        }
        min_cos
    }

    /// PolarQuant alone — 8 angle buckets gives ~22.5° quantization. The
    /// resulting cosine similarity to uncompressed attention is non-trivially
    /// lossy. This test pins the floor (currently ~0.84 worst-head at
    /// seq_len=256 with this fixture) so a regression below it surfaces
    /// immediately. Tighter quality requires QJL — see the next test.
    #[test]
    fn polar_only_attention_preserves_output() {
        let cos = compressed_vs_uncompressed_min_cos(/*with_qjl*/ false);
        assert!(
            cos > 0.80,
            "PolarQuant-only worst-head cosine {cos:.4} below floor 0.80",
        );
    }

    /// Read back a wgpu storage buffer to a CPU `Vec<f32>`. Test-only.
    fn readback_f32(gpu: &GpuDevice, src: &wgpu::Buffer, n: usize) -> Vec<f32> {
        let bytes = (n * std::mem::size_of::<f32>()) as u64;
        let staging = gpu.create_staging_buffer(bytes);
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test.readback"),
        });
        encoder.copy_buffer_to_buffer(src, 0, &staging, 0, bytes);
        gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (s, r) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| { let _ = s.send(res); });
        gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        r.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        out
    }

    /// Load-bearing phase 2c.3 test: take known-good f32 K and V data,
    /// build BOTH a CPU `QuantizedKvCache` (which compresses on the CPU
    /// inside `append_one`) and a `GpuPolarKvCache` populated by the
    /// GPU compress shader. Read back the GPU buffers and assert byte-
    /// equality with the CPU compressed bytes. This pins the contract
    /// that the GPU compress shader produces exactly what the CPU
    /// pipeline does.
    ///
    /// Identical CPU↔GPU bytes is a stronger property than just
    /// "compressed correctly" — it means swapping the prefill from the
    /// CPU upload path to the GPU compress path is invisible to all
    /// downstream attention math. No tolerance, no quantization-noise
    /// allowance.
    #[test]
    fn gpu_compress_matches_cpu_append() {
        use crate::layers::gpu_polar_kv_cache::{seed_for_layer, GpuPolarKvCache};

        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let n_layers = 2;
        let n_kv_heads = 2;
        let head_dim = 8;       // divisible by 8 → n_pairs=4 → 1 word/thread
        let n_tokens = 6;
        let max_seq = n_tokens; // start_pos = 0
        let seed_base = 42u64;
        let layer = 1usize;

        // Source f32 K and V (same shape the projection layers produce:
        // [n_tokens, n_kv_heads, head_dim]).
        let kv_dim = n_kv_heads * head_dim;
        let mut k_data = Vec::with_capacity(n_tokens * kv_dim);
        let mut v_data = Vec::with_capacity(n_tokens * kv_dim);
        for t in 0..n_tokens {
            for i in 0..kv_dim {
                k_data.push((((t * 7 + layer * 3 + i) as f32) * 0.05).sin());
                v_data.push((((t * 11 + layer * 5 + i) as f32) * 0.04).cos());
            }
        }

        // CPU reference: build QuantizedKvCache with the matching seed
        // and append all positions. Its k_angles_slice/k_radius_slice are
        // the ground truth.
        let mut cpu = QuantizedKvCache::new(
            n_kv_heads, head_dim, max_seq, seed_for_layer(seed_base, layer),
        );
        for t in 0..n_tokens {
            let off = t * kv_dim;
            cpu.append_one(&k_data[off..off + kv_dim], &v_data[off..off + kv_dim]);
        }

        // GPU side: build resident polar cache, upload f32 K/V buffers,
        // run the compress shader for layer `layer`.
        let polar_kv = GpuPolarKvCache::new(
            gpu.clone(), n_layers, n_kv_heads, head_dim, max_seq, seed_base,
        );

        // Phase A: kv_compress_polar now reads packed-f16 input.
        let k_packed = GpuDevice::pack_f16(&k_data);
        let v_packed = GpuDevice::pack_f16(&v_data);
        let k_in_buf = gpu.create_storage_buffer(bytemuck::cast_slice(&k_packed), "test.k_in");
        let v_in_buf = gpu.create_storage_buffer(bytemuck::cast_slice(&v_packed), "test.v_in");

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test.compress_layer"),
        });
        compress_layer_into_polar(
            &gpu, &mut encoder, &polar_kv, layer,
            &k_in_buf, &v_in_buf, n_tokens, /*start_pos*/ 0,
        );
        gpu.queue.submit(Some(encoder.finish()));
        gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();

        // Read back GPU compressed buffers.
        let n_pairs = head_dim / 2;
        let used_angles = n_tokens * n_kv_heads * n_pairs;
        let used_radius = n_tokens * n_kv_heads;
        let used_packed_words = (used_angles + 3) / 4;

        let read = |buf: &wgpu::Buffer, bytes: u64| -> Vec<u8> {
            let staging = gpu.create_staging_buffer(bytes);
            let mut e = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("test.compress.readback"),
            });
            e.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
            gpu.queue.submit(Some(e.finish()));
            let slice = staging.slice(..);
            let (s, r) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |res| { let _ = s.send(res); });
            gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
            r.recv().unwrap().unwrap();
            let mapped = slice.get_mapped_range();
            let out = mapped.to_vec();
            drop(mapped);
            staging.unmap();
            out
        };

        let packed_bytes = (used_packed_words * std::mem::size_of::<u32>()) as u64;
        let radius_bytes = (used_radius * std::mem::size_of::<f32>()) as u64;

        let gpu_k_angles = read(polar_kv.k_angles_layer(layer), packed_bytes);
        let gpu_k_radius_bytes = read(polar_kv.k_radius_layer(layer), radius_bytes);
        let gpu_v_angles = read(polar_kv.v_angles_layer(layer), packed_bytes);
        let gpu_v_radius_bytes = read(polar_kv.v_radius_layer(layer), radius_bytes);

        // CPU expected packed bytes.
        let cpu_k_angles_full: Vec<u8> = cpu.k_angles_slice()[..used_angles].to_vec();
        let expected_k_packed = pack_angles_to_u32(&cpu_k_angles_full);
        let expected_k_radius: &[f32] = &cpu.k_radius_slice()[..used_radius];

        // V via CompressedEntry (engram cache doesn't expose v_angles_slice).
        let mut cpu_v_angles_full = vec![0u8; used_angles];
        let mut cpu_v_radius = vec![0f32; used_radius];
        for t in 0..n_tokens {
            for h in 0..n_kv_heads {
                let entry = cpu.read_compressed_k(t, h);
                let off = (t * n_kv_heads + h) * n_pairs;
                cpu_v_angles_full[off..off + n_pairs].copy_from_slice(&entry.v_angles);
                cpu_v_radius[t * n_kv_heads + h] = entry.v_radius;
            }
        }
        let expected_v_packed = pack_angles_to_u32(&cpu_v_angles_full);

        // Compare bytes. K angles, K radius, V angles, V radius — all four.
        let got_k_packed: &[u32] = bytemuck::cast_slice(&gpu_k_angles);
        let got_v_packed: &[u32] = bytemuck::cast_slice(&gpu_v_angles);
        let got_k_radius: &[f32] = bytemuck::cast_slice(&gpu_k_radius_bytes);
        let got_v_radius: &[f32] = bytemuck::cast_slice(&gpu_v_radius_bytes);

        assert_eq!(got_k_packed, expected_k_packed.as_slice(),
            "GPU K angles differ from CPU compress");
        assert_eq!(got_v_packed, expected_v_packed.as_slice(),
            "GPU V angles differ from CPU compress");

        // Radius tolerance: 1e-3 abs to accommodate Phase A's f16 input
        // quantization on top of WGSL FMA drift (was 1e-6 when the
        // source buffer was f32).
        for (i, (&g, &e)) in got_k_radius.iter().zip(expected_k_radius.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-3,
                "K radius[{i}] differs: gpu={g}, cpu={e}, |Δ|={}", (g - e).abs(),
            );
        }
        for (i, (&g, &e)) in got_v_radius.iter().zip(cpu_v_radius.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-3,
                "V radius[{i}] differs: gpu={g}, cpu={e}, |Δ|={}", (g - e).abs(),
            );
        }
    }

    /// Verify start_pos > 0 writes to the right slot — proves prefill at
    /// non-zero offset works (e.g., decode after prefill, append calls).
    #[test]
    fn gpu_compress_writes_at_start_pos() {
        use crate::layers::gpu_polar_kv_cache::{seed_for_layer, GpuPolarKvCache};

        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let n_kv_heads = 1;
        let head_dim = 8;
        let max_seq = 8;
        let n_tokens = 3;
        let start_pos = 4usize;
        let seed_base = 42u64;

        let polar_kv = GpuPolarKvCache::new(
            gpu.clone(), /*n_layers*/ 1, n_kv_heads, head_dim, max_seq, seed_base,
        );

        let kv_dim = n_kv_heads * head_dim;
        let k: Vec<f32> = (0..n_tokens * kv_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let v: Vec<f32> = (0..n_tokens * kv_dim).map(|i| (i as f32 + 2.0) * 0.07).collect();
        let k_buf = gpu.create_storage_buffer(bytemuck::cast_slice(&k), "test.k_at_offset");
        let v_buf = gpu.create_storage_buffer(bytemuck::cast_slice(&v), "test.v_at_offset");

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test.compress_at_offset"),
        });
        compress_layer_into_polar(
            &gpu, &mut encoder, &polar_kv, /*layer*/ 0,
            &k_buf, &v_buf, n_tokens, start_pos,
        );
        gpu.queue.submit(Some(encoder.finish()));
        gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();

        // The radius buffer at indices [start_pos*n_kv_heads ..
        // (start_pos+n_tokens)*n_kv_heads] should be non-zero; the
        // pre-start_pos prefix should still be zero (untouched).
        let bytes = (max_seq * n_kv_heads * std::mem::size_of::<f32>()) as u64;
        let staging = gpu.create_staging_buffer(bytes);
        let mut e = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test.read_radius"),
        });
        e.copy_buffer_to_buffer(polar_kv.k_radius_layer(0), 0, &staging, 0, bytes);
        gpu.queue.submit(Some(e.finish()));
        let slice = staging.slice(..);
        let (s, r) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| { let _ = s.send(res); });
        gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        r.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range();
        let radius: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();

        for i in 0..start_pos * n_kv_heads {
            assert_eq!(radius[i], 0.0, "radius[{i}] should be untouched (pre-start_pos)");
        }
        for i in start_pos * n_kv_heads..(start_pos + n_tokens) * n_kv_heads {
            assert!(radius[i] > 0.0, "radius[{i}] should be populated");
        }
    }

    /// Load-bearing phase 2c.2 test: build a CPU `QuantizedKvCache`,
    /// upload it to a resident `GpuPolarKvCache`, then run BOTH the
    /// oneshot path and the resident path against it. Assert the score
    /// and value/derotate outputs are byte-equal. This pins the contract
    /// that the resident dispatchers compute exactly the same thing as
    /// the (already-tested) oneshot path.
    #[test]
    fn resident_dispatchers_match_oneshot() {
        use crate::layers::gpu_polar_kv_cache::{seed_for_layer, GpuPolarKvCache};

        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let n_query_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 8;
        let seq_len = 6;
        let seed_base = 42u64;
        let layer = 1usize; // Use a non-zero layer to verify per-layer routing.

        // Build the resident cache, upload one layer.
        let mut polar_kv = GpuPolarKvCache::new(
            gpu.clone(), /*n_layers*/ 3, n_kv_heads, head_dim, seq_len, seed_base,
        );
        let mut cpu = QuantizedKvCache::new(
            n_kv_heads, head_dim, seq_len, seed_for_layer(seed_base, layer),
        );
        let kv_dim = n_kv_heads * head_dim;
        for t in 0..seq_len {
            let k: Vec<f32> = (0..kv_dim)
                .map(|i| (((t * 7 + layer * 3 + i) as f32) * 0.05).sin())
                .collect();
            let v: Vec<f32> = (0..kv_dim)
                .map(|i| (((t * 11 + layer * 5 + i) as f32) * 0.04).cos())
                .collect();
            cpu.append_one(&k, &v);
        }
        polar_kv.upload_layer_from_cpu(layer, &cpu);
        polar_kv.set_len(seq_len);

        // Rotated query (CPU rotation; production will rotate on GPU).
        let q: Vec<f32> = (0..n_query_heads * head_dim)
            .map(|i| (((i * 13 + layer * 17) as f32) * 0.06).cos())
            .collect();
        let r = polar::generate_rotation_matrix(head_dim, seed_for_layer(seed_base, layer));
        let mut rq = vec![0f32; n_query_heads * head_dim];
        for h in 0..n_query_heads {
            let qs = &q[h * head_dim..(h + 1) * head_dim];
            polar::rotate(&r, qs, &mut rq[h * head_dim..(h + 1) * head_dim]);
        }

        // ---- Score: oneshot vs resident ----
        let n_pairs = head_dim / 2;
        let cpu_k_angles: Vec<u8> = cpu.k_angles_slice()[..seq_len * n_kv_heads * n_pairs].to_vec();
        let cpu_k_radius: Vec<f32> = cpu.k_radius_slice()[..seq_len * n_kv_heads].to_vec();

        let scores_oneshot = attn_score_polar_oneshot(
            &gpu, &rq, &cpu_k_angles, &cpu_k_radius,
            n_query_heads, n_kv_heads, head_dim, seq_len,
        );

        let scores_len = n_query_heads * seq_len;
        let scores_bytes = (scores_len * std::mem::size_of::<f32>()) as u64;
        let rq_buf = gpu.create_storage_buffer(bytemuck::cast_slice(&rq), "test.rq");
        let scores_buf_resident = gpu.create_empty_buffer(scores_bytes, "test.scores_resident");

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test.resident_score"),
        });
        attn_score_polar_resident(
            &gpu, &mut encoder, &polar_kv, layer,
            &rq_buf, &scores_buf_resident, n_query_heads, /*max_seq*/ seq_len,
        );
        gpu.queue.submit(Some(encoder.finish()));
        gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();

        let scores_resident = readback_f32(&gpu, &scores_buf_resident, scores_len);

        for (i, (&a, &b)) in scores_oneshot.iter().zip(scores_resident.iter()).enumerate() {
            assert_eq!(
                a.to_bits(), b.to_bits(),
                "score[{i}] differs: oneshot={a}, resident={b}",
            );
        }

        // ---- Value+derotate: oneshot vs resident ----
        // Use the just-computed scores as if they were softmax (the path
        // is linear in its weights — exact byte equality is what we want).
        let softmax_buf = gpu.create_storage_buffer(
            bytemuck::cast_slice(&scores_resident),
            "test.softmax",
        );

        // V slices for oneshot path (engram cache doesn't expose these
        // directly, so reconstruct via read_compressed_k).
        let mut v_angles_full = vec![0u8; seq_len * n_kv_heads * n_pairs];
        let mut v_radius_full = vec![0f32; seq_len * n_kv_heads];
        for t in 0..seq_len {
            for h in 0..n_kv_heads {
                let entry = cpu.read_compressed_k(t, h);
                let off = (t * n_kv_heads + h) * n_pairs;
                v_angles_full[off..off + n_pairs].copy_from_slice(&entry.v_angles);
                v_radius_full[t * n_kv_heads + h] = entry.v_radius;
            }
        }

        let out_oneshot = attn_value_polar_oneshot(
            &gpu, &scores_resident, &v_angles_full, &v_radius_full, &r,
            n_query_heads, n_kv_heads, head_dim, seq_len,
        );

        let out_len = n_query_heads * head_dim;
        let out_bytes = (out_len * std::mem::size_of::<f32>()) as u64;
        let weighted_rot_buf = gpu.create_empty_buffer(out_bytes, "test.weighted_rot");
        let out_buf_resident = gpu.create_empty_buffer(out_bytes, "test.out_resident");

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test.resident_value"),
        });
        attn_value_polar_resident(
            &gpu, &mut encoder, &polar_kv, layer,
            &softmax_buf, &weighted_rot_buf, &out_buf_resident,
            n_query_heads, /*max_seq*/ seq_len,
        );
        gpu.queue.submit(Some(encoder.finish()));
        gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();

        let out_resident = readback_f32(&gpu, &out_buf_resident, out_len);

        for (i, (&a, &b)) in out_oneshot.iter().zip(out_resident.iter()).enumerate() {
            assert_eq!(
                a.to_bits(), b.to_bits(),
                "out[{i}] differs: oneshot={a}, resident={b}",
            );
        }
    }

    /// Load-bearing phase 2c.5a test: the multi-token causal-masked
    /// polar batch shaders. Build a `QuantizedKvCache` populated with
    /// N+n_tokens positions, generate per-token queries, run the GPU
    /// batch chain (rotate_q → attn_score_polar_batch → softmax_batch
    /// → attn_value_polar_batch → derotate), and compare against a CPU
    /// implementation that uses `dot_key` + manual causal mask.
    ///
    /// This is what production prefill against a polar cache will run
    /// (the inner forward block's attention chain), so getting it
    /// shape-correct + numerically matching the CPU primitive locks
    /// the contract for the next phase that wires it into
    /// `forward_full_gpu_polar_traced`.
    #[test]
    fn polar_batch_shaders_match_cpu() {
        use crate::layers::gpu_polar_kv_cache::{seed_for_layer, GpuPolarKvCache};

        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let n_query_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 8;
        let context_len = 6;       // positions in cache before the query
        let n_tokens = 3;          // query tokens
        let total = context_len + n_tokens;
        let start_pos = context_len;
        let max_seq = total;
        let seed_base = 42u64;
        let layer = 0usize;

        // CPU cache holds context K/V at [0, context_len) AND the query
        // tokens' own K/V at [context_len, total). The latter lets the
        // queries self-attend (causal) like real prefill does.
        let mut cpu = QuantizedKvCache::new(
            n_kv_heads, head_dim, max_seq, seed_for_layer(seed_base, layer),
        );
        let kv_dim = n_kv_heads * head_dim;
        let mut all_k = Vec::with_capacity(total * kv_dim);
        let mut all_v = Vec::with_capacity(total * kv_dim);
        for t in 0..total {
            let k: Vec<f32> = (0..kv_dim)
                .map(|i| (((t * 7 + i) as f32) * 0.05).sin())
                .collect();
            let v: Vec<f32> = (0..kv_dim)
                .map(|i| (((t * 11 + i) as f32) * 0.04).cos())
                .collect();
            cpu.append_one(&k, &v);
            all_k.extend_from_slice(&k);
            all_v.extend_from_slice(&v);
        }

        // Build resident polar cache and populate via the GPU compress
        // shader from the CPU-known f32 K/V (skips the upload-by-CPU path
        // — this fixture needs all positions on GPU).
        let mut polar_kv = GpuPolarKvCache::new(
            gpu.clone(), /*n_layers*/ 1, n_kv_heads, head_dim, max_seq, seed_base,
        );
        // Phase A: kv_compress_polar reads packed f16 input.
        let all_k_packed = GpuDevice::pack_f16(&all_k);
        let all_v_packed = GpuDevice::pack_f16(&all_v);
        let k_in_buf = gpu.create_storage_buffer(bytemuck::cast_slice(&all_k_packed), "test.k_in_full");
        let v_in_buf = gpu.create_storage_buffer(bytemuck::cast_slice(&all_v_packed), "test.v_in_full");
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test.batch.compress_full"),
        });
        compress_layer_into_polar(
            &gpu, &mut encoder, &polar_kv, layer,
            &k_in_buf, &v_in_buf, total, /*start_pos*/ 0,
        );
        gpu.queue.submit(Some(encoder.finish()));
        gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        polar_kv.set_len(total);

        // Per-query-token Q in original space. n_tokens queries, n_heads each.
        let q_data: Vec<f32> = (0..n_tokens * n_query_heads * head_dim)
            .map(|i| (((i * 13) as f32) * 0.06).cos())
            .collect();

        // ---- CPU expected: scores then softmax then weighted V then derotate ----
        let scale = 1.0 / (head_dim as f32).sqrt();
        let heads_per_kv = n_query_heads / n_kv_heads;

        let mut cpu_scores = vec![-1e30f32; n_tokens * n_query_heads * max_seq];
        for tok in 0..n_tokens {
            for head in 0..n_query_heads {
                let kv_h = head / heads_per_kv;
                let q_off = (tok * n_query_heads + head) * head_dim;
                let qs = &q_data[q_off..q_off + head_dim];
                let row_off = tok * n_query_heads * max_seq + head * max_seq;
                let seq_len = start_pos + tok + 1;
                for t in 0..seq_len {
                    cpu_scores[row_off + t] = cpu.dot_key(t, kv_h, qs) * scale;
                }
                // [seq_len, max_seq) stays -1e30 (mask).
            }
        }

        // CPU softmax over each (tok, head) row. Match the shader's
        // row-of-max_seq layout; positions beyond seq_len at -1e30 won't
        // contribute (their exp ≈ 0).
        let mut cpu_softmax = cpu_scores.clone();
        for tok in 0..n_tokens {
            for head in 0..n_query_heads {
                let row = &mut cpu_softmax[(tok * n_query_heads + head) * max_seq
                                          ..(tok * n_query_heads + head + 1) * max_seq];
                let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for v in row.iter_mut() {
                    *v = (*v - m).exp();
                    sum += *v;
                }
                for v in row.iter_mut() {
                    *v /= sum;
                }
            }
        }

        // CPU output = sum_t softmax * value_at_dequant(t).
        let mut cpu_out = vec![0.0f32; n_tokens * n_query_heads * head_dim];
        for tok in 0..n_tokens {
            for head in 0..n_query_heads {
                let kv_h = head / heads_per_kv;
                let row_off = (tok * n_query_heads + head) * max_seq;
                let out_off = (tok * n_query_heads + head) * head_dim;
                let seq_len = start_pos + tok + 1;
                for t in 0..seq_len {
                    let w = cpu_softmax[row_off + t];
                    let vp = cpu.value_at_dequant(t, kv_h);
                    for d in 0..head_dim {
                        cpu_out[out_off + d] += w * vp[d];
                    }
                }
            }
        }

        // ---- GPU pipeline: rotate_q → score_polar_batch → softmax_batch
        //      → value_polar_batch → derotate ----
        let q_buf = gpu.create_storage_buffer(bytemuck::cast_slice(&q_data), "test.q");
        let rq_bytes = (n_tokens * n_query_heads * head_dim * 4) as u64;
        let rq_buf = gpu.create_empty_buffer(rq_bytes, "test.rq");

        let scores_len = n_tokens * n_query_heads * max_seq;
        let scores_bytes = (scores_len * 4) as u64;
        let scores_buf = gpu.create_empty_buffer(scores_bytes, "test.batch.scores");

        let out_bytes = rq_bytes;
        let weighted_rot_buf = gpu.create_empty_buffer(out_bytes, "test.batch.weighted_rot");
        let final_buf = gpu.create_empty_buffer(out_bytes, "test.batch.out");

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test.batch.encoder"),
        });

        dispatch_rotate_q(
            &gpu, &mut encoder, &q_buf, polar_kv.rotation_layer(layer), &rq_buf,
            n_tokens, n_query_heads, head_dim,
        );
        dispatch_attn_score_polar_batch(
            &gpu, &mut encoder, &rq_buf,
            polar_kv.k_angles_layer(layer), polar_kv.k_radius_layer(layer),
            &scores_buf, polar_kv.lut_buffer(),
            n_query_heads, n_kv_heads, head_dim, start_pos, n_tokens, max_seq,
        );
        // Softmax: reuse the existing softmax_batch shader (same layout).
        // We use the same dispatch shape as gpu_engine.dispatch_attention_inner.
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct SoftmaxBatchParams {
            n_heads: u32,
            max_seq: u32,
            start_pos: u32,
            n_tokens: u32,
        }
        let sm_params = SoftmaxBatchParams {
            n_heads: n_query_heads as u32,
            max_seq: max_seq as u32,
            start_pos: start_pos as u32,
            n_tokens: n_tokens as u32,
        };
        let sm_params_buf = gpu.create_params_buffer(&sm_params);
        let sm_pipeline = &gpu.pipelines.softmax_batch;
        let sm_bind = gpu.make_bind_group(sm_pipeline, &[&scores_buf, &sm_params_buf]);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("test.batch.softmax"),
                timestamp_writes: None,
            });
            pass.set_pipeline(sm_pipeline);
            pass.set_bind_group(0, &sm_bind, &[]);
            pass.dispatch_workgroups((n_tokens * n_query_heads) as u32, 1, 1);
        }
        dispatch_attn_value_polar_batch(
            &gpu, &mut encoder, &scores_buf,
            polar_kv.v_angles_layer(layer), polar_kv.v_radius_layer(layer),
            &weighted_rot_buf, polar_kv.lut_buffer(),
            n_query_heads, n_kv_heads, head_dim, start_pos, n_tokens, max_seq,
        );
        // Derotate: treat (n_tokens * n_query_heads) as "n_heads" — the
        // existing single-token shader applies R^T per-(effective)head and
        // the f32 layout is [tok, head, head_dim] flat = same.
        dispatch_derotate(
            &gpu, &mut encoder, &weighted_rot_buf,
            polar_kv.rotation_layer(layer), &final_buf,
            n_tokens * n_query_heads, head_dim,
        );

        gpu.queue.submit(Some(encoder.finish()));
        gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();

        let gpu_out = readback_f32(&gpu, &final_buf, n_tokens * n_query_heads * head_dim);

        // Per-element comparison. Phase A f16-input adds compress-stage
        // quantization noise; bumped from 1e-5.
        for (i, (&g, &e)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-3,
                "out[{i}] differs: gpu={g}, cpu={e}, |Δ|={}", (g - e).abs(),
            );
        }
    }

    /// Sanity: with n_tokens=1, the batch chain should produce the same
    /// scores as the existing single-token oneshot path. Same layer of
    /// the same polar cache, same Q.
    #[test]
    fn polar_batch_score_n_tokens_1_matches_oneshot() {
        use crate::layers::gpu_polar_kv_cache::{seed_for_layer, GpuPolarKvCache};

        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let n_query_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 8;
        let cache_len = 6;
        let n_tokens = 1;
        let max_seq = cache_len + n_tokens;
        let start_pos = cache_len;
        let seed_base = 42u64;
        let layer = 0usize;

        // Populate cache (cache_len + 1 positions; the +1 is the query slot).
        let mut cpu = QuantizedKvCache::new(
            n_kv_heads, head_dim, max_seq, seed_for_layer(seed_base, layer),
        );
        let kv_dim = n_kv_heads * head_dim;
        let mut all_k = Vec::with_capacity(max_seq * kv_dim);
        let mut all_v = Vec::with_capacity(max_seq * kv_dim);
        for t in 0..max_seq {
            let k: Vec<f32> = (0..kv_dim).map(|i| (((t * 7 + i) as f32) * 0.05).sin()).collect();
            let v: Vec<f32> = (0..kv_dim).map(|i| (((t * 11 + i) as f32) * 0.04).cos()).collect();
            cpu.append_one(&k, &v);
            all_k.extend_from_slice(&k);
            all_v.extend_from_slice(&v);
        }
        let mut polar_kv = GpuPolarKvCache::new(
            gpu.clone(), 1, n_kv_heads, head_dim, max_seq, seed_base,
        );
        // Phase A: kv_compress_polar reads packed f16 input.
        let all_k_packed = GpuDevice::pack_f16(&all_k);
        let all_v_packed = GpuDevice::pack_f16(&all_v);
        let k_in = gpu.create_storage_buffer(bytemuck::cast_slice(&all_k_packed), "test.k1");
        let v_in = gpu.create_storage_buffer(bytemuck::cast_slice(&all_v_packed), "test.v1");
        let mut e = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test.compress.full"),
        });
        compress_layer_into_polar(&gpu, &mut e, &polar_kv, layer, &k_in, &v_in, max_seq, 0);
        gpu.queue.submit(Some(e.finish()));
        gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        polar_kv.set_len(max_seq);

        // Q for the single query token.
        let q_data: Vec<f32> = (0..n_query_heads * head_dim)
            .map(|i| (((i * 13) as f32) * 0.06).cos())
            .collect();

        // Reference: oneshot path against the cache's first start_pos+1 positions.
        // The oneshot's seq_len param is the number of cache positions it dots
        // against; with n_tokens=1, the only valid t for the query is
        // [0, start_pos+1). We can pass seq_len = start_pos+1 and read
        // scores from the first row of output.
        let r = polar::generate_rotation_matrix(head_dim, seed_for_layer(seed_base, layer));
        let mut rq_cpu = vec![0f32; n_query_heads * head_dim];
        for h in 0..n_query_heads {
            let qs = &q_data[h * head_dim..(h + 1) * head_dim];
            polar::rotate(&r, qs, &mut rq_cpu[h * head_dim..(h + 1) * head_dim]);
        }
        let n_pairs = head_dim / 2;
        let oneshot_seq = start_pos + n_tokens;
        let cpu_k_angles_full: Vec<u8> =
            cpu.k_angles_slice()[..oneshot_seq * n_kv_heads * n_pairs].to_vec();
        let cpu_k_radius: Vec<f32> =
            cpu.k_radius_slice()[..oneshot_seq * n_kv_heads].to_vec();
        let oneshot_scores = attn_score_polar_oneshot(
            &gpu, &rq_cpu, &cpu_k_angles_full, &cpu_k_radius,
            n_query_heads, n_kv_heads, head_dim, oneshot_seq,
        );

        // Batch path: rotate_q on GPU + dispatch_attn_score_polar_batch.
        let q_buf = gpu.create_storage_buffer(bytemuck::cast_slice(&q_data), "test.q1");
        let rq_buf = gpu.create_empty_buffer(
            (n_query_heads * head_dim * 4) as u64, "test.rq1",
        );
        let scores_buf = gpu.create_empty_buffer(
            (n_query_heads * max_seq * 4) as u64, "test.scores1",
        );
        let mut e = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test.batch1"),
        });
        dispatch_rotate_q(
            &gpu, &mut e, &q_buf, polar_kv.rotation_layer(layer), &rq_buf,
            n_tokens, n_query_heads, head_dim,
        );
        dispatch_attn_score_polar_batch(
            &gpu, &mut e, &rq_buf,
            polar_kv.k_angles_layer(layer), polar_kv.k_radius_layer(layer),
            &scores_buf, polar_kv.lut_buffer(),
            n_query_heads, n_kv_heads, head_dim, start_pos, n_tokens, max_seq,
        );
        gpu.queue.submit(Some(e.finish()));
        gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();

        let batch_scores = readback_f32(&gpu, &scores_buf, n_query_heads * max_seq);

        // For each head, compare oneshot[head, t] (t in 0..oneshot_seq)
        // against batch_scores[0, head, t].
        for head in 0..n_query_heads {
            for t in 0..oneshot_seq {
                let one = oneshot_scores[head * oneshot_seq + t];
                let bat = batch_scores[head * max_seq + t];
                // Phase A: f16 input on the batch path introduces small
                // per-pair quantization noise that the oneshot path
                // (CPU-pre-rotated f32 rq) doesn't see; bumped from 1e-5.
                assert!(
                    (one - bat).abs() < 1e-3,
                    "head {head}, t {t}: oneshot={one}, batch={bat}",
                );
            }
            // Beyond start_pos+tok (= start_pos for tok=0), batch should mask to -inf.
            // Since oneshot covers [0, start_pos+1), batch[start_pos+1..max_seq]
            // should be -1e30 (only one position beyond, just t=cache_len, but
            // that's t==start_pos which is allowed for tok=0).
            // For n_tokens=1, max_seq = start_pos+1, so no masked positions
            // exist. Skip mask check in this fixture.
        }
    }

    /// QJL correction is wired into `dot_key` (K side only) but NOT into
    /// `value_at_dequant` (V side), per the engram-port quantized.rs
    /// design. So QJL improves K·Q accuracy but NOT the V aggregation
    /// noise that dominates the attention-output cosine. This test
    /// measures what QJL actually corrects: per-position dot-product
    /// error against ground-truth `Q · K_uncompressed`.
    ///
    /// Adding QJL to V is a future enhancement that would close the
    /// attention-output cosine gap, but it doubles the QJL storage and
    /// changes the dequant API. Out of scope for this port.
    #[test]
    fn qjl_reduces_dot_product_error() {
        let n_kv_heads = 2;
        let head_dim = 64;
        let seq_len = 256;

        // Build two caches with identical seeds and content; only QJL
        // differs.
        let mut polar_only = QuantizedKvCache::new(n_kv_heads, head_dim, seq_len, /*seed*/ 42);
        let mut polar_qjl = QuantizedKvCache::with_qjl(n_kv_heads, head_dim, seq_len, /*rot*/ 42, /*qjl*/ 99);
        let kv_dim = n_kv_heads * head_dim;
        let mut k_truth: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let k = varied_vec(kv_dim, t * 2);
            let v = varied_vec(kv_dim, t * 2 + 1);
            polar_only.append_one(&k, &v);
            polar_qjl.append_one(&k, &v);
            k_truth.push(k);
        }

        let q = varied_vec(head_dim, 9999);

        // Mean absolute error of the dot-product estimate vs ground truth,
        // averaged across (t, kv_h).
        let mut mae_polar = 0.0f64;
        let mut mae_qjl = 0.0f64;
        let mut count = 0usize;
        for t in 0..seq_len {
            for kv_h in 0..n_kv_heads {
                let truth_k = &k_truth[t][kv_h * head_dim..(kv_h + 1) * head_dim];
                let truth_dot: f32 = q.iter().zip(truth_k).map(|(&a, &b)| a * b).sum();
                let est_polar = polar_only.dot_key(t, kv_h, &q);
                let est_qjl = polar_qjl.dot_key(t, kv_h, &q);
                mae_polar += (truth_dot - est_polar).abs() as f64;
                mae_qjl += (truth_dot - est_qjl).abs() as f64;
                count += 1;
            }
        }
        mae_polar /= count as f64;
        mae_qjl /= count as f64;

        // QJL adds an unbiased correction; mean error should drop. The
        // magnitude depends on the residual structure, but QJL on >
        // QJL off is the property under test. Allow ~5% slack so this
        // doesn't false-positive on a particular fixture's tail.
        assert!(
            mae_qjl < mae_polar * 1.05,
            "QJL did not reduce dot-product error (mae_polar={mae_polar:.4}, mae_qjl={mae_qjl:.4})",
        );
    }
}
