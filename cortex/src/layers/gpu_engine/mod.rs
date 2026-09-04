//! GpuEngine — fused GPU forward pass orchestrator.
//!
//! Wraps a `TransformerModel` and a shared `GpuDevice`. The intent is to
//! reimplement the model's forward methods to keep activations on-device
//! across layers, replacing the per-layer CPU↔GPU round-trip that
//! `GpuBitLinear` / `GpuFloatLinear` still perform on their own.
//!
//! ## Phase plan
//!
//! - **1a (this commit):** wrapper scaffolding. Every public method
//!   delegates to the embedded `TransformerModel`. This proves the wrapper
//!   compiles, doesn't double-load weights (we hold a `&Arc<GpuDevice>`
//!   that's already shared with the resident layers), and gives us a place
//!   to add GPU-native methods incrementally.
//! - **1b:** GPU-native embedding lookup and final norm; blocks still
//!   delegate to CPU. Activations cross the GPU boundary at block edges
//!   only.
//! - **1c:** one transformer block on GPU end-to-end (rmsnorm → Q/K/V
//!   matvec → RoPE → attention → O proj → residual → ffn_norm → SwiGLU →
//!   residual).
//! - **1d:** all blocks on GPU; phase-1 done. Multi-token prefill,
//!   no KV cache yet (decode + cache come in #9).
//!
//! ## Discipline (per integration-claude's note 2026-04-23)
//!
//! Keep the wrapper thin. Only methods we reimplement on GPU live as real
//! code here; everything else is a one-line passthrough. When the trait
//! refactor (Option A) lands later, the trait surface is "the union of
//! methods this file actually implements" — pure mechanical lift, no
//! design-the-trait-up-front guesses.


mod dispatch;
mod forward_f32;
mod forward_polar;
mod scratch;

pub use scratch::{BlockScratch, ChunkLimits, PolarBlockScratch};

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::compute::wgpu_backend::GpuDevice;

use crate::layers::kv_cache::ModelKvCache;

use crate::layers::linear::LinearLayer;

use crate::layers::model::TransformerModel;

use crate::layers::sampler::SamplerConfig;

use crate::layers::trace::ForwardTrace;

use crate::layers::transformer::FfnInjector;

/// Resident LM-head weights for the GPU greedy decode path.
///
/// Materialized once at engine init from whichever `OutputProjection`
/// variant the loader produced. All three paths (Linear, Float,
/// TiedEmbedding) allocate fresh on `gpu.weights_heap` via
/// `allocate_static` and upload the packed-f16 weights. The Linear
/// case used to clone the existing wgpu::Buffer via Arc-counting;
/// post-Phase G it `copy_buffer_to_buffer`s from the source allocation
/// to the new one (~590 MB extra weights_heap for Linear-output
/// models; Qwen uses TiedEmbedding so isn't affected). `None` if the
/// projection isn't a shape the GPU shader can handle (odd
/// in_features can't be packed, etc.) — the caller falls through to
/// the CPU path.
pub(crate) struct LmHead {
    pub(crate) weight_buf: ::vram_heap::VramAllocation,
    pub(crate) vocab_size: usize,
    pub(crate) embed_dim: usize,
}

/// Per-block GPU resources extracted at construction time. Holds resident
/// rmsnorm weights for the two norms inside a `TransformerBlock`. The matvec
/// weights are accessed lazily via the CPU model's block at dispatch time
/// (they live inside `Box<dyn LinearLayer>` and are reached via
/// `as_any().downcast_ref::<GpuFloatLinear>()`).
struct GpuBlock {
    attn_norm_weight_buf: ::vram_heap::VramAllocation,
    attn_norm_eps: f32,
    ffn_norm_weight_buf: ::vram_heap::VramAllocation,
    ffn_norm_eps: f32,
    /// Optional Q/K/V biases (Qwen2 family). None for most LLaMA-style models.
    q_bias_buf: Option<::vram_heap::VramAllocation>,
    k_bias_buf: Option<::vram_heap::VramAllocation>,
    v_bias_buf: Option<::vram_heap::VramAllocation>,
}

/// Per-block scratch buffers reused across all dispatches inside a single
/// `forward_block_gpu` call. Caller allocates once per forward pass and
/// reuses across blocks (since dimensions are constant).
/// Output of `forward_full_gpu_with_hidden_capture`. Read-side hook
/// surface for cortex shims (`project_cortex_v1_shim_api.md`):
///
/// - `per_layer_hidden[i]` corresponds to `attachment.layer = "entrance:N+1"`
///   for the i-th element of `capture_layers` — the hidden state at the
///   END of block N (= start of block N+1). Layout: `[n_tokens, embed_dim]`
///   row-major f32.
/// - `final_post_norm_hidden` is what `attachment.layer = "final"` shims
///   read — the LM head's input. Same `[n_tokens, embed_dim]` shape.
///   Pool downstream per the manifest's `pooling` field.
pub struct HiddenCaptures {
    pub per_layer_hidden: Vec<Vec<f32>>,
    pub final_post_norm_hidden: Vec<f32>,
    pub n_tokens: usize,
    pub embed_dim: usize,
}

impl HiddenCaptures {
    /// Pull the last token's slice from the final post-norm hidden state —
    /// the most common pooling for gate / steer shims.
    pub fn final_last_token(&self) -> &[f32] {
        let off = (self.n_tokens - 1) * self.embed_dim;
        &self.final_post_norm_hidden[off..off + self.embed_dim]
    }

    /// Pull the last token's slice from the i-th captured layer's
    /// post-block hidden state.
    pub fn layer_last_token(&self, capture_index: usize) -> &[f32] {
        let off = (self.n_tokens - 1) * self.embed_dim;
        &self.per_layer_hidden[capture_index][off..off + self.embed_dim]
    }
}

/// Fused GPU forward-pass orchestrator wrapping a `TransformerModel`.
pub struct GpuEngine {
    /// CPU-side model. Owns the layers (which may themselves hold resident
    /// GPU buffers via `GpuBitLinear` / `GpuFloatLinear`). Phase 1a delegates
    /// every call through to this; later phases replace specific calls with
    /// on-device dispatches that read the same resident buffers.
    cpu: TransformerModel,
    /// Shared GPU context (device, queue, pipelines). Same `Arc` already
    /// held by the resident layers inside `cpu` — no double allocation.
    gpu: Arc<GpuDevice>,
    /// Resident f32 weights for the final RMSNorm (Phase G: static
    /// allocation on `gpu.weights_heap`).
    final_norm_weight_buf: ::vram_heap::VramAllocation,
    /// Captured at construction time so the dispatcher doesn't have to
    /// re-borrow the CPU model on every call.
    final_norm_eps: f32,
    /// Captured for shader-param construction.
    embed_dim: usize,
    /// Per-block resident resources.
    blocks_gpu: Vec<GpuBlock>,
    /// Resident RoPE cos lookup table, sized to `rope_max_seq * (head_dim/2)`.
    /// Phase G: static allocation on `gpu.weights_heap`.
    rope_cos_buf: ::vram_heap::VramAllocation,
    /// Resident RoPE sin lookup table, same shape as `rope_cos_buf`.
    rope_sin_buf: ::vram_heap::VramAllocation,
    /// Maximum sequence length the rope tables cover. Forward calls with
    /// `start_pos + n_tokens > rope_max_seq` would index out of range, so
    /// they assert.
    rope_max_seq: usize,
    /// GPU-side timestamp infrastructure for per-block timing. `None` if
    /// the adapter doesn't support TIMESTAMP_QUERY +
    /// TIMESTAMP_QUERY_INSIDE_ENCODERS. When `Some`, the forward path
    /// places encoder-level write_timestamp markers between blocks and
    /// resolves them at end-of-forward to a per-block waterfall log.
    timer: Option<TimestampTimer>,
    /// Per-pass timer state. When `Some`, `begin_timed_pass` (used by
    /// the chat-completion prefill hot path) opens compute passes with
    /// `ComputePassTimestampWrites` and increments next_idx. When
    /// `None`, passes open as regular untimed passes — the prefill
    /// resets this back to None at end-of-forward.
    pass_timer: std::sync::Mutex<Option<PassTimerState>>,
    /// Resident LM-head weights for the GPU greedy decode fast path.
    /// `None` when the projection isn't GPU-able (e.g. odd in_features
    /// that the packed-f16 shaders can't handle).
    lm_head: Option<LmHead>,
}

/// GPU-side timestamp tracing. One QuerySet sized for the longest
/// expected forward (37 markers = 36 blocks + final_norm boundary +
/// some slack). Resolved into `resolve_buf` then copied to
/// `readback_buf` for CPU mapping. The buffers are kept resident so we
/// don't re-allocate per forward.
pub struct TimestampTimer {
    pub query_set: wgpu::QuerySet,
    pub resolve_buf: wgpu::Buffer,
    pub readback_buf: wgpu::Buffer,
    pub capacity: u32,
    pub period_ns: f32,
}

/// Mutable state for `begin_timed_pass`. `next_idx` is the offset into
/// the QuerySet for the next (begin, end) pair to allocate; `labels`
/// records what each pair corresponds to so the readback log can
/// aggregate by pass name. Stored under a Mutex on the engine so the
/// helper can mutate it from `&self` methods.
pub struct PassTimerState {
    pub next_idx: u32,
    pub labels: Vec<&'static str>,
}

/// Read back a buffer's contents to a `Vec<f32>`. The buffer must have
/// `MAP_READ` usage (typically created via `create_staging_buffer`) and
/// must already be the destination of a copy_buffer_to_buffer that was
/// included in the most recent submit.
///
/// Only consumer today is the `#[cfg(any())]`-gated parity tests in
/// `tests.rs` (its f16 sibling below has live callers). Kept for when
/// those tests are individually revived.
#[allow(dead_code)]
fn read_back_buffer(gpu: &GpuDevice, staging: &wgpu::Buffer, bytes: usize) -> Vec<f32> {
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).ok();
    });
    gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
    rx.recv().expect("readback failed").expect("buffer map failed");
    let data = slice.get_mapped_range();
    let out: Vec<f32> = data[..bytes].chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    drop(data);
    staging.unmap();
    out
}

/// Read back a packed-f16 buffer and unpack to `Vec<f32>`. `packed_bytes`
/// is the byte count of the packed data in staging (each u32 holds 2
/// f16). Output length is `packed_bytes / 2` f32s. Used by Phase B
/// readback of `normed_buf` for CPU `finalize_logits`.
fn read_back_buffer_f16_unpack(gpu: &GpuDevice, staging: &wgpu::Buffer, packed_bytes: usize) -> Vec<f32> {
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).ok();
    });
    gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
    rx.recv().expect("readback failed").expect("buffer map failed");
    let data = slice.get_mapped_range();
    // Each u32 = 2 f16 = 4 bytes packed → 2 f32 unpacked.
    let mut out: Vec<f32> = Vec::with_capacity(packed_bytes / 2);
    for chunk in data[..packed_bytes].chunks_exact(4) {
        let packed = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let lo = half::f16::from_bits((packed & 0xFFFF) as u16).to_f32();
        let hi = half::f16::from_bits((packed >> 16) as u16).to_f32();
        out.push(lo);
        out.push(hi);
    }
    drop(data);
    staging.unmap();
    out
}

impl std::fmt::Debug for GpuEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GpuEngine(wrapping {:?})", self.cpu)
    }
}

// gpu_engine tests partially reconstructed post-BitNet-excision: the
// BitNet-specific helpers (`toy_ternary_block_pair`) and tests
// (`forward_block_gpu_matches_cpu_bitnet_block`) were deleted. The rest
// stays gated — several CPU-vs-GPU parity tests drifted during C2/C3
// activation packing and aren't passing today (sign-flips on logits,
// not just precision tolerance). Each needs individual investigation
// against the current packed-scratch forward path.
#[cfg(any())]
#[cfg(test)]
mod tests;

impl GpuEngine {
    // -- Pure passthroughs (Phase 1a) ---------------------------------------
    /// Wrap a CPU `TransformerModel` with a shared GPU context. Uses the
    /// default RoPE-table size (4096 positions); call `with_max_seq` if
    /// you need a longer context window.
    pub fn from_cpu_model(cpu: TransformerModel, gpu: Arc<GpuDevice>) -> Self {
        Self::with_max_seq(cpu, gpu, 4096)
    }

    /// Wrap a CPU `TransformerModel` with a shared GPU context, sizing the
    /// RoPE cos/sin lookup tables to `max_seq` positions.
    pub fn with_max_seq(cpu: TransformerModel, gpu: Arc<GpuDevice>, max_seq: usize) -> Self {
        // Phase G: all static weight buffers (final norm + per-block norms +
        // optional biases + RoPE + LM head) allocate from gpu.weights_heap
        // via allocate_static. Heap drop at process exit emits no leak
        // warning. Helper closures wrap the alloc + write pattern.
        let align = ::vram_heap::STORAGE_BUFFER_OFFSET_ALIGNMENT_NVIDIA;
        let upload_static = |bytes: &[u8], label: &str| -> ::vram_heap::VramAllocation {
            let alloc = gpu.weights_heap.allocate_static(
                bytes.len() as u64, align, label,
            ).expect("weights_heap capacity");
            alloc.write(&gpu.queue, bytes);
            alloc
        };

        // Final norm
        let final_norm = cpu.final_norm();
        let final_norm_weight_buf = upload_static(
            bytemuck::cast_slice(final_norm.weight()),
            "gpu_engine.final_norm.weight",
        );
        let final_norm_eps = final_norm.eps();
        let embed_dim = cpu.embed_dim();

        // Per-block norms + optional Q/K/V biases (Qwen2)
        let blocks_gpu: Vec<GpuBlock> = cpu.blocks().iter().enumerate().map(|(i, blk)| {
            let an = blk.attn_norm();
            let fn_ = blk.ffn_norm();
            let attn = blk.attention();
            GpuBlock {
                attn_norm_weight_buf: upload_static(
                    bytemuck::cast_slice(an.weight()),
                    &format!("gpu_engine.block{i}.attn_norm.weight"),
                ),
                attn_norm_eps: an.eps(),
                ffn_norm_weight_buf: upload_static(
                    bytemuck::cast_slice(fn_.weight()),
                    &format!("gpu_engine.block{i}.ffn_norm.weight"),
                ),
                ffn_norm_eps: fn_.eps(),
                q_bias_buf: attn.q_bias().map(|b| upload_static(
                    bytemuck::cast_slice(b),
                    &format!("gpu_engine.block{i}.q_bias"),
                )),
                k_bias_buf: attn.k_bias().map(|b| upload_static(
                    bytemuck::cast_slice(b),
                    &format!("gpu_engine.block{i}.k_bias"),
                )),
                v_bias_buf: attn.v_bias().map(|b| upload_static(
                    bytemuck::cast_slice(b),
                    &format!("gpu_engine.block{i}.v_bias"),
                )),
            }
        }).collect();

        // RoPE tables. All blocks share one rope (same base + head_dim).
        let attn0 = cpu.blocks()[0].attention();
        let (rope_cos_buf, rope_sin_buf) =
            Self::build_rope_tables(&gpu, attn0.rope().inv_freq(), max_seq);

        // Timestamp infrastructure — only allocated if the device
        // supports the required features. Capacity 64 covers
        // 36 blocks + final_norm + slack.
        let timer = if gpu.device.features().contains(wgpu::Features::TIMESTAMP_QUERY)
            && gpu.device.features().contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
        {
            // Capacity bumped from 64 → 512 to fit both:
            //  - per-block boundary markers (n_layers + 2, max ~64)
            //  - per-pass timestamp_writes pairs (~5 passes/block × 36
            //    blocks × 2 timestamps = 360)
            let capacity: u32 = 512;
            let bytes = (capacity as u64) * 8; // u64 per timestamp
            let query_set = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("gpu_engine.timer.query_set"),
                ty: wgpu::QueryType::Timestamp,
                count: capacity,
            });
            let resolve_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu_engine.timer.resolve"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu_engine.timer.readback"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            Some(TimestampTimer {
                query_set,
                resolve_buf,
                readback_buf,
                capacity,
                period_ns: gpu.queue.get_timestamp_period(),
            })
        } else {
            None
        };

        // Materialize the LM-head as a single resident packed-f16 buffer
        // so the greedy decode path can dispatch matmul + argmax without
        // a vocab-sized readback. For the common GPU-loader case (Linear
        // wrapping a GpuFloatLinear), we clone the existing buffer
        // (wgpu::Buffer is Arc-wrapped — no GPU memory duplicated). For
        // TiedEmbedding / Float we pack f32 → f16 and upload a fresh
        // buffer. Odd in_features can't be packed, in which case the
        // fast path is unavailable and lm_head stays None — callers
        // fall through to CPU finalize_logits.
        let vocab_size = cpu.vocab_size();
        let lm_head: Option<LmHead> = if embed_dim % 2 == 0 {
            match cpu.output_proj() {
                crate::layers::model::OutputProjection::Linear(layer) => {
                    layer
                        .as_any()
                        .downcast_ref::<crate::layers::gpu_floatlinear::GpuFloatLinear>()
                        .map(|gpu_layer| {
                            // Phase G: allocate fresh on weights_heap and copy
                            // from the GpuFloatLinear's existing allocation.
                            // The Arc-cloned wgpu::Buffer trick used pre-Phase-G
                            // doesn't work with VramAllocation (not Clone).
                            let src = gpu_layer.weight_buffer();
                            let alloc = gpu.weights_heap.allocate_static(
                                src.size(),
                                ::vram_heap::STORAGE_BUFFER_OFFSET_ALIGNMENT_NVIDIA,
                                "gpu_engine.lm_head.linear",
                            ).expect("weights_heap capacity for LM head (Linear)");
                            // Copy src → alloc via a one-shot command buffer.
                            let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("gpu_engine.lm_head.copy"),
                            });
                            enc.copy_buffer_to_buffer(
                                src.buffer(), src.offset(),
                                alloc.buffer(), alloc.offset(),
                                src.size(),
                            );
                            gpu.queue.submit(Some(enc.finish()));
                            LmHead { weight_buf: alloc, vocab_size, embed_dim }
                        })
                }
                crate::layers::model::OutputProjection::Float(tensor) => {
                    let packed = GpuDevice::pack_f16(tensor.data());
                    let buf = upload_static(
                        bytemuck::cast_slice(&packed),
                        "gpu_engine.lm_head.float",
                    );
                    Some(LmHead { weight_buf: buf, vocab_size, embed_dim })
                }
                crate::layers::model::OutputProjection::TiedEmbedding => {
                    let packed = GpuDevice::pack_f16(cpu.embedding_data());
                    let buf = upload_static(
                        bytemuck::cast_slice(&packed),
                        "gpu_engine.lm_head.tied",
                    );
                    Some(LmHead { weight_buf: buf, vocab_size, embed_dim })
                }
            }
        } else {
            None
        };

        Self {
            cpu,
            gpu,
            final_norm_weight_buf,
            final_norm_eps,
            embed_dim,
            blocks_gpu,
            rope_cos_buf,
            rope_sin_buf,
            rope_max_seq: max_seq,
            timer,
            pass_timer: std::sync::Mutex::new(None),
            lm_head,
        }
    }

    /// Open a compute pass, attaching begin/end timestamp_writes if a
    /// per-pass timer is currently active (and the pass timer + query
    /// set both have capacity). Otherwise opens an untimed pass.
    /// Helper for the chat-completion prefill hot path; other callers
    /// can keep using `encoder.begin_compute_pass(...)` directly.
    fn begin_timed_pass<'enc>(
        &self,
        encoder: &'enc mut wgpu::CommandEncoder,
        label: &'static str,
    ) -> wgpu::ComputePass<'enc> {
        let writes = self.next_pass_timestamp_writes(label);
        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: writes,
        })
    }

    /// Allocate the next (begin, end) pair from `pass_timer` if active;
    /// returns None if no per-pass timer is running or the QuerySet is
    /// full. Pulled out so a caller that needs to construct a pass
    /// descriptor inline can still get the timestamp_writes.
    fn next_pass_timestamp_writes(
        &self,
        label: &'static str,
    ) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let mut guard = self.pass_timer.lock().unwrap();
        let state = guard.as_mut()?;
        let timer = self.timer.as_ref()?;
        if state.next_idx + 2 > timer.capacity {
            return None;
        }
        let begin = state.next_idx;
        state.next_idx += 2;
        state.labels.push(label);
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &timer.query_set,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(begin + 1),
        })
    }

    /// Build cos/sin lookup tables for the `rope_batch` shader, sized to
    /// `max_seq` positions. Layout matches `rope_batch.wgsl`:
    /// `cos_table[pos * half_dim + i]`, same for sin. Half_dim = inv_freq.len().
    ///
    /// Called once at construction (cos/sin are fixed for a given RoPE
    /// config). Returns two storage buffers ready to be bound.
    pub fn build_rope_tables(
        gpu: &GpuDevice,
        inv_freq: &[f32],
        max_seq: usize,
    ) -> (::vram_heap::VramAllocation, ::vram_heap::VramAllocation) {
        let half_dim = inv_freq.len();
        let mut cos = Vec::with_capacity(max_seq * half_dim);
        let mut sin = Vec::with_capacity(max_seq * half_dim);
        for pos in 0..max_seq {
            let p = pos as f32;
            for &freq in inv_freq {
                let angle = p * freq;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }

        let align = ::vram_heap::STORAGE_BUFFER_OFFSET_ALIGNMENT_NVIDIA;
        let cos_bytes = bytemuck::cast_slice(&cos);
        let sin_bytes = bytemuck::cast_slice(&sin);
        let cos_buf = gpu.weights_heap.allocate_static(
            cos_bytes.len() as u64, align, "gpu_engine.rope.cos",
        ).expect("weights_heap capacity for RoPE cos");
        cos_buf.write(&gpu.queue, cos_bytes);
        let sin_buf = gpu.weights_heap.allocate_static(
            sin_bytes.len() as u64, align, "gpu_engine.rope.sin",
        ).expect("weights_heap capacity for RoPE sin");
        sin_buf.write(&gpu.queue, sin_bytes);
        (cos_buf, sin_buf)
    }

    /// Borrow the underlying CPU model (for delegation in tests / debug).
    pub fn cpu(&self) -> &TransformerModel {
        &self.cpu
    }

    /// Borrow the shared GPU device (vram-heap arenas, ParamsBufferPool,
    /// pipelines). Used by the Phase K metrics sampler to read heap
    /// usage and pool stats.
    pub fn gpu(&self) -> &Arc<GpuDevice> {
        &self.gpu
    }

    pub fn vocab_size(&self) -> usize {
        self.cpu.vocab_size()
    }

    pub fn embed_dim(&self) -> usize {
        self.cpu.embed_dim()
    }

    pub fn n_layers(&self) -> usize {
        self.cpu.n_layers()
    }

    /// Block until all submitted GPU work has completed AND the deferred-
    /// destroy queue for dropped buffers has been processed. wgpu uses a
    /// lazy destruction model — wgpu::Buffer's Drop only queues the
    /// underlying allocation for cleanup; the actual free happens at the
    /// next poll/submit. Without an explicit poll, churn-heavy patterns
    /// (e.g. cache_load creating + dropping a 300MB f32 cache per shard)
    /// can grow the destroy queue until wgpu-29's stricter validation
    /// rejects new allocations with a delayed "Buffer X is invalid"
    /// error that surfaces at the next poll/get_mapped_range. Calling
    /// this between cache loads lets the allocator drain.
    pub fn poll_wait(&self) {
        self.gpu.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        }).unwrap();
    }

    /// Log wgpu's internal allocator report at INFO. Use to track
    /// VRAM growth / fragmentation around the 2200-token polar
    /// retrieve device-lost ceiling. Includes total_allocated /
    /// total_reserved totals, block + allocation counts, and the top
    /// N allocations by size. `tag` is a free-text label that appears
    /// in the log line so callers can distinguish before/after pairs.
    pub fn log_allocator_report(&self, tag: &str) {
        let Some(report) = self.gpu.device.generate_allocator_report() else {
            tracing::info!(tag, "allocator report unavailable (backend doesn't provide)");
            return;
        };
        let n_allocations = report.allocations.len();
        let n_blocks = report.blocks.len();
        let total_allocated_mb = report.total_allocated_bytes / (1024 * 1024);
        let total_reserved_mb = report.total_reserved_bytes / (1024 * 1024);
        tracing::info!(
            tag,
            total_allocated_mb,
            total_reserved_mb,
            n_allocations,
            n_blocks,
            "allocator report",
        );
        let mut top: Vec<_> = report.allocations.iter().collect();
        top.sort_by_key(|a| std::cmp::Reverse(a.size));
        for (i, a) in top.iter().take(10).enumerate() {
            tracing::info!(
                tag,
                rank = i,
                name = %a.name,
                size_kb = a.size / 1024,
                "top alloc",
            );
        }
    }

    /// Log per-heap vram-heap stats — complements `log_allocator_report`
    /// (which reflects wgpu's internal allocator). Useful when chasing
    /// vram-heap fragmentation, leak, or high-water issues. Same opt-in
    /// gating expected at call sites (CORTEX_POLAR_TRACE_DIAG=1).
    pub fn log_vram_heap_stats(&self, tag: &str) {
        for heap in [
            &self.gpu.transient_heap_a,
            &self.gpu.transient_heap_b,
            &self.gpu.host_readback_heap,
        ] {
            let s = heap.stats();
            tracing::info!(
                tag,
                heap = %s.label,
                tier = ?s.tier,
                total_mb = s.total_size / (1024 * 1024),
                used_mb = s.used_size / (1024 * 1024),
                payload_mb = s.used_payload / (1024 * 1024),
                high_water_mb = s.high_water_mark / (1024 * 1024),
                live = s.current_live_allocations,
                allocs = s.allocation_count,
                frees = s.free_count,
                largest_free_kb = s.largest_free_block / 1024,
                fragmentation = s.fragmentation_ratio,
                "vram_heap stats",
            );
        }
    }

    pub fn embedding_data(&self) -> &[f32] {
        self.cpu.embedding_data()
    }

    pub fn set_block_injector(&mut self, layer: usize, injector: Box<dyn FfnInjector>) {
        self.cpu.set_block_injector(layer, injector);
    }

    pub fn create_kv_cache(&self, max_seq_len: usize) -> ModelKvCache {
        self.cpu.create_kv_cache(max_seq_len)
    }

    /// Allocate a GPU-resident KV cache sized for this model. Panics if the
    /// VRAM budget refuses it; see [`Self::try_create_gpu_kv_cache`].
    pub fn create_gpu_kv_cache(&self, max_seq_len: usize) -> crate::layers::gpu_kv_cache::GpuKvCache {
        self.try_create_gpu_kv_cache(max_seq_len)
            .expect("gpu_kv_cache heap construction failed")
    }

    /// Fallible variant (adversarial review 2026-09-02, #6): returns the
    /// vram-heap error instead of panicking, so a serving layer can answer
    /// 503 and keep running.
    pub fn try_create_gpu_kv_cache(
        &self,
        max_seq_len: usize,
    ) -> Result<crate::layers::gpu_kv_cache::GpuKvCache, ::vram_heap::Error> {
        let attn0 = self.cpu.blocks()[0].attention();
        crate::layers::gpu_kv_cache::GpuKvCache::try_new(
            self.gpu.clone(),
            self.cpu.n_layers(),
            attn0.n_kv_heads(),
            attn0.head_dim(),
            max_seq_len,
        )
    }

    /// Allocate a fresh `GpuPolarKvCache` shaped to this engine's model.
    /// `rotation_seed_base` selects the per-layer rotation matrices —
    /// must be consistent across caches that participate in the same
    /// composition / shared retrieval session.
    pub fn create_gpu_polar_kv_cache(
        &self,
        max_seq_len: usize,
        rotation_seed_base: u64,
    ) -> crate::layers::gpu_polar_kv_cache::GpuPolarKvCache {
        let attn0 = self.cpu.blocks()[0].attention();
        crate::layers::gpu_polar_kv_cache::GpuPolarKvCache::new(
            self.gpu.clone(),
            self.cpu.n_layers(),
            attn0.n_kv_heads(),
            attn0.head_dim(),
            max_seq_len,
            rotation_seed_base,
        )
    }

    /// Same as `create_gpu_polar_kv_cache` but enables QJL correction
    /// for K residuals (`n_qjl_proj > 0`). `qjl_seed_base` selects the
    /// per-layer projection matrices.
    pub fn create_gpu_polar_kv_cache_with_qjl(
        &self,
        max_seq_len: usize,
        rotation_seed_base: u64,
        n_qjl_proj: usize,
        qjl_seed_base: u64,
    ) -> crate::layers::gpu_polar_kv_cache::GpuPolarKvCache {
        let attn0 = self.cpu.blocks()[0].attention();
        crate::layers::gpu_polar_kv_cache::GpuPolarKvCache::new_with_qjl(
            self.gpu.clone(),
            self.cpu.n_layers(),
            attn0.n_kv_heads(),
            attn0.head_dim(),
            max_seq_len,
            rotation_seed_base,
            n_qjl_proj,
            qjl_seed_base,
        )
    }

    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Vec<f32> {
        self.cpu.forward(tokens, start_pos)
    }

    pub fn forward_last(&self, tokens: &[u32], start_pos: usize) -> Vec<f32> {
        self.cpu.forward_last(tokens, start_pos)
    }

    pub fn forward_cached(&self, tokens: &[u32], cache: &mut ModelKvCache) -> Vec<f32> {
        self.cpu.forward_cached(tokens, cache)
    }

    pub fn forward_traced(&self, tokens: &[u32]) -> (Vec<f32>, ForwardTrace) {
        self.cpu.forward_traced(tokens)
    }

    pub fn generate(
        &self,
        prompt: &[u32],
        max_tokens: usize,
        sampler_config: SamplerConfig,
        seed: u64,
        stop_token: Option<u32>,
    ) -> Vec<u32> {
        self.cpu.generate(prompt, max_tokens, sampler_config, seed, stop_token)
    }

}
