//! Polar (PolarQuant KV) forward variants (split from gpu_engine.rs, Phase N).

use super::*;
use super::dispatch::SoftmaxBatchParams;

impl GpuEngine {
    /// Phase O: per-forward Lane C scratch for the V QJL correction's
    /// C accumulator (`[n_tokens, n_heads, n_v_proj]` f32). `None`
    /// when the cache has no QJL. Allocated once per forward and
    /// reused across blocks — see `forward_block_gpu_polar_inner`'s
    /// `qjl_c_buf` param for why per-block allocation would be wrong.
    fn alloc_qjl_c_buf(
        &self,
        polar_cache: &crate::layers::gpu_polar_kv_cache::GpuPolarKvCache,
        n_tokens: usize,
        n_heads: usize,
    ) -> Option<::vram_heap::VramAllocation> {
        if polar_cache.n_qjl_proj() == 0 {
            return None;
        }
        let bytes = (n_tokens * n_heads * polar_cache.n_v_qjl_proj()
            * std::mem::size_of::<f32>()) as u64;
        Some(self.gpu.transient_heap_c.allocate(
            bytes,
            ::vram_heap::STORAGE_BUFFER_OFFSET_ALIGNMENT_NVIDIA,
            "qjl_value_weights.c",
        ).expect("transient_heap_c capacity for qjl C accumulator"))
    }

    /// Polar variant of `forward_full_gpu_with_cache_traced`. Runs the
    /// transformer forward over `query_tokens` against a resident
    /// PolarQuant-compressed KV cache, capturing per-layer pre-softmax
    /// attention scores from `capture_layers`. The query's K/V are
    /// compressed and written into the polar cache at offset
    /// `polar_cache.seq_len()` for self-attention; the cache cursor is
    /// **not** advanced (mirrors the f32 traced forward — same cache can
    /// serve repeated queries).
    ///
    /// All blocks run; non-captured layers are still executed because the
    /// hidden state must propagate through every layer. `capture_layers`
    /// only controls which layers' pre-softmax attention scores are
    /// returned.
    pub fn forward_full_gpu_polar_traced(
        &self,
        query_tokens: &[u32],
        polar_cache: &crate::layers::gpu_polar_kv_cache::GpuPolarKvCache,
        capture_layers: &[usize],
    ) -> Vec<Vec<f32>> {
        let n_tokens = query_tokens.len();
        assert!(n_tokens > 0, "must have at least one query token");

        let n_layers = self.cpu.n_layers();
        assert_eq!(n_layers, polar_cache.n_layers(), "polar cache layer count mismatch");
        for &l in capture_layers {
            assert!(l < n_layers, "capture layer {l} out of range (n_layers={n_layers})");
        }

        let attn0 = self.cpu.blocks()[0].attention();
        assert_eq!(polar_cache.n_kv_heads(), attn0.n_kv_heads(),
            "polar cache n_kv_heads mismatch");
        assert_eq!(polar_cache.head_dim(), attn0.head_dim(),
            "polar cache head_dim mismatch");

        let start_pos = polar_cache.seq_len();
        let attn_max_seq = start_pos + n_tokens;
        assert!(attn_max_seq <= polar_cache.max_seq_len(),
            "polar cache overflow: {} + {} > {}",
            start_pos, n_tokens, polar_cache.max_seq_len());

        // ---- Embedding lookup (CPU) ----
        let embed_data = self.cpu.embedding_data();
        let vocab_size = self.cpu.vocab_size();
        let mut hidden_init: Vec<f32> = Vec::with_capacity(n_tokens * self.embed_dim);
        for &tok in query_tokens {
            assert!((tok as usize) < vocab_size, "token {tok} out of vocab");
            let off = tok as usize * self.embed_dim;
            hidden_init.extend_from_slice(&embed_data[off..off + self.embed_dim]);
        }
        // Phase 0 fix (2026-06-03): forward_block_gpu_polar_inner reads
        // hidden_buf via rmsnorm_packed_to_packed (packed-f16 u32 array).
        // Previously this passed raw f32 bytes — the block was reading
        // misaligned data and producing garbage scores. Mirror the chat
        // orchestrator's pack-then-upload pattern.
        let hidden_packed = GpuDevice::pack_f16(&hidden_init);
        // Phase C (vram-heap): hidden_buf is a sub-allocation of
        // transient_heap_a. RAII Drop returns the range to the free
        // list at function exit; bind groups inside the block loop
        // hold the heap buffer alive until queue.submit completes.
        let hidden_bytes = (hidden_packed.len() * std::mem::size_of::<u32>()) as u64;
        let hidden_buf = self.gpu.transient_heap_a.allocate(
            hidden_bytes,
            ::vram_heap::STORAGE_BUFFER_OFFSET_ALIGNMENT_NVIDIA,
            "forward_polar_traced.hidden",
        ).expect("transient_heap_a capacity");
        hidden_buf.write(&self.gpu.queue, bytemuck::cast_slice(&hidden_packed));

        let intermediate = self.cpu.blocks()[0].ffn().as_any()
            .downcast_ref::<crate::layers::swiglu::SwiGLU>()
            .unwrap_or_else(|| panic!("forward_full_gpu_polar_traced requires SwiGLU FFN"))
            .intermediate_size();
        let n_heads = attn0.n_heads();
        let head_dim = attn0.head_dim();
        // The polar retrieve path is unchunked (unlike cache_load's f32
        // prefill, which chunks via `safe_prefill_chunk_size`). Retrieve
        // queries are normally short, so `scores`
        // (n_tokens · n_heads · attn_max_seq · 4) stays small — but a long
        // query against a large shard can exceed Lane B. Fail with a clear,
        // actionable message instead of the opaque `OutOfMemory` the
        // `PolarBlockScratch::allocate` `.expect` would otherwise emit.
        let scores_bytes =
            (n_tokens * n_heads * attn_max_seq * std::mem::size_of::<f32>()) as u64;
        // Two independent ceilings: the Lane B heap, and the device's
        // single-storage-binding limit (~2 GB) — scores is bound as ONE
        // storage buffer, so it must fit the binding cap even when Lane B
        // is sized larger (Phase M device-derived lanes can exceed 2 GB).
        let lane_b = self.gpu.transient_heap_b.capacity();
        let binding_max = self.gpu.device.limits().max_storage_buffer_binding_size as u64;
        let ceiling = lane_b.min(binding_max);
        assert!(
            scores_bytes <= ceiling,
            "polar retrieve scratch too large: scores need {scores_bytes} B but the \
             ceiling is {ceiling} B (Lane B capacity {lane_b} B, max storage binding \
             {binding_max} B) for {n_tokens} query tokens against a {start_pos}-token \
             shard. Shorten the query or raise CORTEX_VRAM_HEAP_B_MB.",
        );
        // Review #4: this traced forward is unchunked, and softmax dispatches
        // n_tokens·n_heads in one dimension against wgpu's 65535 cap. The
        // handler bounds the query first; this is the engine-level backstop
        // with a clear message instead of a driver validation error.
        let wg_per_token =
            super::scratch::max_workgroups_per_token(n_heads, head_dim, self.embed_dim);
        let max_q = super::scratch::WGPU_MAX_WORKGROUPS_PER_DIM / wg_per_token;
        assert!(
            n_tokens <= max_q,
            "polar retrieve query too long: {n_tokens} tokens > {max_q} \
             (wgpu 65535 dispatch limit / {wg_per_token} workgroups per token). \
             Shorten the query.",
        );
        let scratch = PolarBlockScratch::allocate(
            &self.gpu, n_tokens, self.embed_dim,
            n_heads, attn0.n_kv_heads(), head_dim,
            intermediate, attn_max_seq,
        );

        // Polar-only scratch: reused as both rq (post-rotate_q) and
        // weighted_rotated_V (pre-derotate) inside each block. One
        // allocation per trace call, not per block.
        let rotated_bytes = (n_tokens * n_heads * head_dim * std::mem::size_of::<f32>()) as u64;
        let rotated_buf = self.gpu.transient_heap_a.allocate(
            rotated_bytes,
            ::vram_heap::STORAGE_BUFFER_OFFSET_ALIGNMENT_NVIDIA,
            "forward_polar_traced.rotated",
        ).expect("transient_heap_a capacity");
        let qjl_c_buf = self.alloc_qjl_c_buf(polar_cache, n_tokens, n_heads);

        // Per-captured-layer scores. Same shape as the f32 path:
        // [n_tokens, n_heads, attn_max_seq].
        let scores_bytes = (n_tokens * n_heads * attn_max_seq * std::mem::size_of::<f32>()) as u64;
        let capture_bufs: Vec<wgpu::Buffer> = capture_layers.iter().map(|&l| {
            self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("forward_polar_traced.scores.layer{l}")),
                size: scores_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        }).collect();
        // vram-heap migration: capture stagings are sub-allocations of
        // the host-visible readback heap rather than fresh wgpu
        // staging buffers. Safe because stagings are never bound as
        // compute resources — only copy_buffer_to_buffer destinations
        // and host map targets, neither of which triggers wgpu's
        // per-dispatch buffer-usage tracker. After the overnight
        // vram-heap rewrite, allocations free + coalesce via RAII
        // Drop (no manual reset_transients); the explicit
        // `drop(capture_stagings)` at function exit ensures Drop runs
        // before the next forward call's allocator activity.
        let capture_stagings: Vec<::vram_heap::VramAllocation> = capture_layers.iter()
            .map(|&l| self.gpu.host_readback_heap.allocate(
                scores_bytes,
                ::vram_heap::COPY_BUFFER_ALIGNMENT,
                &format!("forward_polar_traced.stg.layer{l}"),
            ).expect("host_readback_heap capacity"))
            .collect();
        let capture_lookup: std::collections::HashMap<usize, &wgpu::Buffer> =
            capture_layers.iter().zip(capture_bufs.iter())
                .map(|(&l, buf)| (l, buf))
                .collect();

        // ---- All blocks, split into chunks to avoid Windows TDR ----
        // Submitting all 36 layers' polar dispatches at large cache
        // seq_len (e.g. 2271 tokens for a hist-139th-street shard) in
        // one command buffer reliably trips Windows' 2-second TDR —
        // Vulkan reports DeviceError::Lost from queue.submit, wgpu
        // marks the device permanently lost, every subsequent request
        // panics with "Parent device is lost".
        //
        // Confirmed empirically: retrieves on ~989-token polar shards
        // succeed; the same code path on a 2271-token polar shard
        // crashes the device. Chunking the dispatches across multiple
        // smaller submits with poll(Wait) between gives each individual
        // command buffer a tractable budget. Tunable via
        // CORTEX_POLAR_CHUNK_LAYERS; default 9 yields 4 submits for
        // Qwen 36-layer. Set 0 (or >= n_layers) to revert to the
        // original single-submit behavior for A/B / regression testing.
        let chunk_layers: usize = std::env::var("CORTEX_POLAR_CHUNK_LAYERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(9);
        let single_submit = chunk_layers == 0 || chunk_layers >= n_layers;

        // Per-submit diagnostic (gated on CORTEX_POLAR_TRACE_DIAG=1).
        // When the device-lost panic fires from inside a chunk's
        // queue.submit, the LAST tracing line tells us WHICH chunk +
        // layer range was being submitted. First-chunk failure → a
        // specific shader; last-chunk failure → cumulative-in-flight.
        let diag = std::env::var("CORTEX_POLAR_TRACE_DIAG").as_deref() == Ok("1");

        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("forward_polar_traced.encoder.0"),
        });
        let mut layers_in_encoder = 0usize;
        let mut chunk_idx = 0usize;
        let mut chunk_layer_start = 0usize;
        for i in 0..n_layers {
            let capture = capture_lookup.get(&i).copied();
            self.forward_block_gpu_polar_inner(
                &mut encoder, i, &hidden_buf, n_tokens, start_pos, &scratch,
                &rotated_buf, polar_cache, qjl_c_buf.as_ref(), capture, None, None,
            );
            layers_in_encoder += 1;
            if !single_submit
                && layers_in_encoder >= chunk_layers
                && i + 1 < n_layers
            {
                // No poll(Wait) between chunks — wgpu's queue is FIFO so
                // submits run in order, and TDR fires per command buffer
                // not per queue. The final submit + readback poll below
                // covers synchronization. Inter-chunk poll(Wait) was
                // observed to make populate_from_f32_cache_gpu WORSE
                // (cache_load crashed sooner) so we use the same
                // submit-only pattern here for consistency.
                if diag {
                    tracing::info!(
                        chunk = chunk_idx,
                        layers = format!("{}..{}", chunk_layer_start, i + 1),
                        n_tokens, start_pos, attn_max_seq,
                        "polar_traced submitting chunk",
                    );
                }
                self.gpu.queue.submit(Some(encoder.finish()));
                chunk_idx += 1;
                chunk_layer_start = i + 1;
                encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("forward_polar_traced.encoder.{chunk_idx}")),
                });
                layers_in_encoder = 0;
            }
        }
        if diag {
            tracing::info!(
                chunk = chunk_idx,
                layers = format!("{}..{}", chunk_layer_start, n_layers),
                n_tokens, start_pos, attn_max_seq,
                "polar_traced submitting final chunk",
            );
        }
        // Skip final_norm + output projection — retrieval doesn't need logits.
        // Capture-to-staging copies go on the final encoder so they ride
        // the last submit; staging readback below polls again before mapping.
        for (cap_buf, stg) in capture_bufs.iter().zip(capture_stagings.iter()) {
            // vram-heap: stg is a sub-range of the host_readback_heap's
            // backing buffer. Copy to the right offset.
            encoder.copy_buffer_to_buffer(
                cap_buf, 0,
                stg.buffer(), stg.offset(),
                scores_bytes,
            );
        }
        self.gpu.queue.submit(Some(encoder.finish()));

        // vram-heap migration: all capture_stagings share the same
        // backing wgpu::Buffer (the host_readback_heap), so we can't
        // map each sub-range separately (wgpu enforces one map state
        // per buffer). Instead, map ONE range covering all the
        // capture_staging sub-ranges, then slice into the mapped
        // bytes by offset. Allocations are bump-allocated contiguously
        // so the range is exactly [first.offset .. last.end].
        use std::sync::mpsc;
        let span_start = capture_stagings.first().map(|a| a.offset()).unwrap_or(0);
        let span_end = capture_stagings.last()
            .map(|a| a.offset() + a.size())
            .unwrap_or(0);
        let mapped_range = if span_end > span_start {
            let slice = self.gpu.host_readback_heap.buffer().slice(span_start..span_end);
            let (tx, rx) = mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
            self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
            rx.recv().expect("readback channel closed").expect("buffer map failed");
            Some(slice)
        } else {
            self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
            None
        };

        let per_layer_scores: Vec<Vec<f32>> = if let Some(slice) = mapped_range.as_ref() {
            let data = slice.get_mapped_range();
            let out = capture_stagings.iter().map(|stg| {
                let local_off = (stg.offset() - span_start) as usize;
                let bytes = &data[local_off..local_off + scores_bytes as usize];
                bytes.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }).collect();
            drop(data);
            out
        } else {
            Vec::new()
        };

        // Unmap the heap backing (covers the mapped span). Then drop
        // the VramAllocations explicitly: vram-heap's free-list
        // reclaims the ranges and coalesces with neighbors via RAII
        // Drop (no manual reset needed after the overnight rewrite).
        // Explicit drop forces this to happen before the function
        // returns so the freelist state is clean for the next call.
        if mapped_range.is_some() {
            self.gpu.host_readback_heap.buffer().unmap();
        }
        drop(capture_stagings);

        per_layer_scores
    }

    /// Polar counterpart to `forward_full_gpu_with_cache_advance_only`.
    /// Runs all blocks against the polar cache (compressing new K/V at
    /// `[start_pos, start_pos+n_tokens)` via the existing polar block
    /// forward, including QJL encode when the cache has it enabled),
    /// then advances `polar_cache.set_len(...)`. No final RMSNorm, no
    /// readback, no LM head.
    ///
    /// Used by `cache_append` for polar_chat shards — fire-and-forget
    /// extension that doesn't pay the readback cost.
    pub fn forward_full_gpu_polar_with_cache_advance_only(
        &self,
        tokens: &[u32],
        polar_cache: &mut crate::layers::gpu_polar_kv_cache::GpuPolarKvCache,
    ) {
        let n_tokens = tokens.len();
        assert!(n_tokens > 0, "must have at least one token");

        let n_layers = self.cpu.n_layers();
        assert_eq!(n_layers, polar_cache.n_layers(), "polar cache layer count mismatch");

        let attn0 = self.cpu.blocks()[0].attention();
        assert_eq!(polar_cache.n_kv_heads(), attn0.n_kv_heads(),
            "polar cache n_kv_heads mismatch");
        assert_eq!(polar_cache.head_dim(), attn0.head_dim(),
            "polar cache head_dim mismatch");

        let start_pos = polar_cache.seq_len();
        let attn_max_seq = start_pos + n_tokens;
        assert!(attn_max_seq <= polar_cache.max_seq_len(),
            "polar cache overflow: {} + {} > {}",
            start_pos, n_tokens, polar_cache.max_seq_len());

        // Embedding (CPU) — pack to f16 for packed hidden_buf.
        let embed_data = self.cpu.embedding_data();
        let vocab_size = self.cpu.vocab_size();
        let mut hidden_init: Vec<f32> = Vec::with_capacity(n_tokens * self.embed_dim);
        for &tok in tokens {
            assert!((tok as usize) < vocab_size, "token {tok} out of vocab");
            let off = tok as usize * self.embed_dim;
            hidden_init.extend_from_slice(&embed_data[off..off + self.embed_dim]);
        }

        let hidden_packed = GpuDevice::pack_f16(&hidden_init);
        // Phase C (vram-heap): hidden_buf + rotated_buf are sub-allocations
        // of transient_heap_a; RAII Drop reclaims at function exit.
        let hidden_bytes = (hidden_packed.len() * std::mem::size_of::<u32>()) as u64;
        let hidden_buf = self.gpu.transient_heap_a.allocate(
            hidden_bytes,
            ::vram_heap::STORAGE_BUFFER_OFFSET_ALIGNMENT_NVIDIA,
            "forward_polar_advance_only.hidden",
        ).expect("transient_heap_a capacity");
        hidden_buf.write(&self.gpu.queue, bytemuck::cast_slice(&hidden_packed));

        let intermediate = self.cpu.blocks()[0].ffn().as_any()
            .downcast_ref::<crate::layers::swiglu::SwiGLU>()
            .unwrap_or_else(|| panic!("forward_polar_advance_only requires SwiGLU FFN"))
            .intermediate_size();

        let n_heads = attn0.n_heads();
        let head_dim = attn0.head_dim();
        let scratch = PolarBlockScratch::allocate(
            &self.gpu, n_tokens, self.embed_dim,
            n_heads, attn0.n_kv_heads(), head_dim,
            intermediate, attn_max_seq,
        );

        // Polar scratch: shared rotated_buf for rq + weighted V across blocks.
        let rotated_bytes = (n_tokens * n_heads * head_dim * std::mem::size_of::<f32>()) as u64;
        let rotated_buf = self.gpu.transient_heap_a.allocate(
            rotated_bytes,
            ::vram_heap::STORAGE_BUFFER_OFFSET_ALIGNMENT_NVIDIA,
            "forward_polar_advance_only.rotated",
        ).expect("transient_heap_a capacity");
        let qjl_c_buf = self.alloc_qjl_c_buf(polar_cache, n_tokens, n_heads);

        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("forward_polar_advance_only.encoder"),
        });
        for i in 0..n_layers {
            self.forward_block_gpu_polar_inner(
                &mut encoder, i, &hidden_buf, n_tokens, start_pos, &scratch,
                &rotated_buf, &*polar_cache, qjl_c_buf.as_ref(), None, None, None,
            );
        }
        self.gpu.queue.submit(Some(encoder.finish()));

        if std::env::var("CORTEX_SYNC_AFTER_ADVANCE").as_deref() == Ok("1") {
            self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        }

        polar_cache.set_len(start_pos + n_tokens);
    }

    /// Polar-cache chat fast path: same shape as
    /// `forward_full_gpu_with_cache_inject_argmax_greedy` but runs the
    /// polar block forward (which compresses new K/V into the polar
    /// cache as it goes) and advances `polar_cache.set_len(...)` on
    /// success. 4-byte token-id readback.
    ///
    /// Quality target without QJL: ~0.84 cosine on attention output vs
    /// f32 path. Phase 3 adds GPU QJL to close to ~0.95. Use only for
    /// greedy sampling — see `lm_head_greedy_eligible` in cortex-cloud
    /// for the gate.
    ///
    /// Returns `None` if `lm_head` isn't GPU-resident — callers fall
    /// through to the legacy logits-readback path.
    pub fn forward_full_gpu_polar_with_cache_inject_argmax_greedy(
        &self,
        tokens: &[u32],
        polar_cache: &mut crate::layers::gpu_polar_kv_cache::GpuPolarKvCache,
        inject_deltas: &[Option<wgpu::Buffer>],
    ) -> Option<u32> {
        let lm_head = self.lm_head.as_ref()?;

        let n_tokens = tokens.len();
        assert!(n_tokens > 0, "must have at least one token");

        let n_layers = self.cpu.n_layers();
        assert_eq!(n_layers, polar_cache.n_layers(), "polar cache layer count mismatch");

        let attn0 = self.cpu.blocks()[0].attention();
        assert_eq!(polar_cache.n_kv_heads(), attn0.n_kv_heads(),
            "polar cache n_kv_heads mismatch");
        assert_eq!(polar_cache.head_dim(), attn0.head_dim(),
            "polar cache head_dim mismatch");

        let start_pos = polar_cache.seq_len();
        let attn_max_seq = start_pos + n_tokens;
        assert!(attn_max_seq <= polar_cache.max_seq_len(),
            "polar cache overflow: {} + {} > {}",
            start_pos, n_tokens, polar_cache.max_seq_len());

        // Embedding (CPU) — pack to f16 for packed hidden_buf.
        let embed_data = self.cpu.embedding_data();
        let vocab_size = self.cpu.vocab_size();
        let mut hidden_init: Vec<f32> = Vec::with_capacity(n_tokens * self.embed_dim);
        for &tok in tokens {
            assert!((tok as usize) < vocab_size, "token {tok} out of vocab");
            let off = tok as usize * self.embed_dim;
            hidden_init.extend_from_slice(&embed_data[off..off + self.embed_dim]);
        }

        let packed_bytes = (hidden_init.len() * 2) as u64;
        let hidden_packed = GpuDevice::pack_f16(&hidden_init);
        // Phase C (vram-heap): hidden_buf + rotated_buf are sub-allocations
        // of transient_heap_a; RAII Drop reclaims at function exit.
        let hidden_buf = self.gpu.transient_heap_a.allocate(
            packed_bytes,
            ::vram_heap::STORAGE_BUFFER_OFFSET_ALIGNMENT_NVIDIA,
            "forward_polar_argmax.hidden",
        ).expect("transient_heap_a capacity");
        hidden_buf.write(&self.gpu.queue, bytemuck::cast_slice(&hidden_packed));
        let normed_buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("forward_polar_argmax.normed"),
            size: packed_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let intermediate = self.cpu.blocks()[0].ffn().as_any()
            .downcast_ref::<crate::layers::swiglu::SwiGLU>()
            .unwrap_or_else(|| panic!("forward_polar_argmax requires SwiGLU FFN"))
            .intermediate_size();

        let n_heads = attn0.n_heads();
        let head_dim = attn0.head_dim();
        let scratch = PolarBlockScratch::allocate(
            &self.gpu, n_tokens, self.embed_dim,
            n_heads, attn0.n_kv_heads(), head_dim,
            intermediate, attn_max_seq,
        );

        // Polar scratch: rotated_buf reused as both rq (post-rotate_q) and
        // weighted_rotated_V (pre-derotate) inside each block. One
        // allocation per call, not per block.
        let rotated_bytes = (n_tokens * n_heads * head_dim * std::mem::size_of::<f32>()) as u64;
        let rotated_buf = self.gpu.transient_heap_a.allocate(
            rotated_bytes,
            ::vram_heap::STORAGE_BUFFER_OFFSET_ALIGNMENT_NVIDIA,
            "forward_polar_argmax.rotated",
        ).expect("transient_heap_a capacity");
        let qjl_c_buf = self.alloc_qjl_c_buf(polar_cache, n_tokens, n_heads);

        if !inject_deltas.is_empty() {
            assert_eq!(
                inject_deltas.len(), n_layers,
                "inject_deltas must be empty or have length n_layers ({})", n_layers,
            );
        }

        // Last-token slice buffer (packed f16, [embed_dim/2] u32s).
        let last_token_bytes = (self.embed_dim * 2) as u64;
        let last_token_buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("forward_polar_argmax.last_token_packed"),
            size: last_token_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let logits_bytes = (lm_head.vocab_size * std::mem::size_of::<f32>()) as u64;
        let logits_buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("forward_polar_argmax.logits"),
            size: logits_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let token_id_buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("forward_polar_argmax.token_id"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.gpu.create_staging_buffer(4);

        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("forward_polar_argmax.encoder"),
        });

        // Optional per-block hidden-state finite check, mirroring the
        // f32 path's CORTEX_DEBUG_HIDDEN_FINITE. hidden_buf is packed
        // f16 in C3 (n_tokens * embed_dim * 2 bytes); unpack to f32 on
        // readback. Use to localize NaN/Inf propagation across layers
        // for the polar chat gibberish hunt.
        let debug_finite = std::env::var("CORTEX_DEBUG_POLAR_FINITE").is_ok();
        let hidden_bytes = (n_tokens * self.embed_dim * 2) as u64;
        let debug_captures: Vec<wgpu::Buffer> = if debug_finite {
            (0..n_layers).map(|i| self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("polar_debug_finite.{}", i)),
                size: hidden_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })).collect()
        } else { Vec::new() };
        let debug_stagings: Vec<wgpu::Buffer> = if debug_finite {
            (0..n_layers).map(|_| self.gpu.create_staging_buffer(hidden_bytes)).collect()
        } else { Vec::new() };

        // All N blocks via the polar block forward — which compresses
        // new K/V into polar_cache at [start_pos, start_pos+n_tokens) as
        // it goes (line 3126 in forward_block_gpu_polar_inner).
        for i in 0..n_layers {
            let inject = inject_deltas.get(i).and_then(|opt| opt.as_ref());
            let capture = if debug_finite { Some(&debug_captures[i]) } else { None };
            self.forward_block_gpu_polar_inner(
                &mut encoder, i, &hidden_buf, n_tokens, start_pos, &scratch,
                &rotated_buf, &*polar_cache, qjl_c_buf.as_ref(), None, capture, inject,
            );
            if debug_finite {
                encoder.copy_buffer_to_buffer(&debug_captures[i], 0, &debug_stagings[i], 0, hidden_bytes);
            }
        }

        // Final norm: packed → packed. Phase C (vram-heap): hidden_buf is
        // a sub-allocation of transient_heap_a; use the encoder-level
        // _into variant (takes BindingResource) rather than the
        // _in_pass variant (still takes &wgpu::Buffer for non-polar
        // callers).
        self.dispatch_rmsnorm_packed_to_packed_into(
            &mut encoder, hidden_buf.binding(), self.final_norm_weight_buf.binding(), normed_buf.as_entire_binding(),
            self.embed_dim, n_tokens, self.final_norm_eps,
        );

        // Slice the last token's packed row out of normed_buf.
        let last_token_offset = ((n_tokens - 1) * self.embed_dim * 2) as u64;
        encoder.copy_buffer_to_buffer(
            &normed_buf, last_token_offset,
            &last_token_buf, 0,
            last_token_bytes,
        );

        // LM head matmul: 1 × vocab_size logits.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("forward_polar_argmax.lm_head.pass"),
                timestamp_writes: None,
            });
            self.dispatch_lm_head_matmul_pin_in_pass(
                &mut pass, lm_head, &last_token_buf, &logits_buf,
            );
        }

        // Argmax → 4-byte token id.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("forward_polar_argmax.argmax.pass"),
                timestamp_writes: None,
            });
            self.dispatch_argmax_vocab_in_pass(
                &mut pass, &logits_buf, &token_id_buf, lm_head.vocab_size,
            );
        }
        encoder.copy_buffer_to_buffer(&token_id_buf, 0, &staging, 0, 4);

        self.gpu.queue.submit(Some(encoder.finish()));

        // CORTEX_DEBUG_POLAR_FINITE: per-layer hidden-state scan.
        if debug_finite {
            tracing::info!(
                n_tokens, embed_dim = self.embed_dim, n_layers, start_pos,
                "CORTEX_DEBUG_POLAR_FINITE: per-block hidden-state scan",
            );
            for i in 0..n_layers {
                let unpacked = read_back_buffer_f16_unpack(
                    &self.gpu, &debug_stagings[i], hidden_bytes as usize,
                );
                let mut n_inf = 0usize;
                let mut n_nan = 0usize;
                let mut max_abs = 0.0f32;
                let mut first_bad: Option<usize> = None;
                for (idx, &v) in unpacked.iter().enumerate() {
                    if v.is_nan() { n_nan += 1; if first_bad.is_none() { first_bad = Some(idx); } }
                    else if v.is_infinite() { n_inf += 1; if first_bad.is_none() { first_bad = Some(idx); } }
                    else if v.abs() > max_abs { max_abs = v.abs(); }
                }
                tracing::info!(
                    layer = i, n_inf, n_nan, max_abs,
                    first_bad = ?first_bad,
                    total = unpacked.len(),
                    "polar hidden state",
                );
                if n_inf + n_nan > 0 {
                    tracing::warn!(layer = i, "polar: FIRST LAYER with non-finite values");
                    break;
                }
            }
        }

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        rx.recv().expect("token-id readback failed").expect("token-id map failed");
        let data = slice.get_mapped_range();
        let token_id = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        drop(data);
        staging.unmap();

        polar_cache.set_len(start_pos + n_tokens);
        Some(token_id)
    }

    /// Polar variant of `forward_block_gpu_inner`. Same RMSNorm + projections
    /// + RoPE + FFN as the f32 path, but the K/V cache write is replaced by
    /// the GPU compress shader (writing into a `GpuPolarKvCache` layer's
    /// resident buffers) and the score → softmax → value chain runs against
    /// the polar cache via the batch polar dispatchers + softmax_batch.
    ///
    /// Caller provides:
    /// - `polar_cache`: the resident compressed cache (must hold the full
    ///   prefix this attention attends to; the query's K/V are written here
    ///   at offset `start_pos`).
    /// - `rotated_buf`: scratch buffer of size `n_tokens * n_heads * head_dim`
    ///   f32, reused inside the block as both `rq` (post-rotate_q) and
    ///   `weighted_rotated_V` (pre-derotate). One allocation per trace call.
    /// - `pre_softmax_capture`: optional buffer to copy scores into before
    ///   softmax overwrites them — for retrieval-mode tracing.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_block_gpu_polar_inner(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        block_idx: usize,
        hidden_buf: &::vram_heap::VramAllocation,
        n_tokens: usize,
        start_pos: usize,
        scratch: &PolarBlockScratch,
        rotated_buf: &::vram_heap::VramAllocation,
        polar_cache: &crate::layers::gpu_polar_kv_cache::GpuPolarKvCache,
        // Phase O: per-forward scratch for the V QJL correction's C
        // accumulator ([n_tokens, n_heads, n_v_proj] f32, Lane C).
        // `Some` iff the cache has QJL; reused across blocks (each
        // block fully rewrites it before reading — per-block
        // allocation would let RAII recycle the range mid-encoder).
        qjl_c_buf: Option<&::vram_heap::VramAllocation>,
        pre_softmax_capture: Option<&wgpu::Buffer>,
        post_block_hidden_capture: Option<&wgpu::Buffer>,
        pre_block_hidden_inject: Option<&wgpu::Buffer>,
    ) {
        let block = &self.cpu.blocks()[block_idx];
        let block_gpu = &self.blocks_gpu[block_idx];
        let attn = block.attention();

        // Injection-phase hook (#6c). See forward_block_gpu_inner for
        // full discussion — same broadcast-add semantics, just on the
        // polar attention path.
        if let Some(delta_buf) = pre_block_hidden_inject {
            // C3: hidden_buf is packed — use packed broadcast.
            self.dispatch_add_broadcast_into(
                encoder, hidden_buf.binding(), delta_buf, self.embed_dim, n_tokens,
            );
        }

        // BitNet b1.58 sub-norms are routed through the same GpuBlock-
        // resident buffers used by the f32 forward path (#bn-11).
        assert!((block.attn_residual_scale() - 1.0).abs() < f32::EPSILON);
        assert!((block.ffn_residual_scale() - 1.0).abs() < f32::EPSILON);

        let swiglu = block.ffn().as_any()
            .downcast_ref::<crate::layers::swiglu::SwiGLU>()
            .unwrap_or_else(|| panic!("forward_block_gpu_polar requires SwiGLU FFN"));
        // Activation routed at dispatch time (SiLU vs ReLU²).

        let n_heads = attn.n_heads();
        let n_kv_heads = attn.n_kv_heads();
        let head_dim = attn.head_dim();
        let embed_dim = self.embed_dim;
        let intermediate = swiglu.intermediate_size();
        let kv_dim = n_kv_heads * head_dim;
        let attn_max_seq = start_pos + n_tokens;

        assert_eq!(n_kv_heads, polar_cache.n_kv_heads(),
            "polar cache n_kv_heads mismatch");
        assert_eq!(head_dim, polar_cache.head_dim(),
            "polar cache head_dim mismatch");
        assert!(attn_max_seq <= polar_cache.max_seq_len(),
            "polar cache overflow: start_pos {} + n_tokens {} > max_seq {}",
            start_pos, n_tokens, polar_cache.max_seq_len());
        assert!(start_pos + n_tokens <= self.rope_max_seq);

        // ===== ATTENTION SUBLAYER =====

        // 1. attn_norm — C3: hidden packed → normed packed.
        self.dispatch_rmsnorm_packed_to_packed_into(
            encoder, hidden_buf.binding(), block_gpu.attn_norm_weight_buf.binding(), scratch.normed.binding(),
            embed_dim, n_tokens, block_gpu.attn_norm_eps,
        );

        // 2-4. Q, K, V projections — C3: packed input AND packed output.
        self.dispatch_linear_batch_packed_io_into(encoder, attn.q_proj(), scratch.normed.binding(), scratch.q.binding(), n_tokens);
        if let Some(buf) = block_gpu.q_bias_buf.as_ref() {
            self.dispatch_bias_add_packed_into(encoder, scratch.q.binding(), buf.binding(), n_heads * head_dim, n_tokens);
        }
        self.dispatch_linear_batch_packed_io_into(encoder, attn.k_proj(), scratch.normed.binding(), scratch.k.binding(), n_tokens);
        if let Some(buf) = block_gpu.k_bias_buf.as_ref() {
            self.dispatch_bias_add_packed_into(encoder, scratch.k.binding(), buf.binding(), kv_dim, n_tokens);
        }
        self.dispatch_linear_batch_packed_io_into(encoder, attn.v_proj(), scratch.normed.binding(), scratch.v.binding(), n_tokens);
        if let Some(buf) = block_gpu.v_bias_buf.as_ref() {
            self.dispatch_bias_add_packed_into(encoder, scratch.v.binding(), buf.binding(), kv_dim, n_tokens);
        }

        // 5. RoPE on Q and K — C3 packed (in-place).
        // Phase P.2 (diagnostic): skip under CORTEX_RETRIEVE_OFFSET_ZERO
        // so the query is encoded position-free, matching the
        // position-free corpus prefill — scores become pure content
        // dots. See the f32 inner forward for the full rationale + caveat.
        if std::env::var("CORTEX_RETRIEVE_OFFSET_ZERO").is_err() {
            self.dispatch_rope_packed_into(
                encoder, scratch.q.binding(), self.rope_cos_buf.binding(), self.rope_sin_buf.binding(),
                n_heads, head_dim, start_pos, n_tokens,
            );
            self.dispatch_rope_packed_into(
                encoder, scratch.k.binding(), self.rope_cos_buf.binding(), self.rope_sin_buf.binding(),
                n_kv_heads, head_dim, start_pos, n_tokens,
            );
        }

        // 5.5 Compress K and V into the polar cache at [start_pos, start_pos+n_tokens).
        // kv_compress_polar reads packed-f16 input (fixed 2026-05-27).
        crate::layers::gpu_polar::compress_layer_into_polar(
            &self.gpu, encoder, polar_cache, block_idx,
            scratch.k.binding(), scratch.v.binding(), n_tokens, start_pos,
        );

        // 5.6 If QJL is enabled, encode K residual signs for the freshly-
        // compressed positions. Must run after compress (reads angles +
        // radius written above). No-op when polar_cache.n_qjl_proj() == 0.
        crate::layers::gpu_polar::qjl_encode_k_layer(
            &self.gpu, encoder, polar_cache, block_idx,
            scratch.k.binding(), n_tokens, start_pos,
        );

        // 5.7 Phase O: V residual signs + norms for the value-side
        // correction. Same compress-then-encode ordering as K.
        crate::layers::gpu_polar::qjl_encode_v_layer(
            &self.gpu, encoder, polar_cache, block_idx,
            scratch.v.binding(), n_tokens, start_pos,
        );

        // 6a. rotate_q: packed scratch.q → f32 rotated_buf.
        crate::layers::gpu_polar::dispatch_rotate_q_packed(
            &self.gpu, encoder, scratch.q.binding(),
            polar_cache.rotation_layer(block_idx).binding(),
            rotated_buf.binding(),
            n_tokens, n_heads, head_dim,
        );

        // 6b. attn_score: rotated_buf · K_polar → scratch.scores.
        // Use the QJL-corrected variant when the cache has QJL signs;
        // otherwise the plain polar variant. Both write the same
        // scratch.scores layout, so softmax + value paths are unchanged.
        if polar_cache.n_qjl_proj() > 0 {
            crate::layers::gpu_polar::dispatch_attn_score_polar_qjl_batch(
                &self.gpu, encoder, rotated_buf.binding(),
                polar_cache.k_angles_layer(block_idx).binding(),
                polar_cache.k_radius_layer(block_idx).binding(),
                scratch.scores.binding(),
                polar_cache.k_qjl_signs_layer(block_idx)
                    .expect("k_qjl_signs_layer must exist when n_qjl_proj > 0")
                    .binding(),
                polar_cache.k_qjl_projection_layer(block_idx)
                    .expect("k_qjl_projection_layer must exist when n_qjl_proj > 0")
                    .binding(),
                polar_cache.lut_buffer().binding(),
                n_heads, n_kv_heads, head_dim, start_pos, n_tokens, attn_max_seq,
                polar_cache.n_qjl_proj(),
            );
        } else {
            crate::layers::gpu_polar::dispatch_attn_score_polar_batch(
                &self.gpu, encoder, rotated_buf.binding(),
                polar_cache.k_angles_layer(block_idx).binding(),
                polar_cache.k_radius_layer(block_idx).binding(),
                scratch.scores.binding(), polar_cache.lut_buffer().binding(),
                n_heads, n_kv_heads, head_dim, start_pos, n_tokens, attn_max_seq,
            );
        }

        // 6c. (optional) capture pre-softmax scores
        if let Some(capture_buf) = pre_softmax_capture {
            let bytes = (n_tokens * n_heads * attn_max_seq * std::mem::size_of::<f32>()) as u64;
            encoder.copy_buffer_to_buffer(
                scratch.scores.buffer(), scratch.scores.offset(),
                capture_buf, 0, bytes,
            );
        }

        // 6d. softmax in-place on scratch.scores. Reuses the f32 path's softmax_batch
        //     pipeline; same scores buffer layout.
        let sm_params = SoftmaxBatchParams {
            n_heads: n_heads as u32,
            max_seq: attn_max_seq as u32,
            start_pos: start_pos as u32,
            n_tokens: n_tokens as u32,
        };
        let sm_params_buf = self.gpu.create_params_buffer(&sm_params);
        let sm_pipeline = &self.gpu.pipelines.softmax_batch;
        let sm_bind = self.gpu.make_bind_group_with(
            sm_pipeline,
            vec![scratch.scores.binding(), sm_params_buf.as_entire_binding()],
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_engine.polar.softmax.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(sm_pipeline);
            pass.set_bind_group(0, &sm_bind, &[]);
            pass.dispatch_workgroups((n_tokens * n_heads) as u32, 1, 1);
        }

        // 6e. attn_value: scratch.scores * V_polar → rotated_buf
        //     (overwriting rq, no longer needed). Phase O: when the
        //     cache has QJL, run the two-pass corrected variant —
        //     pass A accumulates C_j = Σ_t w_t·rnorm_t·s_tj, pass B
        //     adds the Γ-scaled residual estimate to the weighted sum.
        if polar_cache.n_qjl_proj() > 0 {
            let c_buf = qjl_c_buf
                .expect("qjl_c_buf must be provided when polar_cache has QJL");
            crate::layers::gpu_polar::dispatch_qjl_value_weights(
                &self.gpu, encoder, scratch.scores.binding(),
                polar_cache.v_qjl_signs_layer(block_idx)
                    .expect("v_qjl_signs_layer must exist when n_qjl_proj > 0")
                    .binding(),
                polar_cache.v_qjl_rnorm_layer(block_idx)
                    .expect("v_qjl_rnorm_layer must exist when n_qjl_proj > 0")
                    .binding(),
                c_buf.binding(),
                n_heads, n_kv_heads, polar_cache.n_v_qjl_proj(),
                start_pos, n_tokens, attn_max_seq,
            );
            crate::layers::gpu_polar::dispatch_attn_value_polar_qjl_batch(
                &self.gpu, encoder, scratch.scores.binding(),
                polar_cache.v_angles_layer(block_idx).binding(),
                polar_cache.v_radius_layer(block_idx).binding(),
                c_buf.binding(),
                polar_cache.v_qjl_projection_layer(block_idx)
                    .expect("v_qjl_projection_layer must exist when n_qjl_proj > 0")
                    .binding(),
                rotated_buf.binding(), polar_cache.lut_buffer().binding(),
                n_heads, n_kv_heads, head_dim, start_pos, n_tokens, attn_max_seq,
                polar_cache.n_v_qjl_proj(),
            );
        } else {
            crate::layers::gpu_polar::dispatch_attn_value_polar_batch(
                &self.gpu, encoder, scratch.scores.binding(),
                polar_cache.v_angles_layer(block_idx).binding(),
                polar_cache.v_radius_layer(block_idx).binding(),
                rotated_buf.binding(), polar_cache.lut_buffer().binding(),
                n_heads, n_kv_heads, head_dim, start_pos, n_tokens, attn_max_seq,
            );
        }

        // 6f. derotate: f32 rotated_buf → packed scratch.attn_out (C3).
        //     Treat (n_tokens * n_heads) as effective head count — R is per-layer.
        crate::layers::gpu_polar::dispatch_derotate_packed(
            &self.gpu, encoder, rotated_buf.binding(),
            polar_cache.rotation_layer(block_idx).binding(), scratch.attn_out.binding(),
            n_tokens * n_heads, head_dim,
        );

        // 7. O projection — C3: packed attn_out → packed projected.
        self.dispatch_linear_batch_packed_io_into(encoder, attn.o_proj(), scratch.attn_out.binding(), scratch.projected.binding(), n_tokens);

        // 8. Residual — C3: hidden packed += projected packed.
        self.dispatch_add_packed_into(encoder, hidden_buf.binding(), scratch.projected.binding(), embed_dim, n_tokens);

        // ===== FFN SUBLAYER =====

        // ffn_norm — C3: hidden packed → normed packed.
        self.dispatch_rmsnorm_packed_to_packed_into(
            encoder, hidden_buf.binding(), block_gpu.ffn_norm_weight_buf.binding(), scratch.normed.binding(),
            embed_dim, n_tokens, block_gpu.ffn_norm_eps,
        );
        // C2 polar: gate/up/activated packed; use packed-IO router.
        self.dispatch_linear_batch_packed_io_into(encoder, swiglu.gate_proj(), scratch.normed.binding(), scratch.gate.binding(), n_tokens);
        self.dispatch_linear_batch_packed_io_into(encoder, swiglu.up_proj(),   scratch.normed.binding(), scratch.up.binding(),   n_tokens);
        self.dispatch_gate_mul_packed_into(encoder, scratch.gate.binding(), scratch.up.binding(), scratch.activated.binding(), intermediate, n_tokens, swiglu.activation());

        // C3 down_proj: packed input, packed output.
        self.dispatch_linear_batch_packed_io_into(encoder, swiglu.down_proj(), scratch.activated.binding(), scratch.projected.binding(), n_tokens);
        // C3: hidden packed += projected packed.
        self.dispatch_add_packed_into(encoder, hidden_buf.binding(), scratch.projected.binding(), embed_dim, n_tokens);

        // (optional) post-block hidden state capture — shim hook point.
        // C3: hidden_buf is packed f16; capture is half size.
        if let Some(capture_buf) = post_block_hidden_capture {
            let bytes = (n_tokens * embed_dim * 2) as u64;
            encoder.copy_buffer_to_buffer(
                hidden_buf.buffer(), hidden_buf.offset(),
                capture_buf, 0, bytes,
            );
        }
    }

}
