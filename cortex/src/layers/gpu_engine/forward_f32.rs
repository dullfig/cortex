//! f32-path forward variants (split from gpu_engine.rs, Phase N).

use super::*;

impl GpuEngine {
    /// Forward pass with GPU-native final RMSNorm. Embedding lookup and
    /// transformer blocks still run on CPU; output projection runs on CPU.
    /// Phase 1b checkpoint — proves the rmsnorm dispatch path is correct
    /// before we move attention/FFN into the same orchestration.
    pub fn forward_gpu(&self, tokens: &[u32], start_pos: usize) -> Vec<f32> {
        let pre_norm = self.cpu.forward_pre_norm(tokens, start_pos);
        let normed = self.dispatch_final_norm(&pre_norm, tokens.len());
        self.cpu.finalize_logits(&normed, tokens.len())
    }

    /// Like `forward_full_gpu_traced` but skips the output projection
    /// entirely (no logits computed). Retrieval doesn't need logits; the
    /// per-token GpuFloatLinear vocab projection across 2000+ tokens is the
    /// expensive part and was hanging the server.
    pub fn forward_traced_scores_only(
        &self,
        tokens: &[u32],
        start_pos: usize,
        capture_layers: &[usize],
    ) -> Vec<Vec<f32>> {
        let (_logits, scores) = self.forward_traced_inner(tokens, start_pos, capture_layers, false);
        scores
    }

    /// Forward pass that captures pre-softmax attention scores for the
    /// requested layers. Used by retrieval (memex) — the per-position
    /// attention weight aggregation in `cortex-cloud`'s `/v1/retrieve`
    /// handler reads these scores.
    ///
    /// `capture_layers`: indices of blocks whose pre-softmax attention
    /// scores should be captured. Each capture is sized
    /// `[n_tokens, n_heads, n_tokens]` f32 = O(n_tokens² × n_heads × 4)
    /// bytes per layer. For Qwen 3B at 2300 tokens × 16 heads × 4 bytes,
    /// that's ~340 MB per layer, so callers should keep the set small.
    /// memex architecture suggests "last few layers" carry the retrieval
    /// signal; default in cortex-cloud is the last 4.
    ///
    /// Returns `(logits, per_layer_scores)` where `per_layer_scores[i]`
    /// is the captured pre-softmax tensor for `capture_layers[i]`,
    /// flat as `[n_tokens, n_heads, n_tokens]`.
    pub fn forward_full_gpu_traced(
        &self,
        tokens: &[u32],
        start_pos: usize,
        capture_layers: &[usize],
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        self.forward_traced_inner(tokens, start_pos, capture_layers, true)
    }

    pub(super) fn forward_traced_inner(
        &self,
        tokens: &[u32],
        start_pos: usize,
        capture_layers: &[usize],
        compute_logits: bool,
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        let n_tokens = tokens.len();
        assert!(n_tokens > 0, "must have at least one token");
        let n_layers = self.cpu.n_layers();
        for &l in capture_layers {
            assert!(l < n_layers, "capture layer {l} out of range (n_layers={n_layers})");
        }

        // ---- 1. Embedding lookup (CPU) ----
        let embed_data = self.cpu.embedding_data();
        let vocab_size = self.cpu.vocab_size();
        let mut hidden_init: Vec<f32> = Vec::with_capacity(n_tokens * self.embed_dim);
        for &tok in tokens {
            assert!((tok as usize) < vocab_size, "token {tok} out of vocab");
            let off = tok as usize * self.embed_dim;
            hidden_init.extend_from_slice(&embed_data[off..off + self.embed_dim]);
        }

        // ---- Allocate buffers ----
        let bytes = (hidden_init.len() * std::mem::size_of::<f32>()) as u64;
        let hidden_buf = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("forward_full_traced.hidden"),
            contents: bytemuck::cast_slice(&hidden_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let normed_buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("forward_full_traced.normed"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let normed_staging = self.gpu.create_staging_buffer(bytes);

        let attn0 = self.cpu.blocks()[0].attention();
        let intermediate = self.cpu.blocks()[0].ffn().as_any()
            .downcast_ref::<crate::layers::swiglu::SwiGLU>()
            .unwrap_or_else(|| panic!("forward_full_gpu_traced requires SwiGLU FFN"))
            .intermediate_size();
        let n_heads = attn0.n_heads();
        let scratch = BlockScratch::allocate(
            &self.gpu, n_tokens, self.embed_dim,
            n_heads, attn0.n_kv_heads(), attn0.head_dim(),
            intermediate, n_tokens,
        );

        // Per-captured-layer score storage buffers. Same shape as scratch.scores
        // but persistent across the whole forward.
        let scores_bytes = (n_tokens * n_heads * n_tokens * std::mem::size_of::<f32>()) as u64;
        let capture_bufs: Vec<wgpu::Buffer> = capture_layers.iter().map(|&l| {
            self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("forward_full_traced.scores.layer{l}")),
                size: scores_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        }).collect();
        let capture_stagings: Vec<wgpu::Buffer> = (0..capture_layers.len())
            .map(|_| self.gpu.create_staging_buffer(scores_bytes))
            .collect();

        // Build a layer_idx -> capture_buf lookup for O(1) access in the loop.
        let capture_lookup: std::collections::HashMap<usize, &wgpu::Buffer> =
            capture_layers.iter().zip(capture_bufs.iter())
                .map(|(&l, buf)| (l, buf))
                .collect();

        // ---- 2-3. All blocks + final_norm in one encoder ----
        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("forward_full_traced.encoder"),
        });
        for i in 0..n_layers {
            let capture = capture_lookup.get(&i).copied();
            self.forward_block_gpu_inner(&mut encoder, i, &hidden_buf, n_tokens, start_pos, &scratch, capture, None, None, None);
        }
        self.dispatch_rmsnorm_into(
            &mut encoder, hidden_buf.as_entire_binding(), self.final_norm_weight_buf.binding(), normed_buf.as_entire_binding(),
            self.embed_dim, n_tokens, self.final_norm_eps,
        );
        encoder.copy_buffer_to_buffer(&normed_buf, 0, &normed_staging, 0, bytes);
        for (cap_buf, stg_buf) in capture_bufs.iter().zip(capture_stagings.iter()) {
            encoder.copy_buffer_to_buffer(cap_buf, 0, stg_buf, 0, scores_bytes);
        }
        self.gpu.queue.submit(Some(encoder.finish()));

        // Issue all map_async calls together, then poll once. Sequential
        // poll(Wait) per buffer was hanging — possibly because the wgpu
        // device only fires callbacks inside poll, and re-polling after
        // a buffer's already mapped doesn't re-fire pending callbacks for
        // others in some cases. Single poll drives all of them at once.
        use std::sync::mpsc;
        let mut receivers: Vec<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>> = Vec::with_capacity(1 + capture_stagings.len());
        let normed_slice = normed_staging.slice(..);
        let (tx, rx) = mpsc::channel();
        normed_slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        receivers.push(rx);
        let capture_slices: Vec<wgpu::BufferSlice> = capture_stagings.iter().map(|stg| {
            let slice = stg.slice(..);
            let (tx, rx) = mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
            receivers.push(rx);
            slice
        }).collect();

        self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        for rx in &receivers {
            rx.recv().expect("readback channel closed").expect("buffer map failed");
        }

        // ---- Decode the readbacks ----
        let normed: Vec<f32> = {
            let data = normed_slice.get_mapped_range();
            let v: Vec<f32> = data[..bytes as usize].chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            drop(data);
            normed_staging.unmap();
            v
        };
        let per_layer_scores: Vec<Vec<f32>> = capture_slices.iter().zip(capture_stagings.iter()).map(|(slice, stg)| {
            let data = slice.get_mapped_range();
            let v: Vec<f32> = data[..scores_bytes as usize].chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            drop(data);
            stg.unmap();
            v
        }).collect();

        // ---- 4. Output projection (CPU; vocab matmul deferred). Skipped
        //         when caller doesn't need logits (retrieve path) — that
        //         saves 2000+ per-token GpuFloatLinear calls and the
        //         staging-buffer churn that comes with them. ----
        let logits = if compute_logits {
            self.cpu.finalize_logits(&normed, n_tokens)
        } else {
            Vec::new()
        };

        (logits, per_layer_scores)
    }

    /// Hidden-state capture forward pass — non-cached. Runs the model
    /// forward over `tokens` and returns:
    /// - `per_layer_hidden[i]` — the post-FFN-residual hidden state at
    ///   the END of block `capture_layers[i]`. Shape:
    ///   `n_tokens * embed_dim` f32. Same layout as `entrance:(N+1)`
    ///   inputs in shim manifests.
    /// - `final_post_norm_hidden` — the final RMSNorm output (input to
    ///   the LM head). Shape `n_tokens * embed_dim` f32. This is what
    ///   `attachment.layer = "final"` gate / steer shims read; pool
    ///   downstream (last_token / mean / etc.) per the manifest.
    ///
    /// Read-side hook only — no injection / steering. Those are #5/#6.
    pub fn forward_full_gpu_with_hidden_capture(
        &self,
        tokens: &[u32],
        capture_layers: &[usize],
    ) -> HiddenCaptures {
        let n_tokens = tokens.len();
        assert!(n_tokens > 0);
        let n_layers = self.cpu.n_layers();
        for &l in capture_layers {
            assert!(l < n_layers, "capture layer {l} out of range (n_layers={n_layers})");
        }

        // Embedding lookup (CPU).
        let embed_data = self.cpu.embedding_data();
        let vocab_size = self.cpu.vocab_size();
        let mut hidden_init: Vec<f32> = Vec::with_capacity(n_tokens * self.embed_dim);
        for &tok in tokens {
            assert!((tok as usize) < vocab_size, "token {tok} out of vocab");
            let off = tok as usize * self.embed_dim;
            hidden_init.extend_from_slice(&embed_data[off..off + self.embed_dim]);
        }

        let hidden_bytes = (hidden_init.len() * std::mem::size_of::<f32>()) as u64;
        let hidden_buf = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("forward_hidden_capture.hidden"),
            contents: bytemuck::cast_slice(&hidden_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let normed_buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("forward_hidden_capture.normed"),
            size: hidden_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let normed_staging = self.gpu.create_staging_buffer(hidden_bytes);

        let attn0 = self.cpu.blocks()[0].attention();
        let intermediate = self.cpu.blocks()[0].ffn().as_any()
            .downcast_ref::<crate::layers::swiglu::SwiGLU>()
            .unwrap_or_else(|| panic!("forward_full_gpu_with_hidden_capture requires SwiGLU FFN"))
            .intermediate_size();
        let n_heads = attn0.n_heads();
        let scratch = BlockScratch::allocate(
            &self.gpu, n_tokens, self.embed_dim,
            n_heads, attn0.n_kv_heads(), attn0.head_dim(),
            intermediate, n_tokens,
        );

        // Per-captured-layer hidden buffers (post-FFN-residual). Same shape
        // as hidden_buf: [n_tokens, embed_dim] f32 flat.
        let capture_bufs: Vec<wgpu::Buffer> = capture_layers.iter().map(|&l| {
            self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("forward_hidden_capture.layer{l}")),
                size: hidden_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        }).collect();
        let capture_stagings: Vec<wgpu::Buffer> = (0..capture_layers.len())
            .map(|_| self.gpu.create_staging_buffer(hidden_bytes))
            .collect();
        let capture_lookup: std::collections::HashMap<usize, &wgpu::Buffer> =
            capture_layers.iter().zip(capture_bufs.iter())
                .map(|(&l, buf)| (l, buf))
                .collect();

        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("forward_hidden_capture.encoder"),
        });
        for i in 0..n_layers {
            let post_capture = capture_lookup.get(&i).copied();
            self.forward_block_gpu_inner(
                &mut encoder, i, &hidden_buf, n_tokens, /*start_pos*/ 0, &scratch,
                /*pre_softmax_capture*/ None, /*kv_cache_target*/ None, post_capture,
                /*pre_block_hidden_inject*/ None,
            );
        }
        // Final RMSNorm — gives the final post-norm hidden state shims read.
        self.dispatch_rmsnorm_into(
            &mut encoder, hidden_buf.as_entire_binding(), self.final_norm_weight_buf.binding(), normed_buf.as_entire_binding(),
            self.embed_dim, n_tokens, self.final_norm_eps,
        );
        encoder.copy_buffer_to_buffer(&normed_buf, 0, &normed_staging, 0, hidden_bytes);
        for (cap_buf, stg_buf) in capture_bufs.iter().zip(capture_stagings.iter()) {
            encoder.copy_buffer_to_buffer(cap_buf, 0, stg_buf, 0, hidden_bytes);
        }
        self.gpu.queue.submit(Some(encoder.finish()));

        // Batched readback (single poll for all stagings).
        use std::sync::mpsc;
        let mut receivers: Vec<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>> =
            Vec::with_capacity(1 + capture_stagings.len());
        let normed_slice = normed_staging.slice(..);
        let (tx, rx) = mpsc::channel();
        normed_slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        receivers.push(rx);
        let capture_slices: Vec<wgpu::BufferSlice> = capture_stagings.iter().map(|stg| {
            let slice = stg.slice(..);
            let (tx, rx) = mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
            receivers.push(rx);
            slice
        }).collect();

        self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        for rx in &receivers {
            rx.recv().expect("readback channel closed").expect("buffer map failed");
        }

        let final_post_norm_hidden: Vec<f32> = {
            let data = normed_slice.get_mapped_range();
            let v: Vec<f32> = data[..hidden_bytes as usize].chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            drop(data);
            normed_staging.unmap();
            v
        };
        let per_layer_hidden: Vec<Vec<f32>> = capture_slices.iter().zip(capture_stagings.iter()).map(|(slice, stg)| {
            let data = slice.get_mapped_range();
            let v: Vec<f32> = data[..hidden_bytes as usize].chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            drop(data);
            stg.unmap();
            v
        }).collect();

        HiddenCaptures {
            per_layer_hidden,
            final_post_norm_hidden,
            n_tokens,
            embed_dim: self.embed_dim,
        }
    }

    /// Cached + traced forward: process new `query_tokens` against a
    /// pre-populated `cache`, capturing pre-softmax attention scores for
    /// the requested layers. **Does not advance the cache cursor** — the
    /// cache stays at its original `seq_len`, so this is safe to call
    /// repeatedly with different queries against the same shard cache.
    ///
    /// The query tokens' K/V do get written into the cache buffers at
    /// offset `cache.seq_len()` during the dispatch, but those positions
    /// are logically unallocated (`cache.seq_len()` is the end of the
    /// real prefix), so the next call simply overwrites them. Subsequent
    /// `forward_full_gpu_with_cache` calls would also overwrite that
    /// region — the contract is "K/V at offsets >= seq_len are scratch."
    ///
    /// Returns per-layer pre-softmax score tensors flat as
    /// `[n_query_tokens, n_heads, max_seq]` where `max_seq = cache.seq_len()
    /// + n_query_tokens`. The score from query position q to corpus
    /// position k is `scores[q * n_heads * max_seq + h * max_seq + k]`,
    /// which is what cortex-cloud's retrieve handler aggregates over.
    pub fn forward_full_gpu_with_cache_traced(
        &self,
        query_tokens: &[u32],
        cache: &crate::layers::gpu_kv_cache::GpuKvCache,
        capture_layers: &[usize],
    ) -> Vec<Vec<f32>> {
        let n_tokens = query_tokens.len();
        assert!(n_tokens > 0, "must have at least one query token");

        let n_layers = self.cpu.n_layers();
        assert_eq!(n_layers, cache.n_layers(), "cache layer count mismatch");
        for &l in capture_layers {
            assert!(l < n_layers, "capture layer {l} out of range (n_layers={n_layers})");
        }

        let attn0 = self.cpu.blocks()[0].attention();
        assert_eq!(cache.n_kv_heads(), attn0.n_kv_heads(), "cache n_kv_heads mismatch");
        assert_eq!(cache.head_dim(), attn0.head_dim(), "cache head_dim mismatch");

        let start_pos = cache.seq_len();
        let attn_max_seq = start_pos + n_tokens;
        assert!(
            attn_max_seq <= cache.max_seq_len(),
            "cache overflow: {} + {} > {}",
            start_pos, n_tokens, cache.max_seq_len(),
        );

        // ---- Embedding lookup (CPU) ----
        let embed_data = self.cpu.embedding_data();
        let vocab_size = self.cpu.vocab_size();
        let mut hidden_init: Vec<f32> = Vec::with_capacity(n_tokens * self.embed_dim);
        for &tok in query_tokens {
            assert!((tok as usize) < vocab_size, "token {tok} out of vocab");
            let off = tok as usize * self.embed_dim;
            hidden_init.extend_from_slice(&embed_data[off..off + self.embed_dim]);
        }

        // Phase 0 fix (same class as 7d63396 cache_advance_only +
        // f5b55a2 polar_traced): forward_block_gpu_inner reads hidden_buf
        // as packed f16 (C3). Passing raw f32 here feeds misaligned
        // bytes through the block, garbage K/V get written into the
        // captured score buffers as NaN, and the retrieve aggregator
        // filters them all out — symptom: /v1/chat/completions
        // mode="retrieve" returns {"hits":[]} for every query regardless
        // of corpus or top_k. cache_traced was the last sibling left
        // unpatched after the f5b55a2/7d63396 hunt.
        let hidden_packed = GpuDevice::pack_f16(&hidden_init);
        let hidden_buf = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("forward_traced_with_cache.hidden"),
            contents: bytemuck::cast_slice(&hidden_packed),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let intermediate = self.cpu.blocks()[0].ffn().as_any()
            .downcast_ref::<crate::layers::swiglu::SwiGLU>()
            .unwrap_or_else(|| panic!("forward_full_gpu_with_cache_traced requires SwiGLU FFN"))
            .intermediate_size();
        let n_heads = attn0.n_heads();
        let scratch = BlockScratch::allocate(
            &self.gpu, n_tokens, self.embed_dim,
            n_heads, attn0.n_kv_heads(), attn0.head_dim(),
            intermediate, attn_max_seq,
        );

        // Per-captured-layer score storage. Shape: [n_tokens, n_heads, max_seq]
        // where max_seq = start_pos + n_tokens (the full attention window).
        let scores_bytes = (n_tokens * n_heads * attn_max_seq * std::mem::size_of::<f32>()) as u64;
        let capture_bufs: Vec<wgpu::Buffer> = capture_layers.iter().map(|&l| {
            self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("forward_traced_with_cache.scores.layer{l}")),
                size: scores_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        }).collect();
        let capture_stagings: Vec<wgpu::Buffer> = (0..capture_layers.len())
            .map(|_| self.gpu.create_staging_buffer(scores_bytes))
            .collect();
        let capture_lookup: std::collections::HashMap<usize, &wgpu::Buffer> =
            capture_layers.iter().zip(capture_bufs.iter())
                .map(|(&l, buf)| (l, buf))
                .collect();

        // ---- All blocks in one encoder, with cache target + optional capture ----
        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("forward_traced_with_cache.encoder"),
        });
        for i in 0..n_layers {
            let capture = capture_lookup.get(&i).copied();
            let target = (cache.k_layer(i), cache.v_layer(i));
            self.forward_block_gpu_inner(
                &mut encoder, i, &hidden_buf, n_tokens, start_pos, &scratch,
                capture, Some(target), None, None,
            );
        }
        // Skip final_norm + output projection — retrieval doesn't need
        // logits, only the captured scores.
        for (cap_buf, stg_buf) in capture_bufs.iter().zip(capture_stagings.iter()) {
            encoder.copy_buffer_to_buffer(cap_buf, 0, stg_buf, 0, scores_bytes);
        }
        self.gpu.queue.submit(Some(encoder.finish()));

        // Batched readback (single poll for all stagings).
        use std::sync::mpsc;
        let mut receivers: Vec<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>> = Vec::with_capacity(capture_stagings.len());
        let capture_slices: Vec<wgpu::BufferSlice> = capture_stagings.iter().map(|stg| {
            let slice = stg.slice(..);
            let (tx, rx) = mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
            receivers.push(rx);
            slice
        }).collect();
        self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        for rx in &receivers {
            rx.recv().expect("readback channel closed").expect("buffer map failed");
        }

        let per_layer_scores: Vec<Vec<f32>> = capture_slices.iter().zip(capture_stagings.iter()).map(|(slice, stg)| {
            let data = slice.get_mapped_range();
            let v: Vec<f32> = data[..scores_bytes as usize].chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            drop(data);
            stg.unmap();
            v
        }).collect();

        // NOTE: we deliberately do NOT call cache.advance() — see method docs.
        per_layer_scores
    }

    /// Forward pass that writes new K/V into the supplied `cache` and reads
    /// the full prefix from the cache during attention. Both prefill (cache
    /// initially empty, `cache.seq_len() == 0`) and decode (cache populated,
    /// new tokens at positions `[cache.seq_len(), cache.seq_len() + n_tokens)`)
    /// are handled by the same code path — `cache.seq_len()` becomes the
    /// RoPE/attention `start_pos`, and the cache buffers are sized for the
    /// full prefix.
    ///
    /// On success the cache's write cursor is advanced by `n_tokens`.
    /// Returns logits over vocab for each new token (same shape as
    /// `forward_full_gpu`).
    pub fn forward_full_gpu_with_cache(
        &self,
        tokens: &[u32],
        cache: &mut crate::layers::gpu_kv_cache::GpuKvCache,
    ) -> Vec<f32> {
        let n_tokens = tokens.len();
        let normed = self.forward_full_gpu_with_cache_inject_returning_hidden(tokens, cache, &[]);
        self.cpu.finalize_logits(&normed, n_tokens)
    }

    /// `forward_full_gpu_with_cache` plus injection-phase deltas.
    /// `inject_deltas` is either empty (no injection) or has length
    /// `n_layers()` with `Some(buf)` for each layer that carries a
    /// summed `[embed_dim]` f32 hidden_delta. Each present buffer is
    /// broadcast-added into hidden at that block's entrance.
    pub fn forward_full_gpu_with_cache_inject(
        &self,
        tokens: &[u32],
        cache: &mut crate::layers::gpu_kv_cache::GpuKvCache,
        inject_deltas: &[Option<wgpu::Buffer>],
    ) -> Vec<f32> {
        let n_tokens = tokens.len();
        let normed = self.forward_full_gpu_with_cache_inject_returning_hidden(
            tokens, cache, inject_deltas,
        );
        self.cpu.finalize_logits(&normed, n_tokens)
    }

    /// Same forward as `forward_full_gpu_with_cache` but returns the final
    /// post-norm hidden state ([n_tokens * embed_dim] row-major f32) WITHOUT
    /// running the LM-head projection. The hidden buffer is what
    /// `attachment.layer = "final"` shims read; steer-phase shims (#6b)
    /// mutate this slice and the caller re-projects via
    /// `cpu().finalize_logits(modified, n_tokens)` to get steered logits.
    ///
    /// Skipping the projection here matters when the caller is going to
    /// re-project anyway — for Qwen 3B the projection is the largest CPU
    /// cost in the per-token loop (~50–200ms), so doing it twice would
    /// halve decode throughput. On success the cache's write cursor is
    /// advanced by `n_tokens`.
    pub fn forward_full_gpu_with_cache_returning_hidden(
        &self,
        tokens: &[u32],
        cache: &mut crate::layers::gpu_kv_cache::GpuKvCache,
    ) -> Vec<f32> {
        self.forward_full_gpu_with_cache_inject_returning_hidden(tokens, cache, &[])
    }

    /// Fire-and-forget forward: runs all blocks, writes K/V into the
    /// cache, advances the cache cursor — but does NOT compute the final
    /// RMSNorm, does NOT copy to a staging buffer, and does NOT wait for
    /// the GPU. Callers that discard the hidden state entirely (e.g.
    /// `cache_append` building a shard) should use this to skip the
    /// `device.poll(Maintain::Wait)` round-trip, which empirical
    /// profiling shows costs ~300 ms per call on a 4080 Laptop /
    /// Vulkan independent of token count.
    ///
    /// **Ordering / safety**: wgpu's queue guarantees in-submission-order
    /// execution. The cache cursor is a CPU counter advanced after submit;
    /// any subsequent call that submits more work to the same cache will
    /// observe the previous K/V writes because the GPU executes them in
    /// order. If the GPU work fails (e.g. device lost), the failure
    /// surfaces on the next call that DOES wait — typical late-binding
    /// error reporting. For cache_append's HTTP handler this is
    /// acceptable: success means "queued and cursor advanced," not
    /// "GPU work completed."
    pub fn forward_full_gpu_with_cache_advance_only(
        &self,
        tokens: &[u32],
        cache: &mut crate::layers::gpu_kv_cache::GpuKvCache,
    ) {
        let n_tokens = tokens.len();
        assert!(n_tokens > 0, "must have at least one token");

        let n_layers = self.cpu.n_layers();
        assert_eq!(n_layers, cache.n_layers(), "cache layer count mismatch");

        let attn0 = self.cpu.blocks()[0].attention();
        assert_eq!(cache.n_kv_heads(), attn0.n_kv_heads(), "cache n_kv_heads mismatch");
        assert_eq!(cache.head_dim(), attn0.head_dim(), "cache head_dim mismatch");

        let start_pos = cache.seq_len();
        assert!(
            start_pos + n_tokens <= cache.max_seq_len(),
            "cache overflow: {} + {} > {}",
            start_pos, n_tokens, cache.max_seq_len(),
        );

        let t_start = std::time::Instant::now();

        // Embedding lookup (CPU) — same as full forward.
        let embed_data = self.cpu.embedding_data();
        let vocab_size = self.cpu.vocab_size();
        let mut hidden_init: Vec<f32> = Vec::with_capacity(n_tokens * self.embed_dim);
        for &tok in tokens {
            assert!((tok as usize) < vocab_size, "token {tok} out of vocab");
            let off = tok as usize * self.embed_dim;
            hidden_init.extend_from_slice(&embed_data[off..off + self.embed_dim]);
        }

        // Same Phase 0 fix as trace forward: forward_block_gpu_inner
        // reads hidden_buf as packed-f16 (C3). Passing raw f32 here
        // (the pre-2026-06-03 mistake) feeds the block misaligned bytes
        // and produces garbage K/V in the cache — which then poisons
        // every downstream attention read with NaN. Symptom: chat
        // against a cache_load'd shard outputs `!!!!` regardless of
        // polar/QJL/etc.
        let hidden_packed = GpuDevice::pack_f16(&hidden_init);
        let hidden_buf = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("forward_advance_only.hidden"),
            contents: bytemuck::cast_slice(&hidden_packed),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let intermediate = self.cpu.blocks()[0].ffn().as_any()
            .downcast_ref::<crate::layers::swiglu::SwiGLU>()
            .unwrap_or_else(|| panic!("forward_full_gpu_with_cache_advance_only requires SwiGLU FFN"))
            .intermediate_size();

        let attn_max_seq = start_pos + n_tokens;
        let t_alloc_start = std::time::Instant::now();
        let scratch = BlockScratch::allocate(
            &self.gpu, n_tokens, self.embed_dim,
            attn0.n_heads(), attn0.n_kv_heads(), attn0.head_dim(),
            intermediate, attn_max_seq,
        );
        let t_alloc = t_alloc_start.elapsed();

        let t_record_start = std::time::Instant::now();
        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("forward_advance_only.encoder"),
        });
        let skip_block_forward = std::env::var("CORTEX_SKIP_BLOCK_FORWARD").as_deref() == Ok("1");
        if !skip_block_forward {
            for i in 0..n_layers {
                let target = (cache.k_layer(i), cache.v_layer(i));
                self.forward_block_gpu_inner(
                    &mut encoder, i, &hidden_buf, n_tokens, start_pos, &scratch,
                    None, Some(target), None, None,
                );
            }
        }
        let t_record = t_record_start.elapsed();

        // No final RMSNorm, no copy_buffer_to_buffer, no readback.
        let t_submit_start = std::time::Instant::now();
        self.gpu.queue.submit(Some(encoder.finish()));
        let t_submit = t_submit_start.elapsed();

        // Force a sync so the per-call wall time actually reflects GPU work
        // (otherwise the next call's submit blocks behind ours). Gated:
        // CORTEX_SYNC_AFTER_ADVANCE=1 to enable. Default off (back-compat).
        let t_poll_start = std::time::Instant::now();
        if std::env::var("CORTEX_SYNC_AFTER_ADVANCE").as_deref() == Ok("1") {
            self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        }
        let t_poll = t_poll_start.elapsed();

        // Advance cache cursor (CPU). Subsequent submits will see ordered
        // K/V writes via the wgpu queue's in-order guarantee.
        cache.advance(n_tokens);
        let t_total = t_start.elapsed();
        tracing::info!(
            n_tokens, start_pos, attn_max_seq,
            alloc_us = t_alloc.as_micros() as u64,
            record_us = t_record.as_micros() as u64,
            submit_us = t_submit.as_micros() as u64,
            poll_us = t_poll.as_micros() as u64,
            total_us = t_total.as_micros() as u64,
            "fwd_advance_only stage timings",
        );
    }

    /// Inject-aware variant of `forward_full_gpu_with_cache_returning_hidden`.
    /// `inject_deltas` is either empty (no injection — same path as the
    /// no-inject variant) or has length `n_layers()` with `Some(buf)` for
    /// each layer that carries a [embed_dim] f32 hidden_delta. Each
    /// present buffer is broadcast-added into hidden BEFORE that block's
    /// attn_norm, every forward step (prefill + decode).
    ///
    /// The chat handler (cortex-cloud) computes injection deltas once
    /// per request (running each injection shim against the prompt's
    /// prefill hidden), uploads each summed per-layer delta via
    /// `upload_f32_to_storage`, then threads the same `inject_deltas`
    /// slice through every prefill / decode call for the request.
    pub fn forward_full_gpu_with_cache_inject_returning_hidden(
        &self,
        tokens: &[u32],
        cache: &mut crate::layers::gpu_kv_cache::GpuKvCache,
        inject_deltas: &[Option<wgpu::Buffer>],
    ) -> Vec<f32> {
        let n_tokens = tokens.len();
        assert!(n_tokens > 0, "must have at least one token");

        let n_layers = self.cpu.n_layers();
        assert_eq!(n_layers, cache.n_layers(), "cache layer count mismatch");

        let attn0 = self.cpu.blocks()[0].attention();
        assert_eq!(cache.n_kv_heads(), attn0.n_kv_heads(), "cache n_kv_heads mismatch");
        assert_eq!(cache.head_dim(), attn0.head_dim(), "cache head_dim mismatch");

        let start_pos = cache.seq_len();
        assert!(
            start_pos + n_tokens <= cache.max_seq_len(),
            "cache overflow: {} + {} > {}",
            start_pos, n_tokens, cache.max_seq_len(),
        );

        let t_start = std::time::Instant::now();

        // ---- Embedding lookup (CPU) ----
        let embed_data = self.cpu.embedding_data();
        let vocab_size = self.cpu.vocab_size();
        let mut hidden_init: Vec<f32> = Vec::with_capacity(n_tokens * self.embed_dim);
        for &tok in tokens {
            assert!((tok as usize) < vocab_size, "token {tok} out of vocab");
            let off = tok as usize * self.embed_dim;
            hidden_init.extend_from_slice(&embed_data[off..off + self.embed_dim]);
        }
        let t_embed = t_start.elapsed();

        // Phase C3: hidden_buf and normed_buf both packed f16 (2 per
        // u32). (Option E reverted hidden_buf to f32 for BitNet's
        // residual saturation; BitNet is gone post-2026-05-29 un-merge,
        // so the packed path is restored for the ~9% Qwen prefill win.)
        let packed_bytes = (hidden_init.len() * 2) as u64;
        let hidden_packed = GpuDevice::pack_f16(&hidden_init);
        let hidden_buf = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("forward_with_cache.hidden"),
            contents: bytemuck::cast_slice(&hidden_packed),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let normed_buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("forward_with_cache.normed"),
            size: packed_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.gpu.create_staging_buffer(packed_bytes);
        let t_io_alloc = t_start.elapsed() - t_embed;

        let intermediate = self.cpu.blocks()[0].ffn().as_any()
            .downcast_ref::<crate::layers::swiglu::SwiGLU>()
            .unwrap_or_else(|| panic!("forward_full_gpu_with_cache requires SwiGLU FFN"))
            .intermediate_size();

        // Scratch sized for ATTENTION over the full prefix (max_seq =
        // start_pos + n_tokens). Scores buffer must hold the score grid.
        let attn_max_seq = start_pos + n_tokens;
        let t_pre_scratch = t_start.elapsed();
        let scratch = BlockScratch::allocate(
            &self.gpu, n_tokens, self.embed_dim,
            attn0.n_heads(), attn0.n_kv_heads(), attn0.head_dim(),
            intermediate, attn_max_seq,
        );
        let t_scratch = t_start.elapsed() - t_pre_scratch;

        // ---- All blocks + final_norm in one encoder ----
        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("forward_with_cache.encoder"),
        });
        // Inject deltas: empty slice means no injection (same path as
        // before); non-empty must be exactly n_layers long with
        // Some(buf) for layers that carry a delta.
        if !inject_deltas.is_empty() {
            assert_eq!(
                inject_deltas.len(), n_layers,
                "inject_deltas must be empty or have length n_layers ({})", n_layers,
            );
        }
        // Timestamp markers: one per block boundary + one final.
        // n_markers = n_layers + 2 (start, after each block, after final_norm).
        let n_markers: u32 = (n_layers as u32) + 2;
        // Pass-level timestamps occupy QuerySet indices >= n_markers
        // (per-block markers live at 0..n_markers). Each pass uses 2
        // indices (begin + end), so an upper bound for the pass region
        // is n_markers + 2 * (5 passes/block × n_layers + 1 for final_norm).
        let pass_region_upper: u32 = n_markers + 2 * (5 * (n_layers as u32) + 1);
        let timer_active = self.timer.as_ref().map(|t| pass_region_upper <= t.capacity).unwrap_or(false);

        // Activate per-pass timer for the prefill, starting at index n_markers
        // so we don't collide with the per-block markers.
        if timer_active {
            *self.pass_timer.lock().unwrap() = Some(PassTimerState {
                next_idx: n_markers,
                labels: Vec::with_capacity(pass_region_upper as usize / 2),
            });
        }

        let t_pre_record = t_start.elapsed();
        if let (true, Some(t)) = (timer_active, self.timer.as_ref()) {
            encoder.write_timestamp(&t.query_set, 0);
        }
        // Phase B/C debug: per-block hidden-state finite check. Allocates
        // a packed-sized capture buffer per layer, captures hidden_buf
        // after each block, reads back after submit, counts Inf/NaN per
        // layer. Gated on CORTEX_DEBUG_HIDDEN_FINITE.
        let debug_finite = std::env::var("CORTEX_DEBUG_HIDDEN_FINITE").is_ok();
        // C3: hidden_buf is PACKED f16 (2 per u32), half the f32 size.
        // (The stale "Option E: f32" comment misled the trace into
        // reading raw bytes as f32 floats — every other 2-byte slice
        // got reinterpreted as a high-half of f32, producing fake NaNs.)
        let hidden_bytes = (n_tokens * self.embed_dim * 2) as u64;
        let debug_captures: Vec<wgpu::Buffer> = if debug_finite {
            (0..n_layers).map(|i| self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("debug_finite.capture.{}", i)),
                size: hidden_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })).collect()
        } else { Vec::new() };
        let debug_stagings: Vec<wgpu::Buffer> = if debug_finite {
            (0..n_layers).map(|_| self.gpu.create_staging_buffer(hidden_bytes)).collect()
        } else { Vec::new() };
        for i in 0..n_layers {
            let target = (cache.k_layer(i), cache.v_layer(i));
            let inject = inject_deltas.get(i).and_then(|opt| opt.as_ref());
            let capture = if debug_finite { Some(&debug_captures[i]) } else { None };
            self.forward_block_gpu_inner(
                &mut encoder, i, &hidden_buf, n_tokens, start_pos, &scratch,
                None, Some(target), capture, inject,
            );
            if debug_finite {
                encoder.copy_buffer_to_buffer(&debug_captures[i], 0, &debug_stagings[i], 0, hidden_bytes);
            }
            if let (true, Some(t)) = (timer_active, self.timer.as_ref()) {
                encoder.write_timestamp(&t.query_set, (i as u32) + 1);
            }
        }
        // Final norm — C3: hidden_buf and normed_buf both packed f16.
        {
            let mut pass = self.begin_timed_pass(&mut encoder, "final_norm");
            self.dispatch_rmsnorm_packed_to_packed_in_pass(
                &mut pass, hidden_buf.as_entire_binding(), self.final_norm_weight_buf.binding(), normed_buf.as_entire_binding(),
                self.embed_dim, n_tokens, self.final_norm_eps,
            );
        }
        if let (true, Some(t)) = (timer_active, self.timer.as_ref()) {
            encoder.write_timestamp(&t.query_set, (n_layers as u32) + 1);
        }
        encoder.copy_buffer_to_buffer(&normed_buf, 0, &staging, 0, packed_bytes);
        // Capture the actual range used by the pass-level timer
        // (begin_timed_pass advanced state.next_idx).
        let pass_final_idx: u32 = self.pass_timer.lock().unwrap()
            .as_ref().map(|s| s.next_idx).unwrap_or(n_markers);
        // Resolve all queries into the resolve buffer, then copy to the
        // mappable readback buffer — both in the same encoder so it's
        // one submit.
        if let (true, Some(t)) = (timer_active, self.timer.as_ref()) {
            // Combined range: per-block markers 0..n_markers plus
            // pass-level timestamps n_markers..pass_final_idx.
            encoder.resolve_query_set(&t.query_set, 0..pass_final_idx, &t.resolve_buf, 0);
            let resolve_bytes = (pass_final_idx as u64) * 8;
            encoder.copy_buffer_to_buffer(&t.resolve_buf, 0, &t.readback_buf, 0, resolve_bytes);
        }
        let t_record = t_start.elapsed() - t_pre_record;

        let t_pre_submit = t_start.elapsed();
        self.gpu.queue.submit(Some(encoder.finish()));
        let t_submit = t_start.elapsed() - t_pre_submit;

        let t_pre_readback = t_start.elapsed();
        // Phase B: normed_buf is packed f16. Unpack to Vec<f32> for the
        // CPU finalize_logits (LM head matmul) which still consumes f32.
        let normed = read_back_buffer_f16_unpack(&self.gpu, &staging, packed_bytes as usize);
        let t_readback = t_start.elapsed() - t_pre_readback;

        // Per-block hidden-state finite check (CORTEX_DEBUG_HIDDEN_FINITE).
        // Reads back each layer's post-block hidden, unpacks f16, counts
        // non-finite + tracks the running absmax. Locates the first layer
        // where pack2x16float saturates to Inf (the gut-feel BitNet
        // failure mode).
        if debug_finite {
            tracing::info!(
                n_tokens, embed_dim = self.embed_dim, n_layers,
                "CORTEX_DEBUG_HIDDEN_FINITE: per-block hidden-state scan",
            );
            for i in 0..n_layers {
                // C3: hidden_buf is packed f16. Unpack via the existing
                // helper so we count NaN/Inf on the actual f32 values
                // the next block would read.
                let unpacked = read_back_buffer_f16_unpack(
                    &self.gpu, &debug_stagings[i], hidden_bytes as usize,
                );
                let mut n_inf: usize = 0;
                let mut n_nan: usize = 0;
                let mut max_abs: f32 = 0.0;
                let mut first_bad_idx: Option<usize> = None;
                for (idx, &v) in unpacked.iter().enumerate() {
                    if v.is_nan() {
                        n_nan += 1;
                        if first_bad_idx.is_none() { first_bad_idx = Some(idx); }
                    } else if v.is_infinite() {
                        n_inf += 1;
                        if first_bad_idx.is_none() { first_bad_idx = Some(idx); }
                    } else {
                        let a = v.abs();
                        if a > max_abs { max_abs = a; }
                    }
                }
                tracing::info!(
                    layer = i, n_inf, n_nan, max_abs,
                    first_bad_idx = ?first_bad_idx,
                    total = unpacked.len(),
                    "hidden state",
                );
                if n_inf + n_nan > 0 {
                    tracing::warn!(layer = i,
                        "FIRST LAYER with non-finite values — saturation point");
                }
            }
        }

        // Read back timestamp markers (if active) and log the per-block
        // GPU waterfall + per-pass cumulative times.
        if let (true, Some(t)) = (timer_active, self.timer.as_ref()) {
            let resolve_bytes = (pass_final_idx as usize) * 8;
            let slice = t.readback_buf.slice(0..(resolve_bytes as u64));
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
            self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
            rx.recv().expect("timer readback failed").expect("timer map failed");
            let data = slice.get_mapped_range();
            let ticks: Vec<u64> = data[..resolve_bytes].chunks_exact(8)
                .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            drop(data);
            t.readback_buf.unmap();

            // Per-block timing (from the n_markers boundary markers).
            let mut per_block_us: Vec<u64> = Vec::with_capacity(n_layers);
            for i in 0..n_layers {
                let dt_ticks = ticks[i + 1].saturating_sub(ticks[i]);
                let dt_ns = (dt_ticks as f32) * t.period_ns;
                per_block_us.push((dt_ns / 1000.0) as u64);
            }
            let final_norm_us = {
                let dt_ticks = ticks[n_layers + 1].saturating_sub(ticks[n_layers]);
                let dt_ns = (dt_ticks as f32) * t.period_ns;
                (dt_ns / 1000.0) as u64
            };
            let total_blocks_us: u64 = per_block_us.iter().sum();
            let min_us = per_block_us.iter().min().copied().unwrap_or(0);
            let max_us = per_block_us.iter().max().copied().unwrap_or(0);
            let avg_us = if n_layers > 0 { total_blocks_us / (n_layers as u64) } else { 0 };

            tracing::info!(
                n_tokens, start_pos,
                total_blocks_us, avg_us, min_us, max_us, final_norm_us,
                "fwd_cache GPU waterfall (per-block timestamps)",
            );
            tracing::debug!(?per_block_us, "per-block GPU times");

            // Per-pass timing: pull labels back out of the pass timer,
            // each label corresponds to one (begin, end) pair starting
            // at index n_markers. Aggregate cumulative µs per label.
            let pass_state = self.pass_timer.lock().unwrap().take();
            if let Some(state) = pass_state {
                let mut by_label: std::collections::BTreeMap<&'static str, (u64, u64)>
                    = std::collections::BTreeMap::new();
                for (pass_i, label) in state.labels.iter().enumerate() {
                    let begin_idx = (n_markers as usize) + pass_i * 2;
                    let end_idx = begin_idx + 1;
                    if end_idx >= ticks.len() { break; }
                    let dt_ticks = ticks[end_idx].saturating_sub(ticks[begin_idx]);
                    let dt_ns = (dt_ticks as f32) * t.period_ns;
                    let entry = by_label.entry(*label).or_insert((0, 0));
                    entry.0 += (dt_ns / 1000.0) as u64;
                    entry.1 += 1;
                }
                let pass_summary: Vec<String> = by_label.iter()
                    .map(|(lbl, (cum_us, cnt))| {
                        let avg = if *cnt > 0 { cum_us / cnt } else { 0 };
                        format!("{lbl}={cum_us}us ({cnt}x, avg={avg}us)")
                    })
                    .collect();
                tracing::info!(
                    pass_count = state.labels.len() as u64,
                    summary = pass_summary.join(" "),
                    "fwd_cache GPU per-pass cumulative",
                );
            }
        } else if timer_active {
            // Timer was active but timer somehow gone — clean up state.
            *self.pass_timer.lock().unwrap() = None;
        }

        // Successful forward — bump the cache's write cursor. The caller
        // either runs `cpu().finalize_logits(&normed, n_tokens)` for plain
        // decode, or applies steer hidden_delta to the last token's slice
        // first and then re-projects.
        cache.advance(n_tokens);
        let t_total = t_start.elapsed();
        tracing::info!(
            n_tokens, start_pos, attn_max_seq,
            embed_us = t_embed.as_micros() as u64,
            io_alloc_us = t_io_alloc.as_micros() as u64,
            scratch_us = t_scratch.as_micros() as u64,
            record_us = t_record.as_micros() as u64,
            submit_us = t_submit.as_micros() as u64,
            readback_us = t_readback.as_micros() as u64,
            total_us = t_total.as_micros() as u64,
            "fwd_cache stage timings",
        );
        normed
    }

    /// GPU greedy fast-path: runs the full forward like
    /// `forward_full_gpu_with_cache_inject_returning_hidden`, then
    /// projects ONLY the last token through the LM head and runs
    /// argmax — all on GPU, with a 4-byte readback instead of the
    /// usual ~1.2 MB logits readback.
    ///
    /// Returns `None` if the LM-head isn't GPU-resident (engine init
    /// couldn't materialize a packed-f16 buffer for the projection)
    /// — callers should fall through to the CPU sampler path.
    /// Otherwise returns `Some(token_id)`.
    ///
    /// Only valid for greedy sampling (`temperature <= 0` with
    /// `top_k <= 1` and no repetition penalty). Callers must not use
    /// this when stochastic sampling is requested.
    pub fn forward_full_gpu_with_cache_inject_argmax_greedy(
        &self,
        tokens: &[u32],
        cache: &mut crate::layers::gpu_kv_cache::GpuKvCache,
        inject_deltas: &[Option<wgpu::Buffer>],
    ) -> Option<u32> {
        let lm_head = self.lm_head.as_ref()?;

        let n_tokens = tokens.len();
        assert!(n_tokens > 0, "must have at least one token");

        let n_layers = self.cpu.n_layers();
        assert_eq!(n_layers, cache.n_layers(), "cache layer count mismatch");

        let attn0 = self.cpu.blocks()[0].attention();
        assert_eq!(cache.n_kv_heads(), attn0.n_kv_heads(), "cache n_kv_heads mismatch");
        assert_eq!(cache.head_dim(), attn0.head_dim(), "cache head_dim mismatch");

        let start_pos = cache.seq_len();
        assert!(
            start_pos + n_tokens <= cache.max_seq_len(),
            "cache overflow: {} + {} > {}",
            start_pos, n_tokens, cache.max_seq_len(),
        );

        // Embedding (CPU, same as the returning_hidden path).
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
        let hidden_buf = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("forward_with_cache_argmax.hidden"),
            contents: bytemuck::cast_slice(&hidden_packed),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let normed_buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("forward_with_cache_argmax.normed"),
            size: packed_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let intermediate = self.cpu.blocks()[0].ffn().as_any()
            .downcast_ref::<crate::layers::swiglu::SwiGLU>()
            .unwrap_or_else(|| panic!("forward_with_cache_argmax requires SwiGLU FFN"))
            .intermediate_size();

        let attn_max_seq = start_pos + n_tokens;
        let scratch = BlockScratch::allocate(
            &self.gpu, n_tokens, self.embed_dim,
            attn0.n_heads(), attn0.n_kv_heads(), attn0.head_dim(),
            intermediate, attn_max_seq,
        );

        if !inject_deltas.is_empty() {
            assert_eq!(
                inject_deltas.len(), n_layers,
                "inject_deltas must be empty or have length n_layers ({})", n_layers,
            );
        }

        // Last-token slice buffer (packed f16, [embed_dim/2] u32s).
        let last_token_bytes = (self.embed_dim * 2) as u64;
        let last_token_buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("forward_with_cache_argmax.last_token_packed"),
            size: last_token_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Vocab-sized f32 logits buffer.
        let logits_bytes = (lm_head.vocab_size * std::mem::size_of::<f32>()) as u64;
        let logits_buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("forward_with_cache_argmax.logits"),
            size: logits_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // 4-byte argmax output + matching staging.
        let token_id_buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("forward_with_cache_argmax.token_id"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.gpu.create_staging_buffer(4);

        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("forward_with_cache_argmax.encoder"),
        });

        // All N blocks. Same call shape as the returning_hidden path
        // minus the per-block hidden-state finite capture.
        for i in 0..n_layers {
            let target = (cache.k_layer(i), cache.v_layer(i));
            let inject = inject_deltas.get(i).and_then(|opt| opt.as_ref());
            self.forward_block_gpu_inner(
                &mut encoder, i, &hidden_buf, n_tokens, start_pos, &scratch,
                None, Some(target), None, inject,
            );
        }

        // Final norm: packed → packed.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("forward_with_cache_argmax.final_norm.pass"),
                timestamp_writes: None,
            });
            self.dispatch_rmsnorm_packed_to_packed_in_pass(
                &mut pass, hidden_buf.as_entire_binding(), self.final_norm_weight_buf.binding(), normed_buf.as_entire_binding(),
                self.embed_dim, n_tokens, self.final_norm_eps,
            );
        }

        // Copy the last token's packed row out of normed_buf.
        // normed_buf layout: [n_tokens, embed_dim/2] u32, so the last
        // token starts at byte offset (n_tokens-1) * embed_dim * 2.
        let last_token_offset = ((n_tokens - 1) * self.embed_dim * 2) as u64;
        encoder.copy_buffer_to_buffer(
            &normed_buf, last_token_offset,
            &last_token_buf, 0,
            last_token_bytes,
        );

        // LM-head matmul: 1 × vocab_size logits.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("forward_with_cache_argmax.lm_head.pass"),
                timestamp_writes: None,
            });
            self.dispatch_lm_head_matmul_pin_in_pass(
                &mut pass, lm_head, &last_token_buf, &logits_buf,
            );
        }

        // Argmax → 4-byte token id.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("forward_with_cache_argmax.argmax.pass"),
                timestamp_writes: None,
            });
            self.dispatch_argmax_vocab_in_pass(
                &mut pass, &logits_buf, &token_id_buf, lm_head.vocab_size,
            );
        }
        encoder.copy_buffer_to_buffer(&token_id_buf, 0, &staging, 0, 4);

        self.gpu.queue.submit(Some(encoder.finish()));

        // 4-byte readback (vs ~1.2 MB on the legacy path for Qwen 3B).
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        rx.recv().expect("token-id readback failed").expect("token-id map failed");
        let data = slice.get_mapped_range();
        let token_id = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        drop(data);
        staging.unmap();

        cache.advance(n_tokens);
        Some(token_id)
    }

    /// **Phase 1 close — full forward on GPU.** Embedding lookup runs CPU
    /// (cheap; saves an embedding-gather shader for now), then ALL N blocks
    /// chain into one command encoder against resident weights, then
    /// final_norm runs on GPU in the same encoder. One submit. Output
    /// projection still runs CPU (vocab-sized matmul; deferred to a later
    /// phase that wires GpuFloatLinear into the projection path).
    ///
    /// Same constraints as `forward_block_gpu`: every block must be SwiGLU
    /// + SiLU with no biases / sub-norms / non-1.0 residual scales, every
    /// matvec layer must be `GpuFloatLinear`. Asserts on violations.
    pub fn forward_full_gpu(&self, tokens: &[u32], start_pos: usize) -> Vec<f32> {
        let n_tokens = tokens.len();
        assert!(n_tokens > 0, "must have at least one token");

        // ---- 1. Embedding lookup (CPU) ----
        let embed_data = self.cpu.embedding_data();
        let vocab_size = self.cpu.vocab_size();
        let mut hidden_init: Vec<f32> = Vec::with_capacity(n_tokens * self.embed_dim);
        for &tok in tokens {
            assert!((tok as usize) < vocab_size, "token {tok} out of vocab");
            let off = tok as usize * self.embed_dim;
            hidden_init.extend_from_slice(&embed_data[off..off + self.embed_dim]);
        }

        // ---- Allocate buffers — C3 restored: hidden + normed packed f16 ----
        let packed_bytes = (hidden_init.len() * 2) as u64;
        let hidden_packed = GpuDevice::pack_f16(&hidden_init);
        let hidden_buf = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("forward_full.hidden"),
            contents: bytemuck::cast_slice(&hidden_packed),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let normed_buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("forward_full.normed"),
            size: packed_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.gpu.create_staging_buffer(packed_bytes);

        // Per-block sizing (consistent across blocks for non-MoE models).
        let attn0 = self.cpu.blocks()[0].attention();
        let intermediate = self.cpu.blocks()[0].ffn().as_any()
            .downcast_ref::<crate::layers::swiglu::SwiGLU>()
            .unwrap_or_else(|| panic!("forward_full_gpu requires SwiGLU FFN"))
            .intermediate_size();
        let scratch = BlockScratch::allocate(
            &self.gpu, n_tokens, self.embed_dim,
            attn0.n_heads(), attn0.n_kv_heads(), attn0.head_dim(),
            intermediate, n_tokens,
        );
        // Single encoder for all blocks + final norm = one submit for the
        // whole forward pass. Earlier per-block submit was a workaround for
        // what turned out to be a separate cross-device bug (#16).
        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("forward_full.encoder"),
        });
        for i in 0..self.cpu.n_layers() {
            self.forward_block_gpu(&mut encoder, i, &hidden_buf, n_tokens, start_pos, &scratch);
        }
        // Final norm (Phase B): hidden_buf and normed_buf are both packed.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("forward_full.final_norm.pass"),
                timestamp_writes: None,
            });
            // C3: hidden packed, normed packed.
            self.dispatch_rmsnorm_packed_to_packed_in_pass(
                &mut pass, hidden_buf.as_entire_binding(), self.final_norm_weight_buf.binding(), normed_buf.as_entire_binding(),
                self.embed_dim, n_tokens, self.final_norm_eps,
            );
        }
        encoder.copy_buffer_to_buffer(&normed_buf, 0, &staging, 0, packed_bytes);
        self.gpu.queue.submit(Some(encoder.finish()));

        // ---- Read back final-normed hidden state (Phase B: f16 unpack) ----
        let normed = read_back_buffer_f16_unpack(&self.gpu, &staging, packed_bytes as usize);

        // ---- 4. Output projection (CPU; vocab matmul deferred) ----
        self.cpu.finalize_logits(&normed, n_tokens)
    }

    /// Run one transformer block fully on GPU. Reads `hidden_buf` (shape
    /// `[n_tokens, embed_dim]`), modifies it in place with the post-block
    /// hidden state. All intermediate scratch buffers must be supplied by
    /// the caller — `forward_blocks_gpu` allocates them once and reuses
    /// across all blocks.
    ///
    /// Caveats (will be lifted in later phases):
    /// - Q/K/V biases (Qwen2) ignored: panics if any are set.
    /// - Attention sub-norm (BitNet) ignored: panics if set.
    /// - FFN sub-norm (BitNet) ignored: panics if set.
    /// - Residual scales must be 1.0.
    /// - FFN must be a `SwiGLU` with `SiLU` activation.
    /// - Matvec layers must be `GpuFloatLinear` (ternary fused not yet built).
    /// - For prefill mode (no historical KV cache): `start_pos = 0`. Cached
    ///   decoding lands in #9 along with the resident KV cache.
    pub fn forward_block_gpu(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        block_idx: usize,
        hidden_buf: &wgpu::Buffer,
        n_tokens: usize,
        start_pos: usize,
        scratch: &BlockScratch,
    ) {
        // Phase A: KV cache is now packed f16, and the attention shaders
        // read packed-f16 K/V. The pre-Phase-A "scratch K/V" path is
        // gone (would require parallel f32 attention shaders to keep);
        // forward_block_gpu now allocates a tiny single-block scratch
        // cache and routes through the cached path. This is the only
        // caller — production always passes its own kv_cache_target via
        // forward_block_gpu_inner.
        let attn = self.cpu.blocks()[block_idx].attention();
        let tmp_cache = crate::layers::gpu_kv_cache::GpuKvCache::new(
            Arc::clone(&self.gpu),
            1, attn.n_kv_heads(), attn.head_dim(),
            (start_pos + n_tokens).max(1),
        );
        let target = (tmp_cache.k_layer(0), tmp_cache.v_layer(0));
        self.forward_block_gpu_inner(
            encoder, block_idx, hidden_buf, n_tokens, start_pos, scratch,
            None, Some(target), None, None,
        );
    }

    /// Same as `forward_block_gpu` but with optional pre-softmax score
    /// capture for retrieval / traced forward use, plus optional KV cache
    /// targeting for cached forward (decode + cached prefill), plus
    /// optional post-block hidden state capture for shim hooks.
    ///
    /// When `kv_cache_target` is Some, this block's projected K/V get
    /// written into the supplied cache buffers at offset `start_pos`, and
    /// the attention dispatch reads K/V back from the cache (with
    /// max_seq = start_pos + n_tokens) so the new tokens attend over the
    /// full prefix. When None, K/V live only in scratch and attention reads
    /// scratch (the prefill-only path).
    ///
    /// When `post_block_hidden_capture` is Some, the hidden state at the
    /// END of this block (after the FFN residual add) is copied into the
    /// supplied buffer. This is the natural attachment point for
    /// "entrance:N+1" shim hooks (the input to block N+1) and, when
    /// captured at the last block, the input to the final RMSNorm — i.e.
    /// what gate / steer shims read. Buffer must be sized
    /// `n_tokens * embed_dim * 4` bytes.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_block_gpu_inner(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        block_idx: usize,
        hidden_buf: &wgpu::Buffer,
        n_tokens: usize,
        start_pos: usize,
        scratch: &BlockScratch,
        pre_softmax_capture: Option<&wgpu::Buffer>,
        kv_cache_target: Option<(&::vram_heap::VramAllocation, &::vram_heap::VramAllocation)>,
        post_block_hidden_capture: Option<&wgpu::Buffer>,
        pre_block_hidden_inject: Option<&wgpu::Buffer>,
    ) {
        let block = &self.cpu.blocks()[block_idx];
        let block_gpu = &self.blocks_gpu[block_idx];
        let attn = block.attention();

        // Per-block pass merging (#pass-2): collapse the ~17 sequential
        // `begin_compute_pass` calls per block into 3 grouped passes
        // separated only by encoder-level operations (copy_buffer_to_buffer
        // for capture hooks) and the attention triple (which has its own
        // internal pass splits at score↔softmax↔value hazards). Saves
        // ~14 × ~0.8ms = ~11ms per block × 30 blocks = ~330ms per forward
        // of Vulkan pipeline-barrier overhead.

        // BitNet b1.58 sub-norms (#bn-11): when present on the CPU layer,
        // their resident weight buffers are populated on the GpuBlock at
        // construction. The dispatch sites below insert dispatch_rmsnorm
        // calls before o_proj / down_proj respectively.
        assert!((block.attn_residual_scale() - 1.0).abs() < f32::EPSILON,
            "forward_block_gpu requires attn_residual_scale == 1.0");
        assert!((block.ffn_residual_scale() - 1.0).abs() < f32::EPSILON,
            "forward_block_gpu requires ffn_residual_scale == 1.0");

        let swiglu = block.ffn().as_any()
            .downcast_ref::<crate::layers::swiglu::SwiGLU>()
            .unwrap_or_else(|| panic!("forward_block_gpu requires SwiGLU FFN"));
        // Activation routing: SiLU (LLaMA / Qwen) or ReLU² (BitNet b1.58).

        let n_heads = attn.n_heads();
        let n_kv_heads = attn.n_kv_heads();
        let head_dim = attn.head_dim();
        let embed_dim = self.embed_dim;
        let intermediate = swiglu.intermediate_size();
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        assert!(start_pos + n_tokens <= self.rope_max_seq,
            "start_pos + n_tokens ({}) exceeds rope_max_seq ({})",
            start_pos + n_tokens, self.rope_max_seq);

        // ----- PASS 1: inject (optional) + attn_norm + Q/K/V projs (+biases)
        // + RoPE + kv_write. All dispatches read/write block-local scratch
        // buffers with the implicit ordering wgpu enforces between dispatches
        // in a single pass on most drivers.
        {
            let mut pass = self.begin_timed_pass(encoder, "block.pass1");

            // Injection-phase hook (#6c). Broadcast-add a [embed_dim] delta
            // into every token's hidden BEFORE attn_norm.
            // C3: hidden_buf is packed — use packed broadcast.
            if let Some(delta_buf) = pre_block_hidden_inject {
                self.dispatch_add_broadcast_in_pass(
                    &mut pass, hidden_buf.as_entire_binding(), delta_buf, embed_dim, n_tokens,
                );
            }

            // 1. attn_norm: hidden packed → normed packed.
            self.dispatch_rmsnorm_packed_to_packed_in_pass(
                &mut pass, hidden_buf.as_entire_binding(), block_gpu.attn_norm_weight_buf.binding(), scratch.normed.binding(),
                embed_dim, n_tokens, block_gpu.attn_norm_eps,
            );

            // 2-4. Q, K, V projections (+ optional Qwen-style biases).
            // C3: scratch.q/k/v packed → packed_io dispatcher; bias add
            // and RoPE use their packed variants.
            self.dispatch_linear_batch_packed_io_in_pass(&mut pass, attn.q_proj(), scratch.normed.binding(), scratch.q.binding(), n_tokens);
            if let Some(buf) = block_gpu.q_bias_buf.as_ref() {
                self.dispatch_bias_add_packed_in_pass(&mut pass, scratch.q.binding(), buf.binding(), q_dim, n_tokens);
            }
            self.dispatch_linear_batch_packed_io_in_pass(&mut pass, attn.k_proj(), scratch.normed.binding(), scratch.k.binding(), n_tokens);
            if let Some(buf) = block_gpu.k_bias_buf.as_ref() {
                self.dispatch_bias_add_packed_in_pass(&mut pass, scratch.k.binding(), buf.binding(), kv_dim, n_tokens);
            }
            self.dispatch_linear_batch_packed_io_in_pass(&mut pass, attn.v_proj(), scratch.normed.binding(), scratch.v.binding(), n_tokens);
            if let Some(buf) = block_gpu.v_bias_buf.as_ref() {
                self.dispatch_bias_add_packed_in_pass(&mut pass, scratch.v.binding(), buf.binding(), kv_dim, n_tokens);
            }

            // 5. RoPE on Q and K (in-place packed).
            self.dispatch_rope_packed_in_pass(
                &mut pass, scratch.q.binding(), self.rope_cos_buf.binding(), self.rope_sin_buf.binding(),
                n_heads, head_dim, start_pos, n_tokens,
            );
            self.dispatch_rope_packed_in_pass(
                &mut pass, scratch.k.binding(), self.rope_cos_buf.binding(), self.rope_sin_buf.binding(),
                n_kv_heads, head_dim, start_pos, n_tokens,
            );

            // 5.5 (cached path) Write K/V into the layer's resident cache.
            // Phase F: kv_cache_target carries &VramAllocation. Build a
            // fresh binding per dispatch (BindingResource is move-only,
            // so we re-emit .binding() at each consumption site).
            if let Some((k_cache, v_cache)) = kv_cache_target {
                self.dispatch_kv_write_in_pass(
                    &mut pass, scratch.k.binding(), scratch.v.binding(),
                    k_cache.binding(), v_cache.binding(),
                    kv_dim, start_pos, n_tokens,
                );
            }
            // pass1 ends here (drop)
        }

        // 6. Attention math: Q · K^T, softmax, weighted V.
        // Kept as its own dispatcher because score↔softmax↔value share
        // scratch.scores with conflicting access — needs explicit pass
        // splits inside `dispatch_attention_inner`.
        let (k_for_attn, v_for_attn, attn_max_seq) = match kv_cache_target {
            Some((kc, vc)) => (kc.binding(), vc.binding(), start_pos + n_tokens),
            None => (scratch.k.binding(), scratch.v.binding(), n_tokens),
        };
        // Perf-bisect: see CORTEX_SKIP_SCORE / SOFTMAX / VALUE inside
        // `dispatch_attention_inner` — per-stage skip flags that bypass
        // individual attention dispatches. (The previous CORTEX_SKIP_ATTENTION
        // gate that called `encoder.clear_buffer` here panicked at runtime
        // because `scratch.attn_out` lacks the COPY_DST usage; the per-stage
        // flags are the correct path.)
        self.dispatch_attention_inner(
            encoder,
            scratch.q.binding(), k_for_attn, v_for_attn,
            &scratch.scores, scratch.attn_out.binding(),
            n_heads, n_kv_heads, head_dim,
            start_pos, attn_max_seq, n_tokens,
            pre_softmax_capture,
        );

        // ----- PASS 2: o_sub_norm (BitNet) + o_proj + residual + ffn_norm
        // + gate/up + silu_mul + ffn_sub_norm (BitNet) + down + residual.
        // All purely intra-block; ends at the optional post-block capture
        // (encoder copy outside any pass).
        {
            let mut pass = self.begin_timed_pass(encoder, "block.pass2");

            // 7. O projection — C3 restored: packed attn_out → packed projected.
            // (BitNet sub-norm branches removed with the 2026-05-29 un-merge;
            // float models never had o_sub_norm_weight_buf set.)
            self.dispatch_linear_batch_packed_io_in_pass(&mut pass, attn.o_proj(), scratch.attn_out.binding(), scratch.projected.binding(), n_tokens);

            // 8. Residual: hidden (packed) += projected (packed) — C3.
            self.dispatch_add_packed_in_pass(&mut pass, hidden_buf.as_entire_binding(), scratch.projected.binding(), embed_dim, n_tokens);

            // 9. ffn_norm: hidden packed → normed packed — C3.
            self.dispatch_rmsnorm_packed_to_packed_in_pass(
                &mut pass, hidden_buf.as_entire_binding(), block_gpu.ffn_norm_weight_buf.binding(), scratch.normed.binding(),
                embed_dim, n_tokens, block_gpu.ffn_norm_eps,
            );

            // 10-11. Gate / Up projections.
            // C2: scratch.gate/up are packed; fused dispatcher writes
            // packed via matmul_gate_up_shared (adjacent-pair output).
            // Non-fused fallback routes through the packed-I/O variant
            // (matmul_pin_pout for float decode, quantize+ternary_pout
            // for BitNet).
            let fused_ok = n_tokens >= 16 && self.dispatch_gate_up_fused_in_pass(
                &mut pass, swiglu.gate_proj(), swiglu.up_proj(),
                scratch.normed.binding(), scratch.gate.binding(), scratch.up.binding(), n_tokens,
            );
            if !fused_ok {
                self.dispatch_linear_batch_packed_io_in_pass(&mut pass, swiglu.gate_proj(), scratch.normed.binding(), scratch.gate.binding(), n_tokens);
                self.dispatch_linear_batch_packed_io_in_pass(&mut pass, swiglu.up_proj(),   scratch.normed.binding(), scratch.up.binding(),   n_tokens);
            }

            // 12. silu(gate) * up — packed in C2.
            self.dispatch_gate_mul_packed_in_pass(&mut pass, scratch.gate.binding(), scratch.up.binding(), scratch.activated.binding(), intermediate, n_tokens, swiglu.activation());

            // 13. Down projection — C3: packed input, packed output.
            // (BitNet ffn_sub_norm branch removed with the 2026-05-29 un-merge;
            // float models never had ffn_sub_norm_weight_buf set.)
            self.dispatch_linear_batch_packed_io_in_pass(&mut pass, swiglu.down_proj(), scratch.activated.binding(), scratch.projected.binding(), n_tokens);

            // 14. Residual: hidden (packed) += projected (packed) — C3.
            self.dispatch_add_packed_in_pass(&mut pass, hidden_buf.as_entire_binding(), scratch.projected.binding(), embed_dim, n_tokens);
            // pass2 ends here (drop)
        }

        // 15. (optional) capture post-block hidden state — shim hook point.
        // C3: hidden_buf is packed f16; capture is half size.
        if let Some(capture_buf) = post_block_hidden_capture {
            let bytes = (n_tokens * embed_dim * 2) as u64;
            encoder.copy_buffer_to_buffer(hidden_buf, 0, capture_buf, 0, bytes);
        }
    }

}
