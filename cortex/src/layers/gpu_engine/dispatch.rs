//! GPU dispatch helpers (split from gpu_engine.rs, Phase N).
//!
//! Shader param structs and the `dispatch_*` encoder/pass helpers the
//! forward paths compose. Private helpers are `pub(super)` — visible
//! only within the `gpu_engine` module tree.

use super::*;

/// Params struct for the rmsnorm_batch shader. Layout must match
/// `compute/shaders/rmsnorm_batch.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct RmsNormBatchParams {
    n: u32,
    eps: f32,
    n_tokens: u32,
    _pad: u32,
}

/// Params struct for the rope_batch shader. Layout must match
/// `compute/shaders/rope_batch.wgsl` exactly — eight u32s.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct RopeBatchParams {
    n_heads: u32,
    head_dim: u32,
    start_pos: u32,
    half_dim: u32,
    n_tokens: u32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
}

/// Params struct for the attn_score_batch shader. Twelve u32s; the trailing
/// padding entries are required for std140-ish uniform alignment.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct AttnScoreBatchParams {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    start_pos: u32,
    max_seq: u32,
    heads_per_kv: u32,
    kv_dim: u32,
    scale: f32,
    n_tokens: u32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
}

/// Params struct for the softmax_batch shader. Four u32s.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct SoftmaxBatchParams {
    pub(super) n_heads: u32,
    pub(super) max_seq: u32,
    pub(super) start_pos: u32,
    pub(super) n_tokens: u32,
}

/// Params struct for the attn_value_batch shader. Eight u32s.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct AttnValueBatchParams {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    start_pos: u32,
    max_seq: u32,
    heads_per_kv: u32,
    kv_dim: u32,
    n_tokens: u32,
}

/// Params struct for the silu_mul_batch shader. Two u32s.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct SiluMulBatchParams {
    n: u32,
    n_tokens: u32,
}

/// Params struct for the kv_write_batch shader. Four u32s.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct KvWriteBatchParams {
    kv_dim: u32,
    start_pos: u32,
    n_tokens: u32,
    _pad: u32,
}

/// Params struct for the add_inplace_batch shader. Two u32s.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct AddInplaceBatchParams {
    n: u32,
    n_tokens: u32,
}

/// Params struct for the argmax_vocab shader. Two u32s.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ArgmaxVocabParams {
    n: u32,
    _pad: u32,
}

impl GpuEngine {
    /// Dispatch RMSNorm into `out_buf` from `in_buf`, using `weight_buf` for
    /// the per-feature scale. Both buffers are `[n_tokens, n]` flat. One
    /// workgroup per token via `rmsnorm_batch`.
    ///
    /// Opens its own compute pass. For callers that want to batch multiple
    /// dispatches into a single pass (saves ~0.8ms of Vulkan
    /// pipeline-barrier overhead per dispatch), use
    /// `dispatch_rmsnorm_in_pass`.
    /// Phase G (vram-heap): `weight_buf` is `BindingResource` so
    /// static weights on `gpu.weights_heap` bind at sub-ranges.
    pub fn dispatch_rmsnorm_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        in_buf: wgpu::BindingResource<'_>,
        weight_buf: wgpu::BindingResource<'_>,
        out_buf: wgpu::BindingResource<'_>,
        n: usize,
        n_tokens: usize,
        eps: f32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.rmsnorm.pass"),
            timestamp_writes: None,
        });
        self.dispatch_rmsnorm_in_pass(&mut pass, in_buf, weight_buf, out_buf, n, n_tokens, eps);
    }

    /// Phase C1 encoder-level wrapper for packed→packed rmsnorm.
    ///
    /// Phase C (vram-heap): `in_buf` is a `BindingResource` so hidden_buf
    /// — a sub-allocation of `transient_heap_a` — can be bound at its
    /// sub-range rather than the entire heap. Whole-buffer callers pass
    /// `buf.as_entire_binding()`. The bind group is built inline via
    /// `make_bind_group_with` rather than delegating to `_in_pass`,
    /// because the in-pass variant is still called from non-polar paths
    /// that bind whole buffers.
    pub fn dispatch_rmsnorm_packed_to_packed_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        in_buf: wgpu::BindingResource<'_>,
        weight_buf: wgpu::BindingResource<'_>,
        out_buf: wgpu::BindingResource<'_>,
        n: usize,
        n_tokens: usize,
        eps: f32,
    ) {
        let params = RmsNormBatchParams {
            n: n as u32, eps, n_tokens: n_tokens as u32, _pad: 0,
        };
        let params_buf = self.gpu.create_params_buffer(&params);
        let pipeline = &self.gpu.pipelines.rmsnorm_batch_packed_to_packed;
        let bind = self.gpu.make_bind_group_with(
            pipeline,
            vec![
                in_buf,
                weight_buf,
                out_buf,
                params_buf.as_entire_binding(),
            ],
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.rmsnorm_packed_to_packed.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(n_tokens as u32, 1, 1);
    }

    /// Phase C1 encoder-level wrapper for f32→packed rmsnorm (BitNet o_sub_norm).
    pub fn dispatch_rmsnorm_f32_to_packed_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        in_buf: wgpu::BindingResource<'_>,
        weight_buf: wgpu::BindingResource<'_>,
        out_buf: wgpu::BindingResource<'_>,
        n: usize,
        n_tokens: usize,
        eps: f32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.rmsnorm_f32_to_packed.pass"),
            timestamp_writes: None,
        });
        self.dispatch_rmsnorm_f32_to_packed_in_pass(&mut pass, in_buf, weight_buf, out_buf, n, n_tokens, eps);
    }

    /// In-pass variant of `dispatch_rmsnorm_into`. Records the dispatch
    /// into the caller-supplied compute pass instead of opening a new one.
    /// See `dispatch_rmsnorm_into` for semantics. Both buffers are f32
    /// (used by bitnet sub-norms — Phase B leaves those f32).
    pub fn dispatch_rmsnorm_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        in_buf: wgpu::BindingResource<'_>,
        weight_buf: wgpu::BindingResource<'_>,
        out_buf: wgpu::BindingResource<'_>,
        n: usize,
        n_tokens: usize,
        eps: f32,
    ) {
        let params = RmsNormBatchParams {
            n: n as u32, eps, n_tokens: n_tokens as u32, _pad: 0,
        };
        let params_buf = self.gpu.create_params_buffer(&params);
        let pipeline = &self.gpu.pipelines.rmsnorm_batch;
        let bind = self.gpu.make_bind_group_with(
            pipeline,
            vec![
                in_buf,
                weight_buf,
                out_buf,
                params_buf.as_entire_binding(),
            ],
        );
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(n_tokens as u32, 1, 1);
    }

    /// Phase B: rmsnorm reading packed-f16 input (hidden_buf), writing
    /// f32 output (scratch.normed). Used for per-block attn_norm /
    /// ffn_norm. Same Params struct as the f32 variant; the shader does
    /// the packing math internally.
    pub fn dispatch_rmsnorm_packed_to_f32_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        in_buf: wgpu::BindingResource<'_>,
        weight_buf: wgpu::BindingResource<'_>,
        out_buf: wgpu::BindingResource<'_>,
        n: usize,
        n_tokens: usize,
        eps: f32,
    ) {
        let params = RmsNormBatchParams {
            n: n as u32, eps, n_tokens: n_tokens as u32, _pad: 0,
        };
        let params_buf = self.gpu.create_params_buffer(&params);
        let pipeline = &self.gpu.pipelines.rmsnorm_batch_packed_to_f32;
        let bind = self.gpu.make_bind_group_with(
            pipeline,
            vec![
                in_buf,
                weight_buf,
                out_buf,
                params_buf.as_entire_binding(),
            ],
        );
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(n_tokens as u32, 1, 1);
    }

    /// Phase C1: rmsnorm with f32 input, packed-f16 output. Used by
    /// BitNet o_sub_norm where scratch.attn_out (f32 in C1) feeds the
    /// packed scratch.normed (C1+).
    pub fn dispatch_rmsnorm_f32_to_packed_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        in_buf: wgpu::BindingResource<'_>,
        weight_buf: wgpu::BindingResource<'_>,
        out_buf: wgpu::BindingResource<'_>,
        n: usize,
        n_tokens: usize,
        eps: f32,
    ) {
        let params = RmsNormBatchParams {
            n: n as u32, eps, n_tokens: n_tokens as u32, _pad: 0,
        };
        let params_buf = self.gpu.create_params_buffer(&params);
        let pipeline = &self.gpu.pipelines.rmsnorm_batch_f32_to_packed;
        let bind = self.gpu.make_bind_group_with(
            pipeline,
            vec![
                in_buf,
                weight_buf,
                out_buf,
                params_buf.as_entire_binding(),
            ],
        );
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(n_tokens as u32, 1, 1);
    }

    /// Phase B: rmsnorm with packed-f16 input AND output. Used by the
    /// FINAL norm at the end of forward (hidden_buf → normed_buf, both
    /// packed). Same Params struct; shader does packing math internally.
    pub fn dispatch_rmsnorm_packed_to_packed_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        in_buf: wgpu::BindingResource<'_>,
        weight_buf: wgpu::BindingResource<'_>,
        out_buf: wgpu::BindingResource<'_>,
        n: usize,
        n_tokens: usize,
        eps: f32,
    ) {
        let params = RmsNormBatchParams {
            n: n as u32, eps, n_tokens: n_tokens as u32, _pad: 0,
        };
        let params_buf = self.gpu.create_params_buffer(&params);
        let pipeline = &self.gpu.pipelines.rmsnorm_batch_packed_to_packed;
        let bind = self.gpu.make_bind_group_with(
            pipeline,
            vec![
                in_buf,
                weight_buf,
                out_buf,
                params_buf.as_entire_binding(),
            ],
        );
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(n_tokens as u32, 1, 1);
    }

    /// Dispatch a batch matmul against a `GpuFloatLinear` layer's resident
    /// weights. Input is `[n_tokens, in_features]`, output is
    /// `[n_tokens, out_features]`. Uses the `matmul` (batch) shader which
    /// processes all tokens in one dispatch — the right primitive for prefill.
    pub fn dispatch_matmul_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layer: &dyn LinearLayer,
        in_buf: wgpu::BindingResource<'_>,
        out_buf: wgpu::BindingResource<'_>,
        n_tokens: usize,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.matmul.pass"),
            timestamp_writes: None,
        });
        self.dispatch_matmul_in_pass(&mut pass, layer, in_buf, out_buf, n_tokens);
    }

    /// In-pass variant. See `dispatch_matmul_into`. Routes to the
    /// tiled matmul shader when `n_tokens >= 8` (TILE_M), falling
    /// back to the per-output-element legacy shader for decode-sized
    /// batches where register blocking has nothing to amortize.
    pub fn dispatch_matmul_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        layer: &dyn LinearLayer,
        in_buf: wgpu::BindingResource<'_>,
        out_buf: wgpu::BindingResource<'_>,
        n_tokens: usize,
    ) {
        let float = layer
            .as_any()
            .downcast_ref::<crate::layers::gpu_floatlinear::GpuFloatLinear>()
            .unwrap_or_else(|| {
                panic!(
                    "GpuEngine.dispatch_matmul_in_pass: layer is not GpuFloatLinear \
                     (concrete type: {:?})",
                    layer
                )
            });

        // Route by n_tokens. The shared-memory tiled shader needs
        // TILE_N=16 tokens to amortize its workgroup-scoped loads;
        // for decode (n_tokens=1) the per-output legacy shader wins.
        // Threshold of 16 = TILE_N is the natural cutoff.
        // Overridable for A/B testing via CORTEX_MATMUL_SHARED_THRESHOLD
        // (set to usize::MAX-equivalent like 999999 to force legacy).
        let threshold: usize = std::env::var("CORTEX_MATMUL_SHARED_THRESHOLD")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(16);
        if n_tokens >= threshold {
            self.dispatch_matmul_shared_inner_in_pass(pass, float, in_buf, out_buf, n_tokens);
        } else {
            self.dispatch_matmul_legacy_inner_in_pass(pass, float, in_buf, out_buf, n_tokens);
        }
    }

    /// Shared-memory tiled matmul. See `shaders/matmul_shared.wgsl` for
    /// the kernel design. Same Params struct as the legacy variant.
    /// Phase C1: matmul_shared variant that reads PACKED f16 input
    /// (Q/K/V/O/gate/up projections after scratch.normed is packed).
    /// Output stays f32 (scratch.q/k/v/projected/gate/up still f32
    /// in C1). Same Params, same dispatch math.
    pub(super) fn dispatch_matmul_shared_pin_fout_inner_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        float: &crate::layers::gpu_floatlinear::GpuFloatLinear,
        in_buf: wgpu::BindingResource<'_>,
        out_buf: wgpu::BindingResource<'_>,
        n_tokens: usize,
    ) {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct MatmulParams { rows: u32, cols: u32, n_tokens: u32, _pad: u32 }
        let params = MatmulParams {
            rows: float.out_features() as u32,
            cols: float.in_features() as u32,
            n_tokens: n_tokens as u32,
            _pad: 0,
        };
        let params_buf = self.gpu.create_params_buffer(&params);

        let pipeline = &self.gpu.pipelines.matmul_shared_pin_fout;
        let bind = self.gpu.make_bind_group_with(
            pipeline,
            vec![
                float.weight_buffer().binding(),
                in_buf,
                out_buf,
                params_buf.as_entire_binding(),
            ],
        );

        let rows = float.out_features();
        const TILE_M: usize = 32;
        const TILE_N: usize = 16;
        let dx = ((rows + TILE_M - 1) / TILE_M) as u32;
        let dy = ((n_tokens + TILE_N - 1) / TILE_N) as u32;

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(dx, dy, 1);
    }

    pub(super) fn dispatch_matmul_shared_inner_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        float: &crate::layers::gpu_floatlinear::GpuFloatLinear,
        in_buf: wgpu::BindingResource<'_>,
        out_buf: wgpu::BindingResource<'_>,
        n_tokens: usize,
    ) {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct MatmulParams { rows: u32, cols: u32, n_tokens: u32, _pad: u32 }
        let params = MatmulParams {
            rows: float.out_features() as u32,
            cols: float.in_features() as u32,
            n_tokens: n_tokens as u32,
            _pad: 0,
        };
        let params_buf = self.gpu.create_params_buffer(&params);

        let pipeline = &self.gpu.pipelines.matmul_shared;
        let bind = self.gpu.make_bind_group_with(
            pipeline,
            vec![
                float.weight_buffer().binding(),
                in_buf,
                out_buf,
                params_buf.as_entire_binding(),
            ],
        );

        // workgroup_id.x = row tile (one per TILE_M=32 output rows;
        // each thread of the 16×16 WG computes 2 outputs stride-16 apart)
        // workgroup_id.y = token tile (one per TILE_N=16 tokens)
        let rows = float.out_features();
        const TILE_M: usize = 32;
        const TILE_N: usize = 16;
        let dx = ((rows + TILE_M - 1) / TILE_M) as u32;
        let dy = ((n_tokens + TILE_N - 1) / TILE_N) as u32;

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(dx, dy, 1);
    }

    /// Fused gate+up dispatch. Both projections must share the same
    /// (rows, cols) shape (true for all SwiGLU FFNs we support). One
    /// dispatch, one input load per K-step from shared memory,
    /// computes both gate and up outputs.
    ///
    /// Returns false if shapes mismatch or if either layer isn't
    /// `GpuFloatLinear`; caller falls back to two sequential
    /// matmul_shared dispatches.
    pub(super) fn dispatch_gate_up_fused_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        gate_layer: &dyn LinearLayer,
        up_layer: &dyn LinearLayer,
        in_buf: wgpu::BindingResource<'_>,
        gate_out: wgpu::BindingResource<'_>,
        up_out: wgpu::BindingResource<'_>,
        n_tokens: usize,
    ) -> bool {
        let gate_float = match gate_layer.as_any()
            .downcast_ref::<crate::layers::gpu_floatlinear::GpuFloatLinear>()
        {
            Some(l) => l,
            None => return false,
        };
        let up_float = match up_layer.as_any()
            .downcast_ref::<crate::layers::gpu_floatlinear::GpuFloatLinear>()
        {
            Some(l) => l,
            None => return false,
        };
        if gate_float.out_features() != up_float.out_features()
            || gate_float.in_features() != up_float.in_features()
        {
            return false;
        }

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params { rows: u32, cols: u32, n_tokens: u32, _pad: u32 }
        let params = Params {
            rows: gate_float.out_features() as u32,
            cols: gate_float.in_features() as u32,
            n_tokens: n_tokens as u32,
            _pad: 0,
        };
        let params_buf = self.gpu.create_params_buffer(&params);

        let pipeline = &self.gpu.pipelines.matmul_gate_up_shared;
        let bind = self.gpu.make_bind_group_with(
            pipeline,
            vec![
                gate_float.weight_buffer().binding(),
                up_float.weight_buffer().binding(),
                in_buf,
                gate_out,
                up_out,
                params_buf.as_entire_binding(),
            ],
        );

        let rows = gate_float.out_features();
        const TILE_M: usize = 32;
        const TILE_N: usize = 16;
        let dx = ((rows + TILE_M - 1) / TILE_M) as u32;
        let dy = ((n_tokens + TILE_N - 1) / TILE_N) as u32;

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(dx, dy, 1);
        true
    }

    /// Phase C1: packed-input decode-path matmul (per-output legacy
    /// shader, packed scratch.normed input → f32 output). Picked when
    /// caller has packed input and n_tokens < TILE_N=16.
    pub(super) fn dispatch_matmul_pin_inner_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        float: &crate::layers::gpu_floatlinear::GpuFloatLinear,
        in_buf: wgpu::BindingResource<'_>,
        out_buf: wgpu::BindingResource<'_>,
        n_tokens: usize,
    ) {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct MatmulParams { rows: u32, cols: u32, n_tokens: u32, _pad: u32 }
        let params = MatmulParams {
            rows: float.out_features() as u32,
            cols: float.in_features() as u32,
            n_tokens: n_tokens as u32,
            _pad: 0,
        };
        let params_buf = self.gpu.create_params_buffer(&params);

        let pipeline = &self.gpu.pipelines.matmul_pin;
        let bind = self.gpu.make_bind_group_with(
            pipeline,
            vec![
                float.weight_buffer().binding(),
                in_buf,
                out_buf,
                params_buf.as_entire_binding(),
            ],
        );

        let rows = float.out_features();
        let dx = (rows.min(65535)) as u32;
        let dy = ((rows + 65534) / 65535) as u32;
        let dz = n_tokens as u32;

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(dx, dy, dz);
    }

    /// Phase C3: prefill packed-input + packed-output matmul (shared-
    /// memory tiled). Same dispatch shape as matmul_shared / _pin_fout,
    /// adjacent-pair output packing (rows × n_tokens, one u32 per pair).
    pub(super) fn dispatch_matmul_shared_pin_pout_inner_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        float: &crate::layers::gpu_floatlinear::GpuFloatLinear,
        in_buf: wgpu::BindingResource<'_>,
        out_buf: wgpu::BindingResource<'_>,
        n_tokens: usize,
    ) {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct MatmulParams { rows: u32, cols: u32, n_tokens: u32, _pad: u32 }
        let params = MatmulParams {
            rows: float.out_features() as u32,
            cols: float.in_features() as u32,
            n_tokens: n_tokens as u32,
            _pad: 0,
        };
        let params_buf = self.gpu.create_params_buffer(&params);

        let pipeline = &self.gpu.pipelines.matmul_shared_pin_pout;
        let bind = self.gpu.make_bind_group_with(
            pipeline,
            vec![
                float.weight_buffer().binding(),
                in_buf,
                out_buf,
                params_buf.as_entire_binding(),
            ],
        );

        let rows = float.out_features();
        const TILE_M: usize = 32;
        const TILE_N: usize = 16;
        let dx = ((rows + TILE_M - 1) / TILE_M) as u32;
        let dy = ((n_tokens + TILE_N - 1) / TILE_N) as u32;

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(dx, dy, 1);
    }

    /// Phase C2: packed-input + packed-output decode-path matmul.
    /// One WG per (row_pair, tok); 256 threads cooperatively sum two
    /// adjacent rows and pack them into a single output u32.
    pub(super) fn dispatch_matmul_pin_pout_inner_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        float: &crate::layers::gpu_floatlinear::GpuFloatLinear,
        in_buf: wgpu::BindingResource<'_>,
        out_buf: wgpu::BindingResource<'_>,
        n_tokens: usize,
    ) {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct MatmulParams { rows: u32, cols: u32, n_tokens: u32, _pad: u32 }
        let params = MatmulParams {
            rows: float.out_features() as u32,
            cols: float.in_features() as u32,
            n_tokens: n_tokens as u32,
            _pad: 0,
        };
        let params_buf = self.gpu.create_params_buffer(&params);

        let pipeline = &self.gpu.pipelines.matmul_pin_pout;
        let bind = self.gpu.make_bind_group_with(
            pipeline,
            vec![
                float.weight_buffer().binding(),
                in_buf,
                out_buf,
                params_buf.as_entire_binding(),
            ],
        );

        let row_pairs = (float.out_features() + 1) / 2;
        let dx = (row_pairs.min(65535)) as u32;
        let dy = ((row_pairs + 65534) / 65535) as u32;
        let dz = n_tokens as u32;

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(dx, dy, dz);
    }

    /// Linear-batch with PACKED input AND PACKED output. Float-only
    /// after the BitNet un-merge — routes float prefill to
    /// matmul_shared_pin_pout, decode to matmul_pin_pout.
    pub fn dispatch_linear_batch_packed_io_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        layer: &dyn LinearLayer,
        in_packed_buf: wgpu::BindingResource<'_>,
        out_packed_buf: wgpu::BindingResource<'_>,
        n_tokens: usize,
    ) {
        let float = layer
            .as_any()
            .downcast_ref::<crate::layers::gpu_floatlinear::GpuFloatLinear>()
            .unwrap_or_else(|| panic!(
                "dispatch_linear_batch_packed_io_in_pass: expected GpuFloatLinear \
                 (concrete type: {:?})", layer));
        let threshold = std::env::var("CORTEX_MATMUL_SHARED_THRESHOLD")
            .ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(16);
        if n_tokens >= threshold {
            self.dispatch_matmul_shared_pin_pout_inner_in_pass(pass, float, in_packed_buf, out_packed_buf, n_tokens);
        } else {
            self.dispatch_matmul_pin_pout_inner_in_pass(pass, float, in_packed_buf, out_packed_buf, n_tokens);
        }
    }

    /// Phase C1 routing fn: dispatch matmul with PACKED f16 input.
    /// Routes to matmul_shared_pin_fout (prefill, n_tokens >= 16) or
    /// matmul_pin (decode). Output stays f32 in C1.
    pub fn dispatch_matmul_packed_input_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        layer: &dyn LinearLayer,
        in_buf: wgpu::BindingResource<'_>,
        out_buf: wgpu::BindingResource<'_>,
        n_tokens: usize,
    ) {
        let float = layer
            .as_any()
            .downcast_ref::<crate::layers::gpu_floatlinear::GpuFloatLinear>()
            .unwrap_or_else(|| {
                panic!(
                    "dispatch_matmul_packed_input_in_pass: layer is not GpuFloatLinear \
                     (concrete type: {:?})", layer)
            });
        let threshold: usize = std::env::var("CORTEX_MATMUL_SHARED_THRESHOLD")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(16);
        if n_tokens >= threshold {
            self.dispatch_matmul_shared_pin_fout_inner_in_pass(pass, float, in_buf, out_buf, n_tokens);
        } else {
            self.dispatch_matmul_pin_inner_in_pass(pass, float, in_buf, out_buf, n_tokens);
        }
    }

    pub(super) fn dispatch_matmul_legacy_inner_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        float: &crate::layers::gpu_floatlinear::GpuFloatLinear,
        in_buf: wgpu::BindingResource<'_>,
        out_buf: wgpu::BindingResource<'_>,
        n_tokens: usize,
    ) {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct MatmulParams { rows: u32, cols: u32, n_tokens: u32, _pad: u32 }
        let params = MatmulParams {
            rows: float.out_features() as u32,
            cols: float.in_features() as u32,
            n_tokens: n_tokens as u32,
            _pad: 0,
        };
        let params_buf = self.gpu.create_params_buffer(&params);

        let pipeline = &self.gpu.pipelines.matmul;
        let bind = self.gpu.make_bind_group_with(
            pipeline,
            vec![
                float.weight_buffer().binding(),
                in_buf,
                out_buf,
                params_buf.as_entire_binding(),
            ],
        );

        let rows = float.out_features();
        let dx = (rows.min(65535)) as u32;
        let dy = ((rows + 65534) / 65535) as u32;
        let dz = n_tokens as u32;

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(dx, dy, dz);
    }

    /// Unified linear-layer dispatcher used by `forward_block_gpu_inner`.
    /// Float-only after the BitNet un-merge.
    pub fn dispatch_linear_batch_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layer: &dyn LinearLayer,
        in_buf: wgpu::BindingResource<'_>,
        out_buf: wgpu::BindingResource<'_>,
        n_tokens: usize,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.linear_batch.pass"),
            timestamp_writes: None,
        });
        self.dispatch_linear_batch_in_pass(&mut pass, layer, in_buf, out_buf, n_tokens);
    }

    /// Phase C2 encoder-level wrapper for packed-IO linear-batch.
    ///
    /// Phase D (vram-heap): `in_packed_buf` and `out_packed_buf` are
    /// `BindingResource` so PolarBlockScratch sub-allocations can be
    /// bound at their sub-ranges. Bind group built inline via
    /// `make_bind_group_with` rather than delegating to `_in_pass`
    /// (the in-pass variant still serves the f32 path's PASS-bundled
    /// dispatches that bind whole BlockScratch buffers).
    pub fn dispatch_linear_batch_packed_io_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layer: &dyn LinearLayer,
        in_packed_buf: wgpu::BindingResource<'_>,
        out_packed_buf: wgpu::BindingResource<'_>,
        n_tokens: usize,
    ) {
        let float = layer
            .as_any()
            .downcast_ref::<crate::layers::gpu_floatlinear::GpuFloatLinear>()
            .unwrap_or_else(|| panic!(
                "dispatch_linear_batch_packed_io_into: expected GpuFloatLinear \
                 (concrete type: {:?})", layer));
        let threshold = std::env::var("CORTEX_MATMUL_SHARED_THRESHOLD")
            .ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(16);

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct MatmulParams { rows: u32, cols: u32, n_tokens: u32, _pad: u32 }
        let params = MatmulParams {
            rows: float.out_features() as u32,
            cols: float.in_features() as u32,
            n_tokens: n_tokens as u32,
            _pad: 0,
        };
        let params_buf = self.gpu.create_params_buffer(&params);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.linear_batch_packed_io.pass"),
            timestamp_writes: None,
        });

        if n_tokens >= threshold {
            let pipeline = &self.gpu.pipelines.matmul_shared_pin_pout;
            let bind = self.gpu.make_bind_group_with(
                pipeline,
                vec![
                    float.weight_buffer().binding(),
                    in_packed_buf,
                    out_packed_buf,
                    params_buf.as_entire_binding(),
                ],
            );
            const TILE_M: usize = 32;
            const TILE_N: usize = 16;
            let rows = float.out_features();
            let dx = ((rows + TILE_M - 1) / TILE_M) as u32;
            let dy = ((n_tokens + TILE_N - 1) / TILE_N) as u32;
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(dx, dy, 1);
        } else {
            let pipeline = &self.gpu.pipelines.matmul_pin_pout;
            let bind = self.gpu.make_bind_group_with(
                pipeline,
                vec![
                    float.weight_buffer().binding(),
                    in_packed_buf,
                    out_packed_buf,
                    params_buf.as_entire_binding(),
                ],
            );
            let row_pairs = (float.out_features() + 1) / 2;
            let dx = (row_pairs.min(65535)) as u32;
            let dy = ((row_pairs + 65534) / 65535) as u32;
            let dz = n_tokens as u32;
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(dx, dy, dz);
        }
    }

    /// Phase C2 encoder-level wrapper for packed gate_mul.
    ///
    /// Phase D (vram-heap): gate/up/out args are `BindingResource` so
    /// PolarBlockScratch sub-allocations bind at sub-ranges. Bind
    /// group built inline (the `_in_pass` variant still serves f32).
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_gate_mul_packed_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gate_buf: wgpu::BindingResource<'_>,
        up_buf: wgpu::BindingResource<'_>,
        out_buf: wgpu::BindingResource<'_>,
        n: usize,
        n_tokens: usize,
        activation: crate::layers::swiglu::GateActivation,
    ) {
        let params = SiluMulBatchParams { n: n as u32, n_tokens: n_tokens as u32 };
        let params_buf = self.gpu.create_params_buffer(&params);
        let _ = activation;
        let pipeline = &self.gpu.pipelines.silu_mul_batch_packed;
        let bind = self.gpu.make_bind_group_with(
            pipeline,
            vec![gate_buf, up_buf, out_buf, params_buf.as_entire_binding()],
        );
        let total_packed = ((n * n_tokens) / 2) as u32;
        let groups = (total_packed + 255) / 256;
        let dx = groups.min(65535);
        let dy = (groups + 65534) / 65535;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.gate_mul_packed.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(dx, dy, 1);
    }

    /// Phase C1 encoder-level wrapper for packed-input linear-batch.
    pub fn dispatch_linear_batch_packed_input_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layer: &dyn LinearLayer,
        in_packed_buf: wgpu::BindingResource<'_>,
        out_buf: wgpu::BindingResource<'_>,
        n_tokens: usize,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.linear_batch_packed_input.pass"),
            timestamp_writes: None,
        });
        self.dispatch_linear_batch_packed_input_in_pass(&mut pass, layer, in_packed_buf, out_buf, n_tokens);
    }

    /// In-pass variant. See `dispatch_linear_batch_into`. Float-only.
    pub fn dispatch_linear_batch_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        layer: &dyn LinearLayer,
        in_buf: wgpu::BindingResource<'_>,
        out_buf: wgpu::BindingResource<'_>,
        n_tokens: usize,
    ) {
        self.dispatch_matmul_in_pass(pass, layer, in_buf, out_buf, n_tokens);
    }

    /// Packed-f16 input variant — float-only after BitNet un-merge.
    pub fn dispatch_linear_batch_packed_input_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        layer: &dyn LinearLayer,
        in_packed_buf: wgpu::BindingResource<'_>,
        out_buf: wgpu::BindingResource<'_>,
        n_tokens: usize,
    ) {
        self.dispatch_matmul_packed_input_in_pass(pass, layer, in_packed_buf, out_buf, n_tokens);
    }

    /// GPU-native dispatch of the per-token final RMSNorm using the
    /// `rmsnorm_batch` pipeline. Kept private so callers compose it via
    /// `forward_gpu` rather than reach in directly.
    pub(super) fn dispatch_final_norm(&self, pre_norm: &[f32], seq_len: usize) -> Vec<f32> {
        assert_eq!(pre_norm.len(), seq_len * self.embed_dim, "shape mismatch");
        let total_bytes = (pre_norm.len() * std::mem::size_of::<f32>()) as u64;

        let input_buf = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_engine.final_norm.input"),
            contents: bytemuck::cast_slice(pre_norm),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_engine.final_norm.output"),
            size: total_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = self.gpu.create_staging_buffer(total_bytes);

        let params = RmsNormBatchParams {
            n: self.embed_dim as u32,
            eps: self.final_norm_eps,
            n_tokens: seq_len as u32,
            _pad: 0,
        };
        let params_buf = self.gpu.create_params_buffer(&params);

        let pipeline = &self.gpu.pipelines.rmsnorm_batch;
        let bind_group = self.gpu.make_bind_group_with(
            pipeline,
            vec![
                input_buf.as_entire_binding(),
                self.final_norm_weight_buf.binding(),
                output_buf.as_entire_binding(),
                params_buf.as_entire_binding(),
            ],
        );

        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu_engine.final_norm.encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_engine.final_norm.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One workgroup per token (rmsnorm_batch indexes tokens by workgroup_id.x).
            pass.dispatch_workgroups(seq_len as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, total_bytes);
        self.gpu.queue.submit(Some(encoder.finish()));

        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        rx.recv().expect("GPU readback failed").expect("buffer map failed");

        let data = slice.get_mapped_range();
        let out: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        drop(data);
        staging_buf.unmap();
        out
    }

    /// Internal: matmul of the resident LM-head weights against an input
    /// packed-f16 buffer. Same wiring as `dispatch_matmul_pin_inner_in_pass`
    /// but takes raw (buffer, rows, cols) rather than a `GpuFloatLinear`
    /// reference — the LM-head is stored on the engine, not behind the
    /// LinearLayer trait. n_tokens is always 1 in the greedy path (we
    /// project just the last token's slice).
    pub(super) fn dispatch_lm_head_matmul_pin_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        lm_head: &LmHead,
        in_packed_buf: &wgpu::Buffer,
        out_buf: &wgpu::Buffer,
    ) {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct MatmulParams { rows: u32, cols: u32, n_tokens: u32, _pad: u32 }
        let params = MatmulParams {
            rows: lm_head.vocab_size as u32,
            cols: lm_head.embed_dim as u32,
            n_tokens: 1,
            _pad: 0,
        };
        let params_buf = self.gpu.create_params_buffer(&params);
        let pipeline = &self.gpu.pipelines.matmul_pin;
        let bind = self.gpu.make_bind_group_with(
            pipeline,
            vec![
                lm_head.weight_buf.binding(),
                in_packed_buf.as_entire_binding(),
                out_buf.as_entire_binding(),
                params_buf.as_entire_binding(),
            ],
        );
        // matmul_pin uses (wid.x + wid.y * 65535) row indexing — Qwen
        // 151k vocab requires dy = 3.
        let rows = lm_head.vocab_size;
        let dx = (rows.min(65535)) as u32;
        let dy = ((rows + 65534) / 65535) as u32;
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(dx, dy, 1);
    }

    /// Internal: single-WG argmax reduction over `logits_buf` (length
    /// `vocab_size` f32). Writes one u32 to `out_id_buf[0]`.
    pub(super) fn dispatch_argmax_vocab_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        logits_buf: &wgpu::Buffer,
        out_id_buf: &wgpu::Buffer,
        vocab_size: usize,
    ) {
        let params = ArgmaxVocabParams { n: vocab_size as u32, _pad: 0 };
        let params_buf = self.gpu.create_params_buffer(&params);
        let pipeline = &self.gpu.pipelines.argmax_vocab;
        let bind = self.gpu.make_bind_group(
            pipeline, &[logits_buf, out_id_buf, &params_buf],
        );
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Dispatch RoPE in place on `x_buf`, which must hold f32 values laid
    /// out as `[n_tokens, n_heads, head_dim]`. Token `t` is rotated for
    /// position `start_pos + t`. Halved (NeoX/HF) layout — Qwen and BitNet
    /// both use this; interleaved (older llama.cpp) is not supported by the
    /// shader yet.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_rope_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        x_buf: &wgpu::Buffer,
        cos_buf: wgpu::BindingResource<'_>,
        sin_buf: wgpu::BindingResource<'_>,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
        n_tokens: usize,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.rope.pass"),
            timestamp_writes: None,
        });
        self.dispatch_rope_in_pass(&mut pass, x_buf.as_entire_binding(), cos_buf, sin_buf, n_heads, head_dim, start_pos, n_tokens);
    }

    /// In-pass variant. See `dispatch_rope_into`.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_rope_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        x_buf: wgpu::BindingResource<'_>,
        cos_buf: wgpu::BindingResource<'_>,
        sin_buf: wgpu::BindingResource<'_>,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
        n_tokens: usize,
    ) {
        assert!(head_dim % 2 == 0, "RoPE head_dim must be even");
        let half_dim = head_dim / 2;

        let params = RopeBatchParams {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            start_pos: start_pos as u32,
            half_dim: half_dim as u32,
            n_tokens: n_tokens as u32,
            _p1: 0, _p2: 0, _p3: 0,
        };
        let params_buf = self.gpu.create_params_buffer(&params);

        let pipeline = &self.gpu.pipelines.rope_batch;
        let bind_group = self.gpu.make_bind_group_with(
            pipeline,
            vec![
                x_buf,
                cos_buf,
                sin_buf,
                params_buf.as_entire_binding(),
            ],
        );

        let total_threads = (n_tokens * n_heads * half_dim) as u32;
        let dispatch_x = (total_threads + 63) / 64;

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dispatch_x, 1, 1);
    }

    /// Phase C3 encoder-level wrapper for packed RoPE.
    /// Phase D (vram-heap): `x_buf` is `BindingResource` so the
    /// in-place RW target (scratch.q or scratch.k) binds at its
    /// sub-range. Bind group inlined; `_in_pass` still serves f32.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_rope_packed_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        x_buf: wgpu::BindingResource<'_>,
        cos_buf: wgpu::BindingResource<'_>,
        sin_buf: wgpu::BindingResource<'_>,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
        n_tokens: usize,
    ) {
        assert!(head_dim % 2 == 0, "RoPE head_dim must be even");
        assert!((head_dim / 2) % 2 == 0,
            "Packed RoPE requires half_dim even (head_dim divisible by 4)");
        let half_dim = head_dim / 2;
        let params = RopeBatchParams {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            start_pos: start_pos as u32,
            half_dim: half_dim as u32,
            n_tokens: n_tokens as u32,
            _p1: 0, _p2: 0, _p3: 0,
        };
        let params_buf = self.gpu.create_params_buffer(&params);
        let pipeline = &self.gpu.pipelines.rope_batch_packed;
        let bind_group = self.gpu.make_bind_group_with(
            pipeline,
            vec![
                x_buf,
                cos_buf,
                sin_buf,
                params_buf.as_entire_binding(),
            ],
        );
        let total_threads = (n_tokens * n_heads * (half_dim / 2)) as u32;
        let dispatch_x = (total_threads + 63) / 64;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.rope_packed.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dispatch_x, 1, 1);
    }

    /// Phase C3: packed-f16 RoPE. x_buf is packed; thread count halves
    /// (one thread per pair-pair, processing 2 adjacent rotation pairs
    /// that share two u32 slots).
    pub fn dispatch_rope_packed_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        x_buf: wgpu::BindingResource<'_>,
        cos_buf: wgpu::BindingResource<'_>,
        sin_buf: wgpu::BindingResource<'_>,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
        n_tokens: usize,
    ) {
        assert!(head_dim % 2 == 0, "RoPE head_dim must be even");
        assert!((head_dim / 2) % 2 == 0,
            "Packed RoPE requires half_dim even (head_dim divisible by 4)");
        let half_dim = head_dim / 2;

        let params = RopeBatchParams {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            start_pos: start_pos as u32,
            half_dim: half_dim as u32,
            n_tokens: n_tokens as u32,
            _p1: 0, _p2: 0, _p3: 0,
        };
        let params_buf = self.gpu.create_params_buffer(&params);

        let pipeline = &self.gpu.pipelines.rope_batch_packed;
        let bind_group = self.gpu.make_bind_group_with(
            pipeline,
            vec![
                x_buf,
                cos_buf,
                sin_buf,
                params_buf.as_entire_binding(),
            ],
        );

        // Half the original thread count (one per pair-pair).
        let total_threads = (n_tokens * n_heads * (half_dim / 2)) as u32;
        let dispatch_x = (total_threads + 63) / 64;

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dispatch_x, 1, 1);
    }

    /// Dispatch GQA attention math (attn_score → softmax → attn_value)
    /// against pre-projected, RoPE-rotated Q/K/V buffers. Records all three
    /// dispatches into the supplied `encoder`.
    ///
    /// Buffer layouts:
    /// - `q_buf`:  [n_tokens, n_heads * head_dim]  f32
    /// - `k_buf`:  [max_seq,  n_kv_heads * head_dim]  f32 (cache-shaped)
    /// - `v_buf`:  [max_seq,  n_kv_heads * head_dim]  f32 (cache-shaped)
    /// - `scores_buf`: [n_tokens, n_heads, max_seq] f32 (scratch; written
    ///   by attn_score, read+written by softmax, read by attn_value)
    /// - `out_buf`: [n_tokens, n_heads * head_dim]  f32 (output)
    ///
    /// Causal mask is applied in `attn_score_batch` (positions > start_pos+tok
    /// are written as -inf so softmax zeros them). For prefill mode pass
    /// `start_pos=0` and size K/V buffers as `max_seq=n_tokens`.
    pub fn dispatch_attention_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        q_buf: &wgpu::Buffer,
        k_buf: &wgpu::Buffer,
        v_buf: &wgpu::Buffer,
        scores_buf: &::vram_heap::VramAllocation,
        out_buf: &wgpu::Buffer,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        start_pos: usize,
        max_seq: usize,
        n_tokens: usize,
    ) {
        self.dispatch_attention_inner(
            encoder,
            q_buf.as_entire_binding(),
            k_buf.as_entire_binding(),
            v_buf.as_entire_binding(),
            scores_buf,
            out_buf.as_entire_binding(),
            n_heads, n_kv_heads, head_dim, start_pos, max_seq, n_tokens,
            None,
        );
    }

    /// Same as `dispatch_attention_into` but if `pre_softmax_capture` is
    /// `Some`, the pre-softmax `scores_buf` contents are copied into it
    /// after the attn_score dispatch and before softmax overwrites them.
    /// Used by the retrieval / traced forward path to extract per-layer
    /// raw attention scores.
    #[allow(clippy::too_many_arguments)]
    /// Phase F (vram-heap): `k_buf` and `v_buf` are `BindingResource` so
    /// GpuKvCache sub-allocations (cached path) bind at sub-ranges; the
    /// prefill path passes `scratch.k.binding()` / `scratch.v.binding()`.
    ///
    /// Phase I: `q_buf`, `scores_buf`, `out_buf` become `BindingResource`
    /// too — they're all BlockScratch fields on the f32 path.
    /// `scores_buf` is consumed three times internally (attn_score W,
    /// optional capture R, softmax RW, attn_value R); the function takes
    /// `&VramAllocation` so it can call `.binding()` at each consumption
    /// site (BindingResource is move-only).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_attention_inner(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        q_buf: wgpu::BindingResource<'_>,
        k_buf: wgpu::BindingResource<'_>,
        v_buf: wgpu::BindingResource<'_>,
        scores_buf: &::vram_heap::VramAllocation,
        out_buf: wgpu::BindingResource<'_>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        start_pos: usize,
        max_seq: usize,
        n_tokens: usize,
        pre_softmax_capture: Option<&wgpu::Buffer>,
    ) {
        assert!(n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads");
        let heads_per_kv = n_heads / n_kv_heads;
        let kv_dim = n_kv_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Dual-path routing.
        //
        // Legacy 3-shader path is the DEFAULT — it parallelizes
        // massively across (tok, head, t) (millions of threads on a
        // 2000-token prefill) and was heavily tuned through the
        // C-series work. The FlashAttention-1 fused shader exists for
        // numerical-parity experiments and as infrastructure for
        // future split-K / FA2 optimization, but at current per-tile
        // barrier overhead it's ~2x slower than legacy on RTX 4080
        // Laptop at 500w (132ms vs 60ms cumulative attention).
        //
        // Opt-in via CORTEX_ATTN_BACKEND=fused. Forced to legacy
        // (regardless of env) when pre_softmax_capture is set, because
        // the fused path can't materialize the full scores matrix that
        // retrieval trace mode needs.
        let force_fused = std::env::var("CORTEX_ATTN_BACKEND")
            .as_deref() == Ok("fused");
        let needs_capture = pre_softmax_capture.is_some();
        let use_fused = head_dim == 128 && !needs_capture && force_fused;
        if use_fused {
            let params = AttnScoreBatchParams {
                n_heads: n_heads as u32,
                n_kv_heads: n_kv_heads as u32,
                head_dim: head_dim as u32,
                start_pos: start_pos as u32,
                max_seq: max_seq as u32,
                heads_per_kv: heads_per_kv as u32,
                kv_dim: kv_dim as u32,
                scale,
                n_tokens: n_tokens as u32,
                _p1: 0, _p2: 0, _p3: 0,
            };
            let params_buf = self.gpu.create_params_buffer(&params);
            let pipeline = &self.gpu.pipelines.attn_fused_batch;
            let bind = self.gpu.make_bind_group_with(
                pipeline,
                vec![
                    q_buf,
                    k_buf,
                    v_buf,
                    out_buf,
                    params_buf.as_entire_binding(),
                ],
            );
            let mut pass = self.begin_timed_pass(encoder, "attn_fused");
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            // Dispatch: one workgroup per (head, tok).
            pass.dispatch_workgroups(n_heads as u32, n_tokens as u32, 1);
            return;
        }

        // ---- Legacy 3-shader path (trace mode / fallback) ----

        // Bisect within attention (attn-3 followup): env-gated skip of
        // individual stages, to identify which of score/softmax/value
        // actually dominates wall time. Stage that, when skipped, gives
        // back the most time is the next optimization target.
        // Values: CORTEX_SKIP_SCORE / SOFTMAX / VALUE = 1 to skip.
        let skip_score = std::env::var("CORTEX_SKIP_SCORE").as_deref() == Ok("1");
        let skip_softmax = std::env::var("CORTEX_SKIP_SOFTMAX").as_deref() == Ok("1");
        let skip_value = std::env::var("CORTEX_SKIP_VALUE").as_deref() == Ok("1");

        // ---- 1. attn_score: Q · K^T * scale, with causal mask ----
        let score_params = AttnScoreBatchParams {
            n_heads: n_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            head_dim: head_dim as u32,
            start_pos: start_pos as u32,
            max_seq: max_seq as u32,
            heads_per_kv: heads_per_kv as u32,
            kv_dim: kv_dim as u32,
            scale,
            n_tokens: n_tokens as u32,
            _p1: 0, _p2: 0, _p3: 0,
        };
        let score_params_buf = self.gpu.create_params_buffer(&score_params);
        let score_pipeline = &self.gpu.pipelines.attn_score_batch;
        let score_bind = self.gpu.make_bind_group_with(
            score_pipeline,
            vec![
                q_buf,
                k_buf,
                scores_buf.binding(),
                score_params_buf.as_entire_binding(),
            ],
        );
        // 256-thread workgroups over (head*max_seq, tok); gid.x covers
        // (head, t), gid.y covers tok.
        let inner_threads = (n_heads * max_seq) as u32;
        let score_groups_x = (inner_threads + 255) / 256;
        let score_groups_y = n_tokens as u32;
        if !skip_score {
            let mut pass = self.begin_timed_pass(encoder, "attn_score");
            pass.set_pipeline(score_pipeline);
            pass.set_bind_group(0, &score_bind, &[]);
            pass.dispatch_workgroups(score_groups_x, score_groups_y, 1);
        }

        // ---- 1.5. (optional) capture pre-softmax scores ----
        if let Some(capture_buf) = pre_softmax_capture {
            let bytes = (n_tokens * n_heads * max_seq * std::mem::size_of::<f32>()) as u64;
            encoder.copy_buffer_to_buffer(
                scores_buf.buffer(), scores_buf.offset(),
                capture_buf, 0, bytes,
            );
        }

        // ---- 2. softmax: in-place over scores ----
        let softmax_params = SoftmaxBatchParams {
            n_heads: n_heads as u32,
            max_seq: max_seq as u32,
            start_pos: start_pos as u32,
            n_tokens: n_tokens as u32,
        };
        let softmax_params_buf = self.gpu.create_params_buffer(&softmax_params);
        let softmax_pipeline = &self.gpu.pipelines.softmax_batch;
        let softmax_bind = self.gpu.make_bind_group_with(
            softmax_pipeline,
            vec![
                scores_buf.binding(),
                softmax_params_buf.as_entire_binding(),
            ],
        );
        // One workgroup per (tok, head) pair.
        let softmax_groups = (n_tokens * n_heads) as u32;
        if !skip_softmax {
            let mut pass = self.begin_timed_pass(encoder, "softmax");
            pass.set_pipeline(softmax_pipeline);
            pass.set_bind_group(0, &softmax_bind, &[]);
            pass.dispatch_workgroups(softmax_groups, 1, 1);
        }

        // ---- 3. attn_value: weighted sum of V ----
        let value_params = AttnValueBatchParams {
            n_heads: n_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            head_dim: head_dim as u32,
            start_pos: start_pos as u32,
            max_seq: max_seq as u32,
            heads_per_kv: heads_per_kv as u32,
            kv_dim: kv_dim as u32,
            n_tokens: n_tokens as u32,
        };
        let value_params_buf = self.gpu.create_params_buffer(&value_params);
        let value_pipeline = &self.gpu.pipelines.attn_value_batch;
        let value_bind = self.gpu.make_bind_group_with(
            value_pipeline,
            vec![
                scores_buf.binding(),
                v_buf,
                out_buf,
                value_params_buf.as_entire_binding(),
            ],
        );
        // One thread per (tok, head, d); workgroup_size=256.
        let total_value_threads = (n_tokens * n_heads * head_dim) as u32;
        let value_groups = (total_value_threads + 255) / 256;
        if !skip_value {
            let mut pass = self.begin_timed_pass(encoder, "attn_value");
            pass.set_pipeline(value_pipeline);
            pass.set_bind_group(0, &value_bind, &[]);
            pass.dispatch_workgroups(value_groups, 1, 1);
        }
    }

    /// Dispatch element-wise SiLU(gate) * up into `out_buf`. Sized as
    /// `[n_tokens, n]` flat. Used by the SwiGLU FFN. SiLU activation only;
    /// ReLU² (BitNet variant) needs a separate shader and is deferred until
    /// the ternary fused path lands.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_silu_mul_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gate_buf: wgpu::BindingResource<'_>,
        up_buf: wgpu::BindingResource<'_>,
        out_buf: wgpu::BindingResource<'_>,
        n: usize,
        n_tokens: usize,
    ) {
        self.dispatch_gate_mul_into(
            encoder, gate_buf, up_buf, out_buf, n, n_tokens,
            crate::layers::swiglu::GateActivation::SiLU,
        );
    }

    /// Activation-aware gate*up dispatcher: routes to `silu_mul_batch`
    /// or `relu2_mul_batch` based on the SwiGLU's activation kind.
    /// Both pipelines share the same binding layout, so the buffer
    /// arguments are interchangeable.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_gate_mul_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gate_buf: wgpu::BindingResource<'_>,
        up_buf: wgpu::BindingResource<'_>,
        out_buf: wgpu::BindingResource<'_>,
        n: usize,
        n_tokens: usize,
        activation: crate::layers::swiglu::GateActivation,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.gate_mul.pass"),
            timestamp_writes: None,
        });
        self.dispatch_gate_mul_in_pass(&mut pass, gate_buf, up_buf, out_buf, n, n_tokens, activation);
    }

    /// In-pass variant. See `dispatch_gate_mul_into`.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_gate_mul_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        gate_buf: wgpu::BindingResource<'_>,
        up_buf: wgpu::BindingResource<'_>,
        out_buf: wgpu::BindingResource<'_>,
        n: usize,
        n_tokens: usize,
        activation: crate::layers::swiglu::GateActivation,
    ) {
        let params = SiluMulBatchParams { n: n as u32, n_tokens: n_tokens as u32 };
        let params_buf = self.gpu.create_params_buffer(&params);

        // Float-only after BitNet un-merge: SiLU is the only activation.
        let _ = activation;
        let pipeline = &self.gpu.pipelines.silu_mul_batch;
        let bind = self.gpu.make_bind_group_with(
            pipeline,
            vec![gate_buf, up_buf, out_buf, params_buf.as_entire_binding()],
        );

        let total = (n * n_tokens) as u32;
        let groups = (total + 255) / 256;
        let dx = groups.min(65535);
        let dy = (groups + 65534) / 65535;

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(dx, dy, 1);
    }

    /// Phase C2: packed-f16 variant of dispatch_gate_mul_in_pass.
    /// Float-only after BitNet un-merge: SiLU only.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_gate_mul_packed_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        gate_buf: wgpu::BindingResource<'_>,
        up_buf: wgpu::BindingResource<'_>,
        out_buf: wgpu::BindingResource<'_>,
        n: usize,
        n_tokens: usize,
        activation: crate::layers::swiglu::GateActivation,
    ) {
        let params = SiluMulBatchParams { n: n as u32, n_tokens: n_tokens as u32 };
        let params_buf = self.gpu.create_params_buffer(&params);

        let _ = activation;
        let pipeline = &self.gpu.pipelines.silu_mul_batch_packed;
        let bind = self.gpu.make_bind_group_with(
            pipeline,
            vec![gate_buf, up_buf, out_buf, params_buf.as_entire_binding()],
        );

        // Half the elements since each u32 packs 2.
        let total_packed = ((n * n_tokens) / 2) as u32;
        let groups = (total_packed + 255) / 256;
        let dx = groups.min(65535);
        let dy = (groups + 65534) / 65535;

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(dx, dy, 1);
    }

    /// Dispatch kv_write_batch: copy K and V vectors for `n_tokens` new
    /// positions from per-block scratch buffers into the layer's resident
    /// cache buffers, starting at offset `start_pos` (in tokens). Used by
    /// the cached forward path.
    /// Phase F (vram-heap): `k_cache` and `v_cache` are `BindingResource`
    /// so GpuKvCache sub-allocations bind at sub-ranges. `k_src` and
    /// `v_src` stay as `&wgpu::Buffer` because f32-path BlockScratch
    /// is not migrated until Phase I.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_kv_write_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        k_src: wgpu::BindingResource<'_>,
        v_src: wgpu::BindingResource<'_>,
        k_cache: wgpu::BindingResource<'_>,
        v_cache: wgpu::BindingResource<'_>,
        kv_dim: usize,
        start_pos: usize,
        n_tokens: usize,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.kv_write.pass"),
            timestamp_writes: None,
        });
        self.dispatch_kv_write_in_pass(&mut pass, k_src, v_src, k_cache, v_cache, kv_dim, start_pos, n_tokens);
    }

    /// In-pass variant. See `dispatch_kv_write_into`.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_kv_write_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        k_src: wgpu::BindingResource<'_>,
        v_src: wgpu::BindingResource<'_>,
        k_cache: wgpu::BindingResource<'_>,
        v_cache: wgpu::BindingResource<'_>,
        kv_dim: usize,
        start_pos: usize,
        n_tokens: usize,
    ) {
        let params = KvWriteBatchParams {
            kv_dim: kv_dim as u32,
            start_pos: start_pos as u32,
            n_tokens: n_tokens as u32,
            _pad: 0,
        };
        let params_buf = self.gpu.create_params_buffer(&params);

        let pipeline = &self.gpu.pipelines.kv_write_batch;
        let bind = self.gpu.make_bind_group_with(
            pipeline,
            vec![
                k_src,
                v_src,
                k_cache,
                v_cache,
                params_buf.as_entire_binding(),
            ],
        );

        // Phase A: K/V cache is packed f16 (2 per u32). The shader writes
        // ONE u32 per thread = 2 f32s packed. Halves the thread count.
        let total = ((kv_dim / 2) * n_tokens) as u32;
        let groups = (total + 127) / 128;
        let dx = groups.min(65535);
        let dy = (groups + 65534) / 65535;

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(dx, dy, 1);
    }

    /// Dispatch broadcast bias add: `a[tok, i] += bias[i]` for all tokens.
    /// Used for Q/K/V projection biases in Qwen-family models.
    pub fn dispatch_bias_add_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a_buf: &wgpu::Buffer,
        bias_buf: wgpu::BindingResource<'_>,
        n: usize,
        n_tokens: usize,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.bias_add.pass"),
            timestamp_writes: None,
        });
        self.dispatch_bias_add_in_pass(&mut pass, a_buf.as_entire_binding(), bias_buf, n, n_tokens);
    }

    /// In-pass variant. See `dispatch_bias_add_into`.
    pub fn dispatch_bias_add_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        a_buf: wgpu::BindingResource<'_>,
        bias_buf: wgpu::BindingResource<'_>,
        n: usize,
        n_tokens: usize,
    ) {
        let params = AddInplaceBatchParams { n: n as u32, n_tokens: n_tokens as u32 };
        let params_buf = self.gpu.create_params_buffer(&params);

        let pipeline = &self.gpu.pipelines.bias_add_batch;
        let bind = self.gpu.make_bind_group_with(
            pipeline,
            vec![
                a_buf,
                bias_buf,
                params_buf.as_entire_binding(),
            ],
        );

        let total = (n * n_tokens) as u32;
        let groups = (total + 255) / 256;

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }

    /// Phase C3 encoder-level wrapper for packed bias_add.
    /// Phase D (vram-heap): `a_buf` is `BindingResource` so the
    /// in-place RW target (a PolarBlockScratch projection field) binds
    /// at its sub-range. Bind group inlined; the `_in_pass` variant
    /// still serves the f32 path.
    pub fn dispatch_bias_add_packed_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a_buf: wgpu::BindingResource<'_>,
        bias_buf: wgpu::BindingResource<'_>,
        n: usize,
        n_tokens: usize,
    ) {
        let params = AddInplaceBatchParams { n: n as u32, n_tokens: n_tokens as u32 };
        let params_buf = self.gpu.create_params_buffer(&params);
        let pipeline = &self.gpu.pipelines.bias_add_batch_packed;
        let bind = self.gpu.make_bind_group_with(
            pipeline,
            vec![a_buf, bias_buf, params_buf.as_entire_binding()],
        );
        let total_slots = (n * n_tokens / 2) as u32;
        let groups = (total_slots + 255) / 256;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.bias_add_packed.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }

    /// Phase C3: packed-`a` variant of bias_add. Bias buffer stays f32.
    /// Halves thread count (one per u32 slot = 2 columns).
    pub fn dispatch_bias_add_packed_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        a_buf: wgpu::BindingResource<'_>,
        bias_buf: wgpu::BindingResource<'_>,
        n: usize,
        n_tokens: usize,
    ) {
        let params = AddInplaceBatchParams { n: n as u32, n_tokens: n_tokens as u32 };
        let params_buf = self.gpu.create_params_buffer(&params);
        let pipeline = &self.gpu.pipelines.bias_add_batch_packed;
        let bind = self.gpu.make_bind_group_with(
            pipeline,
            vec![
                a_buf,
                bias_buf,
                params_buf.as_entire_binding(),
            ],
        );
        let total_slots = (n * n_tokens / 2) as u32;
        let groups = (total_slots + 255) / 256;
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }

    /// Dispatch element-wise in-place add: `a_buf[i] += b_buf[i]`. Used for
    /// residual connections (post-attention and post-FFN).
    pub fn dispatch_add_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a_buf: &wgpu::Buffer,
        b_buf: &wgpu::Buffer,
        n: usize,
        n_tokens: usize,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.add.pass"),
            timestamp_writes: None,
        });
        self.dispatch_add_in_pass(&mut pass, a_buf.as_entire_binding(), b_buf.as_entire_binding(), n, n_tokens);
    }

    /// Phase C3: packed-`a` packed-`b` variant of dispatch_add_in_pass.
    /// Used for `hidden += projected` when scratch.projected is packed.
    pub fn dispatch_add_packed_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        a_buf: wgpu::BindingResource<'_>,
        b_buf: wgpu::BindingResource<'_>,
        n: usize,
        n_tokens: usize,
    ) {
        let params = AddInplaceBatchParams { n: n as u32, n_tokens: n_tokens as u32 };
        let params_buf = self.gpu.create_params_buffer(&params);
        let pipeline = &self.gpu.pipelines.add_inplace_batch_packed;
        let bind = self.gpu.make_bind_group_with(
            pipeline,
            vec![a_buf, b_buf, params_buf.as_entire_binding()],
        );
        let total = (n * n_tokens / 2) as u32;
        let groups = (total + 255) / 256;
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }

    /// Phase C3 encoder-level wrapper for packed add.
    ///
    /// Phase C (vram-heap): `a_buf` is a `BindingResource` so hidden_buf
    /// — a sub-allocation of `transient_heap_a` — can be bound at its
    /// sub-range rather than the entire heap. Bind group built inline
    /// rather than delegating to `_in_pass` (the in-pass variant has
    /// non-polar callers that bind whole buffers).
    pub fn dispatch_add_packed_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a_buf: wgpu::BindingResource<'_>,
        b_buf: wgpu::BindingResource<'_>,
        n: usize,
        n_tokens: usize,
    ) {
        let params = AddInplaceBatchParams { n: n as u32, n_tokens: n_tokens as u32 };
        let params_buf = self.gpu.create_params_buffer(&params);
        let pipeline = &self.gpu.pipelines.add_inplace_batch_packed;
        let bind = self.gpu.make_bind_group_with(
            pipeline,
            vec![a_buf, b_buf, params_buf.as_entire_binding()],
        );
        let total = (n * n_tokens / 2) as u32;
        let groups = (total + 255) / 256;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.add_packed.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }

    /// In-pass variant. See `dispatch_add_into`.
    pub fn dispatch_add_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        a_buf: wgpu::BindingResource<'_>,
        b_buf: wgpu::BindingResource<'_>,
        n: usize,
        n_tokens: usize,
    ) {
        let params = AddInplaceBatchParams { n: n as u32, n_tokens: n_tokens as u32 };
        let params_buf = self.gpu.create_params_buffer(&params);

        let pipeline = &self.gpu.pipelines.add_inplace_batch;
        let bind = self.gpu.make_bind_group_with(
            pipeline,
            vec![a_buf, b_buf, params_buf.as_entire_binding()],
        );

        // Phase B: `a` is packed f16 (2 per u32). Dispatch one thread
        // per u32, processing 2 elements per thread.
        let total = (n * n_tokens / 2) as u32;
        let groups = (total + 255) / 256;

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }

    /// Broadcast in-place add: every row of `a` ([n_tokens, n]) gets the
    /// same `delta` ([n]) added. Used by injection-phase shims (#6c) —
    /// one [embed_dim] delta is computed per request and applied at the
    /// chosen layer's entrance during every forward step (n_tokens=1
    /// during decode, prompt_len during prefill).
    pub fn dispatch_add_broadcast_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a_buf: wgpu::BindingResource<'_>,
        delta_buf: &wgpu::Buffer,
        n: usize,
        n_tokens: usize,
    ) {
        let params = AddInplaceBatchParams { n: n as u32, n_tokens: n_tokens as u32 };
        let params_buf = self.gpu.create_params_buffer(&params);
        let pipeline = &self.gpu.pipelines.add_broadcast_batch;
        let bind = self.gpu.make_bind_group_with(
            pipeline,
            vec![a_buf, delta_buf.as_entire_binding(), params_buf.as_entire_binding()],
        );
        let total = (n * n_tokens / 2) as u32;
        let groups = (total + 255) / 256;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.add_broadcast.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }

    /// In-pass variant. See `dispatch_add_broadcast_into`.
    pub fn dispatch_add_broadcast_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        a_buf: wgpu::BindingResource<'_>,
        delta_buf: &wgpu::Buffer,
        n: usize,
        n_tokens: usize,
    ) {
        let params = AddInplaceBatchParams { n: n as u32, n_tokens: n_tokens as u32 };
        let params_buf = self.gpu.create_params_buffer(&params);

        let pipeline = &self.gpu.pipelines.add_broadcast_batch;
        let bind = self.gpu.make_bind_group_with(
            pipeline,
            vec![
                a_buf,
                delta_buf.as_entire_binding(),
                params_buf.as_entire_binding(),
            ],
        );

        // Phase B: `a` is packed f16 (2 per u32). Halve dispatch count.
        let total = (n * n_tokens / 2) as u32;
        let groups = (total + 255) / 256;

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }

    /// Upload an f32 slice to a GPU storage buffer. Used by callers
    /// (cortex-cloud) to materialize injection-phase hidden_delta
    /// vectors as resident buffers before threading them through
    /// `forward_full_gpu_with_cache_inject_returning_hidden`.
    pub fn upload_f32_to_storage(&self, data: &[f32], label: &str) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        })
    }

}
