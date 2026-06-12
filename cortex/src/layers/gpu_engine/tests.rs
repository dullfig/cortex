    use super::*;
    use crate::layers::attention::MultiHeadAttention;
    use crate::layers::ffn::FeedForward;
    use crate::layers::floatlinear::FloatLinear;
    use crate::layers::model::OutputProjection;
    use crate::layers::rmsnorm::RmsNorm;
    use crate::layers::swiglu::SwiGLU;
    use crate::layers::transformer::TransformerBlock;
    use crate::tensor::FloatTensor;

    /// Build a FloatLinear from raw i8 ternary-style values × scale. The
    /// original tests used BitLinear (post-2026-05-29 un-merge: gone);
    /// since i8 weights in {-1, 0, +1} dequantize exactly, FloatLinear
    /// gives bit-identical forward output to BitLinear-on-CPU here.
    fn floatlinear(values: &[i8], rows: usize, cols: usize, scale: f32) -> FloatLinear {
        let weights: Vec<f32> = values.iter().map(|&v| (v as f32) * scale).collect();
        FloatLinear::new(weights, rows, cols)
    }

    /// Build a single-block, single-head, vocab=4 toy model. Same shape as
    /// the existing `TransformerModel` tests use, just enough to exercise a
    /// real forward pass.
    fn toy_model() -> TransformerModel {
        let embed_dim = 4;
        let n_heads = 1;
        let head_dim = embed_dim;
        let intermediate = 4;
        let vocab_size = 4;

        let embedding = FloatTensor::new(
            (0..vocab_size * embed_dim).map(|i| (i as f32) * 0.1).collect(),
            vec![vocab_size, embed_dim],
        );

        let q = floatlinear(&[1, 0, -1, 0, 0, 1, 0, -1, 1, 1, 0, 0, 0, 0, 1, -1], embed_dim, embed_dim, 0.1);
        let k = floatlinear(&[0, 1, 0, -1, 1, 0, -1, 0, 0, -1, 1, 0, -1, 0, 0, 1], embed_dim, embed_dim, 0.1);
        let v = floatlinear(&[1, -1, 1, -1, 0, 1, 0, 1, -1, 0, 1, 0, 0, 0, -1, 1], embed_dim, embed_dim, 0.1);
        let o = floatlinear(&[1, 0, 0, 1, 0, 1, 1, 0, -1, 0, 1, 0, 0, -1, 0, 1], embed_dim, embed_dim, 0.1);

        let attention = MultiHeadAttention::new(
            Box::new(q), Box::new(k), Box::new(v), Box::new(o),
            n_heads, n_heads, head_dim, 10000.0,
        );

        let gate = floatlinear(&[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], intermediate, embed_dim, 0.1);
        let up = floatlinear(&[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0], intermediate, embed_dim, 0.1);
        let down = floatlinear(&[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], embed_dim, intermediate, 0.1);
        let ffn: Box<dyn FeedForward> = Box::new(SwiGLU::new(Box::new(gate), Box::new(up), Box::new(down)));

        let attn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let ffn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let block = TransformerBlock::new(attn_norm, attention, ffn_norm, ffn);

        let final_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        TransformerModel::new(embedding, vec![block], final_norm, OutputProjection::TiedEmbedding)
    }

    #[test]
    fn wrapper_forward_matches_cpu() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let cpu = toy_model();
        let cpu_logits = cpu.forward(&[0, 1, 2], 0);

        let engine = GpuEngine::from_cpu_model(toy_model(), gpu);
        let gpu_logits = engine.forward(&[0, 1, 2], 0);

        assert_eq!(cpu_logits.len(), gpu_logits.len());
        for (i, (a, b)) in cpu_logits.iter().zip(&gpu_logits).enumerate() {
            assert!((a - b).abs() < 1e-6, "logit {i}: cpu={a} gpu={b}");
        }
    }

    #[test]
    fn wrapper_forward_traced_matches_cpu() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let cpu = toy_model();
        let (cpu_logits, cpu_trace) = cpu.forward_traced(&[1, 2]);

        let engine = GpuEngine::from_cpu_model(toy_model(), gpu);
        let (gpu_logits, gpu_trace) = engine.forward_traced(&[1, 2]);

        assert_eq!(cpu_logits, gpu_logits);
        assert_eq!(cpu_trace.n_layers, gpu_trace.n_layers);
        assert_eq!(cpu_trace.hidden_states.len(), gpu_trace.hidden_states.len());
    }

    #[test]
    fn forward_gpu_matches_cpu() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let cpu = toy_model();
        let cpu_logits = cpu.forward(&[0, 1, 2], 0);

        let engine = GpuEngine::from_cpu_model(toy_model(), gpu);
        let gpu_logits = engine.forward_gpu(&[0, 1, 2], 0);

        assert_eq!(cpu_logits.len(), gpu_logits.len());
        // f32 rmsnorm on GPU vs CPU — same precision, very tight tolerance.
        for (i, (a, b)) in cpu_logits.iter().zip(&gpu_logits).enumerate() {
            assert!((a - b).abs() < 1e-4, "logit {i}: cpu={a} gpu={b}");
        }
    }

    #[test]
    fn forward_gpu_single_token() {
        // Smoke-test the seq_len=1 dispatch path (one workgroup).
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let cpu_logits = toy_model().forward(&[3], 0);
        let engine = GpuEngine::from_cpu_model(toy_model(), gpu);
        let gpu_logits = engine.forward_gpu(&[3], 0);

        for (i, (a, b)) in cpu_logits.iter().zip(&gpu_logits).enumerate() {
            assert!((a - b).abs() < 1e-4, "logit {i}: cpu={a} gpu={b}");
        }
    }

    #[test]
    fn dispatch_rope_matches_cpu_rope() {
        use crate::layers::rope::{RoPE, RoPELayout};

        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let n_heads = 2;
        let head_dim = 8;
        let n_tokens = 3;
        let start_pos = 0;
        let max_seq = 16;

        let rope = RoPE::with_layout(head_dim, 10000.0, RoPELayout::Halved);

        // Build a deterministic input: [n_tokens, n_heads, head_dim] as f32.
        let mut x: Vec<f32> = Vec::with_capacity(n_tokens * n_heads * head_dim);
        for i in 0..(n_tokens * n_heads * head_dim) {
            x.push((i as f32) * 0.01 - 0.5);
        }

        // CPU reference: rotate each (tok, head) head_dim slice at pos start_pos+tok.
        let mut cpu_out = x.clone();
        for tok in 0..n_tokens {
            for h in 0..n_heads {
                let off = tok * n_heads * head_dim + h * head_dim;
                let slice = &x[off..off + head_dim].to_vec();
                let rotated = rope.forward(slice, start_pos + tok);
                cpu_out[off..off + head_dim].copy_from_slice(&rotated);
            }
        }

        // GPU path
        let (cos_buf, sin_buf) = GpuEngine::build_rope_tables(&gpu, rope.inv_freq(), max_seq);
        let total_bytes = (x.len() * std::mem::size_of::<f32>()) as u64;
        let x_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rope_test.x"),
            contents: bytemuck::cast_slice(&x),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let staging = gpu.create_staging_buffer(total_bytes);

        let engine = GpuEngine::from_cpu_model(toy_model(), gpu.clone());
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rope_test.encoder"),
        });
        engine.dispatch_rope_into(
            &mut encoder, &x_buf, &cos_buf, &sin_buf,
            n_heads, head_dim, start_pos, n_tokens,
        );
        encoder.copy_buffer_to_buffer(&x_buf, 0, &staging, 0, total_bytes);
        gpu.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let gpu_out: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        drop(data);
        staging.unmap();

        assert_eq!(cpu_out.len(), gpu_out.len());
        for (i, (c, g)) in cpu_out.iter().zip(&gpu_out).enumerate() {
            assert!(
                (c - g).abs() < 1e-5,
                "elem {i}: cpu={c} gpu={g} (diff={})",
                (c - g).abs()
            );
        }
    }

    #[test]
    fn dispatch_rope_nonzero_start_pos() {
        // Smoke-test that start_pos plumbs through to the right cos/sin row.
        use crate::layers::rope::{RoPE, RoPELayout};
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let n_heads = 1;
        let head_dim = 4;
        let n_tokens = 2;
        let start_pos = 5;
        let max_seq = 16;

        let rope = RoPE::with_layout(head_dim, 10000.0, RoPELayout::Halved);
        let x: Vec<f32> = (0..(n_tokens * n_heads * head_dim))
            .map(|i| (i as f32) * 0.1)
            .collect();

        let mut cpu_out = x.clone();
        for tok in 0..n_tokens {
            for h in 0..n_heads {
                let off = tok * n_heads * head_dim + h * head_dim;
                let rotated = rope.forward(&x[off..off + head_dim].to_vec(), start_pos + tok);
                cpu_out[off..off + head_dim].copy_from_slice(&rotated);
            }
        }

        let (cos_buf, sin_buf) = GpuEngine::build_rope_tables(&gpu, rope.inv_freq(), max_seq);
        let total_bytes = (x.len() * std::mem::size_of::<f32>()) as u64;
        let x_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rope_test2.x"),
            contents: bytemuck::cast_slice(&x),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let staging = gpu.create_staging_buffer(total_bytes);
        let engine = GpuEngine::from_cpu_model(toy_model(), gpu.clone());
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rope_test2.encoder"),
        });
        engine.dispatch_rope_into(
            &mut encoder, &x_buf, &cos_buf, &sin_buf,
            n_heads, head_dim, start_pos, n_tokens,
        );
        encoder.copy_buffer_to_buffer(&x_buf, 0, &staging, 0, total_bytes);
        gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let gpu_out: Vec<f32> = data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        drop(data); staging.unmap();
        for (i, (c, g)) in cpu_out.iter().zip(&gpu_out).enumerate() {
            assert!((c - g).abs() < 1e-5, "elem {i}: cpu={c} gpu={g}");
        }
    }

    /// CPU reference for the GPU attention chain: causal scaled dot-product
    /// attention from pre-projected, RoPE-rotated Q/K/V to per-token output.
    fn cpu_attention_reference(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_tokens: usize,
    ) -> Vec<f32> {
        let kv_dim = n_kv_heads * head_dim;
        let q_dim = n_heads * head_dim;
        let heads_per_kv = n_heads / n_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0f32; n_tokens * q_dim];

        for tok in 0..n_tokens {
            let seq_len = tok + 1;
            for h in 0..n_heads {
                let kv_h = h / heads_per_kv;
                let q_off = tok * q_dim + h * head_dim;

                let mut scores = vec![0.0f32; seq_len];
                for t in 0..seq_len {
                    let k_off = t * kv_dim + kv_h * head_dim;
                    let dot: f32 = (0..head_dim).map(|d| q[q_off + d] * k[k_off + d]).sum();
                    scores[t] = dot * scale;
                }

                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scores.iter().map(|&s| (s - max).exp()).sum();
                for s in &mut scores { *s = (*s - max).exp() / exp_sum; }

                for t in 0..seq_len {
                    let v_off = t * kv_dim + kv_h * head_dim;
                    for d in 0..head_dim {
                        out[tok * q_dim + h * head_dim + d] += scores[t] * v[v_off + d];
                    }
                }
            }
        }
        out
    }

    #[test]
    fn dispatch_attention_matches_cpu_gqa() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        // GQA: 4 query heads, 2 KV heads, head_dim=8, 5 tokens, prefill mode.
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 8;
        let n_tokens = 5;
        let max_seq = n_tokens; // prefill: K/V buffer sized to current sequence
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        // Deterministic Q/K/V values.
        let mut rng: u64 = 0xA77E_BEEFu64;
        let mut roll = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as i32 % 1000 - 500) as f32 * 0.001
        };
        let q: Vec<f32> = (0..n_tokens * q_dim).map(|_| roll()).collect();
        let k: Vec<f32> = (0..max_seq * kv_dim).map(|_| roll()).collect();
        let v: Vec<f32> = (0..max_seq * kv_dim).map(|_| roll()).collect();

        let cpu_out = cpu_attention_reference(&q, &k, &v, n_heads, n_kv_heads, head_dim, n_tokens);

        // Phase A: KV cache packed f16. Phase C3: Q is also packed f16
        // (attn_score reads packed Q) and out_buf is packed f16
        // (attn_value writes packed). Test packs inputs and unpacks
        // output on readback.
        let q_packed = GpuDevice::pack_f16(&q);
        let k_packed = GpuDevice::pack_f16(&k);
        let v_packed = GpuDevice::pack_f16(&v);

        // GPU path: upload Q/K/V (all packed), allocate scores+out scratch, dispatch.
        let q_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("attn_test.q"),
            contents: bytemuck::cast_slice(&q_packed),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let k_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("attn_test.k"),
            contents: bytemuck::cast_slice(&k_packed),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let v_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("attn_test.v"),
            contents: bytemuck::cast_slice(&v_packed),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let scores_bytes = (n_tokens * n_heads * max_seq * std::mem::size_of::<f32>()) as u64;
        let scores_buf = gpu.transient_heap_b.allocate(
            scores_bytes,
            ::vram_heap::STORAGE_BUFFER_OFFSET_ALIGNMENT_NVIDIA,
            "attn_test.scores",
        ).expect("transient_heap_b capacity for attn_test.scores");
        // C3: out buffer packed (half size).
        let out_bytes = (n_tokens * q_dim * 2) as u64;
        let out_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("attn_test.out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = gpu.create_staging_buffer(out_bytes);

        let engine = GpuEngine::from_cpu_model(toy_model(), gpu.clone());
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("attn_test.encoder"),
        });
        engine.dispatch_attention_into(
            &mut encoder, &q_buf, &k_buf, &v_buf, &scores_buf, &out_buf,
            n_heads, n_kv_heads, head_dim, /*start_pos*/ 0, max_seq, n_tokens,
        );
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_bytes);
        gpu.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        // C3: output packed; unpack 2 f16 per u32.
        let mut gpu_out: Vec<f32> = Vec::with_capacity(data.len() / 2);
        for chunk in data.chunks_exact(4) {
            let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            gpu_out.push(half::f16::from_bits((bits & 0xFFFF) as u16).to_f32());
            gpu_out.push(half::f16::from_bits(((bits >> 16) & 0xFFFF) as u16).to_f32());
        }
        drop(data); staging.unmap();

        assert_eq!(cpu_out.len(), gpu_out.len());
        // Phase A: K/V stored as f16 packed. CPU reference is f32. f16
        // quantization noise on K and V flows into scores and the value
        // sum; tolerance loosens from 1e-4 to 1e-2 to absorb ~10-bit
        // mantissa rounding. Softmax in shader still runs f32.
        for (i, (c, g)) in cpu_out.iter().zip(&gpu_out).enumerate() {
            assert!(
                (c - g).abs() < 1e-2,
                "elem {i}: cpu={c} gpu={g} (diff={})",
                (c - g).abs()
            );
        }
    }

    #[test]
    fn dispatch_attention_single_token() {
        // Smoke test the seq_len=1 case (most degenerate softmax: a single
        // value -> probability 1.0).
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let n_heads = 2;
        let n_kv_heads = 1;
        let head_dim = 4;
        let n_tokens = 1;
        let max_seq = 1;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let q: Vec<f32> = (0..n_tokens * q_dim).map(|i| i as f32 * 0.1).collect();
        let k: Vec<f32> = (0..max_seq * kv_dim).map(|i| i as f32 * 0.2 - 0.5).collect();
        let v: Vec<f32> = (0..max_seq * kv_dim).map(|i| i as f32 * -0.05 + 0.3).collect();

        let cpu_out = cpu_attention_reference(&q, &k, &v, n_heads, n_kv_heads, head_dim, n_tokens);

        let q_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("attn_test1.q"), contents: bytemuck::cast_slice(&q),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let k_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("attn_test1.k"), contents: bytemuck::cast_slice(&k),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let v_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("attn_test1.v"), contents: bytemuck::cast_slice(&v),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let scores_bytes = (n_tokens * n_heads * max_seq * 4) as u64;
        let scores_buf = gpu.transient_heap_b.allocate(
            scores_bytes,
            ::vram_heap::STORAGE_BUFFER_OFFSET_ALIGNMENT_NVIDIA,
            "attn_test1.scores",
        ).expect("transient_heap_b capacity for attn_test1.scores");
        let out_bytes = (n_tokens * q_dim * 4) as u64;
        let out_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("attn_test1.out"), size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = gpu.create_staging_buffer(out_bytes);

        let engine = GpuEngine::from_cpu_model(toy_model(), gpu.clone());
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("attn_test1.encoder"),
        });
        engine.dispatch_attention_into(
            &mut encoder, &q_buf, &k_buf, &v_buf, &scores_buf, &out_buf,
            n_heads, n_kv_heads, head_dim, 0, max_seq, n_tokens,
        );
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_bytes);
        gpu.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let gpu_out: Vec<f32> = data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        drop(data); staging.unmap();

        for (i, (c, g)) in cpu_out.iter().zip(&gpu_out).enumerate() {
            assert!((c - g).abs() < 1e-4, "elem {i}: cpu={c} gpu={g}");
        }
    }

    /// Build a TransformerModel where every linear layer is GpuFloatLinear,
    /// suitable for the fused-forward path. Same shape as `toy_model()`
    /// but with float weights instead of ternary, and resident on GPU.
    fn toy_float_model_for_gpu(gpu: Arc<GpuDevice>) -> TransformerModel {
        use crate::layers::attention::MultiHeadAttention;
        use crate::layers::ffn::FeedForward;
        use crate::layers::floatlinear::FloatLinear;
        use crate::layers::gpu_floatlinear::GpuFloatLinear;
        use crate::layers::model::OutputProjection;
        use crate::layers::rmsnorm::RmsNorm;
        use crate::layers::swiglu::SwiGLU;
        use crate::layers::transformer::TransformerBlock;

        let embed_dim = 4;
        let n_heads = 1;
        let head_dim = embed_dim;
        let intermediate = 4;
        let vocab_size = 4;

        // Deterministic small float weights.
        let mut rng: u64 = 0xF1A7_FACE;
        let mut roll = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as i32 % 200 - 100) as f32 * 0.01
        };
        let mk_tensor = |rows: usize, cols: usize, roll: &mut dyn FnMut() -> f32| -> FloatTensor {
            FloatTensor::new((0..rows * cols).map(|_| roll()).collect(), vec![rows, cols])
        };
        let mk_gpu = |rows: usize, cols: usize, roll: &mut dyn FnMut() -> f32| -> Box<dyn crate::layers::linear::LinearLayer> {
            Box::new(GpuFloatLinear::from_float_tensor(gpu.clone(), mk_tensor(rows, cols, roll)))
        };

        let embedding = mk_tensor(vocab_size, embed_dim, &mut roll);

        let q = mk_gpu(embed_dim, embed_dim, &mut roll);
        let k = mk_gpu(embed_dim, embed_dim, &mut roll);
        let v = mk_gpu(embed_dim, embed_dim, &mut roll);
        let o = mk_gpu(embed_dim, embed_dim, &mut roll);
        let attention = MultiHeadAttention::with_rope_layout(
            q, k, v, o, n_heads, n_heads, head_dim, 10000.0,
            crate::layers::rope::RoPELayout::Halved,
        );

        let gate = mk_gpu(intermediate, embed_dim, &mut roll);
        let up   = mk_gpu(intermediate, embed_dim, &mut roll);
        let down = mk_gpu(embed_dim, intermediate, &mut roll);
        let ffn: Box<dyn FeedForward> = Box::new(SwiGLU::new(gate, up, down));

        let attn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let ffn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let block = TransformerBlock::new(attn_norm, attention, ffn_norm, ffn);

        let final_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        // Use a separate float output projection so we don't tangle the test
        // with tied-embedding logic that runs CPU-side anyway.
        let out_proj = mk_tensor(vocab_size, embed_dim, &mut roll);
        TransformerModel::new(embedding, vec![block], final_norm, OutputProjection::Float(out_proj))
    }

    /// Build a CPU-equivalent TransformerModel sharing the same weights as
    /// `toy_float_model_for_gpu` — needed to compare GPU forward output to
    /// a CPU reference. The two models use the same `roll` seed and produce
    /// identical-tolerance outputs (subject to f16 rounding in GpuFloatLinear).
    fn toy_float_model_cpu_reference() -> TransformerModel {
        use crate::layers::attention::MultiHeadAttention;
        use crate::layers::ffn::FeedForward;
        use crate::layers::floatlinear::FloatLinear;
        use crate::layers::model::OutputProjection;
        use crate::layers::rmsnorm::RmsNorm;
        use crate::layers::swiglu::SwiGLU;
        use crate::layers::transformer::TransformerBlock;

        let embed_dim = 4;
        let n_heads = 1;
        let head_dim = embed_dim;
        let intermediate = 4;
        let vocab_size = 4;

        let mut rng: u64 = 0xF1A7_FACE;
        let mut roll = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as i32 % 200 - 100) as f32 * 0.01
        };
        let mk_tensor = |rows: usize, cols: usize, roll: &mut dyn FnMut() -> f32| -> FloatTensor {
            FloatTensor::new((0..rows * cols).map(|_| roll()).collect(), vec![rows, cols])
        };
        let mk_cpu = |rows: usize, cols: usize, roll: &mut dyn FnMut() -> f32| -> Box<dyn crate::layers::linear::LinearLayer> {
            Box::new(FloatLinear::from_float_tensor(mk_tensor(rows, cols, roll)))
        };

        let embedding = mk_tensor(vocab_size, embed_dim, &mut roll);
        let q = mk_cpu(embed_dim, embed_dim, &mut roll);
        let k = mk_cpu(embed_dim, embed_dim, &mut roll);
        let v = mk_cpu(embed_dim, embed_dim, &mut roll);
        let o = mk_cpu(embed_dim, embed_dim, &mut roll);
        let attention = MultiHeadAttention::with_rope_layout(
            q, k, v, o, n_heads, n_heads, head_dim, 10000.0,
            crate::layers::rope::RoPELayout::Halved,
        );

        let gate = mk_cpu(intermediate, embed_dim, &mut roll);
        let up   = mk_cpu(intermediate, embed_dim, &mut roll);
        let down = mk_cpu(embed_dim, intermediate, &mut roll);
        let ffn: Box<dyn FeedForward> = Box::new(SwiGLU::new(gate, up, down));

        let attn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let ffn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let block = TransformerBlock::new(attn_norm, attention, ffn_norm, ffn);

        let final_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let out_proj = mk_tensor(vocab_size, embed_dim, &mut roll);
        TransformerModel::new(embedding, vec![block], final_norm, OutputProjection::Float(out_proj))
    }

    /// Build a multi-block GPU model. `n_blocks` independent transformer
    /// blocks, each with its own freshly-rolled weights. Used to validate
    /// the block-stacking loop in `forward_full_gpu`.
    fn toy_float_multi_block_for_gpu(gpu: Arc<GpuDevice>, n_blocks: usize) -> TransformerModel {
        use crate::layers::attention::MultiHeadAttention;
        use crate::layers::ffn::FeedForward;
        use crate::layers::gpu_floatlinear::GpuFloatLinear;
        use crate::layers::model::OutputProjection;
        use crate::layers::rmsnorm::RmsNorm;
        use crate::layers::swiglu::SwiGLU;
        use crate::layers::transformer::TransformerBlock;

        let embed_dim = 4;
        let n_heads = 1;
        let head_dim = embed_dim;
        let intermediate = 4;
        let vocab_size = 4;

        let mut rng: u64 = 0xABCD_1234;
        let mut roll = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as i32 % 200 - 100) as f32 * 0.01
        };
        let mk_tensor = |rows: usize, cols: usize, roll: &mut dyn FnMut() -> f32| -> FloatTensor {
            FloatTensor::new((0..rows * cols).map(|_| roll()).collect(), vec![rows, cols])
        };
        let mk_gpu = |rows: usize, cols: usize, roll: &mut dyn FnMut() -> f32| -> Box<dyn crate::layers::linear::LinearLayer> {
            Box::new(GpuFloatLinear::from_float_tensor(gpu.clone(), mk_tensor(rows, cols, roll)))
        };

        let embedding = mk_tensor(vocab_size, embed_dim, &mut roll);

        let mut blocks = Vec::with_capacity(n_blocks);
        for _ in 0..n_blocks {
            let q = mk_gpu(embed_dim, embed_dim, &mut roll);
            let k = mk_gpu(embed_dim, embed_dim, &mut roll);
            let v = mk_gpu(embed_dim, embed_dim, &mut roll);
            let o = mk_gpu(embed_dim, embed_dim, &mut roll);
            let attention = MultiHeadAttention::with_rope_layout(
                q, k, v, o, n_heads, n_heads, head_dim, 10000.0,
                crate::layers::rope::RoPELayout::Halved,
            );
            let gate = mk_gpu(intermediate, embed_dim, &mut roll);
            let up   = mk_gpu(intermediate, embed_dim, &mut roll);
            let down = mk_gpu(embed_dim, intermediate, &mut roll);
            let ffn: Box<dyn FeedForward> = Box::new(SwiGLU::new(gate, up, down));
            let attn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
            let ffn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
            blocks.push(TransformerBlock::new(attn_norm, attention, ffn_norm, ffn));
        }

        let final_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let out_proj = mk_tensor(vocab_size, embed_dim, &mut roll);
        TransformerModel::new(embedding, blocks, final_norm, OutputProjection::Float(out_proj))
    }

    /// Polar-compatible toy: same shape as `toy_float_multi_block_for_gpu`
    /// but with `head_dim = 32` so PolarQuant has enough pairs (16) for
    /// the radius averaging to be stable. head_dim=8 lands in PolarQuant's
    /// pathologically-lossy regime (only 4 pairs → 22.5° per-pair angle
    /// noise, single-vector radius averaging) and produces directional
    /// agreement near 0.3, which doesn't usefully test correctness vs
    /// noise.
    fn toy_float_multi_block_for_gpu_polar(gpu: Arc<GpuDevice>, n_blocks: usize) -> TransformerModel {
        use crate::layers::attention::MultiHeadAttention;
        use crate::layers::ffn::FeedForward;
        use crate::layers::gpu_floatlinear::GpuFloatLinear;
        use crate::layers::model::OutputProjection;
        use crate::layers::rmsnorm::RmsNorm;
        use crate::layers::swiglu::SwiGLU;
        use crate::layers::transformer::TransformerBlock;

        let embed_dim = 32;
        let n_heads = 1;
        let head_dim = embed_dim;
        let intermediate = 32;
        let vocab_size = 4;

        let mut rng: u64 = 0xABCD_1234;
        let mut roll = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as i32 % 200 - 100) as f32 * 0.01
        };
        let mk_tensor = |rows: usize, cols: usize, roll: &mut dyn FnMut() -> f32| -> FloatTensor {
            FloatTensor::new((0..rows * cols).map(|_| roll()).collect(), vec![rows, cols])
        };
        let mk_gpu = |rows: usize, cols: usize, roll: &mut dyn FnMut() -> f32| -> Box<dyn crate::layers::linear::LinearLayer> {
            Box::new(GpuFloatLinear::from_float_tensor(gpu.clone(), mk_tensor(rows, cols, roll)))
        };

        let embedding = mk_tensor(vocab_size, embed_dim, &mut roll);

        let mut blocks = Vec::with_capacity(n_blocks);
        for _ in 0..n_blocks {
            let q = mk_gpu(embed_dim, embed_dim, &mut roll);
            let k = mk_gpu(embed_dim, embed_dim, &mut roll);
            let v = mk_gpu(embed_dim, embed_dim, &mut roll);
            let o = mk_gpu(embed_dim, embed_dim, &mut roll);
            let attention = MultiHeadAttention::with_rope_layout(
                q, k, v, o, n_heads, n_heads, head_dim, 10000.0,
                crate::layers::rope::RoPELayout::Halved,
            );
            let gate = mk_gpu(intermediate, embed_dim, &mut roll);
            let up   = mk_gpu(intermediate, embed_dim, &mut roll);
            let down = mk_gpu(embed_dim, intermediate, &mut roll);
            let ffn: Box<dyn FeedForward> = Box::new(SwiGLU::new(gate, up, down));
            let attn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
            let ffn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
            blocks.push(TransformerBlock::new(attn_norm, attention, ffn_norm, ffn));
        }
        let final_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let out_proj = mk_tensor(vocab_size, embed_dim, &mut roll);
        TransformerModel::new(embedding, blocks, final_norm, OutputProjection::Float(out_proj))
    }

    /// CPU-equivalent multi-block model with the same seeded weights.
    fn toy_float_multi_block_cpu(n_blocks: usize) -> TransformerModel {
        use crate::layers::attention::MultiHeadAttention;
        use crate::layers::ffn::FeedForward;
        use crate::layers::floatlinear::FloatLinear;
        use crate::layers::model::OutputProjection;
        use crate::layers::rmsnorm::RmsNorm;
        use crate::layers::swiglu::SwiGLU;
        use crate::layers::transformer::TransformerBlock;

        let embed_dim = 4;
        let n_heads = 1;
        let head_dim = embed_dim;
        let intermediate = 4;
        let vocab_size = 4;

        let mut rng: u64 = 0xABCD_1234;
        let mut roll = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as i32 % 200 - 100) as f32 * 0.01
        };
        let mk_tensor = |rows: usize, cols: usize, roll: &mut dyn FnMut() -> f32| -> FloatTensor {
            FloatTensor::new((0..rows * cols).map(|_| roll()).collect(), vec![rows, cols])
        };
        let mk_cpu = |rows: usize, cols: usize, roll: &mut dyn FnMut() -> f32| -> Box<dyn crate::layers::linear::LinearLayer> {
            Box::new(FloatLinear::from_float_tensor(mk_tensor(rows, cols, roll)))
        };

        let embedding = mk_tensor(vocab_size, embed_dim, &mut roll);

        let mut blocks = Vec::with_capacity(n_blocks);
        for _ in 0..n_blocks {
            let q = mk_cpu(embed_dim, embed_dim, &mut roll);
            let k = mk_cpu(embed_dim, embed_dim, &mut roll);
            let v = mk_cpu(embed_dim, embed_dim, &mut roll);
            let o = mk_cpu(embed_dim, embed_dim, &mut roll);
            let attention = MultiHeadAttention::with_rope_layout(
                q, k, v, o, n_heads, n_heads, head_dim, 10000.0,
                crate::layers::rope::RoPELayout::Halved,
            );
            let gate = mk_cpu(intermediate, embed_dim, &mut roll);
            let up   = mk_cpu(intermediate, embed_dim, &mut roll);
            let down = mk_cpu(embed_dim, intermediate, &mut roll);
            let ffn: Box<dyn FeedForward> = Box::new(SwiGLU::new(gate, up, down));
            let attn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
            let ffn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
            blocks.push(TransformerBlock::new(attn_norm, attention, ffn_norm, ffn));
        }

        let final_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let out_proj = mk_tensor(vocab_size, embed_dim, &mut roll);
        TransformerModel::new(embedding, blocks, final_norm, OutputProjection::Float(out_proj))
    }

    /// Build CPU + GPU one-block models with Q/K/V biases set, sharing the
    /// same seeded weights (deterministic RNG state). Used to validate the
    /// bias dispatch path lights up correctly.
    fn toy_with_biases_pair(gpu: Arc<GpuDevice>) -> (TransformerModel, TransformerModel) {
        let cpu = build_toy_with_biases(None);
        let gpu_model = build_toy_with_biases(Some(gpu));
        (cpu, gpu_model)
    }

    /// One-block model with biases. If `gpu` is Some, layers are GpuFloatLinear;
    /// otherwise CPU FloatLinear. Same RNG seed → identical weights.
    fn build_toy_with_biases(gpu: Option<Arc<GpuDevice>>) -> TransformerModel {
        use crate::layers::attention::MultiHeadAttention;
        use crate::layers::ffn::FeedForward;
        use crate::layers::floatlinear::FloatLinear;
        use crate::layers::gpu_floatlinear::GpuFloatLinear;
        use crate::layers::model::OutputProjection;
        use crate::layers::rmsnorm::RmsNorm;
        use crate::layers::swiglu::SwiGLU;
        use crate::layers::transformer::TransformerBlock;

        let embed_dim = 4;
        let n_heads = 1;
        let head_dim = embed_dim;
        let intermediate = 4;
        let vocab_size = 4;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_heads * head_dim;

        let mut rng: u64 = 0xB1A5_C0DE;
        let mut roll = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as i32 % 200 - 100) as f32 * 0.01
        };

        let mk_tensor = |rows: usize, cols: usize, roll: &mut dyn FnMut() -> f32| -> FloatTensor {
            FloatTensor::new((0..rows * cols).map(|_| roll()).collect(), vec![rows, cols])
        };
        let mk_layer = |rows: usize, cols: usize, roll: &mut dyn FnMut() -> f32|
            -> Box<dyn crate::layers::linear::LinearLayer> {
            let t = mk_tensor(rows, cols, roll);
            match &gpu {
                Some(g) => Box::new(GpuFloatLinear::from_float_tensor(g.clone(), t)),
                None    => Box::new(FloatLinear::from_float_tensor(t)),
            }
        };

        let embedding = mk_tensor(vocab_size, embed_dim, &mut roll);
        let q = mk_layer(embed_dim, embed_dim, &mut roll);
        let k = mk_layer(embed_dim, embed_dim, &mut roll);
        let v = mk_layer(embed_dim, embed_dim, &mut roll);
        let o = mk_layer(embed_dim, embed_dim, &mut roll);
        let mut attention = MultiHeadAttention::with_rope_layout(
            q, k, v, o, n_heads, n_heads, head_dim, 10000.0,
            crate::layers::rope::RoPELayout::Halved,
        );
        let q_bias: Vec<f32> = (0..q_dim).map(|_| roll()).collect();
        let k_bias: Vec<f32> = (0..kv_dim).map(|_| roll()).collect();
        let v_bias: Vec<f32> = (0..kv_dim).map(|_| roll()).collect();
        attention.set_biases(q_bias, k_bias, v_bias);

        let gate = mk_layer(intermediate, embed_dim, &mut roll);
        let up   = mk_layer(intermediate, embed_dim, &mut roll);
        let down = mk_layer(embed_dim, intermediate, &mut roll);
        let ffn: Box<dyn FeedForward> = Box::new(SwiGLU::new(gate, up, down));
        let attn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let ffn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let block = TransformerBlock::new(attn_norm, attention, ffn_norm, ffn);
        let final_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let out_proj = mk_tensor(vocab_size, embed_dim, &mut roll);
        TransformerModel::new(embedding, vec![block], final_norm, OutputProjection::Float(out_proj))
    }

    #[test]
    fn forward_full_gpu_matches_cpu_with_biases() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let (cpu_ref, gpu_model) = toy_with_biases_pair(gpu.clone());
        let engine = GpuEngine::with_max_seq(gpu_model, gpu, 16);

        let tokens = vec![0u32, 1, 2];
        let cpu_logits = cpu_ref.forward(&tokens, 0);
        let gpu_logits = engine.forward_full_gpu(&tokens, 0);

        assert_eq!(cpu_logits.len(), gpu_logits.len());
        for (i, (c, g)) in cpu_logits.iter().zip(&gpu_logits).enumerate() {
            assert!(
                (c - g).abs() < 0.05,
                "logit {i}: cpu={c} gpu={g} (diff={})",
                (c - g).abs()
            );
        }
    }

    #[test]
    fn forward_full_gpu_traced_matches_cpu_traced() {
        // Validates that the captured pre-softmax scores match what CPU
        // forward_traced would produce. Same toy model in both paths
        // (CPU FloatLinear vs GPU GpuFloatLinear, identical seeded weights).
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let cpu_ref = toy_float_multi_block_cpu(2);
        let gpu_model = toy_float_multi_block_for_gpu(gpu.clone(), 2);
        let engine = GpuEngine::with_max_seq(gpu_model, gpu.clone(), 16);

        let tokens = vec![1u32, 2, 3];
        let n_tokens = tokens.len();
        let n_layers = 2;
        let n_heads = 1;

        let (cpu_logits, cpu_trace) = cpu_ref.forward_traced(&tokens);
        let (gpu_logits, gpu_per_layer) =
            engine.forward_full_gpu_traced(&tokens, 0, &(0..n_layers).collect::<Vec<_>>());

        // Logits should match within tolerance (same as forward_full_gpu test).
        assert_eq!(cpu_logits.len(), gpu_logits.len());
        for (i, (c, g)) in cpu_logits.iter().zip(&gpu_logits).enumerate() {
            assert!((c - g).abs() < 0.1, "logit {i}: cpu={c} gpu={g}");
        }

        // Pre-softmax scores: per-layer comparison.
        // CPU layout: [n_heads, seq_len, seq_len] flat; row(layer, h, q) returns &[seq_len].
        // GPU layout: [n_tokens, n_heads, n_tokens] flat (matches attn_score_batch shader).
        // These are different orderings — we have to index carefully.
        for layer in 0..n_layers {
            let gpu_scores = &gpu_per_layer[layer];
            assert_eq!(gpu_scores.len(), n_tokens * n_heads * n_tokens);
            for h in 0..n_heads {
                for q in 0..n_tokens {
                    let cpu_row = cpu_trace.pre_score_row(layer, h, q);
                    // Causal: only positions 0..=q are real; rest -inf -> filtered by callers.
                    for k in 0..=q {
                        let gpu_idx = q * n_heads * n_tokens + h * n_tokens + k;
                        let gpu_v = gpu_scores[gpu_idx];
                        let cpu_v = cpu_row[k];
                        assert!(
                            (cpu_v - gpu_v).abs() < 0.05,
                            "layer={layer} h={h} q={q} k={k}: cpu={cpu_v} gpu={gpu_v}",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn forward_with_cache_prefill_matches_no_cache() {
        // Cached forward starting from an EMPTY cache should be identical
        // to plain forward_full_gpu — both run the same dispatches over
        // start_pos=0, n_tokens=N. Validates the kv_write + cache-read path.
        use crate::layers::gpu_kv_cache::GpuKvCache;

        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let gpu_model = toy_float_multi_block_for_gpu(gpu.clone(), 2);
        let attn0 = gpu_model.blocks()[0].attention();
        let n_layers = gpu_model.n_layers();
        let n_kv_heads = attn0.n_kv_heads();
        let head_dim = attn0.head_dim();

        let engine = GpuEngine::with_max_seq(gpu_model, gpu.clone(), 16);

        let tokens = vec![1u32, 2, 3];
        let no_cache_logits = engine.forward_full_gpu(&tokens, 0);

        let mut cache = GpuKvCache::new(gpu, n_layers, n_kv_heads, head_dim, 16);
        let cached_logits = engine.forward_full_gpu_with_cache(&tokens, &mut cache);

        assert_eq!(cache.seq_len(), tokens.len());
        assert_eq!(no_cache_logits.len(), cached_logits.len());
        for (i, (a, b)) in no_cache_logits.iter().zip(&cached_logits).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "logit {i}: no_cache={a} cached={b} (diff={})",
                (a - b).abs()
            );
        }
    }

    #[test]
    fn forward_traced_with_cache_matches_fresh_traced() {
        // Compare:
        //   A) prefill shard tokens into cache, then forward_full_gpu_with_cache_traced
        //      on the query tokens with that cache
        //   B) forward_full_gpu_traced over [shard_tokens; query_tokens] from
        //      scratch (the path retrieve uses today)
        //
        // The pre-softmax scores at the QUERY positions should match.
        use crate::layers::gpu_kv_cache::GpuKvCache;

        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let gpu_model = toy_float_multi_block_for_gpu(gpu.clone(), 2);
        let attn0 = gpu_model.blocks()[0].attention();
        let n_layers = gpu_model.n_layers();
        let n_heads = attn0.n_heads();
        let n_kv_heads = attn0.n_kv_heads();
        let head_dim = attn0.head_dim();

        let engine = GpuEngine::with_max_seq(gpu_model, gpu.clone(), 16);

        let shard_tokens = vec![1u32, 2, 3];
        let query_tokens = vec![0u32, 1];
        let all: Vec<u32> = shard_tokens.iter().chain(query_tokens.iter()).copied().collect();

        let capture_layers: Vec<usize> = (0..n_layers).collect();

        // Path A: cached
        let mut cache = GpuKvCache::new(gpu, n_layers, n_kv_heads, head_dim, 16);
        let _ = engine.forward_full_gpu_with_cache(&shard_tokens, &mut cache);
        assert_eq!(cache.seq_len(), shard_tokens.len());
        let cached_scores = engine.forward_full_gpu_with_cache_traced(
            &query_tokens, &cache, &capture_layers,
        );

        // Path B: fresh
        let (_, fresh_scores) = engine.forward_full_gpu_traced(&all, 0, &capture_layers);

        // Compare scores at query positions. Cached layout is
        // [n_query=2, n_heads, max_seq=5]; fresh layout is
        // [n_total=5, n_heads, max_seq=5]. For query token q in [0, 2),
        // global position is shard.len() + q. Compare row-by-row.
        let n_query = query_tokens.len();
        let max_seq = shard_tokens.len() + query_tokens.len();
        let n_total = all.len();
        for layer in 0..n_layers {
            for q in 0..n_query {
                for h in 0..n_heads {
                    let global_pos = shard_tokens.len() + q;
                    for k in 0..=global_pos { // causal
                        let cached_idx = q * n_heads * max_seq + h * max_seq + k;
                        let fresh_idx = global_pos * n_heads * n_total + h * n_total + k;
                        let c = cached_scores[layer][cached_idx];
                        let f = fresh_scores[layer][fresh_idx];
                        assert!(
                            (c - f).abs() < 0.05,
                            "layer={layer} q={q} (global {global_pos}) h={h} k={k}: cached={c} fresh={f}",
                        );
                    }
                }
            }
        }

        // Cache must NOT have advanced.
        assert_eq!(cache.seq_len(), shard_tokens.len(),
            "forward_full_gpu_with_cache_traced should not advance the cache");
    }

    /// Load-bearing phase 2c.5b test: end-to-end polar trace forward.
    ///
    /// Compare:
    ///   A) prefill shard tokens into f32 cache, then
    ///      forward_full_gpu_with_cache_traced (existing f32 path)
    ///   B) prefill shard tokens into f32 cache, convert to polar via
    ///      populate_from_f32_cache_gpu, then forward_full_gpu_polar_traced
    ///      (new polar path)
    ///
    /// Both paths run the same model on the same query against the same
    /// underlying K/V data; the only difference is the polar quantization
    /// in path B. Per-layer pre-softmax scores at query positions should
    /// agree by cosine similarity (PolarQuant is lossy, so we check
    /// directional agreement, not numeric equality). This is the
    /// load-bearing test that the polar attention chain in each block is
    /// computing what a polar-attention forward should compute.
    /// Hidden-state extraction hook: capture per-layer post-block hidden
    /// states + final post-norm hidden via `forward_full_gpu_with_hidden_capture`.
    /// Sanity-check shapes are right, all values are finite, and the
    /// last-token slicer helpers return what they should.
    #[test]
    fn hidden_capture_returns_finite_states() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let gpu_model = toy_float_multi_block_for_gpu(gpu.clone(), 2);
        let n_layers = gpu_model.n_layers();
        let embed_dim = 4; // toy model embed_dim
        let engine = GpuEngine::with_max_seq(gpu_model, gpu, 16);

        let tokens = vec![1u32, 2, 3];
        let n_tokens = tokens.len();
        let capture_layers: Vec<usize> = (0..n_layers).collect();

        let hc = engine.forward_full_gpu_with_hidden_capture(&tokens, &capture_layers);

        assert_eq!(hc.n_tokens, n_tokens);
        assert_eq!(hc.embed_dim, embed_dim);
        assert_eq!(hc.per_layer_hidden.len(), n_layers);
        for (i, layer) in hc.per_layer_hidden.iter().enumerate() {
            assert_eq!(layer.len(), n_tokens * embed_dim,
                "layer {i} hidden size");
            assert!(layer.iter().all(|v| v.is_finite()),
                "layer {i} hidden contains non-finite values");
        }
        assert_eq!(hc.final_post_norm_hidden.len(), n_tokens * embed_dim);
        assert!(hc.final_post_norm_hidden.iter().all(|v| v.is_finite()));

        // The post-FFN hidden of the LAST block is the input to the final
        // RMSNorm. After RMSNorm, magnitudes change but direction is
        // preserved. Sanity: both have nonzero norm at the last token.
        let last_layer_last_tok = hc.layer_last_token(n_layers - 1);
        let final_last_tok = hc.final_last_token();
        let mag_last_layer: f32 = last_layer_last_tok.iter().map(|x| x*x).sum::<f32>().sqrt();
        let mag_final: f32 = final_last_tok.iter().map(|x| x*x).sum::<f32>().sqrt();
        assert!(mag_last_layer > 0.0, "last-block last-token hidden should be nonzero");
        assert!(mag_final > 0.0, "final post-norm last-token hidden should be nonzero");
    }

    /// Hooks-are-optional invariant: forward without capture (capture_layers
    /// empty) should still produce a valid final post-norm hidden, and not
    /// allocate any per-layer capture buffers.
    #[test]
    fn hidden_capture_with_empty_layers_works() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let gpu_model = toy_float_multi_block_for_gpu(gpu.clone(), 2);
        let engine = GpuEngine::with_max_seq(gpu_model, gpu, 16);

        let hc = engine.forward_full_gpu_with_hidden_capture(&[1u32, 2], &[]);
        assert!(hc.per_layer_hidden.is_empty());
        assert!(hc.final_post_norm_hidden.iter().all(|v| v.is_finite()));
        assert_eq!(hc.final_post_norm_hidden.len(), 2 * 4); // n_tokens * embed_dim
    }

    #[test]
    fn polar_traced_matches_f32_traced_directionally() {
        use crate::layers::gpu_kv_cache::GpuKvCache;
        use crate::layers::gpu_polar_kv_cache::GpuPolarKvCache;

        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        // head_dim=8 polar-compat fixture (toy_float_multi_block_for_gpu's
        // head_dim=4 doesn't satisfy the polar shader's head_dim%8==0).
        let gpu_model = toy_float_multi_block_for_gpu_polar(gpu.clone(), 2);
        let attn0 = gpu_model.blocks()[0].attention();
        let n_layers = gpu_model.n_layers();
        let n_heads = attn0.n_heads();
        let n_kv_heads = attn0.n_kv_heads();
        let head_dim = attn0.head_dim();

        let engine = GpuEngine::with_max_seq(gpu_model, gpu.clone(), 16);

        let shard_tokens = vec![1u32, 2, 3];
        let query_tokens = vec![0u32, 1];
        let capture_layers: Vec<usize> = (0..n_layers).collect();

        // Path A: f32 prefill + f32 traced forward.
        let mut f32_cache = GpuKvCache::new(gpu.clone(), n_layers, n_kv_heads, head_dim, 16);
        let _ = engine.forward_full_gpu_with_cache(&shard_tokens, &mut f32_cache);
        assert_eq!(f32_cache.seq_len(), shard_tokens.len());
        let f32_scores = engine.forward_full_gpu_with_cache_traced(
            &query_tokens, &f32_cache, &capture_layers,
        );

        // Path B: convert f32→polar on GPU, then polar traced forward.
        let mut polar_cache = GpuPolarKvCache::new(
            gpu.clone(), n_layers, n_kv_heads, head_dim, /*max_seq*/ 16, /*seed_base*/ 42,
        );
        polar_cache.populate_from_f32_cache_gpu(&f32_cache);
        assert_eq!(polar_cache.seq_len(), shard_tokens.len());
        let polar_scores = engine.forward_full_gpu_polar_traced(
            &query_tokens, &polar_cache, &capture_layers,
        );

        // Cache must NOT have advanced (mirrors f32 trace semantics).
        assert_eq!(polar_cache.seq_len(), shard_tokens.len(),
            "forward_full_gpu_polar_traced should not advance the cache");

        // Per-layer cosine similarity of valid (causally-unmasked) score
        // rows. Each layer captures [n_query, n_heads, max_seq] scores;
        // for query token q, valid positions are [0, shard_len + q + 1).
        let n_query = query_tokens.len();
        let max_seq = shard_tokens.len() + n_query;
        for layer in 0..n_layers {
            for q in 0..n_query {
                for h in 0..n_heads {
                    let row_off = q * n_heads * max_seq + h * max_seq;
                    let valid_len = shard_tokens.len() + q + 1;
                    let f_row = &f32_scores[layer][row_off..row_off + valid_len];
                    let p_row = &polar_scores[layer][row_off..row_off + valid_len];

                    let dot: f32 = f_row.iter().zip(p_row).map(|(&a, &b)| a * b).sum();
                    let na: f32 = f_row.iter().map(|&x| x * x).sum::<f32>().sqrt();
                    let nb: f32 = p_row.iter().map(|&x| x * x).sum::<f32>().sqrt();
                    let cos = dot / (na * nb).max(1e-12);

                    // PolarQuant is lossy; agreement is directional, not
                    // exact. Worst rows in this fixture land around 0.5 —
                    // pin a 0.4 floor to catch sign-flips / dimension
                    // shuffles, the failure modes that actually matter.
                    assert!(
                        cos > 0.4,
                        "layer {layer} q {q} h {h}: cos={cos:.4} below 0.4 floor",
                    );
                }
            }
        }
    }

    #[test]
    fn forward_with_cache_decode_matches_full_forward() {
        // Validates the decode path: prefill [1,2,3] then decode [4,5]
        // through the cache should produce logits matching positions 3
        // and 4 of forward_full_gpu([1,2,3,4,5]).
        use crate::layers::gpu_kv_cache::GpuKvCache;

        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let gpu_model = toy_float_multi_block_for_gpu(gpu.clone(), 2);
        let attn0 = gpu_model.blocks()[0].attention();
        let n_layers = gpu_model.n_layers();
        let n_kv_heads = attn0.n_kv_heads();
        let head_dim = attn0.head_dim();
        let vocab_size = gpu_model.vocab_size();

        let engine = GpuEngine::with_max_seq(gpu_model, gpu.clone(), 16);

        let prefill = vec![1u32, 2, 3];
        let decode = vec![0u32, 1];
        let all: Vec<u32> = prefill.iter().chain(decode.iter()).copied().collect();

        // Reference: single big forward over [1,2,3,0,1].
        let ref_logits = engine.forward_full_gpu(&all, 0);

        // Cached path: prefill then decode.
        let mut cache = GpuKvCache::new(gpu, n_layers, n_kv_heads, head_dim, 16);
        let _ = engine.forward_full_gpu_with_cache(&prefill, &mut cache);
        assert_eq!(cache.seq_len(), 3);
        let decode_logits = engine.forward_full_gpu_with_cache(&decode, &mut cache);
        assert_eq!(cache.seq_len(), 5);

        // decode_logits is logits for the 2 decode tokens (positions 3 and 4
        // of the full sequence). They should match the corresponding rows
        // of ref_logits.
        for tok_idx in 0..decode.len() {
            let global_pos = prefill.len() + tok_idx;
            let ref_row = &ref_logits[global_pos * vocab_size..(global_pos + 1) * vocab_size];
            let decode_row = &decode_logits[tok_idx * vocab_size..(tok_idx + 1) * vocab_size];
            for (i, (r, d)) in ref_row.iter().zip(decode_row).enumerate() {
                assert!(
                    (r - d).abs() < 1e-2,
                    "tok {tok_idx} (global pos {global_pos}) logit {i}: ref={r} decoded={d}",
                );
            }
        }
    }

    #[test]
    fn forward_full_gpu_traced_capture_subset() {
        // Capturing only a subset of layers: check we get back exactly
        // those layers' worth of buffers.
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let gpu_model = toy_float_multi_block_for_gpu(gpu.clone(), 4);
        let engine = GpuEngine::with_max_seq(gpu_model, gpu, 16);

        let tokens = vec![0u32, 1, 2];
        // Capture only layers 1 and 3.
        let (_logits, per_layer) = engine.forward_full_gpu_traced(&tokens, 0, &[1, 3]);
        assert_eq!(per_layer.len(), 2);
        // Each capture is [n_tokens=3, n_heads=1, n_tokens=3] = 9 floats.
        assert_eq!(per_layer[0].len(), 9);
        assert_eq!(per_layer[1].len(), 9);
    }

    #[test]
    fn forward_full_gpu_matches_cpu_single_block() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let cpu_ref = toy_float_multi_block_cpu(1);
        let gpu_model = toy_float_multi_block_for_gpu(gpu.clone(), 1);
        let engine = GpuEngine::with_max_seq(gpu_model, gpu.clone(), 16);

        let tokens = vec![0u32, 1, 2, 3];
        let cpu_logits = cpu_ref.forward(&tokens, 0);
        let gpu_logits = engine.forward_full_gpu(&tokens, 0);

        assert_eq!(cpu_logits.len(), gpu_logits.len());
        for (i, (c, g)) in cpu_logits.iter().zip(&gpu_logits).enumerate() {
            assert!(
                (c - g).abs() < 0.05,
                "logit {i}: cpu={c} gpu={g} (diff={})",
                (c - g).abs()
            );
        }
    }

    #[test]
    fn forward_full_gpu_thirty_blocks_no_crash() {
        // Reproducer for the validation error that hits the bench at ~block
        // 26 with Qwen 3B. Just runs forward without comparing to CPU
        // (32-block deep f16 chains diverge precision-wise; this test only
        // cares about "does it complete without a wgpu validation panic").
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let gpu_model = toy_float_multi_block_for_gpu(gpu.clone(), 30);
        let engine = GpuEngine::with_max_seq(gpu_model, gpu.clone(), 16);

        let tokens = vec![0u32, 1, 2];
        let _ = engine.forward_full_gpu(&tokens, 0); // must not panic
    }

    #[test]
    fn forward_full_gpu_fifty_blocks_no_crash() {
        // Bisection probe for the Qwen failure: if dispatch count is the
        // culprit (~494 dispatches at the Qwen failure point), 50 toy
        // blocks × 16 dispatches = 800 dispatches should reproduce.
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let gpu_model = toy_float_multi_block_for_gpu(gpu.clone(), 50);
        let engine = GpuEngine::with_max_seq(gpu_model, gpu.clone(), 16);

        let tokens = vec![0u32, 1, 2];
        let _ = engine.forward_full_gpu(&tokens, 0);
    }

    /// Multi-block toy model WITH Q/K/V biases set.
    fn toy_float_multi_block_with_biases_for_gpu(
        gpu: Arc<GpuDevice>, n_blocks: usize,
    ) -> TransformerModel {
        use crate::layers::attention::MultiHeadAttention;
        use crate::layers::ffn::FeedForward;
        use crate::layers::gpu_floatlinear::GpuFloatLinear;
        use crate::layers::model::OutputProjection;
        use crate::layers::rmsnorm::RmsNorm;
        use crate::layers::swiglu::SwiGLU;
        use crate::layers::transformer::TransformerBlock;

        let embed_dim = 4;
        let n_heads = 1;
        let head_dim = embed_dim;
        let intermediate = 4;
        let vocab_size = 4;
        let q_dim = n_heads * head_dim;

        let mut rng: u64 = 0xBEAD_BEAD;
        let mut roll = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as i32 % 200 - 100) as f32 * 0.01
        };
        let mk_tensor = |rows: usize, cols: usize, roll: &mut dyn FnMut() -> f32| -> FloatTensor {
            FloatTensor::new((0..rows * cols).map(|_| roll()).collect(), vec![rows, cols])
        };
        let mk_gpu = |rows: usize, cols: usize, roll: &mut dyn FnMut() -> f32| -> Box<dyn crate::layers::linear::LinearLayer> {
            Box::new(GpuFloatLinear::from_float_tensor(gpu.clone(), mk_tensor(rows, cols, roll)))
        };

        let embedding = mk_tensor(vocab_size, embed_dim, &mut roll);
        let mut blocks = Vec::with_capacity(n_blocks);
        for _ in 0..n_blocks {
            let q = mk_gpu(embed_dim, embed_dim, &mut roll);
            let k = mk_gpu(embed_dim, embed_dim, &mut roll);
            let v = mk_gpu(embed_dim, embed_dim, &mut roll);
            let o = mk_gpu(embed_dim, embed_dim, &mut roll);
            let mut attention = MultiHeadAttention::with_rope_layout(
                q, k, v, o, n_heads, n_heads, head_dim, 10000.0,
                crate::layers::rope::RoPELayout::Halved,
            );
            let qb: Vec<f32> = (0..q_dim).map(|_| roll()).collect();
            let kb: Vec<f32> = (0..q_dim).map(|_| roll()).collect();
            let vb: Vec<f32> = (0..q_dim).map(|_| roll()).collect();
            attention.set_biases(qb, kb, vb);

            let gate = mk_gpu(intermediate, embed_dim, &mut roll);
            let up   = mk_gpu(intermediate, embed_dim, &mut roll);
            let down = mk_gpu(embed_dim, intermediate, &mut roll);
            let ffn: Box<dyn FeedForward> = Box::new(SwiGLU::new(gate, up, down));
            let attn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
            let ffn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
            blocks.push(TransformerBlock::new(attn_norm, attention, ffn_norm, ffn));
        }
        let final_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let out_proj = mk_tensor(vocab_size, embed_dim, &mut roll);
        TransformerModel::new(embedding, blocks, final_norm, OutputProjection::Float(out_proj))
    }

    /// Larger-tensor multi-block model. n_heads != n_kv_heads (GQA).
    fn toy_gqa_multi_block_for_gpu(
        gpu: Arc<GpuDevice>, n_blocks: usize,
        embed_dim: usize, intermediate: usize,
        n_heads: usize, n_kv_heads: usize,
    ) -> TransformerModel {
        toy_gqa_multi_block_for_gpu_inner(gpu, n_blocks, embed_dim, intermediate, n_heads, n_kv_heads, /*with_biases*/ false)
    }

    fn toy_gqa_multi_block_for_gpu_inner(
        gpu: Arc<GpuDevice>, n_blocks: usize,
        embed_dim: usize, intermediate: usize,
        n_heads: usize, n_kv_heads: usize,
        with_biases: bool,
    ) -> TransformerModel {
        use crate::layers::attention::MultiHeadAttention;
        use crate::layers::ffn::FeedForward;
        use crate::layers::gpu_floatlinear::GpuFloatLinear;
        use crate::layers::model::OutputProjection;
        use crate::layers::rmsnorm::RmsNorm;
        use crate::layers::swiglu::SwiGLU;
        use crate::layers::transformer::TransformerBlock;

        let head_dim = embed_dim / n_heads;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let vocab_size = 32;

        let mut rng: u64 = 0xCAFE_F00D;
        let mut roll = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as i32 % 200 - 100) as f32 * 0.001
        };
        let mk_tensor = |rows: usize, cols: usize, roll: &mut dyn FnMut() -> f32| -> FloatTensor {
            FloatTensor::new((0..rows * cols).map(|_| roll()).collect(), vec![rows, cols])
        };
        let mk_gpu = |rows: usize, cols: usize, roll: &mut dyn FnMut() -> f32| -> Box<dyn crate::layers::linear::LinearLayer> {
            Box::new(GpuFloatLinear::from_float_tensor(gpu.clone(), mk_tensor(rows, cols, roll)))
        };

        let embedding = mk_tensor(vocab_size, embed_dim, &mut roll);
        let mut blocks = Vec::with_capacity(n_blocks);
        for _ in 0..n_blocks {
            let q = mk_gpu(q_dim, embed_dim, &mut roll);
            let k = mk_gpu(kv_dim, embed_dim, &mut roll);
            let v = mk_gpu(kv_dim, embed_dim, &mut roll);
            let o = mk_gpu(embed_dim, q_dim, &mut roll);
            let mut attention = MultiHeadAttention::with_rope_layout(
                q, k, v, o, n_heads, n_kv_heads, head_dim, 10000.0,
                crate::layers::rope::RoPELayout::Halved,
            );
            if with_biases {
                let qb: Vec<f32> = (0..q_dim).map(|_| roll()).collect();
                let kb: Vec<f32> = (0..kv_dim).map(|_| roll()).collect();
                let vb: Vec<f32> = (0..kv_dim).map(|_| roll()).collect();
                attention.set_biases(qb, kb, vb);
            }
            let gate = mk_gpu(intermediate, embed_dim, &mut roll);
            let up   = mk_gpu(intermediate, embed_dim, &mut roll);
            let down = mk_gpu(embed_dim, intermediate, &mut roll);
            let ffn: Box<dyn FeedForward> = Box::new(SwiGLU::new(gate, up, down));
            let attn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
            let ffn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
            blocks.push(TransformerBlock::new(attn_norm, attention, ffn_norm, ffn));
        }
        let final_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let out_proj = mk_tensor(vocab_size, embed_dim, &mut roll);
        TransformerModel::new(embedding, blocks, final_norm, OutputProjection::Float(out_proj))
    }

    #[test]
    fn forward_full_gpu_qwen_shaped_with_biases_no_crash() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let gpu_model = toy_gqa_multi_block_for_gpu_inner(
            gpu.clone(), 36,
            /*embed*/ 2048, /*intermediate*/ 11008,
            /*n_heads*/ 16, /*n_kv_heads*/ 2,
            /*with_biases*/ true,
        );
        let engine = GpuEngine::with_max_seq(gpu_model, gpu.clone(), 16);
        let tokens = vec![0u32, 1, 2];
        let _ = engine.forward_full_gpu(&tokens, 0);
    }

    #[test]
    #[ignore = "loads Qwen 3B from disk; run with --ignored on workstation"]
    fn forward_full_gpu_real_qwen3b_no_crash() {
        // Loads the actual Qwen 2.5-3B Q4_K_M from disk (same path the bench
        // uses). If this crashes inside the test framework, we have a
        // standalone reproducer of the bench failure.
        let path = "C:\\Users\\danu\\AppData\\Roaming\\memory-rlm\\models\\model-qwen2.5-3b-q4km.gguf";
        if !std::path::Path::new(path).exists() { return; }

        let loaded = crate::load_model(path).expect("load_model");
        let gpu = loaded.gpu.clone().expect("model loaded without GPU");
        let engine = GpuEngine::with_max_seq(loaded.model, gpu, 4096);
        let tokens: Vec<u32> = (0..16u32).collect();
        let _ = engine.forward_full_gpu(&tokens, 0);
    }

    #[test]
    fn forward_full_gpu_qwen_shape_with_gpu_output_proj_no_crash() {
        // Final bisection: same shape as the bench, biases on, output proj
        // routed through GpuFloatLinear (matches what loader.rs does for
        // float models with GPU available). If THIS panics, we've isolated
        // the trigger to "GpuFloatLinear on the output projection".
        use crate::layers::attention::MultiHeadAttention;
        use crate::layers::ffn::FeedForward;
        use crate::layers::gpu_floatlinear::GpuFloatLinear;
        use crate::layers::model::OutputProjection;
        use crate::layers::rmsnorm::RmsNorm;
        use crate::layers::swiglu::SwiGLU;
        use crate::layers::transformer::TransformerBlock;

        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        // Smaller vocab to keep memory in check; the bug should still
        // trigger if output-proj-via-GpuFloatLinear is the cause.
        let embed_dim = 2048;
        let n_heads = 16;
        let n_kv_heads = 2;
        let head_dim = embed_dim / n_heads;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let intermediate = 11008;
        let vocab_size = 151_936; // Qwen 2.5 vocab
        let n_blocks = 36;

        let mut rng: u64 = 0xDEAD_F00D;
        let mut roll = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as i32 % 200 - 100) as f32 * 0.001
        };
        let mk_tensor = |rows: usize, cols: usize, roll: &mut dyn FnMut() -> f32| -> FloatTensor {
            FloatTensor::new((0..rows * cols).map(|_| roll()).collect(), vec![rows, cols])
        };
        let mk_gpu = |rows: usize, cols: usize, roll: &mut dyn FnMut() -> f32| -> Box<dyn crate::layers::linear::LinearLayer> {
            Box::new(GpuFloatLinear::from_float_tensor(gpu.clone(), mk_tensor(rows, cols, roll)))
        };

        let embedding = mk_tensor(vocab_size, embed_dim, &mut roll);
        let mut blocks = Vec::with_capacity(n_blocks);
        for _ in 0..n_blocks {
            let q = mk_gpu(q_dim, embed_dim, &mut roll);
            let k = mk_gpu(kv_dim, embed_dim, &mut roll);
            let v = mk_gpu(kv_dim, embed_dim, &mut roll);
            let o = mk_gpu(embed_dim, q_dim, &mut roll);
            let mut attention = MultiHeadAttention::with_rope_layout(
                q, k, v, o, n_heads, n_kv_heads, head_dim, 10000.0,
                crate::layers::rope::RoPELayout::Halved,
            );
            let qb: Vec<f32> = (0..q_dim).map(|_| roll()).collect();
            let kb: Vec<f32> = (0..kv_dim).map(|_| roll()).collect();
            let vb: Vec<f32> = (0..kv_dim).map(|_| roll()).collect();
            attention.set_biases(qb, kb, vb);
            let gate = mk_gpu(intermediate, embed_dim, &mut roll);
            let up   = mk_gpu(intermediate, embed_dim, &mut roll);
            let down = mk_gpu(embed_dim, intermediate, &mut roll);
            let ffn: Box<dyn FeedForward> = Box::new(SwiGLU::new(gate, up, down));
            let attn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
            let ffn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
            blocks.push(TransformerBlock::new(attn_norm, attention, ffn_norm, ffn));
        }
        let final_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);

        // The candidate trigger: output proj through GpuFloatLinear.
        let out_proj_gpu = mk_gpu(vocab_size, embed_dim, &mut roll);

        let model = TransformerModel::new(embedding, blocks, final_norm, OutputProjection::Linear(out_proj_gpu));
        let engine = GpuEngine::with_max_seq(model, gpu.clone(), 16);
        let tokens = vec![0u32, 1, 2];
        let _ = engine.forward_full_gpu(&tokens, 0);
    }

    #[test]
    fn forward_full_gpu_qwen_shaped_no_crash() {
        // Qwen 2.5-3B shape (embed=2048, GQA 16/2, intermediate=11008,
        // 36 blocks) — the actual case that crashes via cortex-bench-fwd.
        // If THIS panics in tests, the bug is reproducible without the
        // bench harness; if it doesn't, something about the bench's
        // execution path matters (warmup ordering, finalize_logits, etc.).
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let gpu_model = toy_gqa_multi_block_for_gpu(
            gpu.clone(), 36,
            /*embed*/ 2048, /*intermediate*/ 11008,
            /*n_heads*/ 16, /*n_kv_heads*/ 2,
        );
        let engine = GpuEngine::with_max_seq(gpu_model, gpu.clone(), 16);

        let tokens = vec![0u32, 1, 2];
        let _ = engine.forward_full_gpu(&tokens, 0);
    }

    #[test]
    fn forward_full_gpu_fifty_blocks_with_biases_no_crash() {
        // Probe: does adding biases (extra 3 dispatches per block) trigger
        // the Qwen failure with toy-size tensors?
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let gpu_model = toy_float_multi_block_with_biases_for_gpu(gpu.clone(), 50);
        let engine = GpuEngine::with_max_seq(gpu_model, gpu.clone(), 16);

        let tokens = vec![0u32, 1, 2];
        let _ = engine.forward_full_gpu(&tokens, 0);
    }

    #[test]
    fn forward_full_gpu_matches_cpu_two_blocks() {
        // Validates the block-stacking loop: that hidden state correctly
        // flows from block 0 -> block 1 inside one encoder.
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let cpu_ref = toy_float_multi_block_cpu(2);
        let gpu_model = toy_float_multi_block_for_gpu(gpu.clone(), 2);
        let engine = GpuEngine::with_max_seq(gpu_model, gpu.clone(), 16);

        let tokens = vec![1u32, 2, 3];
        let cpu_logits = cpu_ref.forward(&tokens, 0);
        let gpu_logits = engine.forward_full_gpu(&tokens, 0);

        assert_eq!(cpu_logits.len(), gpu_logits.len());
        // 2 blocks ~24 dispatches; error compounds slightly.
        for (i, (c, g)) in cpu_logits.iter().zip(&gpu_logits).enumerate() {
            assert!(
                (c - g).abs() < 0.1,
                "logit {i}: cpu={c} gpu={g} (diff={})",
                (c - g).abs()
            );
        }
    }

    #[test]
    fn forward_block_gpu_matches_cpu_block() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let cpu_ref = toy_float_model_cpu_reference();
        let n_tokens = 3;
        let tokens: Vec<u32> = (0..n_tokens as u32).collect();

        // CPU reference: embedding lookup + one block.forward.
        let embed_data = cpu_ref.embedding_data();
        let embed_dim = cpu_ref.embed_dim();
        let mut hidden_cpu: Vec<f32> = Vec::with_capacity(n_tokens * embed_dim);
        for &t in &tokens {
            let off = t as usize * embed_dim;
            hidden_cpu.extend_from_slice(&embed_data[off..off + embed_dim]);
        }
        let cpu_out = cpu_ref.blocks()[0].forward(&hidden_cpu, n_tokens, /*start_pos*/ 0);

        // GPU path
        let gpu_model = toy_float_model_for_gpu(gpu.clone());
        let engine = GpuEngine::with_max_seq(gpu_model, gpu.clone(), 16);

        // Upload the same hidden-state input. Option E: hidden_buf is f32.
        let bytes = (n_tokens * embed_dim * std::mem::size_of::<f32>()) as u64;
        let hidden_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test.hidden"),
            contents: bytemuck::cast_slice(&hidden_cpu),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let staging = gpu.create_staging_buffer(bytes);

        let attn0 = engine.cpu().blocks()[0].attention();
        let intermediate = engine.cpu().blocks()[0].ffn().out_features(); // == embed_dim
        let _ = intermediate;
        // FFN intermediate = SwiGLU.intermediate_size; access via downcast.
        let intermediate = engine.cpu().blocks()[0].ffn().as_any()
            .downcast_ref::<crate::layers::swiglu::SwiGLU>().unwrap()
            .intermediate_size();

        let scratch = BlockScratch::allocate(
            &gpu, n_tokens, embed_dim,
            attn0.n_heads(), attn0.n_kv_heads(), attn0.head_dim(),
            intermediate, /*max_seq*/ n_tokens,
        );

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test.encoder"),
        });
        engine.forward_block_gpu(&mut encoder, 0, &hidden_buf, n_tokens, 0, &scratch);
        encoder.copy_buffer_to_buffer(&hidden_buf, 0, &staging, 0, bytes);
        gpu.queue.submit(Some(encoder.finish()));

        // Option E: hidden_buf f32; read back directly.
        let gpu_out = read_back_buffer(&gpu, &staging, bytes as usize);

        assert_eq!(cpu_out.len(), gpu_out.len(), "shape mismatch");
        // Tolerance covers f16 quantization on weights and packed scratch.
        for (i, (c, g)) in cpu_out.iter().zip(&gpu_out).enumerate() {
            assert!(
                (c - g).abs() < 0.05,
                "elem {i}: cpu={c} gpu={g} (diff={})",
                (c - g).abs()
            );
        }
    }

    #[test]
    fn matmul_shared_matches_legacy() {
        // Numerical parity: shared-memory tiled matmul vs legacy matmul,
        // n_tokens=16 (above threshold) on a non-trivial shape.
        // Uses a single GpuFloatLinear's weight buffer, dispatches both
        // shaders against the same inputs, compares outputs.
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let rows = 32usize;   // output features (must be small to keep test fast)
        let cols = 64usize;   // input features (even for f16 packing)
        let n_tokens = 16usize;

        // Random weights + inputs.
        let mut rng: u64 = 0xCAFE_F00D;
        let mut roll = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as i32 % 1000 - 500) as f32 * 0.001
        };
        let weights_data: Vec<f32> = (0..rows * cols).map(|_| roll()).collect();
        let input_data: Vec<f32> = (0..n_tokens * cols).map(|_| roll()).collect();

        let weight_tensor = crate::tensor::FloatTensor::new(weights_data, vec![rows, cols]);
        let layer = crate::layers::gpu_floatlinear::GpuFloatLinear::from_float_tensor(
            gpu.clone(), weight_tensor,
        );

        // Upload input (f32; matmul shaders unchanged in Phase B scope).
        let in_bytes = (n_tokens * cols * 4) as u64;
        let in_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test.matmul.in"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let out_bytes = (n_tokens * rows * 4) as u64;
        let mk_out = || gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test.matmul.out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_legacy = mk_out();
        let out_shared = mk_out();
        let staging_legacy = gpu.create_staging_buffer(out_bytes);
        let staging_shared = gpu.create_staging_buffer(out_bytes);

        // We need a GpuEngine to dispatch through. Build a minimal one
        // by wrapping a toy CPU model — we only use its dispatch helpers.
        let cpu_model = toy_float_model_cpu_reference();
        let engine = GpuEngine::with_max_seq(cpu_model, gpu.clone(), 16);

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test.matmul.enc"),
        });

        // Legacy: explicitly call the legacy dispatcher.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("test.matmul.legacy.pass"),
                timestamp_writes: None,
            });
            engine.dispatch_matmul_legacy_inner_in_pass(&mut pass, &layer, &in_buf, &out_legacy, n_tokens);
        }
        encoder.copy_buffer_to_buffer(&out_legacy, 0, &staging_legacy, 0, out_bytes);

        // Shared: explicitly call the shared-memory tiled dispatcher.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("test.matmul.shared.pass"),
                timestamp_writes: None,
            });
            engine.dispatch_matmul_shared_inner_in_pass(&mut pass, &layer, &in_buf, &out_shared, n_tokens);
        }
        encoder.copy_buffer_to_buffer(&out_shared, 0, &staging_shared, 0, out_bytes);

        gpu.queue.submit(Some(encoder.finish()));

        let read = |staging: &wgpu::Buffer| -> Vec<f32> {
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
            gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
            rx.recv().unwrap().unwrap();
            let data = slice.get_mapped_range();
            let out: Vec<f32> = data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            drop(data); staging.unmap();
            out
        };
        let v_legacy = read(&staging_legacy);
        let v_shared = read(&staging_shared);

        assert_eq!(v_legacy.len(), v_shared.len());
        let mut max_diff = 0.0_f32;
        for (i, (a, b)) in v_legacy.iter().zip(&v_shared).enumerate() {
            let d = (a - b).abs();
            if d > max_diff { max_diff = d; }
            assert!(
                d < 1e-2,
                "elem {i}: legacy={a} shared={b} diff={d}",
            );
        }
        eprintln!("matmul_shared_matches_legacy (Phase B f16 input): rows={rows} cols={cols} n_tokens={n_tokens} max_diff={max_diff:.6e}");
    }

    #[test]
    #[ignore = "benchmark; run with --ignored to compare matmul shaders at Qwen prefill size"]
    fn matmul_shared_vs_legacy_bench_qwen_prefill() {
        // Microbenchmark: dispatch the same matmul through both legacy
        // and shared-memory shaders, time wall-clock. Shape chosen to
        // match Qwen 3B Q_proj at 500w prompt: rows=2048, cols=2048,
        // n_tokens=526. Each iteration submits + polls so total wall
        // includes the actual GPU work for that single dispatch.
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let rows = 2048usize;
        let cols = 2048usize;
        let n_tokens = 526usize;
        let iters = 5usize;

        let mut rng: u64 = 0xBEEF_FACE;
        let mut roll = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as i32 % 1000 - 500) as f32 * 0.001
        };
        let weights_data: Vec<f32> = (0..rows * cols).map(|_| roll()).collect();
        let input_data: Vec<f32> = (0..n_tokens * cols).map(|_| roll()).collect();

        let weight_tensor = crate::tensor::FloatTensor::new(weights_data, vec![rows, cols]);
        let layer = crate::layers::gpu_floatlinear::GpuFloatLinear::from_float_tensor(
            gpu.clone(), weight_tensor,
        );

        let in_bytes = (n_tokens * cols * 4) as u64;
        let in_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bench.in"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let out_bytes = (n_tokens * rows * 4) as u64;
        let out_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bench.out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cpu_model = toy_float_model_cpu_reference();
        let engine = GpuEngine::with_max_seq(cpu_model, gpu.clone(), 16);

        let run = |use_shared: bool| -> std::time::Duration {
            // Warmup
            for _ in 0..2 {
                let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("bench.warmup.enc"),
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("bench.warmup.pass"),
                        timestamp_writes: None,
                    });
                    if use_shared {
                        engine.dispatch_matmul_shared_inner_in_pass(&mut pass, &layer, &in_buf, &out_buf, n_tokens);
                    } else {
                        engine.dispatch_matmul_legacy_inner_in_pass(&mut pass, &layer, &in_buf, &out_buf, n_tokens);
                    }
                }
                gpu.queue.submit(Some(encoder.finish()));
                gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
            }

            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("bench.enc"),
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("bench.pass"),
                        timestamp_writes: None,
                    });
                    if use_shared {
                        engine.dispatch_matmul_shared_inner_in_pass(&mut pass, &layer, &in_buf, &out_buf, n_tokens);
                    } else {
                        engine.dispatch_matmul_legacy_inner_in_pass(&mut pass, &layer, &in_buf, &out_buf, n_tokens);
                    }
                }
                gpu.queue.submit(Some(encoder.finish()));
                gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
            }
            t0.elapsed()
        };

        let legacy_time = run(false);
        let shared_time = run(true);
        let per_legacy = legacy_time / iters as u32;
        let per_shared = shared_time / iters as u32;
        let speedup = legacy_time.as_secs_f64() / shared_time.as_secs_f64();

        eprintln!(
            "matmul bench (rows={rows}, cols={cols}, n_tokens={n_tokens}, iters={iters}):\n  \
             legacy: {legacy_time:?} ({per_legacy:?}/iter)\n  \
             shared: {shared_time:?} ({per_shared:?}/iter)\n  \
             speedup: {speedup:.2}x"
        );
    }

    #[test]
    fn matmul_gate_up_fused_matches_two_dispatches() {
        // Numerical parity: the fused gate+up shader should produce
        // outputs byte-equivalent (within fp rounding) to running two
        // separate matmul_shared dispatches against the same inputs.
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let rows = 64usize;     // FFN intermediate-shaped, kept small for speed
        let cols = 32usize;     // embed-shaped
        let n_tokens = 16usize;

        let mut rng: u64 = 0xCAFE_BEEF;
        let mut roll = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as i32 % 1000 - 500) as f32 * 0.001
        };
        let gate_w: Vec<f32> = (0..rows * cols).map(|_| roll()).collect();
        let up_w:   Vec<f32> = (0..rows * cols).map(|_| roll()).collect();
        let input:  Vec<f32> = (0..n_tokens * cols).map(|_| roll()).collect();

        let gate_layer = crate::layers::gpu_floatlinear::GpuFloatLinear::from_float_tensor(
            gpu.clone(),
            crate::tensor::FloatTensor::new(gate_w, vec![rows, cols]),
        );
        let up_layer = crate::layers::gpu_floatlinear::GpuFloatLinear::from_float_tensor(
            gpu.clone(),
            crate::tensor::FloatTensor::new(up_w, vec![rows, cols]),
        );

        let in_bytes = (n_tokens * cols * 4) as u64;
        let in_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test.fused.in"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        // C1: fused gate+up shader takes packed-f16 input.
        let input_packed = GpuDevice::pack_f16(&input);
        let in_packed_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test.fused.in_packed"),
            contents: bytemuck::cast_slice(&input_packed),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let out_bytes = (n_tokens * rows * 4) as u64;
        // C2: fused outputs are packed f16 (half size). Reference is f32.
        let out_packed_bytes = (n_tokens * rows * 2) as u64;
        let mk_out = |label: &'static str, sz: u64| gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: sz,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let gate_ref = mk_out("test.fused.gate_ref", out_bytes);
        let up_ref   = mk_out("test.fused.up_ref", out_bytes);
        let gate_fused = mk_out("test.fused.gate_fused", out_packed_bytes);
        let up_fused   = mk_out("test.fused.up_fused", out_packed_bytes);
        let staging = |label: &'static str, sz: u64| gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: sz,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let stg_gr = staging("test.fused.stg_gr", out_bytes);
        let stg_ur = staging("test.fused.stg_ur", out_bytes);
        let stg_gf = staging("test.fused.stg_gf", out_packed_bytes);
        let stg_uf = staging("test.fused.stg_uf", out_packed_bytes);

        let cpu_model = toy_float_model_cpu_reference();
        let engine = GpuEngine::with_max_seq(cpu_model, gpu.clone(), 16);

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test.fused.enc"),
        });

        // Reference: two separate shared dispatches with the same
        // packed input the fused shader reads (C1: matmul_shared_pin_fout).
        // Pinning both to the packed-input variant keeps the comparison
        // apples-to-apples and avoids f16 quantization error showing up
        // as a "fused vs reference" gap.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("test.fused.ref_gate"),
                timestamp_writes: None,
            });
            engine.dispatch_matmul_shared_pin_fout_inner_in_pass(&mut pass, &gate_layer, &in_packed_buf, &gate_ref, n_tokens);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("test.fused.ref_up"),
                timestamp_writes: None,
            });
            engine.dispatch_matmul_shared_pin_fout_inner_in_pass(&mut pass, &up_layer, &in_packed_buf, &up_ref, n_tokens);
        }
        encoder.copy_buffer_to_buffer(&gate_ref, 0, &stg_gr, 0, out_bytes);
        encoder.copy_buffer_to_buffer(&up_ref, 0, &stg_ur, 0, out_bytes);

        // Fused.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("test.fused.fused"),
                timestamp_writes: None,
            });
            let ok = engine.dispatch_gate_up_fused_in_pass(
                &mut pass, &gate_layer, &up_layer, &in_packed_buf, &gate_fused, &up_fused, n_tokens,
            );
            assert!(ok, "fused dispatch should succeed on matching GpuFloatLinear pair");
        }
        encoder.copy_buffer_to_buffer(&gate_fused, 0, &stg_gf, 0, out_packed_bytes);
        encoder.copy_buffer_to_buffer(&up_fused, 0, &stg_uf, 0, out_packed_bytes);

        gpu.queue.submit(Some(encoder.finish()));

        let read_f32 = |stg: &wgpu::Buffer| -> Vec<f32> {
            let slice = stg.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
            gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
            rx.recv().unwrap().unwrap();
            let data = slice.get_mapped_range();
            let out: Vec<f32> = data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            drop(data); stg.unmap();
            out
        };
        // C2: fused outputs are packed f16; unpack on readback.
        let read_packed = |stg: &wgpu::Buffer| -> Vec<f32> {
            let slice = stg.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
            gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
            rx.recv().unwrap().unwrap();
            let data = slice.get_mapped_range();
            let mut out: Vec<f32> = Vec::with_capacity(data.len() / 2);
            for chunk in data.chunks_exact(4) {
                let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let lo = half::f16::from_bits((bits & 0xFFFF) as u16).to_f32();
                let hi = half::f16::from_bits(((bits >> 16) & 0xFFFF) as u16).to_f32();
                out.push(lo);
                out.push(hi);
            }
            drop(data); stg.unmap();
            out
        };
        let g_ref = read_f32(&stg_gr);
        let u_ref = read_f32(&stg_ur);
        let g_fused = read_packed(&stg_gf);
        let u_fused = read_packed(&stg_uf);

        // C2 tolerance: f16 output adds quantization noise on top of
        // the f16 input. Bumped to 1e-2 vs C1's 1e-3.
        let check = |label: &str, a: &[f32], b: &[f32]| {
            assert_eq!(a.len(), b.len(), "{label} length mismatch");
            let mut max_diff = 0.0_f32;
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                let d = (x - y).abs();
                if d > max_diff { max_diff = d; }
                assert!(d < 1e-2, "{label} elem {i}: ref={x} fused={y} diff={d}");
            }
            eprintln!("{label}: max_diff = {max_diff:.6e}");
        };
        check("gate", &g_ref, &g_fused);
        check("up", &u_ref, &u_fused);
    }

    #[test]
    #[ignore = "benchmark; run with --ignored to compare matmul shaders at Qwen FFN gate/up shape"]
    fn matmul_shared_vs_legacy_bench_ffn_shape() {
        // FFN shape for Qwen 0.5B: gate_proj / up_proj are
        // rows=4864 (intermediate), cols=896 (embed_dim), n_tokens=526
        // for a 500w prompt. This is the dominant matmul cost in the
        // forward (block.pass2 = 87% of prefill GPU time per per-pass
        // timestamps). Speedup here matters MUCH more than at Q_proj
        // shape because FFN matmuls run 3× per block.
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let rows = 4864usize;
        let cols = 896usize;
        let n_tokens = 526usize;
        let iters = 5usize;

        let mut rng: u64 = 0xFFFF_FACE;
        let mut roll = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as i32 % 1000 - 500) as f32 * 0.001
        };
        let weights_data: Vec<f32> = (0..rows * cols).map(|_| roll()).collect();
        let input_data: Vec<f32> = (0..n_tokens * cols).map(|_| roll()).collect();

        let weight_tensor = crate::tensor::FloatTensor::new(weights_data, vec![rows, cols]);
        let layer = crate::layers::gpu_floatlinear::GpuFloatLinear::from_float_tensor(
            gpu.clone(), weight_tensor,
        );

        let in_bytes = (n_tokens * cols * 4) as u64;
        let in_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bench.ffn.in"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let out_bytes = (n_tokens * rows * 4) as u64;
        let out_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bench.ffn.out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cpu_model = toy_float_model_cpu_reference();
        let engine = GpuEngine::with_max_seq(cpu_model, gpu.clone(), 16);

        let run = |use_shared: bool| -> std::time::Duration {
            for _ in 0..2 {
                let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("bench.ffn.warmup.enc"),
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("bench.ffn.warmup.pass"),
                        timestamp_writes: None,
                    });
                    if use_shared {
                        engine.dispatch_matmul_shared_inner_in_pass(&mut pass, &layer, &in_buf, &out_buf, n_tokens);
                    } else {
                        engine.dispatch_matmul_legacy_inner_in_pass(&mut pass, &layer, &in_buf, &out_buf, n_tokens);
                    }
                }
                gpu.queue.submit(Some(encoder.finish()));
                gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
            }

            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("bench.ffn.enc"),
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("bench.ffn.pass"),
                        timestamp_writes: None,
                    });
                    if use_shared {
                        engine.dispatch_matmul_shared_inner_in_pass(&mut pass, &layer, &in_buf, &out_buf, n_tokens);
                    } else {
                        engine.dispatch_matmul_legacy_inner_in_pass(&mut pass, &layer, &in_buf, &out_buf, n_tokens);
                    }
                }
                gpu.queue.submit(Some(encoder.finish()));
                gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
            }
            t0.elapsed()
        };

        let legacy_time = run(false);
        let shared_time = run(true);
        let per_legacy = legacy_time / iters as u32;
        let per_shared = shared_time / iters as u32;
        let speedup = legacy_time.as_secs_f64() / shared_time.as_secs_f64();

        eprintln!(
            "matmul FFN bench (rows={rows}, cols={cols}, n_tokens={n_tokens}, iters={iters}):\n  \
             legacy: {legacy_time:?} ({per_legacy:?}/iter)\n  \
             shared: {shared_time:?} ({per_shared:?}/iter)\n  \
             speedup: {speedup:.2}x"
        );
    }

    // `toy_ternary_block_pair` and `forward_block_gpu_matches_cpu_bitnet_block`
    // were removed with the 2026-05-29 BitNet un-merge — they validated
    // `BitLinear` / `GpuBitLinear` (now in ternary-rs). The float side of
    // the equivalent forward-block-matches-cpu coverage is provided by
    // `forward_block_gpu_matches_cpu_block` above.

    #[test]
    fn dispatch_silu_mul_matches_cpu() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let n = 16;
        let n_tokens = 3;
        let total = n * n_tokens;

        let gate: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1 - 0.7).collect();
        let up:   Vec<f32> = (0..total).map(|i| (i as f32) * -0.05 + 0.4).collect();

        // CPU reference: silu(g) * u  where silu(x) = x / (1 + exp(-x))
        let cpu_out: Vec<f32> = gate.iter().zip(&up)
            .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
            .collect();

        let bytes = (total * 4) as u64;
        let g_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("silu.g"), contents: bytemuck::cast_slice(&gate),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let u_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("silu.u"), contents: bytemuck::cast_slice(&up),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let o_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("silu.o"), size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = gpu.create_staging_buffer(bytes);

        let engine = GpuEngine::from_cpu_model(toy_model(), gpu.clone());
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("silu.encoder"),
        });
        engine.dispatch_silu_mul_into(&mut encoder, &g_buf, &u_buf, &o_buf, n, n_tokens);
        encoder.copy_buffer_to_buffer(&o_buf, 0, &staging, 0, bytes);
        gpu.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let gpu_out: Vec<f32> = data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        drop(data); staging.unmap();

        for (i, (c, g)) in cpu_out.iter().zip(&gpu_out).enumerate() {
            assert!((c - g).abs() < 1e-5, "elem {i}: cpu={c} gpu={g}");
        }
    }

    #[test]
    fn dispatch_add_in_place_matches_cpu() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let n = 8;
        let n_tokens = 4;
        let total = n * n_tokens;

        let mut a: Vec<f32> = (0..total).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32>     = (0..total).map(|i| (total - i) as f32 * -0.05).collect();
        let cpu_a: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x + y).collect();

        let bytes = (total * 4) as u64;
        let a_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("add.a"), contents: bytemuck::cast_slice(&a),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let b_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("add.b"), contents: bytemuck::cast_slice(&b),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let staging = gpu.create_staging_buffer(bytes);

        let engine = GpuEngine::from_cpu_model(toy_model(), gpu.clone());
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("add.encoder"),
        });
        engine.dispatch_add_into(&mut encoder, &a_buf, &b_buf, n, n_tokens);
        encoder.copy_buffer_to_buffer(&a_buf, 0, &staging, 0, bytes);
        gpu.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let gpu_a: Vec<f32> = data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        drop(data); staging.unmap();

        let _ = a; // silence unused-mut warning in case of future refactor
        for (i, (c, g)) in cpu_a.iter().zip(&gpu_a).enumerate() {
            assert!((c - g).abs() < 1e-6, "elem {i}: cpu={c} gpu={g}");
        }
    }

    #[test]
    fn wrapper_metadata_passes_through() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let engine = GpuEngine::from_cpu_model(toy_model(), gpu);
        assert_eq!(engine.vocab_size(), 4);
        assert_eq!(engine.embed_dim(), 4);
        assert_eq!(engine.n_layers(), 1);
    }
