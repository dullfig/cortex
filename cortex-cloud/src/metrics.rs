//! Prometheus-text-format metrics for cortex-server.
//!
//! Two flavors of metric land here:
//!
//! 1. **Request-level** (Phase 1 — already shipped): per-endpoint
//!    counters, latency histograms, TTFT. Recorded inline by handlers
//!    via `RequestTimer`. Each metric here MUST be both recorded by a
//!    handler AND emitted by `render_prometheus()` — phantom metrics
//!    are forbidden.
//!
//! 2. **Substrate/concurrency tripwires** (Phase K): the gauges that
//!    tell us when the operating envelope is approaching the next
//!    scaling stage. `cortex_concurrent_requests` for in-flight load,
//!    `cortex_cache_pool_*` for cortex-cloud pool depth,
//!    `cortex_vram_heap_bytes` for substrate pressure across all 5
//!    heaps, `cortex_params_pool_acquired_total` for the
//!    ParamsBufferPool wrap rate, `cortex_gpu_busy_micros_total` for
//!    GPU utilization (Prometheus `rate()` over the counter gives
//!    busy-fraction-per-second, i.e. utilization %). The sampler task
//!    in `start_metrics_sampler` snapshots the read-only sources;
//!    `RequestTimer` and the engine push the push-style ones.
//!
//! Tripwire/alert logic lives OUTSIDE cortex (Prometheus alert rules
//! or Grafana). cortex emits; the dashboard decides when to fire.
//! This keeps cortex oblivious to operational thresholds that should
//! be tunable per-deployment.
//!
//! Wire format: <https://prometheus.io/docs/instrumenting/exposition_formats/>
//! served via `GET /metrics` with content-type `text/plain; version=0.0.4`.

use std::fmt::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Histogram bucket boundaries in seconds. Cover sub-millisecond decode
/// ticks up to multi-minute long-prompt prefills.
pub const DURATION_BUCKETS_S: &[f64] = &[
    0.005, 0.025, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
];

/// Endpoints we measure. Add a variant + a constructor field below + a
/// render line in `render_prometheus()`. Don't add a variant unless you
/// also wire the handler to record it.
#[derive(Copy, Clone, Debug)]
pub enum Endpoint {
    ChatCompletions,
    CacheLoad,
    CacheAppend,
}

impl Endpoint {
    fn label(self) -> &'static str {
        match self {
            Endpoint::ChatCompletions => "chat_completions",
            Endpoint::CacheLoad => "cache_load",
            Endpoint::CacheAppend => "cache_append",
        }
    }
}

/// One histogram (bucket counts + sum + count). Cumulative bucket counts
/// per the Prometheus convention.
struct Histogram {
    buckets: Vec<AtomicU64>,
    sum_micros: AtomicU64,
    count: AtomicU64,
}

impl Histogram {
    fn new() -> Self {
        Self {
            buckets: (0..DURATION_BUCKETS_S.len()).map(|_| AtomicU64::new(0)).collect(),
            sum_micros: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    fn observe(&self, seconds: f64) {
        let micros = (seconds * 1_000_000.0).round().max(0.0) as u64;
        self.sum_micros.fetch_add(micros, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        for (i, &boundary) in DURATION_BUCKETS_S.iter().enumerate() {
            if seconds <= boundary {
                self.buckets[i].fetch_add(1, Ordering::Relaxed);
            }
        }
        // +Inf bucket is implicit (== count).
    }

    fn render(&self, name: &str, labels: &str, out: &mut String) {
        for (i, &boundary) in DURATION_BUCKETS_S.iter().enumerate() {
            let extra = if labels.is_empty() {
                format!("le=\"{boundary}\"")
            } else {
                format!("{labels},le=\"{boundary}\"")
            };
            let _ = writeln!(out, "{name}_bucket{{{extra}}} {}", self.buckets[i].load(Ordering::Relaxed));
        }
        let count = self.count.load(Ordering::Relaxed);
        let plus_inf_labels = if labels.is_empty() {
            "le=\"+Inf\"".to_string()
        } else {
            format!("{labels},le=\"+Inf\"")
        };
        let _ = writeln!(out, "{name}_bucket{{{plus_inf_labels}}} {count}");
        let sum_s = self.sum_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        let label_brace = if labels.is_empty() { String::new() } else { format!("{{{labels}}}") };
        let _ = writeln!(out, "{name}_sum{label_brace} {sum_s}");
        let _ = writeln!(out, "{name}_count{label_brace} {count}");
    }
}

/// Identifier for one of the 5 vram-heaps cortex uses. The label
/// matches the heap name in boot logs and the metric label exposed
/// to Prometheus.
#[derive(Copy, Clone, Debug)]
pub enum VramHeapLabel {
    TransientA,
    TransientB,
    TransientC,
    Weights,
    HostReadback,
}

impl VramHeapLabel {
    pub fn label(self) -> &'static str {
        match self {
            VramHeapLabel::TransientA => "transient_a",
            VramHeapLabel::TransientB => "transient_b",
            VramHeapLabel::TransientC => "transient_c",
            VramHeapLabel::Weights => "weights",
            VramHeapLabel::HostReadback => "host_readback",
        }
    }

    fn index(self) -> usize {
        match self {
            VramHeapLabel::TransientA => 0,
            VramHeapLabel::TransientB => 1,
            VramHeapLabel::TransientC => 2,
            VramHeapLabel::Weights => 3,
            VramHeapLabel::HostReadback => 4,
        }
    }

    /// Iterate over all heap labels in stable order. Used by the
    /// sampler to thread heap stats into the per-label gauges and by
    /// `render_prometheus` to emit them.
    pub fn all() -> [VramHeapLabel; 5] {
        [
            VramHeapLabel::TransientA,
            VramHeapLabel::TransientB,
            VramHeapLabel::TransientC,
            VramHeapLabel::Weights,
            VramHeapLabel::HostReadback,
        ]
    }
}

/// All cortex-server metrics. Constructed once at startup, shared via
/// `Arc<Metrics>` on `AppState`.
pub struct Metrics {
    start_time: Instant,

    // Per-endpoint counters (one per endpoint × {ok, err}).
    chat_completions_ok: AtomicU64,
    chat_completions_err: AtomicU64,
    cache_load_ok: AtomicU64,
    cache_load_err: AtomicU64,
    cache_append_ok: AtomicU64,
    cache_append_err: AtomicU64,

    // Token totals (chat_completions only — cache_* doesn't generate).
    prompt_tokens_total: AtomicU64,
    completion_tokens_total: AtomicU64,

    // Per-endpoint duration histograms.
    chat_completions_duration: Histogram,
    cache_load_duration: Histogram,
    cache_append_duration: Histogram,

    // Time-to-first-token (chat completions only).
    ttft: Histogram,

    // Phase K: substrate/concurrency tripwire gauges.
    //
    // `concurrent_requests` is push-style (RequestTimer inc/dec on
    // new/Drop); the rest are pull-style (sampler task reads them
    // periodically from cortex-cloud state + GpuDevice).
    //
    // GPU utilization % is NOT emitted as its own counter. The
    // existing `cortex_request_duration_seconds` histogram's `_sum`
    // field already captures cumulative time spent inside the
    // GPU-bound endpoints (chat_completions, cache_load, cache_append).
    // Use `rate(cortex_request_duration_seconds_sum[1m])` in
    // Prometheus to get "wall-time fraction spent in GPU work" =
    // utilization fraction. The 40%-utilization tripwire is
    // `rate(...) > 0.4`. No new counter needed; adding one would
    // duplicate the signal with worse semantics (can't distinguish
    // per-endpoint busy time).
    concurrent_requests: AtomicU64,
    cache_pool_size: AtomicU64,
    cache_pool_tokens_total: AtomicU64,
    vram_heap_bytes: [AtomicU64; 5],
    params_pool_acquired_total: AtomicU64,

    // Static labels for the model_info gauge.
    model_name: String,
    build_version: String,
}

impl Metrics {
    pub fn new(model_name: String, build_version: String) -> Self {
        Self {
            start_time: Instant::now(),
            chat_completions_ok: AtomicU64::new(0),
            chat_completions_err: AtomicU64::new(0),
            cache_load_ok: AtomicU64::new(0),
            cache_load_err: AtomicU64::new(0),
            cache_append_ok: AtomicU64::new(0),
            cache_append_err: AtomicU64::new(0),
            prompt_tokens_total: AtomicU64::new(0),
            completion_tokens_total: AtomicU64::new(0),
            chat_completions_duration: Histogram::new(),
            cache_load_duration: Histogram::new(),
            cache_append_duration: Histogram::new(),
            ttft: Histogram::new(),
            concurrent_requests: AtomicU64::new(0),
            cache_pool_size: AtomicU64::new(0),
            cache_pool_tokens_total: AtomicU64::new(0),
            vram_heap_bytes: [
                AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
                AtomicU64::new(0), AtomicU64::new(0),
            ],
            params_pool_acquired_total: AtomicU64::new(0),
            model_name,
            build_version,
        }
    }

    /// Increment the in-flight requests gauge. Wired by `RequestTimer::new`.
    pub fn record_concurrent_inc(&self) {
        self.concurrent_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement the in-flight requests gauge. Wired by `RequestTimer::Drop`.
    pub fn record_concurrent_dec(&self) {
        self.concurrent_requests.fetch_sub(1, Ordering::Relaxed);
    }

    /// Update the cache pool gauges. Called by the sampler task.
    pub fn record_cache_pool(&self, size: u64, tokens_total: u64) {
        self.cache_pool_size.store(size, Ordering::Relaxed);
        self.cache_pool_tokens_total.store(tokens_total, Ordering::Relaxed);
    }

    /// Update one heap's used-bytes gauge. Called by the sampler task
    /// once per heap, per tick.
    pub fn record_vram_heap(&self, label: VramHeapLabel, used_bytes: u64) {
        self.vram_heap_bytes[label.index()].store(used_bytes, Ordering::Relaxed);
    }

    /// Update the ParamsBufferPool cumulative acquire counter. Called
    /// by the sampler task. Prometheus computes the wrap rate from
    /// `rate()` over this counter.
    pub fn record_params_pool(&self, total_acquired: u64) {
        self.params_pool_acquired_total.store(total_acquired, Ordering::Relaxed);
    }

    /// Record one completed request: increments the endpoint × status
    /// counter and the endpoint's duration histogram.
    pub fn record_request(&self, endpoint: Endpoint, ok: bool, duration_s: f64) {
        let counter = match (endpoint, ok) {
            (Endpoint::ChatCompletions, true) => &self.chat_completions_ok,
            (Endpoint::ChatCompletions, false) => &self.chat_completions_err,
            (Endpoint::CacheLoad, true) => &self.cache_load_ok,
            (Endpoint::CacheLoad, false) => &self.cache_load_err,
            (Endpoint::CacheAppend, true) => &self.cache_append_ok,
            (Endpoint::CacheAppend, false) => &self.cache_append_err,
        };
        counter.fetch_add(1, Ordering::Relaxed);
        let histo = match endpoint {
            Endpoint::ChatCompletions => &self.chat_completions_duration,
            Endpoint::CacheLoad => &self.cache_load_duration,
            Endpoint::CacheAppend => &self.cache_append_duration,
        };
        histo.observe(duration_s);
    }

    pub fn record_tokens(&self, prompt: u64, completion: u64) {
        self.prompt_tokens_total.fetch_add(prompt, Ordering::Relaxed);
        self.completion_tokens_total.fetch_add(completion, Ordering::Relaxed);
    }

    pub fn record_ttft(&self, duration_s: f64) {
        self.ttft.observe(duration_s);
    }

    pub fn render_prometheus(&self) -> String {
        let mut out = String::with_capacity(2048);

        let uptime = self.start_time.elapsed().as_secs_f64();
        let _ = writeln!(out, "# HELP cortex_uptime_seconds Time since server start.");
        let _ = writeln!(out, "# TYPE cortex_uptime_seconds gauge");
        let _ = writeln!(out, "cortex_uptime_seconds {uptime}");

        let _ = writeln!(out, "# HELP cortex_model_info Loaded model (value always 1).");
        let _ = writeln!(out, "# TYPE cortex_model_info gauge");
        let _ = writeln!(out, "cortex_model_info{{name=\"{}\"}} 1", escape(&self.model_name));

        let _ = writeln!(out, "# HELP cortex_build_info Server build (value always 1).");
        let _ = writeln!(out, "# TYPE cortex_build_info gauge");
        let _ = writeln!(out, "cortex_build_info{{version=\"{}\"}} 1", escape(&self.build_version));

        // requests_total: one line per endpoint × status.
        let _ = writeln!(out, "# HELP cortex_requests_total Total HTTP requests by endpoint and status.");
        let _ = writeln!(out, "# TYPE cortex_requests_total counter");
        for (endpoint, ok_counter, err_counter) in [
            (Endpoint::ChatCompletions, &self.chat_completions_ok, &self.chat_completions_err),
            (Endpoint::CacheLoad, &self.cache_load_ok, &self.cache_load_err),
            (Endpoint::CacheAppend, &self.cache_append_ok, &self.cache_append_err),
        ] {
            let _ = writeln!(out, "cortex_requests_total{{endpoint=\"{}\",status=\"ok\"}} {}",
                endpoint.label(), ok_counter.load(Ordering::Relaxed));
            let _ = writeln!(out, "cortex_requests_total{{endpoint=\"{}\",status=\"err\"}} {}",
                endpoint.label(), err_counter.load(Ordering::Relaxed));
        }

        let _ = writeln!(out, "# HELP cortex_tokens_total Total tokens processed by kind.");
        let _ = writeln!(out, "# TYPE cortex_tokens_total counter");
        let _ = writeln!(out, "cortex_tokens_total{{kind=\"prompt\"}} {}",
            self.prompt_tokens_total.load(Ordering::Relaxed));
        let _ = writeln!(out, "cortex_tokens_total{{kind=\"completion\"}} {}",
            self.completion_tokens_total.load(Ordering::Relaxed));

        let _ = writeln!(out, "# HELP cortex_request_duration_seconds End-to-end request wall time.");
        let _ = writeln!(out, "# TYPE cortex_request_duration_seconds histogram");
        for (endpoint, histo) in [
            (Endpoint::ChatCompletions, &self.chat_completions_duration),
            (Endpoint::CacheLoad, &self.cache_load_duration),
            (Endpoint::CacheAppend, &self.cache_append_duration),
        ] {
            histo.render("cortex_request_duration_seconds",
                &format!("endpoint=\"{}\"", endpoint.label()), &mut out);
        }

        let _ = writeln!(out, "# HELP cortex_ttft_seconds Time from chat_completions request to first content token.");
        let _ = writeln!(out, "# TYPE cortex_ttft_seconds histogram");
        self.ttft.render("cortex_ttft_seconds", "", &mut out);

        // ---- Phase K: substrate/concurrency tripwire gauges ----

        let _ = writeln!(out, "# HELP cortex_concurrent_requests In-flight HTTP requests, push-updated by RequestTimer.");
        let _ = writeln!(out, "# TYPE cortex_concurrent_requests gauge");
        let _ = writeln!(out, "cortex_concurrent_requests {}",
            self.concurrent_requests.load(Ordering::Relaxed));

        let _ = writeln!(out, "# HELP cortex_cache_pool_size Number of named KV caches currently held in the pool.");
        let _ = writeln!(out, "# TYPE cortex_cache_pool_size gauge");
        let _ = writeln!(out, "cortex_cache_pool_size {}",
            self.cache_pool_size.load(Ordering::Relaxed));

        let _ = writeln!(out, "# HELP cortex_cache_pool_tokens Total tokens across all KV caches in the pool.");
        let _ = writeln!(out, "# TYPE cortex_cache_pool_tokens gauge");
        let _ = writeln!(out, "cortex_cache_pool_tokens {}",
            self.cache_pool_tokens_total.load(Ordering::Relaxed));

        let _ = writeln!(out, "# HELP cortex_vram_heap_bytes Used bytes per vram-heap.");
        let _ = writeln!(out, "# TYPE cortex_vram_heap_bytes gauge");
        for heap in VramHeapLabel::all() {
            let _ = writeln!(out, "cortex_vram_heap_bytes{{heap=\"{}\"}} {}",
                heap.label(),
                self.vram_heap_bytes[heap.index()].load(Ordering::Relaxed));
        }

        let _ = writeln!(out, "# HELP cortex_params_pool_acquired_total Cumulative ParamsBufferPool slot acquires. rate() over [1m] gives acquire rate; sustained growth means the ring is being stressed.");
        let _ = writeln!(out, "# TYPE cortex_params_pool_acquired_total counter");
        let _ = writeln!(out, "cortex_params_pool_acquired_total {}",
            self.params_pool_acquired_total.load(Ordering::Relaxed));

        out
    }
}

/// RAII helper: stamp start time on construction, record endpoint
/// duration on drop. Default `success = false` (i.e. error) — call
/// `mark_success()` once we know the handler is returning Ok.
///
/// Holds an `Arc<Metrics>` so handlers can both keep the timer alive
/// and move/clone the underlying server state through async boundaries.
///
/// Use this in handlers with multiple early-return / `?` paths so we
/// don't have to thread a counter through every branch.
pub struct RequestTimer {
    metrics: std::sync::Arc<Metrics>,
    endpoint: Endpoint,
    start: Instant,
    success: bool,
}

impl RequestTimer {
    pub fn new(metrics: std::sync::Arc<Metrics>, endpoint: Endpoint) -> Self {
        metrics.record_concurrent_inc();
        Self { metrics, endpoint, start: Instant::now(), success: false }
    }

    pub fn mark_success(&mut self) {
        self.success = true;
    }
}

impl Drop for RequestTimer {
    fn drop(&mut self) {
        let duration_s = self.start.elapsed().as_secs_f64();
        self.metrics.record_request(self.endpoint, self.success, duration_s);
        self.metrics.record_concurrent_dec();
    }
}

/// Escape Prometheus label-value characters: \\, \n, \".
fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '"' => out.push_str("\\\""),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_bucket_monotone() {
        let h = Histogram::new();
        h.observe(0.001);
        h.observe(0.05);
        h.observe(1.5);
        // 0.001 hits every bucket. 0.05 hits 0.1+ buckets. 1.5 hits 2.5+ buckets.
        let counts: Vec<u64> = h.buckets.iter().map(|b| b.load(Ordering::Relaxed)).collect();
        for i in 1..counts.len() {
            assert!(counts[i] >= counts[i - 1], "bucket {i} count {} < prev {}", counts[i], counts[i - 1]);
        }
        assert_eq!(h.count.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn renders_valid_prometheus_lines() {
        let m = Metrics::new("test-model".to_string(), "0.1.0".to_string());
        m.record_request(Endpoint::ChatCompletions, true, 0.05);
        m.record_request(Endpoint::CacheAppend, false, 1.2);
        m.record_tokens(42, 17);
        m.record_ttft(0.01);
        // Phase K tripwire setters.
        m.record_cache_pool(3, 4500);
        m.record_vram_heap(VramHeapLabel::TransientA, 12_000_000);
        m.record_vram_heap(VramHeapLabel::Weights, 6_000_000_000);
        m.record_params_pool(123_456);
        let out = m.render_prometheus();
        // Spot-check: every metric name appears.
        for needle in [
            "cortex_uptime_seconds ",
            "cortex_model_info{name=\"test-model\"} 1",
            "cortex_build_info{version=\"0.1.0\"} 1",
            "cortex_requests_total{endpoint=\"chat_completions\",status=\"ok\"} 1",
            "cortex_requests_total{endpoint=\"cache_append\",status=\"err\"} 1",
            "cortex_tokens_total{kind=\"prompt\"} 42",
            "cortex_tokens_total{kind=\"completion\"} 17",
            "cortex_request_duration_seconds_bucket{endpoint=\"chat_completions\",le=\"0.1\"} 1",
            "cortex_request_duration_seconds_count{endpoint=\"chat_completions\"} 1",
            "cortex_ttft_seconds_bucket{le=\"0.025\"} 1",
            "cortex_ttft_seconds_count 1",
            // Phase K
            "cortex_concurrent_requests 0",
            "cortex_cache_pool_size 3",
            "cortex_cache_pool_tokens 4500",
            "cortex_vram_heap_bytes{heap=\"transient_a\"} 12000000",
            "cortex_vram_heap_bytes{heap=\"weights\"} 6000000000",
            "cortex_vram_heap_bytes{heap=\"host_readback\"} 0",
            "cortex_params_pool_acquired_total 123456",
        ] {
            assert!(out.contains(needle), "missing line: {needle}\nfull output:\n{out}");
        }
    }

    #[test]
    fn concurrent_inc_dec_balanced() {
        let m = std::sync::Arc::new(Metrics::new("t".to_string(), "0".to_string()));
        // Five RequestTimers in flight simultaneously.
        let timers: Vec<RequestTimer> = (0..5)
            .map(|_| RequestTimer::new(m.clone(), Endpoint::ChatCompletions))
            .collect();
        let out = m.render_prometheus();
        assert!(out.contains("cortex_concurrent_requests 5"),
            "expected gauge at 5 with timers alive\nfull output:\n{out}");
        drop(timers);
        let out = m.render_prometheus();
        assert!(out.contains("cortex_concurrent_requests 0"),
            "expected gauge back to 0 after drops\nfull output:\n{out}");
    }

    #[test]
    fn label_escape() {
        assert_eq!(escape("plain"), "plain");
        assert_eq!(escape(r#"a"b"#), r#"a\"b"#);
        assert_eq!(escape("a\\b"), r"a\\b");
        assert_eq!(escape("a\nb"), r"a\nb");
    }
}
