//! Prometheus-text-format metrics for cortex-server.
//!
//! MVP scope: instruments the request handlers we actively care about
//! (chat_completions, cache_load, cache_append). Each metric here MUST be
//! both recorded by a handler AND emitted by `render_prometheus()`. New
//! metrics that aren't paired on both ends are phantom and should not
//! land.
//!
//! Wire format: <https://prometheus.io/docs/instrumenting/exposition_formats/>
//! served via `GET /metrics` with content-type `text/plain; version=0.0.4`.

use std::fmt::Write;
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
            model_name,
            build_version,
        }
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
        ] {
            assert!(out.contains(needle), "missing line: {needle}\nfull output:\n{out}");
        }
    }

    #[test]
    fn label_escape() {
        assert_eq!(escape("plain"), "plain");
        assert_eq!(escape(r#"a"b"#), r#"a\"b"#);
        assert_eq!(escape("a\\b"), r"a\\b");
        assert_eq!(escape("a\nb"), r"a\nb");
    }
}
