//! dump_features — pinky experiment 1, feature extraction stage
//!
//! Runs forward_traced on a labeled fixture and dumps per-position
//! features as CSV. Each row is one token position with:
//!
//!   - metadata (fixture, position, token_id, token_text, source)
//!   - auto-generated label (1 if this is a source-change position, else 0)
//!   - per-layer features extracted from the trace:
//!       * post_attn_l{n}: post-softmax mean attention TO this position
//!         from a future window of subsequent positions, averaged over heads
//!       * post_left_l{n}: post-softmax mean attention from those future
//!         positions back to any position BEFORE this one, normalized
//!         per leftward position
//!       * pre_attn_l{n}: same but from the raw pre-softmax Q·K^T matrix
//!       * pre_left_l{n}: same, from pre-softmax
//!
//! For Qwen2.5-0.5B (24 layers, 14 heads) this produces 4 × 24 = 96
//! feature columns per row, plus 5 metadata columns and the label.
//!
//! The CSV is the input to the training script in
//! pinky/concept-boundaries-train/.
//!
//! Usage:
//!   dump-features \
//!     --model /path/to/qwen.gguf \
//!     --fixture path1.txt --fixture path2.txt ... \
//!     --output features.csv \
//!     [--window 12]

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use cortex::{ForwardTrace, Tokenizer};

// ---------------------------------------------------------------------------
// Source / Region / ProvenancedToken (duplicated from main.rs — pinky's
// looser-standards rule applies; refactor to lib.rs when the third binary
// needs these)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    System,
    User,
    Doc,
    Tool,
}

impl Source {
    fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "system" => Ok(Source::System),
            "user" => Ok(Source::User),
            "doc" => Ok(Source::Doc),
            "tool" => Ok(Source::Tool),
            other => Err(anyhow!("unknown source tag: {other}")),
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Source::System => "system",
            Source::User => "user",
            Source::Doc => "doc",
            Source::Tool => "tool",
        }
    }
}

struct Region {
    source: Source,
    content: String,
}

#[derive(Debug, Clone, Copy)]
struct ProvenancedToken {
    id: u32,
    source: Source,
}

fn parse_fixture(path: &PathBuf) -> Result<Vec<Region>> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("reading fixture {}", path.display()))?;

    let mut regions = Vec::new();
    for (lineno, line) in raw.lines().enumerate() {
        let line = line.trim_end();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (tag, content) = line
            .split_once(':')
            .ok_or_else(|| anyhow!("line {}: missing ':' separator", lineno + 1))?;
        let source = Source::parse(tag)
            .with_context(|| format!("line {}", lineno + 1))?;
        regions.push(Region {
            source,
            content: content.trim().to_string(),
        });
    }
    Ok(regions)
}

fn encode_with_provenance(
    tokenizer: &Tokenizer,
    regions: &[Region],
) -> Vec<ProvenancedToken> {
    let mut out = Vec::new();
    for region in regions {
        let ids = tokenizer.encode(&region.content, false);
        for id in ids {
            out.push(ProvenancedToken {
                id,
                source: region.source,
            });
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Feature extraction
// ---------------------------------------------------------------------------

/// Compute per-layer features for one position from a forward trace.
///
/// Returns 4 × n_layers values: post_attn[..], post_left[..], pre_attn[..],
/// pre_left[..], in that order.
fn extract_position_features(
    trace: &ForwardTrace,
    pos: usize,
    window: usize,
) -> Vec<f32> {
    let s = trace.seq_len;
    let n_layers = trace.n_layers;
    let n_heads = trace.n_heads;

    let q_lo = pos + 1;
    let q_hi = (pos + 1 + window).min(s);

    let mut post_attn = vec![0.0f32; n_layers];
    let mut post_left = vec![0.0f32; n_layers];
    let mut pre_attn = vec![0.0f32; n_layers];
    let mut pre_left = vec![0.0f32; n_layers];

    if q_hi <= q_lo || pos == 0 {
        // Degenerate position — return zeros. Position 0 has no leftward
        // baseline; positions near the end have no future window.
        let mut all = Vec::with_capacity(4 * n_layers);
        all.extend(post_attn);
        all.extend(post_left);
        all.extend(pre_attn);
        all.extend(pre_left);
        return all;
    }

    let mut count = 0usize;
    for layer in 0..n_layers {
        let mut sum_post_attn = 0.0f32;
        let mut sum_post_left = 0.0f32;
        let mut sum_pre_attn = 0.0f32;
        let mut sum_pre_left = 0.0f32;
        let mut local_count = 0usize;

        for h in 0..n_heads {
            let post_row = trace.attention_row(layer, h, 0); // placeholder, replaced below
            let _ = post_row;
            for q in q_lo..q_hi {
                let post_row = trace.attention_row(layer, h, q);
                let pre_row = trace.pre_score_row(layer, h, q);

                sum_post_attn += post_row[pos];
                sum_pre_attn += pre_row[pos];

                let post_left_sum: f32 = post_row[..pos].iter().sum();
                let pre_left_sum: f32 = pre_row[..pos].iter().sum();
                sum_post_left += post_left_sum / pos as f32;
                sum_pre_left += pre_left_sum / pos as f32;

                local_count += 1;
            }
        }

        if local_count > 0 {
            post_attn[layer] = sum_post_attn / local_count as f32;
            post_left[layer] = sum_post_left / local_count as f32;
            pre_attn[layer] = sum_pre_attn / local_count as f32;
            pre_left[layer] = sum_pre_left / local_count as f32;
        }
        count = count.max(local_count);
    }

    let _ = count;

    let mut all = Vec::with_capacity(4 * n_layers);
    all.extend(post_attn);
    all.extend(post_left);
    all.extend(pre_attn);
    all.extend(pre_left);
    all
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "dump-features")]
struct Cli {
    /// Path to a GGUF model.
    #[arg(long)]
    model: PathBuf,

    /// Fixture file(s) to dump features for. Can be repeated.
    #[arg(long)]
    fixture: Vec<PathBuf>,

    /// Output CSV path. Will be overwritten.
    #[arg(long)]
    output: PathBuf,

    /// Future-token window for feature extraction. Should match the
    /// window used by the boundary scorer for consistency.
    #[arg(long, default_value = "12")]
    window: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.fixture.is_empty() {
        return Err(anyhow!("at least one --fixture is required"));
    }

    println!("loading model: {}", cli.model.display());
    let loaded = cortex::load_model(cli.model.to_str().unwrap())
        .map_err(|e| anyhow!("load_model failed: {e}"))?;
    let n_layers = loaded.config.n_layers as usize;
    println!("  n_layers = {n_layers}");
    println!();

    // Open output CSV and write header
    let mut out = fs::File::create(&cli.output)
        .with_context(|| format!("creating {}", cli.output.display()))?;

    // Header: metadata columns + label + 4 × n_layers feature columns
    let mut header = String::from("fixture,position,token_id,token_text,source,label");
    for layer in 0..n_layers {
        header.push_str(&format!(",post_attn_l{:02}", layer));
    }
    for layer in 0..n_layers {
        header.push_str(&format!(",post_left_l{:02}", layer));
    }
    for layer in 0..n_layers {
        header.push_str(&format!(",pre_attn_l{:02}", layer));
    }
    for layer in 0..n_layers {
        header.push_str(&format!(",pre_left_l{:02}", layer));
    }
    header.push('\n');
    out.write_all(header.as_bytes())?;

    let mut total_rows = 0usize;
    let mut total_positives = 0usize;

    for fixture_path in &cli.fixture {
        println!("processing fixture: {}", fixture_path.display());
        let regions = parse_fixture(fixture_path)?;
        let tokens = encode_with_provenance(&loaded.tokenizer, &regions);
        let bare_ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();

        println!("  encoded {} tokens, running forward_traced...", tokens.len());
        let t_start = std::time::Instant::now();
        let (_logits, trace) = loaded.model.forward_traced(&bare_ids);
        println!("  forward_traced took {:.2}s", t_start.elapsed().as_secs_f32());

        let s = trace.seq_len;
        let fixture_name = fixture_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        let mut fixture_positives = 0usize;
        for pos in 0..s {
            // Auto-generated label: 1 if this position is a source-change
            // (trust boundary), 0 otherwise. Position 0 is always 0
            // because there's no previous token to compare against.
            let label: u8 = if pos > 0 && tokens[pos].source != tokens[pos - 1].source {
                fixture_positives += 1;
                1
            } else {
                0
            };

            let features = extract_position_features(&trace, pos, cli.window);

            let token_text = loaded.tokenizer.decode(&[tokens[pos].id])
                .replace('"', "'")
                .replace('\n', "\\n")
                .replace(',', " ")
                .replace('\r', "");

            let mut row = String::new();
            row.push_str(fixture_name);
            row.push(',');
            row.push_str(&pos.to_string());
            row.push(',');
            row.push_str(&tokens[pos].id.to_string());
            row.push_str(",\"");
            row.push_str(&token_text);
            row.push_str("\",");
            row.push_str(tokens[pos].source.label());
            row.push(',');
            row.push_str(&label.to_string());
            for f in &features {
                row.push(',');
                row.push_str(&format!("{:.6}", f));
            }
            row.push('\n');
            out.write_all(row.as_bytes())?;
            total_rows += 1;
        }

        println!(
            "  wrote {} rows ({} positives)",
            s, fixture_positives
        );
        total_positives += fixture_positives;
    }

    println!();
    println!(
        "DONE: {} total rows, {} positives ({:.1}%)",
        total_rows,
        total_positives,
        100.0 * total_positives as f32 / total_rows as f32,
    );
    println!("CSV: {}", cli.output.display());

    Ok(())
}
