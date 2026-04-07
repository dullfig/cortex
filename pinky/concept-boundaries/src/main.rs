//! concept-boundaries — pinky experiment 0
//!
//! What this is testing
//! --------------------
//! Whether per-token provenance is structurally sound: can we tag every
//! token in the input stream with its source region (system / user /
//! doc / tool) at tokenization time, carry the tags through model
//! consumption as a parallel read-only channel, and expose a clean
//! per-token trace at the output? If yes, this is the substrate that
//! later experiments can build attention-discovered concept boundaries
//! and provenance-aware refusal policies on top of.
//!
//! What this binary does today
//! ---------------------------
//! 1. Loads a fixture file with `<source>: <content>` lines
//! 2. Tokenizes each region separately with cortex's tokenizer
//! 3. Concatenates the token IDs and builds a parallel `Vec<Source>`
//! 4. Optionally loads a GGUF model (--model) and runs a forward pass
//!    over the concatenated tokens, just to prove the pipeline survives
//!    end-to-end. The forward output (final logits) is summarized.
//! 5. Prints a per-token table: idx | token_id | source | text
//!
//! What this binary CANNOT do today (blocked on cortex instrumentation)
//! --------------------------------------------------------------------
//! - Dump per-layer attention scores (cortex's MultiHeadAttention::forward
//!   returns the post-O-projection output, not the attention weights)
//! - Dump per-layer hidden states (cortex's TransformerBlock::forward
//!   only returns the final block output)
//! - Identify concept boundaries via attention pattern shifts
//!
//! See README.md for the cortex changes the next experiment needs.

use std::fs;
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use cortex::Tokenizer;

// ---------------------------------------------------------------------------
// Source tags — the parallel provenance channel
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

/// One labeled region of input text.
struct Region {
    source: Source,
    content: String,
}

/// A token that knows where it came from.
///
/// This is the minimal version of the per-token provenance idea: a `u32`
/// token ID and a `Source` tag, kept in lockstep. The downstream model
/// only consumes `id` (via the embedding lookup); `source` rides along
/// in a parallel buffer that the attention mechanism never touches.
#[derive(Debug, Clone, Copy)]
struct ProvenancedToken {
    id: u32,
    source: Source,
}

// ---------------------------------------------------------------------------
// Fixture parsing
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Provenanced encoding
// ---------------------------------------------------------------------------

/// Encode a sequence of labeled regions into a flat `Vec<ProvenancedToken>`.
///
/// Each region is tokenized independently (no BOS prepended — the caller
/// can prepend one BOS at the very start if desired) and the resulting
/// token IDs inherit the region's source tag. The output is a single
/// flat sequence whose order matches the input regions' order.
///
/// This is the load-bearing function: it's where bytes acquire trust
/// labels, before any model touches them. Once a token has its source
/// tag, no downstream component can rewrite that tag — it travels with
/// the token through the rest of the pipeline as read-only metadata.
fn encode_with_provenance(
    tokenizer: &Tokenizer,
    regions: &[Region],
) -> Vec<ProvenancedToken> {
    let mut out = Vec::new();
    for region in regions {
        // We deliberately do NOT add BOS here. Each region is encoded as
        // a continuation; the model loop is responsible for any BOS at
        // the very start of the full sequence.
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
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "concept-boundaries")]
struct Cli {
    /// Path to the fixture file (lines of "<source>: <content>").
    #[arg(long, default_value = "fixtures/mixed-trust.txt")]
    fixture: PathBuf,

    /// Path to a GGUF model. Required because cortex's tokenizer is loaded
    /// from GGUF metadata and there's no standalone tokenizer constructor.
    #[arg(long)]
    model: Option<PathBuf>,

    /// Future-token window for boundary scoring. For each candidate
    /// boundary i, we look at the next `window` query positions.
    #[arg(long, default_value = "12")]
    window: usize,

    /// Weight on the "leftward baseline" penalty in the boundary score.
    /// Higher alpha demands a stronger anchor signal at i relative to
    /// the average leftward position.
    #[arg(long, default_value = "1.0")]
    alpha: f32,

    /// Number of top-scored boundary candidates to print and evaluate
    /// against the fixture's known trust-region edges.
    #[arg(long, default_value = "8")]
    top_k: usize,

    /// Which layers to average attention over: "middle" (middle third),
    /// "late" (last third), "all" (every layer).
    #[arg(long, default_value = "middle")]
    layers: String,

    /// Non-max suppression radius. Boundary candidates within ±nms of a
    /// higher-scoring candidate are suppressed before reporting. Set to 0
    /// to disable.
    #[arg(long, default_value = "2")]
    nms: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let regions = parse_fixture(&cli.fixture)?;
    println!(
        "loaded {} region(s) from {}",
        regions.len(),
        cli.fixture.display()
    );
    for r in &regions {
        println!("  [{:>6}] {}", r.source.label(), r.content);
    }
    println!();

    // The model is needed to construct a tokenizer because cortex's
    // tokenizer is loaded from GGUF metadata. There is no standalone
    // tokenizer constructor today — that's a piece of cortex API
    // surface that pinky exposes as a friction point.
    let model_path = cli
        .model
        .as_ref()
        .ok_or_else(|| anyhow!("--model is required (cortex tokenizers come from GGUF)"))?;

    println!("loading model: {}", model_path.display());
    let loaded = cortex::load_model(model_path.to_str().unwrap())
        .map_err(|e| anyhow!("load_model failed: {e}"))?;
    println!(
        "  vocab_size = {}, n_layers = {}, embed_dim = {}",
        loaded.config.vocab_size, loaded.config.n_layers, loaded.config.embedding_dim,
    );
    println!();

    let tokens = encode_with_provenance(&loaded.tokenizer, &regions);
    println!("encoded {} token(s) with provenance:", tokens.len());
    println!("  {:>4}  {:>6}  {:<8}  text", "idx", "id", "source");
    println!("  {}", "-".repeat(40));
    for (i, t) in tokens.iter().enumerate() {
        let text = loaded.tokenizer.decode(&[t.id]);
        // Escape whitespace and newlines so the table stays aligned
        let text = text
            .replace('\n', "\\n")
            .replace('\t', "\\t");
        println!(
            "  {:>4}  {:>6}  {:<8}  {:?}",
            i,
            t.id,
            t.source.label(),
            text,
        );
    }
    println!();

    // Run forward_traced over the bare token IDs. The provenance buffer
    // is parallel metadata that the model never touches; we'll join it
    // back to the boundary scores below.
    let bare_ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();
    println!("running forward_traced over {} tokens...", bare_ids.len());
    let t_start = std::time::Instant::now();
    let (_logits, trace) = loaded.model.forward_traced(&bare_ids);
    let elapsed = t_start.elapsed();
    println!(
        "  done in {:.2}s — captured {} layers, {} heads, seq_len={}",
        elapsed.as_secs_f32(),
        trace.n_layers,
        trace.n_heads,
        trace.seq_len,
    );
    println!();

    // ----------------------------------------------------------------------
    // Boundary scoring (anchored, per-position normalized)
    // ----------------------------------------------------------------------
    //
    // For each candidate boundary position i, score how strongly i acts as
    // an anchor for the next few tokens:
    //
    //   s_i = mean over q ∈ (i, i+window], heads, selected layers of:
    //           attn(q, i)                       <-- look-back attention TO i
    //         - α · (1/i) · Σ_{k ∈ [0, i)} attn(q, k)   <-- avg per-leftward-position
    //
    // Both terms are "attention weight per single position" so they have
    // the same units, and the position-in-sequence dilution cancels out.
    // A real concept boundary is a position that future tokens look back
    // to disproportionately, more than they look back to any typical
    // earlier position.
    //
    // Layer selection options:
    //   "middle" — middle third of layers (relational structure)
    //   "late"   — last third (task-conditioned, more abstract)
    //   "all"    — every layer averaged
    //
    // Degenerate boundaries are skipped: i = 0 has no leftward baseline,
    // and the last position has no future tokens to score it.

    let n_layers = trace.n_layers;
    let selected_layers: Vec<usize> = match cli.layers.as_str() {
        "middle" => {
            let lo = n_layers / 3;
            let hi = (2 * n_layers / 3).max(lo + 1);
            (lo..hi).collect()
        }
        "late" => {
            let lo = 2 * n_layers / 3;
            (lo..n_layers).collect()
        }
        "all" => (0..n_layers).collect(),
        other => {
            return Err(anyhow!(
                "--layers must be 'middle', 'late', or 'all', got: {other}"
            ));
        }
    };

    let window = cli.window;
    let alpha = cli.alpha;
    let s = trace.seq_len;

    println!(
        "scoring boundaries: layers {} ({:?}), future window = {}, alpha = {}",
        cli.layers, selected_layers, window, alpha,
    );

    // f32::NEG_INFINITY for skipped positions so they sort to the bottom
    let mut scores = vec![f32::NEG_INFINITY; s];
    for i in 1..(s - 1) {
        let q_lo = i + 1;
        let q_hi = (i + 1 + window).min(s);
        if q_hi <= q_lo {
            continue;
        }

        let mut total = 0.0f32;
        let mut count = 0usize;

        for &layer in &selected_layers {
            for h in 0..trace.n_heads {
                for q in q_lo..q_hi {
                    let row = trace.attention_row(layer, h, q);
                    let attn_to_i = row[i];
                    let left_sum: f32 = row[..i].iter().sum();
                    let avg_left = left_sum / i as f32;
                    total += attn_to_i - alpha * avg_left;
                    count += 1;
                }
            }
        }

        scores[i] = if count > 0 { total / count as f32 } else { f32::NEG_INFINITY };
    }

    // ----------------------------------------------------------------------
    // Trust-region edges from the fixture (ground-truth boundaries)
    // ----------------------------------------------------------------------
    //
    // A trust-region edge is any position i where tokens[i].source !=
    // tokens[i-1].source. These are the boundaries we KNOW should exist
    // because they were imposed by the fixture's region structure.
    let trust_edges: Vec<usize> = (1..s)
        .filter(|&i| tokens[i].source != tokens[i - 1].source)
        .collect();

    println!();
    println!("trust-region edges (from fixture): {:?}", trust_edges);
    for &edge in &trust_edges {
        let prev_text = loaded.tokenizer.decode(&[tokens[edge - 1].id])
            .replace('\n', "\\n");
        let next_text = loaded.tokenizer.decode(&[tokens[edge].id])
            .replace('\n', "\\n");
        println!(
            "  pos {:>3}: {:>6} → {:<6}  ({:?} → {:?})",
            edge,
            tokens[edge - 1].source.label(),
            tokens[edge].source.label(),
            prev_text,
            next_text,
        );
    }
    println!();

    // ----------------------------------------------------------------------
    // Sort, then non-max suppress
    // ----------------------------------------------------------------------
    //
    // The model's "noticing" of a boundary tends to peak 1-2 tokens AFTER
    // the actual boundary, so adjacent positions often score similarly.
    // NMS collapses each cluster of nearby high-scoring positions into a
    // single "this is one boundary" finding, by walking the score-ranked
    // list and keeping a candidate only if no higher-scoring candidate
    // has already been kept within ±nms positions of it.

    let mut indexed: Vec<(usize, f32)> = scores
        .iter()
        .enumerate()
        .filter(|(_, &v)| v.is_finite())
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let nms_radius = cli.nms;
    let mut suppressed: Vec<(usize, f32)> = Vec::new();
    for &(pos, score) in &indexed {
        let collides = suppressed
            .iter()
            .any(|&(kept_pos, _)| pos.abs_diff(kept_pos) <= nms_radius);
        if !collides {
            suppressed.push((pos, score));
        }
    }

    let top_k = cli.top_k.min(suppressed.len());
    println!(
        "top-{top_k} boundary candidates after NMS (radius ±{nms_radius}):"
    );
    println!(
        "  {:>4}  {:>9}  {:>6}  {:<8}  {:<8}  text",
        "pos", "score", "trust?", "src_prev", "src_next"
    );
    println!("  {}", "-".repeat(60));
    for &(pos, score) in suppressed.iter().take(top_k) {
        // A candidate "hits" a trust edge if it's within ±nms_radius of one
        let is_trust_near = trust_edges
            .iter()
            .any(|&e| pos.abs_diff(e) <= nms_radius);
        let is_trust_exact = trust_edges.contains(&pos);
        let trust_marker = if is_trust_exact {
            "✓"
        } else if is_trust_near {
            "~"
        } else {
            " "
        };
        let prev_src = if pos > 0 {
            tokens[pos - 1].source.label()
        } else {
            "-"
        };
        let next_src = tokens[pos].source.label();
        let text = loaded.tokenizer.decode(&[tokens[pos].id])
            .replace('\n', "\\n");
        println!(
            "  {:>4}  {:>9.4}  {:>6}  {:<8}  {:<8}  {:?}",
            pos, score, trust_marker, prev_src, next_src, text,
        );
    }
    println!("  (✓ = exact match, ~ = within ±{nms_radius} of a trust edge)");
    println!();

    // ----------------------------------------------------------------------
    // Alignment summary: precision/recall against trust edges, with the
    // "near" tolerance reflecting that boundary signal can peak 1-2 tokens
    // off the exact edge.
    // ----------------------------------------------------------------------
    let trust_set: std::collections::HashSet<usize> =
        trust_edges.iter().copied().collect();

    let mut hits_exact = 0;
    let mut hits_near = 0;
    let mut matched_edges_exact: std::collections::HashSet<usize> =
        std::collections::HashSet::new();
    let mut matched_edges_near: std::collections::HashSet<usize> =
        std::collections::HashSet::new();

    for &(pos, _) in suppressed.iter().take(top_k) {
        if trust_set.contains(&pos) {
            hits_exact += 1;
            matched_edges_exact.insert(pos);
        }
        if let Some(&edge) = trust_edges.iter().find(|&&e| pos.abs_diff(e) <= nms_radius) {
            hits_near += 1;
            matched_edges_near.insert(edge);
        }
    }

    let precision_exact = if top_k > 0 { hits_exact as f32 / top_k as f32 } else { 0.0 };
    let precision_near = if top_k > 0 { hits_near as f32 / top_k as f32 } else { 0.0 };
    let recall_exact = if !trust_set.is_empty() {
        matched_edges_exact.len() as f32 / trust_set.len() as f32
    } else { 0.0 };
    let recall_near = if !trust_set.is_empty() {
        matched_edges_near.len() as f32 / trust_set.len() as f32
    } else { 0.0 };

    println!("alignment vs ground-truth trust edges:");
    println!("  trust edges        = {:?}", trust_edges);
    println!("  exact precision    = {:.2}  ({}/{} top-K candidates land on a trust edge)",
        precision_exact, hits_exact, top_k);
    println!("  ±{} precision      = {:.2}  ({}/{} top-K candidates within ±{} of a trust edge)",
        nms_radius, precision_near, hits_near, top_k, nms_radius);
    println!("  exact recall       = {:.2}  ({}/{} trust edges hit exactly)",
        recall_exact, matched_edges_exact.len(), trust_set.len());
    println!("  ±{} recall         = {:.2}  ({}/{} trust edges hit within ±{})",
        nms_radius, recall_near, matched_edges_near.len(), trust_set.len(), nms_radius);
    println!();

    // Provenance summary by region (kept from earlier)
    let mut counts = [0usize; 4];
    for t in &tokens {
        let idx = match t.source {
            Source::System => 0,
            Source::User => 1,
            Source::Doc => 2,
            Source::Tool => 3,
        };
        counts[idx] += 1;
    }
    println!("provenance summary:");
    for (label, n) in [
        ("system", counts[0]),
        ("user", counts[1]),
        ("doc", counts[2]),
        ("tool", counts[3]),
    ] {
        if n > 0 {
            println!("  {:<8} {} tokens", label, n);
        }
    }

    Ok(())
}
