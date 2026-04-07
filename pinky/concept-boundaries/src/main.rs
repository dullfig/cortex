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
    /// boundary i, we look at the next `window` query positions and
    /// measure how cleanly their attention stops at i.
    #[arg(long, default_value = "6")]
    window: usize,

    /// Weight on the "leftward bleed" penalty in the boundary score.
    /// Higher alpha penalizes future tokens that still attend to positions
    /// before the candidate boundary.
    #[arg(long, default_value = "1.0")]
    alpha: f32,

    /// Number of top-scored boundary candidates to print and evaluate
    /// against the fixture's known trust-region edges.
    #[arg(long, default_value = "8")]
    top_k: usize,
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
    // Boundary scoring (DynSplit-KV-style)
    // ----------------------------------------------------------------------
    //
    // For each candidate boundary position i, score how cleanly the
    // attention pattern of future positions stops at i:
    //
    //   s_i = mean over q ∈ (i, i+window], heads, selected layers of:
    //           Σ attn(q, k) for k ∈ [i, q]      <-- "rightward focus"
    //         - α · Σ attn(q, k) for k ∈ [0, i)  <-- "leftward bleed"
    //
    // High s_i means future positions are mostly attending to things at
    // or after i, ignoring everything before i. That's the operational
    // signature of a concept boundary: the future loses interest in the
    // past at exactly that point.
    //
    // Layer selection: we average over the middle third of the network.
    // Early layers are too syntactic (token n-grams), late layers are
    // too task-specific (next-token prediction). Middle layers carry
    // the abstract relational structure that makes "concept" meaningful.

    let n_layers = trace.n_layers;
    let layer_lo = n_layers / 3;
    let layer_hi = (2 * n_layers / 3).max(layer_lo + 1);
    let selected_layers: Vec<usize> = (layer_lo..layer_hi).collect();

    let window = cli.window;
    let alpha = cli.alpha;
    let s = trace.seq_len;

    println!(
        "scoring boundaries: layers {:?}, future window = {}, alpha = {}",
        selected_layers, window, alpha,
    );

    let mut scores = vec![0.0f32; s];
    for i in 0..s {
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
                    // Rightward focus: attention from q to keys in [i, q].
                    // (k > q is masked to 0 by causality, so safe to slice up
                    //  to s; but we cap at q+1 to be explicit.)
                    let right_sum: f32 = row[i..=q].iter().sum();
                    // Leftward bleed: attention from q to keys in [0, i).
                    let left_sum: f32 = if i > 0 { row[..i].iter().sum() } else { 0.0 };
                    total += right_sum - alpha * left_sum;
                    count += 1;
                }
            }
        }

        scores[i] = if count > 0 { total / count as f32 } else { 0.0 };
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
    // Top-K boundary candidates
    // ----------------------------------------------------------------------
    let mut indexed: Vec<(usize, f32)> = scores
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top_k = cli.top_k.min(s);
    println!("top-{top_k} boundary candidates by score:");
    println!(
        "  {:>4}  {:>9}  {:>6}  {:<8}  {:<8}  text",
        "pos", "score", "trust?", "src_prev", "src_next"
    );
    println!("  {}", "-".repeat(60));
    for &(pos, score) in indexed.iter().take(top_k) {
        let is_trust = trust_edges.contains(&pos);
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
            pos,
            score,
            if is_trust { "✓" } else { " " },
            prev_src,
            next_src,
            text,
        );
    }
    println!();

    // ----------------------------------------------------------------------
    // Alignment summary: how many of the top-K candidates are real edges?
    // ----------------------------------------------------------------------
    let top_set: std::collections::HashSet<usize> =
        indexed.iter().take(top_k).map(|&(p, _)| p).collect();
    let trust_set: std::collections::HashSet<usize> =
        trust_edges.iter().copied().collect();
    let hits = top_set.intersection(&trust_set).count();
    let precision = if top_k > 0 { hits as f32 / top_k as f32 } else { 0.0 };
    let recall = if !trust_set.is_empty() {
        hits as f32 / trust_set.len() as f32
    } else {
        0.0
    };

    println!("alignment vs ground-truth trust edges:");
    println!("  hits     = {hits} / {top_k} top candidates");
    println!("  trust    = {} edges in fixture", trust_set.len());
    println!("  precision = {:.2}  (top-K hits / top-K)", precision);
    println!("  recall    = {:.2}  (top-K hits / trust edges)", recall);
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
