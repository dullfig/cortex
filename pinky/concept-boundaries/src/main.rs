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

    /// Provenance bonus added to the boundary score at every position
    /// where the source tag changes (tokens[i].source != tokens[i-1].source).
    /// Zero (default) ignores provenance entirely and uses only the
    /// attention-derived score, which is the right way to test what the
    /// content alone supports. A positive value treats source changes as
    /// evidence of a concept boundary, biasing the extractor toward the
    /// "trust is a constituent of the concept" architecture.
    ///
    /// Tune by inspecting the magnitude of the raw attention scores in
    /// the report — start at the same order of magnitude as the top-K
    /// scores you see without the bonus.
    #[arg(long, default_value = "0.0")]
    provenance_bonus: f32,

    /// Optional disclosure fixture for the reframing experiment.
    ///
    /// When provided, the binary runs forward_traced TWICE:
    ///   pass 1: just the main fixture (the "conversation")
    ///   pass 2: disclosure tokens + main fixture tokens
    ///
    /// It then measures, for each token position in the conversation,
    /// how much the model's attention over that position reorganized
    /// between the two passes. Tokens whose attention patterns shifted
    /// the most are the ones whose interpretation was reframed by the
    /// disclosure — which is the operational test of "the same bytes
    /// produce a different concept structure under different context."
    #[arg(long)]
    disclosure: Option<PathBuf>,

    /// Optional control disclosure for the reframing experiment.
    ///
    /// When provided alongside --disclosure, the binary runs THREE
    /// forward passes: pass 1 (no disclosure), pass 2 (test disclosure),
    /// and pass 3 (control disclosure — should be irrelevant to the
    /// conversation). It then computes per-token DELTAS between the
    /// test and control reorganization scores, and ranks tokens by
    /// (test - control). This isolates the topic-specific reframing
    /// signal from generic prefix-effect attention drift.
    ///
    /// Use this to confirm that an apparent reframing finding isn't
    /// just a side effect of having any prefix at all.
    #[arg(long)]
    control_disclosure: Option<PathBuf>,
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
        "scoring boundaries: layers {} ({:?}), future window = {}, alpha = {}, provenance_bonus = {}",
        cli.layers, selected_layers, window, alpha, cli.provenance_bonus,
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

        let attention_score = if count > 0 { total / count as f32 } else { f32::NEG_INFINITY };

        // Provenance bonus: if the source tag changes at this position,
        // add the configured bonus. Zero by default (pure attention-based
        // boundary discovery). Positive values fold provenance into the
        // boundary signal as a prior — see the architectural argument
        // about why trust should be a constituent of concepts, not just
        // a parallel label.
        let provenance_bonus = if i > 0 && tokens[i].source != tokens[i - 1].source {
            cli.provenance_bonus
        } else {
            0.0
        };

        scores[i] = attention_score + provenance_bonus;
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

    // ----------------------------------------------------------------------
    // Reframing experiment (only if --disclosure was provided)
    // ----------------------------------------------------------------------
    //
    // Pass 2: encode the disclosure, prepend its tokens to the main
    // fixture's tokens, run forward_traced again. The disclosure adds
    // context that the model can attend to from the main-fixture
    // positions. We then compare the attention patterns over the main-
    // fixture tokens between pass 1 (no disclosure) and pass 2 (with
    // disclosure prefix) to see how much they reorganized.
    if let Some(ref disclosure_path) = cli.disclosure {
        println!();
        println!("================================================================");
        println!("REFRAMING EXPERIMENT");
        println!("================================================================");
        println!();

        let disclosure_regions = parse_fixture(disclosure_path)?;
        println!(
            "loaded disclosure: {} region(s) from {}",
            disclosure_regions.len(),
            disclosure_path.display(),
        );
        for r in &disclosure_regions {
            println!("  [{:>6}] {}", r.source.label(), r.content);
        }
        println!();

        let disclosure_tokens = encode_with_provenance(&loaded.tokenizer, &disclosure_regions);
        let m = disclosure_tokens.len();
        println!("disclosure tokens: {m}");

        // Build pass-2 token sequence: disclosure followed by main fixture
        let pass2_ids: Vec<u32> = disclosure_tokens
            .iter()
            .chain(tokens.iter())
            .map(|t| t.id)
            .collect();
        println!(
            "pass 2 sequence: {} tokens (disclosure {} + conversation {})",
            pass2_ids.len(),
            m,
            tokens.len(),
        );

        println!("running pass 2 forward_traced...");
        let t2_start = std::time::Instant::now();
        let (_logits2, trace2) = loaded.model.forward_traced(&pass2_ids);
        println!(
            "  done in {:.2}s",
            t2_start.elapsed().as_secs_f32(),
        );
        println!();

        // ------------------------------------------------------------------
        // Reframing score per conversation token
        // ------------------------------------------------------------------
        //
        // For each conversation token at position q in pass 1, the
        // corresponding position in pass 2 is q + m (because m disclosure
        // tokens were prepended). We compare the model's attention over
        // that position between the two passes by looking at the
        // *within-conversation* attention rows.
        //
        // Specifically, for query position q in pass 1, the attention row
        // covers keys 0..=q (causal). For query position q + m in pass 2,
        // the row covers keys 0..=(q + m), where keys [0, m) are the
        // disclosure tokens and keys [m, m + q] are the conversation
        // tokens.
        //
        // We measure two things:
        //   - frac_to_disclosure: how much of the attention in pass 2 is
        //     spent on the disclosure tokens (vs the conversation). High
        //     values mean the model is "consulting" the disclosure when
        //     processing this conversation token.
        //   - cosine_dist: cosine distance between pass 1's full row and
        //     the conversation portion of pass 2's row, normalized so
        //     they sum to 1. High values mean even after factoring out
        //     the disclosure pull, the within-conversation attention
        //     pattern reorganized.
        //
        // Both averaged over heads and selected layers.

        let s1 = trace.seq_len;
        assert_eq!(s1, tokens.len(), "pass 1 seq_len should match token count");

        let mut reframe_data: Vec<(usize, f32, f32)> = Vec::with_capacity(s1);

        for q in 0..s1 {
            let q2 = q + m;
            let mut total_frac = 0.0f32;
            let mut total_cos_dist = 0.0f32;
            let mut count = 0usize;

            for &layer in &selected_layers {
                for h in 0..trace.n_heads {
                    let row1 = trace.attention_row(layer, h, q);
                    let row2 = trace2.attention_row(layer, h, q2);

                    // Pass 1 attention is over [0, q]; rest is causal-zero.
                    // Pass 2 attention is over [0, q2] = [0, m+q]; first m
                    // entries are over the disclosure, next q+1 entries are
                    // over the conversation.
                    let row1_conv = &row1[..=q]; // length q+1
                    let row2_disc = &row2[..m];  // length m
                    let row2_conv = &row2[m..=q2]; // length q+1

                    let frac_to_disc: f32 = row2_disc.iter().sum();
                    total_frac += frac_to_disc;

                    // Renormalize the conversation portion of pass 2 to sum
                    // to 1 (since pass 1's row also sums to 1) and compare.
                    let conv_sum: f32 = row2_conv.iter().sum();
                    if conv_sum > 1e-8 {
                        let mut dot = 0.0f32;
                        let mut n1 = 0.0f32;
                        let mut n2 = 0.0f32;
                        for (a, b) in row1_conv.iter().zip(row2_conv.iter()) {
                            let b_norm = b / conv_sum;
                            dot += a * b_norm;
                            n1 += a * a;
                            n2 += b_norm * b_norm;
                        }
                        let denom = (n1.sqrt() * n2.sqrt()).max(1e-8);
                        let cos_sim = (dot / denom).clamp(-1.0, 1.0);
                        total_cos_dist += 1.0 - cos_sim;
                    }
                    count += 1;
                }
            }

            let avg_frac = total_frac / count as f32;
            let avg_cos_dist = total_cos_dist / count as f32;
            reframe_data.push((q, avg_frac, avg_cos_dist));
        }

        // Skip position 0 (no meaningful comparison) and rank by
        // cosine distance — that's the reorganization signal independent
        // of the absolute pull from the disclosure.
        let mut ranked: Vec<(usize, f32, f32)> = reframe_data
            .iter()
            .copied()
            .filter(|&(q, _, _)| q > 0)
            .collect();
        ranked.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        let top_k = cli.top_k.min(ranked.len());
        println!("top-{top_k} reframed tokens (highest within-conversation reorganization):");
        println!(
            "  {:>4}  {:>10}  {:>10}  text",
            "pos", "cos_dist", "frac_disc"
        );
        println!("  {}", "-".repeat(60));
        for &(q, frac, cd) in ranked.iter().take(top_k) {
            let text = loaded.tokenizer.decode(&[tokens[q].id])
                .replace('\n', "\\n");
            println!(
                "  {:>4}  {:>10.4}  {:>10.4}  {:?}",
                q, cd, frac, text,
            );
        }
        println!();

        // Also report the raw per-token table so we can see the full
        // pattern, not just the top.
        println!("per-token reframing trace (cos_dist = within-conv reorganization, frac_disc = attention to disclosure):");
        println!(
            "  {:>4}  {:>10}  {:>10}  text",
            "pos", "cos_dist", "frac_disc"
        );
        println!("  {}", "-".repeat(60));
        for &(q, frac, cd) in &reframe_data {
            let text = loaded.tokenizer.decode(&[tokens[q].id])
                .replace('\n', "\\n");
            println!(
                "  {:>4}  {:>10.4}  {:>10.4}  {:?}",
                q, cd, frac, text,
            );
        }
        println!();

        let mean_cos_dist: f32 = reframe_data.iter().skip(1).map(|x| x.2).sum::<f32>()
            / (reframe_data.len() - 1) as f32;
        let mean_frac: f32 = reframe_data.iter().skip(1).map(|x| x.1).sum::<f32>()
            / (reframe_data.len() - 1) as f32;
        println!("summary:");
        println!("  mean cos_dist  = {:.4}  (avg within-conversation reorganization)", mean_cos_dist);
        println!("  mean frac_disc = {:.4}  (avg attention pulled to disclosure)", mean_frac);

        // ------------------------------------------------------------------
        // Control disclosure comparison
        // ------------------------------------------------------------------
        if let Some(ref control_path) = cli.control_disclosure {
            println!();
            println!("----- CONTROL COMPARISON -----");
            println!();

            let control_regions = parse_fixture(control_path)?;
            println!(
                "loaded control disclosure: {} region(s) from {}",
                control_regions.len(),
                control_path.display(),
            );
            for r in &control_regions {
                println!("  [{:>6}] {}", r.source.label(), r.content);
            }

            let control_tokens = encode_with_provenance(&loaded.tokenizer, &control_regions);
            let mc = control_tokens.len();
            println!("control disclosure tokens: {mc}");

            let pass3_ids: Vec<u32> = control_tokens
                .iter()
                .chain(tokens.iter())
                .map(|t| t.id)
                .collect();
            println!("running pass 3 (control) forward_traced...");
            let t3_start = std::time::Instant::now();
            let (_logits3, trace3) = loaded.model.forward_traced(&pass3_ids);
            println!(
                "  done in {:.2}s",
                t3_start.elapsed().as_secs_f32(),
            );
            println!();

            // Compute control reframe data using the same logic as the
            // test disclosure (just with mc instead of m, and trace3
            // instead of trace2).
            let mut control_data: Vec<(usize, f32, f32)> = Vec::with_capacity(s1);
            for q in 0..s1 {
                let q3 = q + mc;
                let mut total_frac = 0.0f32;
                let mut total_cos_dist = 0.0f32;
                let mut count = 0usize;
                for &layer in &selected_layers {
                    for h in 0..trace.n_heads {
                        let row1 = trace.attention_row(layer, h, q);
                        let row3 = trace3.attention_row(layer, h, q3);
                        let row1_conv = &row1[..=q];
                        let row3_disc = &row3[..mc];
                        let row3_conv = &row3[mc..=q3];
                        let frac_to_disc: f32 = row3_disc.iter().sum();
                        total_frac += frac_to_disc;
                        let conv_sum: f32 = row3_conv.iter().sum();
                        if conv_sum > 1e-8 {
                            let mut dot = 0.0f32;
                            let mut n1 = 0.0f32;
                            let mut n2 = 0.0f32;
                            for (a, b) in row1_conv.iter().zip(row3_conv.iter()) {
                                let b_norm = b / conv_sum;
                                dot += a * b_norm;
                                n1 += a * a;
                                n2 += b_norm * b_norm;
                            }
                            let denom = (n1.sqrt() * n2.sqrt()).max(1e-8);
                            let cos_sim = (dot / denom).clamp(-1.0, 1.0);
                            total_cos_dist += 1.0 - cos_sim;
                        }
                        count += 1;
                    }
                }
                let avg_frac = total_frac / count as f32;
                let avg_cos_dist = total_cos_dist / count as f32;
                control_data.push((q, avg_frac, avg_cos_dist));
            }

            // Compute deltas: test cos_dist minus control cos_dist.
            // Positive delta = the test disclosure produced MORE reorganization
            // at this position than the control did = topic-specific reframing.
            let mut deltas: Vec<(usize, f32, f32, f32)> = Vec::with_capacity(s1);
            for q in 1..s1 {
                let test_cd = reframe_data[q].2;
                let ctrl_cd = control_data[q].2;
                let delta = test_cd - ctrl_cd;
                deltas.push((q, test_cd, ctrl_cd, delta));
            }

            // Sort by delta descending — top positions are where the test
            // disclosure reframed and the control did not.
            deltas.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

            let top_k = cli.top_k.min(deltas.len());
            println!(
                "top-{top_k} TOPIC-SPECIFIC reframed tokens (test cos_dist - control cos_dist):"
            );
            println!(
                "  {:>4}  {:>10}  {:>10}  {:>10}  text",
                "pos", "delta", "test_cd", "ctrl_cd"
            );
            println!("  {}", "-".repeat(70));
            for &(q, test_cd, ctrl_cd, delta) in deltas.iter().take(top_k) {
                let text = loaded.tokenizer.decode(&[tokens[q].id])
                    .replace('\n', "\\n");
                println!(
                    "  {:>4}  {:>+10.4}  {:>10.4}  {:>10.4}  {:?}",
                    q, delta, test_cd, ctrl_cd, text,
                );
            }
            println!();

            // Bottom-K: positions where the CONTROL reframed more than the
            // test. These should be conversation tokens that are intrinsically
            // related to the control topic but unrelated to the test topic.
            // For our dog vs bridge example, these might surface words like
            // "without" or other words that have spatial/architectural
            // associations.
            println!(
                "bottom-{top_k} TOPIC-SPECIFIC tokens (control reframed MORE than test):"
            );
            println!(
                "  {:>4}  {:>10}  {:>10}  {:>10}  text",
                "pos", "delta", "test_cd", "ctrl_cd"
            );
            println!("  {}", "-".repeat(70));
            for &(q, test_cd, ctrl_cd, delta) in deltas.iter().rev().take(top_k) {
                let text = loaded.tokenizer.decode(&[tokens[q].id])
                    .replace('\n', "\\n");
                println!(
                    "  {:>4}  {:>+10.4}  {:>10.4}  {:>10.4}  {:?}",
                    q, delta, test_cd, ctrl_cd, text,
                );
            }
            println!();

            let mean_test: f32 = reframe_data.iter().skip(1).map(|x| x.2).sum::<f32>()
                / (s1 - 1) as f32;
            let mean_ctrl: f32 = control_data.iter().skip(1).map(|x| x.2).sum::<f32>()
                / (s1 - 1) as f32;
            let mean_delta = mean_test - mean_ctrl;
            println!("control comparison summary:");
            println!("  mean test cos_dist     = {:.4}", mean_test);
            println!("  mean control cos_dist  = {:.4}", mean_ctrl);
            println!("  mean delta             = {:+.4}  (positive = test reframed more)", mean_delta);
        }
    }

    Ok(())
}
