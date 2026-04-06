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

    /// Optional path to a GGUF model. If provided, the binary runs a
    /// forward pass over the encoded tokens to prove end-to-end plumbing.
    /// If omitted, only the encoding + provenance trace is printed.
    #[arg(long)]
    model: Option<PathBuf>,
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

    // Run a forward pass over the bare token IDs (provenance is parallel
    // metadata — the model consumes only the IDs). This is just to prove
    // the pipeline survives end-to-end; we don't do anything with the
    // logits yet.
    let bare_ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();
    let logits = loaded.model.forward_last(&bare_ids, 0);
    let (top_id, top_logit) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, &v)| (i as u32, v))
        .unwrap();
    let top_text = loaded.tokenizer.decode(&[top_id]);
    println!(
        "forward_last: top token = {} ({:?}, logit = {:.4})",
        top_id, top_text, top_logit,
    );
    println!();

    // Provenance summary by region
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
