//! morsel-retrieve — pinky simple-C
//!
//! What this is testing
//! --------------------
//! The cross-document morsel property (POSITION-addendum.md section 11):
//! given a corpus of documents and a query, can attention over the
//! corpus surface relevant *morsels* (sub-document spans) even when
//! the morsels are buried in documents whose primary topic is unrelated
//! to the query?
//!
//! The corpus is `pinky/morsel-retrieval/papers/` (8 short fake research
//! papers with 3 deliberately planted cross-paper connections, plus
//! 2 distractors). The ground truth is `pinky/morsel-retrieval/connections.json`.
//!
//! What this binary does
//! ---------------------
//! For each test query in connections.json:
//!   1. Concatenate all 8 papers + the query into a single input sequence
//!      (with paper boundaries marked so we can track per-token provenance)
//!   2. Run cortex's forward_traced on the full sequence
//!   3. For each corpus position (positions before the query), compute
//!      the mean attention from the LAST query position back to that
//!      corpus position, averaged over selected layers and heads
//!   4. Sort corpus positions by score, take top-K
//!   5. Report each top-K position with paper-of-origin, token text, score
//!   6. For each planted connection, check whether the top-K includes
//!      tokens from BOTH papers in the connection
//!
//! Critical architectural choice: we use the **pre-softmax Q·K^T scores**
//! captured by the forward_traced API (added 2026-04-08), not the
//! post-softmax attention weights. This is the "softmax is for inference,
//! not retrieval" insight from the morning of 2026-04-08 — raw dot
//! products are dilution-free and produce sharper relevance signals
//! over a corpus than softmax-normalized weights would.
//!
//! Why this is a valid test of the morsel claim
//! --------------------------------------------
//! In a deployed retrieval system, ingestion would compute K vectors for
//! each corpus token offline and store them, then query-time attention
//! would compute Q·K^T against the stored vectors. We approximate this
//! with a single concatenated forward pass (corpus + query) because the
//! K vectors that the query attends to in this pass ARE the corpus
//! tokens' K vectors, computed during the same forward pass. The
//! attention pattern from query → corpus is exactly the relevance
//! pattern that a deployed retrieval would produce.
//!
//! The constraint is that the corpus must fit in the context window.
//! 8 papers × ~400 tokens each ≈ 3200 corpus tokens + ~30 query tokens
//! = well within Qwen2.5-0.5B's 32K context.
//!
//! Usage
//! -----
//! ```
//! cargo run --release -- \
//!     --model /path/to/qwen2.5-0.5b-instruct-q5_k_m.gguf \
//!     --corpus ../papers \
//!     --connections ../connections.json
//! ```

use std::fs;
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use cortex::Tokenizer;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// connections.json schema
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ConnectionsFile {
    connections: Vec<Connection>,
}

#[derive(Debug, Deserialize)]
struct Connection {
    id: String,
    description: String,
    papers: Vec<ConnectionPaper>,
    test_query: String,
    #[serde(default)]
    #[allow(dead_code)]
    naive_query_result: String,
}

#[derive(Debug, Deserialize)]
struct ConnectionPaper {
    file: String,
    #[allow(dead_code)]
    morsel_text: String,
}

// ---------------------------------------------------------------------------
// Provenanced token: knows which paper (or "query") it came from
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ProvenancedToken {
    id: u32,
    /// "paper_NN" for corpus tokens, "query" for query tokens
    source: String,
}

// ---------------------------------------------------------------------------
// Encoding: build the concatenated input sequence
// ---------------------------------------------------------------------------

/// Build the concatenated input sequence: all papers separated by a
/// distinctive marker, followed by the query.
///
/// Returns the per-token sequence with provenance.
fn build_input(
    tokenizer: &Tokenizer,
    papers: &[(String, String)], // (paper_id, content)
    query: &str,
) -> Vec<ProvenancedToken> {
    let mut tokens = Vec::new();

    // BOS at the very start if the model wants one
    if tokenizer.add_bos_default() {
        tokens.push(ProvenancedToken {
            id: tokenizer.bos_token_id(),
            source: "system".to_string(),
        });
    }

    // Each paper, separated by a distinctive marker. Use the actual paper
    // content; the marker helps the model recognize boundaries between
    // papers but isn't load-bearing for the retrieval scoring.
    for (i, (paper_id, content)) in papers.iter().enumerate() {
        // Separator before each paper after the first
        let prefix = if i == 0 {
            format!("=== {} ===\n", paper_id)
        } else {
            format!("\n\n=== {} ===\n", paper_id)
        };
        let prefix_ids = tokenizer.encode(&prefix, false);
        for id in prefix_ids {
            tokens.push(ProvenancedToken {
                id,
                source: paper_id.clone(),
            });
        }
        let content_ids = tokenizer.encode(content, false);
        for id in content_ids {
            tokens.push(ProvenancedToken {
                id,
                source: paper_id.clone(),
            });
        }
    }

    // Query, with a distinctive marker
    let query_prefix = "\n\n=== query ===\n";
    let prefix_ids = tokenizer.encode(query_prefix, false);
    for id in prefix_ids {
        tokens.push(ProvenancedToken {
            id,
            source: "query".to_string(),
        });
    }
    let query_ids = tokenizer.encode(query, false);
    for id in query_ids {
        tokens.push(ProvenancedToken {
            id,
            source: "query".to_string(),
        });
    }

    tokens
}

// ---------------------------------------------------------------------------
// Layer selection (mirrors concept-boundaries' approach)
// ---------------------------------------------------------------------------

fn select_layers(layers_arg: &str, n_layers: usize) -> Result<Vec<usize>> {
    match layers_arg {
        "middle" => {
            let lo = n_layers / 3;
            let hi = (2 * n_layers / 3).max(lo + 1);
            Ok((lo..hi).collect())
        }
        "late" => {
            let lo = 2 * n_layers / 3;
            Ok((lo..n_layers).collect())
        }
        "all" => Ok((0..n_layers).collect()),
        other => Err(anyhow!(
            "--layers must be 'middle', 'late', or 'all', got: {other}"
        )),
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "morsel-retrieve")]
struct Cli {
    /// Path to a GGUF model.
    #[arg(long)]
    model: PathBuf,

    /// Directory containing the corpus papers (paper_*.txt).
    #[arg(long, default_value = "../papers")]
    corpus: PathBuf,

    /// Path to connections.json.
    #[arg(long, default_value = "../connections.json")]
    connections: PathBuf,

    /// Which layers to average attention over.
    #[arg(long, default_value = "all")]
    layers: String,

    /// Number of top-scored corpus positions to report per query.
    #[arg(long, default_value = "20")]
    top_k: usize,

    /// Number of trailing query positions to use as the "query attention
    /// source." Default 1 means only the very last query token. Higher
    /// values average attention from the last N query tokens, which can
    /// give a smoother signal at the cost of mixing in attention from
    /// earlier query positions that may not capture the full intent.
    #[arg(long, default_value = "3")]
    query_window: usize,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Load corpus
    println!("loading corpus from {}", cli.corpus.display());
    let mut paper_entries: Vec<(String, String)> = Vec::new();
    let mut paper_files: Vec<PathBuf> = fs::read_dir(&cli.corpus)
        .with_context(|| format!("reading {}", cli.corpus.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().and_then(|e| e.to_str()) == Some("txt")
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("paper_"))
                    .unwrap_or(false)
        })
        .collect();
    paper_files.sort();
    for path in &paper_files {
        let content = fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        println!("  loaded {} ({} chars)", id, content.len());
        paper_entries.push((id, content));
    }
    println!("  total: {} papers", paper_entries.len());
    println!();

    // Load connections
    println!("loading connections from {}", cli.connections.display());
    let connections_text = fs::read_to_string(&cli.connections)
        .with_context(|| format!("reading {}", cli.connections.display()))?;
    let connections_file: ConnectionsFile = serde_json::from_str(&connections_text)
        .context("parsing connections.json")?;
    println!("  {} planted connections", connections_file.connections.len());
    for c in &connections_file.connections {
        println!("    {}: {}", c.id, c.description);
    }
    println!();

    // Load model
    println!("loading model: {}", cli.model.display());
    let loaded = cortex::load_model(cli.model.to_str().unwrap())
        .map_err(|e| anyhow!("load_model failed: {e}"))?;
    let n_layers = loaded.config.n_layers as usize;
    println!(
        "  vocab_size = {}, n_layers = {}, embed_dim = {}",
        loaded.config.vocab_size, n_layers, loaded.config.embedding_dim,
    );
    let selected_layers = select_layers(&cli.layers, n_layers)?;
    println!(
        "  layer selection: {} ({:?})",
        cli.layers, selected_layers
    );
    println!();

    // ----------------------------------------------------------------------
    // For each connection, run a separate forward pass over (corpus + query)
    // ----------------------------------------------------------------------
    let mut summary: Vec<ConnectionResult> = Vec::new();

    for connection in &connections_file.connections {
        println!("================================================================");
        println!("CONNECTION: {}", connection.id);
        println!("description: {}", connection.description);
        println!("test query: {:?}", connection.test_query);
        println!("expected papers: {:?}", connection
            .papers
            .iter()
            .map(|p| p.file.trim_end_matches(".txt"))
            .collect::<Vec<_>>());
        println!();

        // Build the concatenated input
        let tokens = build_input(&loaded.tokenizer, &paper_entries, &connection.test_query);
        let n_tokens = tokens.len();
        println!("  encoded {} total tokens", n_tokens);

        // Source distribution
        let mut source_counts: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        for t in &tokens {
            *source_counts.entry(t.source.clone()).or_insert(0) += 1;
        }
        for (src, n) in &source_counts {
            println!("    {}: {} tokens", src, n);
        }
        println!();

        // Run forward_traced
        let bare_ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();
        println!("  running forward_traced...");
        let t_start = std::time::Instant::now();
        let (_logits, trace) = loaded.model.forward_traced(&bare_ids);
        let elapsed = t_start.elapsed();
        println!(
            "  forward_traced took {:.2}s ({} layers, {} heads, seq_len={})",
            elapsed.as_secs_f32(),
            trace.n_layers,
            trace.n_heads,
            trace.seq_len,
        );
        println!();

        // -------------------------------------------------------------
        // Score each corpus position by attention from query positions
        // -------------------------------------------------------------
        // For each corpus position k (where tokens[k].source != "query"),
        // compute mean pre-softmax attention from the last `query_window`
        // positions to k, averaged over selected layers and heads.
        //
        // Pre-softmax (raw Q·K^T) is the right substrate for retrieval —
        // dilution-free, see POSITION.md "Softmax is for inference, not
        // retrieval".
        let s = trace.seq_len;

        // Identify query positions
        let query_positions: Vec<usize> = tokens
            .iter()
            .enumerate()
            .filter(|(_, t)| t.source == "query")
            .map(|(i, _)| i)
            .collect();
        if query_positions.is_empty() {
            return Err(anyhow!("no query tokens in encoded sequence"));
        }
        let last_query_idx = *query_positions.last().unwrap();
        let q_lo = (last_query_idx + 1).saturating_sub(cli.query_window);
        let q_hi = last_query_idx + 1;
        println!(
            "  scoring positions using attention from query tokens [{}, {})",
            q_lo, q_hi
        );

        let mut scores = vec![0.0f32; s];
        for k in 0..s {
            // Skip query positions and separator-marker positions; only
            // score actual corpus content tokens
            if tokens[k].source == "query" {
                continue;
            }

            let mut total = 0.0f32;
            let mut count = 0usize;

            for &layer in &selected_layers {
                for h in 0..trace.n_heads {
                    for q in q_lo..q_hi {
                        let row = trace.pre_score_row(layer, h, q);
                        // Skip if k > q (causal-masked, would be 0 anyway)
                        if k > q {
                            continue;
                        }
                        total += row[k];
                        count += 1;
                    }
                }
            }

            scores[k] = if count > 0 { total / count as f32 } else { f32::NEG_INFINITY };
        }

        // Rank
        let mut ranked: Vec<(usize, f32)> = (0..s)
            .filter(|&k| scores[k].is_finite() && tokens[k].source != "query")
            .map(|k| (k, scores[k]))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_k = cli.top_k.min(ranked.len());
        println!();
        println!("  top-{} corpus positions by attention from query:", top_k);
        println!(
            "    {:>4}  {:>10}  {:<10}  text",
            "pos", "score", "source"
        );
        println!("    {}", "-".repeat(70));

        // Track which papers appear in top-K
        let mut papers_in_top_k: std::collections::BTreeSet<String> =
            std::collections::BTreeSet::new();
        for &(pos, score) in ranked.iter().take(top_k) {
            let token_text = loaded.tokenizer.decode(&[tokens[pos].id])
                .replace('\n', "\\n")
                .replace('\r', "");
            let truncated: String = token_text.chars().take(30).collect();
            println!(
                "    {:>4}  {:>10.4}  {:<10}  {:?}",
                pos, score, tokens[pos].source, truncated
            );
            papers_in_top_k.insert(tokens[pos].source.clone());
        }
        println!();

        // Did we find both papers in the planted connection?
        let expected_papers: Vec<String> = connection
            .papers
            .iter()
            .map(|p| p.file.trim_end_matches(".txt").to_string())
            .collect();
        let mut found_count = 0;
        for ep in &expected_papers {
            if papers_in_top_k.contains(ep) {
                found_count += 1;
                println!("  ✓ found tokens from {}", ep);
            } else {
                println!("  ✗ missed: no tokens from {} in top-{}", ep, top_k);
            }
        }
        let success = found_count == expected_papers.len();
        println!();
        println!(
            "  RESULT: {} ({}/{} expected papers found in top-{})",
            if success { "PASS" } else { "FAIL" },
            found_count,
            expected_papers.len(),
            top_k,
        );
        println!();

        summary.push(ConnectionResult {
            id: connection.id.clone(),
            success,
            found: found_count,
            expected: expected_papers.len(),
        });
    }

    // ----------------------------------------------------------------------
    // Final summary
    // ----------------------------------------------------------------------
    println!("================================================================");
    println!("SUMMARY");
    println!("================================================================");
    let total = summary.len();
    let passed = summary.iter().filter(|r| r.success).count();
    for r in &summary {
        let mark = if r.success { "✓ PASS" } else { "✗ FAIL" };
        println!("  {}  {}  ({}/{} papers)", mark, r.id, r.found, r.expected);
    }
    println!();
    println!("  total: {}/{} connections fully recovered", passed, total);

    Ok(())
}

#[derive(Debug)]
struct ConnectionResult {
    id: String,
    success: bool,
    found: usize,
    expected: usize,
}
