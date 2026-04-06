//! Memory trait — optional persistent memory for the transformer.
//!
//! When `memory: true`, the transformer's KV cache becomes a persistent,
//! compressed, hierarchical memory that survives across conversations.
//! The same model that generates text also encodes and retrieves memories
//! — one embedding space, one mind.
//!
//! This is the trait boundary. ternary-rs defines the interface;
//! engram provides the compressed implementation.
//!
//! # Two modes, one model
//!
//! ```text
//! TransformerModel
//!   ├── generate()   → causal attention, appends to cache, produces logits
//!   └── retrieve()   → bidirectional attention, read-only on cache, returns scores
//! ```
//!
//! The only differences between generate and retrieve:
//! - Retrieve has no causal mask (query attends to ALL cached positions)
//! - Retrieve returns attention scores, not logits
//! - Retrieve does not append to the cache (read-only)
//! - Retrieve does not use V projections (scores come from Q·K only)

/// A retrieved memory span with its source text and relevance.
#[derive(Debug, Clone)]
pub struct MemoryResult {
    /// Original text of this memory span.
    pub text: String,
    /// Role of the speaker (user, assistant, system, tool).
    pub role: MemoryRole,
    /// Conversation turn this came from (if tracked).
    pub turn_id: Option<u64>,
    /// Relevance score (higher = more relevant to the query).
    pub score: f32,
    /// Which memory tier this result came from.
    pub tier: MemoryTier,
}

/// Who said this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryRole {
    User,
    Assistant,
    System,
    Tool,
}

/// Which tier of the memory hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTier {
    /// Working memory — recent, full resolution.
    L1,
    /// Session memory — chunk summaries from L1.
    L2,
    /// Archive — episode summaries from L2.
    L3,
}

/// Statistics about the memory state.
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Positions currently in L1 (working memory).
    pub l1_len: usize,
    pub l1_capacity: usize,
    /// Positions in L2 (session summaries).
    pub l2_len: usize,
    pub l2_capacity: usize,
    /// Positions in L3 (archive).
    pub l3_len: usize,
    pub l3_capacity: usize,
    /// Normalized attention entropy from last query (0.0=focused, 1.0=diffuse).
    /// This is the "drowsiness signal" — high means consolidation needed.
    pub entropy: f32,
    /// Total memory usage in bytes across all tiers.
    pub memory_bytes: usize,
}

/// Configuration for the memory system.
///
/// Maps to YAML config in an AgentOS organism:
/// ```yaml
/// memory:
///   compression: polarquant   # or "none" for f32
///   tiers: 3
///   consolidation: entropy
///   l1_capacity: 4096
///   l2_capacity: 2048
///   l3_capacity: 512
///   chunk_size: 64
///   entropy_threshold: 0.85
///   pressure_threshold: 0.80
/// ```
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// L1 working memory capacity (positions).
    pub l1_capacity: usize,
    /// L2 session memory capacity (positions).
    pub l2_capacity: usize,
    /// L3 archive capacity (positions).
    pub l3_capacity: usize,
    /// Chunk size for consolidation (positions per chunk).
    pub chunk_size: usize,
    /// Entropy threshold for sleep-triggered consolidation.
    pub entropy_threshold: f32,
    /// Fill ratio threshold for pressure-triggered consolidation.
    pub pressure_threshold: f32,
    /// Whether to use compressed storage (PolarQuant 3-bit).
    /// false = standard f32 KV cache (no memory savings, but simpler).
    pub compressed: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            l1_capacity: 4096,
            l2_capacity: 2048,
            l3_capacity: 512,
            chunk_size: 64,
            entropy_threshold: 0.85,
            pressure_threshold: 0.80,
            compressed: true,
        }
    }
}

/// Report from a consolidation pass.
#[derive(Debug, Clone)]
pub struct ConsolidationReport {
    /// How many positions were evicted from L1.
    pub l1_evicted: usize,
    /// How many summary positions were added to L2.
    pub l2_added: usize,
    /// Whether L2→L3 cascade also happened.
    pub l3_cascaded: bool,
    /// What triggered this consolidation.
    pub trigger: ConsolidationTrigger,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsolidationTrigger {
    /// Entropy exceeded threshold — attention is too diffuse.
    Entropy,
    /// L1 fill ratio exceeded threshold.
    Pressure,
    /// Both entropy and pressure.
    Both,
    /// Manually triggered (e.g., session boundary).
    Manual,
    /// No consolidation needed.
    None,
}

// ---------------------------------------------------------------------------
// The trait
// ---------------------------------------------------------------------------

/// Persistent memory backend for a transformer model.
///
/// Implementors provide compressed storage, tiered retrieval, and
/// consolidation. The transformer calls these methods instead of
/// (or in addition to) its standard KV cache operations.
///
/// # Contract
///
/// - `ingest()` encodes text using the **same model's** Q/K projections.
///   The implementor receives pre-projected K vectors, not raw text.
///   This ensures the memory lives in the same embedding space as inference.
///
/// - `retrieve()` performs bidirectional attention (no causal mask)
///   over the compressed cache and returns ranked text spans.
///
/// - `consolidate()` runs the sleep cycle: evict noise from L1,
///   migrate summaries to L2/L3. Can be called explicitly or
///   triggered automatically after `retrieve()`.
pub trait TransformerMemory: Send + Sync {
    /// Ingest pre-projected K vectors into L1 working memory.
    ///
    /// `keys`: flat `[n_tokens, n_kv_heads * head_dim]` — already RoPE'd.
    /// `text`: the original text these keys were projected from.
    /// `role`: who said it.
    ///
    /// The transformer calls this after projecting K through its own layers.
    /// This is NOT raw text — it's the model's own key vectors.
    fn ingest(
        &mut self,
        keys: &[f32],
        n_tokens: usize,
        text: &str,
        role: MemoryRole,
        turn_id: Option<u64>,
    );

    /// Retrieve relevant memories for a query.
    ///
    /// `queries`: flat `[n_tokens, n_heads * head_dim]` — already RoPE'd.
    /// These are the model's own Q projections for the new input.
    ///
    /// Returns the top-k most relevant memory spans, ranked by
    /// attention score across all tiers.
    ///
    /// As a side effect, updates entropy and chunk scores for
    /// consolidation decisions.
    fn retrieve(
        &self,
        queries: &[f32],
        n_tokens: usize,
        n_heads: usize,
        top_k: usize,
    ) -> Vec<MemoryResult>;

    /// Check the current entropy (drowsiness signal).
    ///
    /// `queries`: same as retrieve. Computes attention entropy
    /// without doing a full retrieval pass.
    fn entropy(
        &self,
        queries: &[f32],
        n_tokens: usize,
        n_heads: usize,
    ) -> f32;

    /// Run consolidation if thresholds are met.
    ///
    /// Returns a report describing what happened (or `None` if
    /// no consolidation was needed).
    fn consolidate(&mut self) -> Option<ConsolidationReport>;

    /// Force consolidation regardless of thresholds.
    /// Use at session boundaries or explicit "dream" triggers.
    fn force_consolidate(&mut self) -> ConsolidationReport;

    /// Memory statistics across all tiers.
    fn stats(&self) -> MemoryStats;

    /// Reset all memory (clear all tiers).
    fn clear(&mut self);
}

// ---------------------------------------------------------------------------
// Integration point: how the transformer uses memory
// ---------------------------------------------------------------------------

/// Extension methods for a transformer model with optional memory.
///
/// When memory is enabled, the generation loop becomes:
///
/// ```text
/// 1. User sends message
/// 2. Transformer projects Q/K for the new tokens (standard)
/// 3. Memory.retrieve(Q) → relevant past context
/// 4. Inject retrieved context into the prompt (or attention)
/// 5. Transformer generates response (standard)
/// 6. Memory.ingest(K, response_text) → remember this turn
/// 7. Memory.consolidate() → sleep if needed
/// ```
///
/// Steps 3, 6, 7 are the only additions to a standard generate loop.
/// The Q and K vectors come from the model's own projections — same
/// embedding space, same weights, same understanding.
///
/// This is NOT implemented as a trait because TransformerModel is concrete.
/// Instead, it's a guide for how to wire memory into the generation loop.
///
/// In AgentOS, the organism YAML controls whether memory is active:
/// ```yaml
/// listeners:
///   - name: local-llm
///     handler: ternary-rs
///     config:
///       model: models/bitnet-3b.gguf
///       memory:
///         enabled: true
///         compressed: true
///         l1_capacity: 4096
///         l2_capacity: 2048
///         l3_capacity: 512
///         consolidation: entropy
///         entropy_threshold: 0.85
/// ```
pub struct MemoryIntegration;

impl MemoryIntegration {
    /// Example: how to wire memory into a generation loop.
    ///
    /// This is pseudocode showing the integration pattern, not a real method.
    /// The actual wiring happens in AgentOS's pipeline builder.
    ///
    /// ```ignore
    /// fn generate_with_memory(
    ///     model: &TransformerModel,
    ///     memory: &mut dyn TransformerMemory,
    ///     user_input: &str,
    ///     max_tokens: usize,
    /// ) -> String {
    ///     let input_tokens = model.tokenizer.encode(user_input);
    ///
    ///     // Step 1: Project Q/K for the input (one forward pass through layer 0)
    ///     let (q_vectors, k_vectors) = model.project_qk(&input_tokens);
    ///
    ///     // Step 2: Retrieve relevant memories
    ///     let memories = memory.retrieve(&q_vectors, input_tokens.len(), model.n_heads, 10);
    ///
    ///     // Step 3: Build augmented prompt
    ///     let mut prompt = String::new();
    ///     if !memories.is_empty() {
    ///         prompt.push_str("<context>\n");
    ///         for m in &memories {
    ///             prompt.push_str(&format!("[{}] {}\n", m.score, m.text));
    ///         }
    ///         prompt.push_str("</context>\n\n");
    ///     }
    ///     prompt.push_str(user_input);
    ///
    ///     // Step 4: Generate response (standard)
    ///     let response_tokens = model.generate(&model.tokenizer.encode(&prompt), max_tokens);
    ///     let response = model.tokenizer.decode(&response_tokens);
    ///
    ///     // Step 5: Remember this exchange
    ///     memory.ingest(&k_vectors, input_tokens.len(), user_input, MemoryRole::User, None);
    ///     // (also ingest the response's K vectors after generation)
    ///
    ///     // Step 6: Sleep if needed
    ///     memory.consolidate();
    ///
    ///     response
    /// }
    /// ```
    fn _example() {}
}
