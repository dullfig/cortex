//! EngramMemory — implements `TransformerMemory` using engram's compressed
//! hierarchical cache.
//!
//! This adapter bridges cortex's memory trait with engram's storage and
//! retrieval primitives. Unlike engram's `Engine`, this takes pre-projected
//! K/Q vectors directly from cortex's own model — no separate projection
//! model needed.

use engram::cache::consolidator;
use engram::cache::hierarchical::{
    ConsolidationTrigger as EngramTrigger, HierarchicalCache, HierarchicalConfig,
};
use engram::cache::position_map::Role;
use engram::cache::tiered_retrieve;
use engram::retrieve;

use crate::layers::memory::{
    ConsolidationReport, ConsolidationTrigger, MemoryConfig, MemoryResult, MemoryRole, MemoryStats,
    MemoryTier, TransformerMemory,
};

/// Persistent memory backed by engram's compressed hierarchical cache.
///
/// Stores K vectors in PolarQuant 3-bit compressed format across three tiers
/// (L1 working / L2 session / L3 archive). Retrieval uses bidirectional
/// attention over the compressed cache.
pub struct EngramMemory {
    cache: HierarchicalCache,
    /// Chunk scores from the last retrieval pass (for consolidation).
    last_chunk_scores: Vec<f32>,
}

impl EngramMemory {
    /// Create a new memory from a cortex `MemoryConfig`.
    pub fn new(config: &MemoryConfig, n_kv_heads: usize, head_dim: usize) -> Self {
        let hier_config = HierarchicalConfig {
            l1_capacity: config.l1_capacity,
            l2_capacity: config.l2_capacity,
            l3_capacity: config.l3_capacity,
            chunk_size: config.chunk_size,
            threshold: config.pressure_threshold,
            entropy_threshold: config.entropy_threshold,
            ..Default::default()
        };

        let cache = HierarchicalCache::new(hier_config, n_kv_heads, head_dim, (42, 99));
        Self {
            cache,
            last_chunk_scores: Vec::new(),
        }
    }

    /// Create with explicit engram config.
    pub fn with_config(
        config: HierarchicalConfig,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let cache = HierarchicalCache::new(config, n_kv_heads, head_dim, (42, 99));
        Self {
            cache,
            last_chunk_scores: Vec::new(),
        }
    }

    /// Access the underlying hierarchical cache (for advanced use).
    pub fn cache(&self) -> &HierarchicalCache {
        &self.cache
    }

    /// Run consolidation and map engram's report to cortex's report.
    fn run_consolidation(&mut self) -> ConsolidationReport {
        let scores = if self.last_chunk_scores.is_empty() {
            None
        } else {
            Some(self.last_chunk_scores.as_slice())
        };

        let report = consolidator::consolidate_with_scores(&mut self.cache, scores);

        ConsolidationReport {
            l1_evicted: report.l1_drained,
            l2_added: report.l2_added,
            l3_cascaded: report.l3_added > 0,
            trigger: to_cortex_trigger(report.trigger),
        }
    }
}

/// Map cortex MemoryRole to engram Role.
fn to_engram_role(role: MemoryRole) -> Role {
    match role {
        MemoryRole::User => Role::User,
        MemoryRole::Assistant => Role::Assistant,
        MemoryRole::System => Role::System,
        MemoryRole::Tool => Role::Tool,
    }
}

/// Map engram tier to cortex MemoryTier.
fn to_cortex_tier(tier: tiered_retrieve::Tier) -> MemoryTier {
    match tier {
        tiered_retrieve::Tier::L1 => MemoryTier::L1,
        tiered_retrieve::Tier::L2 => MemoryTier::L2,
        tiered_retrieve::Tier::L3 => MemoryTier::L3,
    }
}

/// Map engram trigger to cortex trigger.
fn to_cortex_trigger(trigger: EngramTrigger) -> ConsolidationTrigger {
    match trigger {
        EngramTrigger::Sleep => ConsolidationTrigger::Entropy,
        EngramTrigger::Pressure => ConsolidationTrigger::Pressure,
        EngramTrigger::Both => ConsolidationTrigger::Both,
        EngramTrigger::None => ConsolidationTrigger::None,
    }
}

impl TransformerMemory for EngramMemory {
    fn ingest(
        &mut self,
        keys: &[f32],
        n_tokens: usize,
        text: &str,
        role: MemoryRole,
        turn_id: Option<u64>,
    ) {
        let kv_dim = self.cache.n_kv_heads() * self.cache.head_dim();
        assert_eq!(keys.len(), n_tokens * kv_dim, "key shape mismatch");

        let start_pos = self.cache.l1.cache.len();

        // Append each token's K vector to the compressed L1 cache.
        // V is dummy — retrieval only uses K (text is in the PositionMap).
        let dummy_v = vec![0.0f32; kv_dim];
        for t in 0..n_tokens {
            let k = &keys[t * kv_dim..(t + 1) * kv_dim];
            self.cache.append_to_l1(k, &dummy_v);
        }

        let end_pos = self.cache.l1.cache.len();
        self.cache.record_span(
            start_pos,
            end_pos,
            text.to_string(),
            to_engram_role(role),
            turn_id,
            None,
        );
    }

    fn retrieve(
        &self,
        queries: &[f32],
        n_tokens: usize,
        n_heads: usize,
        top_k: usize,
    ) -> Vec<MemoryResult> {
        if self.cache.l1.cache.is_empty()
            && self.cache.l2.cache.is_empty()
            && self.cache.l3.cache.is_empty()
        {
            return vec![];
        }

        let results = tiered_retrieve::tiered_retrieve(
            &self.cache, queries, n_tokens, n_heads, top_k,
        );

        results
            .into_iter()
            .map(|r| MemoryResult {
                text: r.text,
                role: MemoryRole::System, // tiered results don't carry original role yet
                turn_id: r.turn_id,
                score: r.score,
                tier: to_cortex_tier(r.tier),
            })
            .collect()
    }

    fn entropy(
        &self,
        queries: &[f32],
        n_tokens: usize,
        n_heads: usize,
    ) -> f32 {
        if self.cache.l1.cache.len() <= 1 {
            return 0.0;
        }
        retrieve::attention_entropy(queries, n_tokens, n_heads, &self.cache.l1.cache)
    }

    fn consolidate(&mut self) -> Option<ConsolidationReport> {
        if !self.cache.needs_consolidation() {
            return None;
        }

        Some(self.run_consolidation())
    }

    fn force_consolidate(&mut self) -> ConsolidationReport {
        self.run_consolidation()
    }

    fn stats(&self) -> MemoryStats {
        MemoryStats {
            l1_len: self.cache.l1.cache.len(),
            l1_capacity: self.cache.config.l1_capacity,
            l2_len: self.cache.l2.cache.len(),
            l2_capacity: self.cache.config.l2_capacity,
            l3_len: self.cache.l3.cache.len(),
            l3_capacity: self.cache.config.l3_capacity,
            entropy: self.cache.last_entropy(),
            memory_bytes: self.cache.l1.cache.memory_bytes()
                + self.cache.l2.cache.memory_bytes()
                + self.cache.l3.cache.memory_bytes(),
        }
    }

    fn clear(&mut self) {
        self.cache.l1.cache.clear();
        self.cache.l1.map.clear();
        self.cache.l2.cache.clear();
        self.cache.l2.map.clear();
        self.cache.l3.cache.clear();
        self.cache.l3.map.clear();
        self.last_chunk_scores.clear();
    }
}
