//! BPE tokenizer — loads vocabulary from GGUF metadata.
//!
//! Supports two BPE variants:
//! - **SentencePiece** (LLaMA): score-based merges, `▁` for spaces, `<0xHH>` byte fallback
//! - **GPT-2** (BitNet b1.58): merge-list based, byte-level Unicode mapping, regex pre-tokenization
//!
//! The variant is auto-detected from `tokenizer.ggml.model` metadata.

use std::collections::HashMap;

use crate::gguf::{GgufError, GgufFile};

/// Token type flags from GGUF metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Unused = 5,
    Byte = 6,
}

impl TokenType {
    fn from_i32(v: i32) -> Self {
        match v {
            1 => TokenType::Normal,
            2 => TokenType::Unknown,
            3 => TokenType::Control,
            4 => TokenType::UserDefined,
            5 => TokenType::Unused,
            6 => TokenType::Byte,
            _ => TokenType::Normal,
        }
    }
}

/// Which BPE variant this tokenizer uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BpeMode {
    /// SentencePiece BPE: score-based merges, `▁` for spaces.
    SentencePiece,
    /// GPT-2 BPE: merge-list based, byte-level Unicode mapping.
    Gpt2,
}

/// Pre-tokenizer type — controls how text is split before BPE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreTokenizerType {
    /// Standard GPT-2 regex: `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ...`
    Gpt2,
    /// LLaMA3/Falcon3 regex: case-insensitive contractions, 3-digit numbers.
    Llama3,
}

/// A BPE tokenizer loaded from GGUF metadata.
pub struct Tokenizer {
    /// Token ID → token string.
    vocab: Vec<String>,
    /// Token ID → merge score (SentencePiece) or 0.0 (GPT-2).
    scores: Vec<f32>,
    /// Token ID → token type.
    token_types: Vec<TokenType>,
    /// Token string → token ID (for encoding).
    token_to_id: HashMap<String, u32>,
    /// Byte fallback tokens: byte value → token ID.
    byte_to_token: [u32; 256],
    /// Beginning of sequence token ID.
    bos_token_id: u32,
    /// End of sequence token ID.
    eos_token_id: u32,
    /// BPE variant.
    mode: BpeMode,
    /// Pre-tokenizer type (how text is split before BPE).
    pre_type: PreTokenizerType,
    /// GPT-2 merge list: `merge_key(left, right)` → rank (lower = merge
    /// first). One `String` key so a lookup borrows a scratch buffer
    /// instead of allocating a pair per adjacent symbol (review #10).
    merge_ranks: HashMap<String, u32>,
    /// Whether to add BOS token by default (from `tokenizer.ggml.add_bos_token`).
    /// Default: true (LLaMA). Qwen2 sets this to false.
    add_bos_default: bool,
    /// Control-type tokens (e.g. `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`)
    /// that must be preserved as single token IDs during encoding rather
    /// than broken into BPE pieces. Populated from any vocab entry whose
    /// `TokenType` is `Control`. Sorted by string length descending so
    /// longest-match-wins behavior is natural during the encode pre-pass.
    /// Empty strings filtered out.
    ///
    /// Without this pre-pass, `<|im_start|>` (id 151644 in Qwen 2.5) gets
    /// tokenized as 7 separate characters via plain BPE — the model then
    /// never sees a real chat-turn marker and produces gibberish for any
    /// chat-template-formatted input.
    special_tokens: Vec<(String, u32)>,
}

impl Tokenizer {
    /// Load tokenizer from GGUF file metadata.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, GgufError> {
        // Detect tokenizer model type
        let model_type = gguf
            .get_metadata("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .unwrap_or("llama")
            .to_string();

        let mode = if model_type == "gpt2" {
            BpeMode::Gpt2
        } else {
            BpeMode::SentencePiece
        };

        // Detect pre-tokenizer type from tokenizer.ggml.pre metadata
        let pre_str = gguf
            .get_metadata("tokenizer.ggml.pre")
            .and_then(|v| v.as_str())
            .unwrap_or("default");
        let pre_type = match pre_str {
            "llama3" | "falcon3" | "llama-v3" | "llama-bpe" => PreTokenizerType::Llama3,
            _ => PreTokenizerType::Gpt2,
        };

        // Extract token strings
        let tokens_meta = gguf
            .get_metadata("tokenizer.ggml.tokens")
            .ok_or_else(|| GgufError::MissingMetadata("tokenizer.ggml.tokens".into()))?;
        let tokens_arr = tokens_meta
            .as_array()
            .ok_or_else(|| GgufError::MetadataTypeMismatch {
                key: "tokenizer.ggml.tokens".into(),
                expected: "array",
            })?;

        // Review #21: every array must describe the vocabulary exactly, and
        // an element of the wrong type is an error, not a silent default
        // (a U32-typed token_type array used to become all-Normal and drop
        // every special token, including the ChatML markers).
        let bad = |field: &'static str, message: String| GgufError::InvalidTokenizer { field, message };
        let mut vocab: Vec<String> = Vec::with_capacity(tokens_arr.len());
        for (i, v) in tokens_arr.iter().enumerate() {
            let s = v
                .as_str()
                .ok_or_else(|| bad("tokenizer.ggml.tokens", format!("element {i} is not a string")))?;
            vocab.push(s.to_string());
        }
        let vocab_size = vocab.len();

        // Scores: absent is a valid file shape (all zero); present must be
        // a float array of exactly vocab_size.
        let scores: Vec<f32> = match gguf.get_metadata("tokenizer.ggml.scores") {
            None => vec![0.0; vocab_size],
            Some(meta) => {
                let arr = meta
                    .as_array()
                    .ok_or_else(|| bad("tokenizer.ggml.scores", "not an array".to_string()))?;
                let mut out = Vec::with_capacity(arr.len());
                for (i, v) in arr.iter().enumerate() {
                    out.push(v.as_f32().ok_or_else(|| {
                        bad("tokenizer.ggml.scores", format!("element {i} is not f32"))
                    })?);
                }
                out
            }
        };

        // Token types: absent is a valid (if unusual) shape — all Normal,
        // no special tokens; say so. Present must be an integer array of
        // exactly vocab_size; any integer width is accepted.
        let token_types: Vec<TokenType> = match gguf.get_metadata("tokenizer.ggml.token_type") {
            None => {
                tracing::warn!("tokenizer.ggml.token_type is absent: no special tokens (chat markers will be split by BPE)");
                vec![TokenType::Normal; vocab_size]
            }
            Some(meta) => {
                let arr = meta
                    .as_array()
                    .ok_or_else(|| bad("tokenizer.ggml.token_type", "not an array".to_string()))?;
                let mut out = Vec::with_capacity(arr.len());
                for (i, v) in arr.iter().enumerate() {
                    let t = v.as_int().ok_or_else(|| {
                        bad("tokenizer.ggml.token_type", format!("element {i} is not an integer"))
                    })?;
                    out.push(TokenType::from_i32(i32::try_from(t).unwrap_or(1)));
                }
                out
            }
        };

        // Extract merge list for GPT-2
        let merges: Vec<String> = if mode == BpeMode::Gpt2 {
            match gguf.get_metadata("tokenizer.ggml.merges") {
                None => Vec::new(),
                Some(meta) => {
                    let arr = meta
                        .as_array()
                        .ok_or_else(|| bad("tokenizer.ggml.merges", "not an array".to_string()))?;
                    let mut out = Vec::with_capacity(arr.len());
                    for (i, v) in arr.iter().enumerate() {
                        out.push(v.as_str().ok_or_else(|| {
                            bad("tokenizer.ggml.merges", format!("element {i} is not a string"))
                        })?.to_string());
                    }
                    out
                }
            }
        } else {
            Vec::new()
        };

        // Special tokens. Read EOS first so the BOS fallback can use
        // it as a sane sentinel for models that don't define a BOS
        // (e.g. Qwen 2.5 — chat-only, no BOS metadata). Previous
        // fallback to magic id=1 produced whatever vocab entry happened
        // to land there: in Qwen that's `"` (double-quote), which got
        // repeated SINK_TOKENS times at cache start and broke chat by
        // making the model parse 4 quote characters before the real
        // chat-template markers.
        // Ids may be written with any integer width; a present key of a
        // non-integer type is an error rather than a silent fallback.
        let read_id = |key: &'static str| -> Result<Option<u32>, GgufError> {
            match gguf.get_metadata(key) {
                None => Ok(None),
                Some(v) => {
                    let n = v.as_int().ok_or_else(|| bad(key, "not an integer".to_string()))?;
                    u32::try_from(n)
                        .map(Some)
                        .map_err(|_| bad(key, format!("{n} is not a valid token id")))
                }
            }
        };
        let eos_token_id = read_id("tokenizer.ggml.eos_token_id")?.unwrap_or(2);
        let bos_token_id = read_id("tokenizer.ggml.bos_token_id")?.unwrap_or(eos_token_id);

        // Check if model explicitly disables BOS token
        let add_bos_default = gguf
            .get_metadata("tokenizer.ggml.add_bos_token")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let mut tok = Self::from_parts_with_mode(
            vocab, scores, token_types, bos_token_id, eos_token_id, mode, pre_type, &merges,
        )?;
        tok.add_bos_default = add_bos_default;
        Ok(tok)
    }

    /// Build tokenizer from raw parts (SentencePiece mode, for testing).
    pub fn from_parts(
        vocab: Vec<String>,
        scores: Vec<f32>,
        token_types: Vec<TokenType>,
        bos_token_id: u32,
        eos_token_id: u32,
    ) -> Result<Self, GgufError> {
        Self::from_parts_with_mode(
            vocab,
            scores,
            token_types,
            bos_token_id,
            eos_token_id,
            BpeMode::SentencePiece,
            PreTokenizerType::Gpt2,
            &[],
        )
    }

    /// Build tokenizer from raw parts (GPT-2 mode, for testing).
    pub fn from_parts_gpt2(
        vocab: Vec<String>,
        token_types: Vec<TokenType>,
        bos_token_id: u32,
        eos_token_id: u32,
        merges: &[String],
    ) -> Result<Self, GgufError> {
        let scores = vec![0.0; vocab.len()];
        Self::from_parts_with_mode(
            vocab,
            scores,
            token_types,
            bos_token_id,
            eos_token_id,
            BpeMode::Gpt2,
            PreTokenizerType::Gpt2,
            merges,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_parts_with_mode(
        vocab: Vec<String>,
        scores: Vec<f32>,
        token_types: Vec<TokenType>,
        bos_token_id: u32,
        eos_token_id: u32,
        mode: BpeMode,
        pre_type: PreTokenizerType,
        merges: &[String],
    ) -> Result<Self, GgufError> {
        // Review #21: the three vectors must describe the same vocabulary
        // and the ids must be in it. A short token_type array used to drop
        // the trailing specials silently (for Qwen 2.5 that is exactly
        // <|im_start|> / <|im_end|>, the last ids) and make `decode`
        // index-panic on any high id; a short scores array panicked inside
        // the merge loop; an out-of-vocab BOS asserted in the engine on
        // the first request, since every prompt starts with it.
        let bad = |field: &'static str, message: String| GgufError::InvalidTokenizer { field, message };
        if vocab.is_empty() {
            return Err(bad("tokenizer.ggml.tokens", "vocabulary is empty".to_string()));
        }
        if scores.len() != vocab.len() {
            return Err(bad(
                "tokenizer.ggml.scores",
                format!("{} entries for a {}-token vocabulary", scores.len(), vocab.len()),
            ));
        }
        if token_types.len() != vocab.len() {
            return Err(bad(
                "tokenizer.ggml.token_type",
                format!("{} entries for a {}-token vocabulary", token_types.len(), vocab.len()),
            ));
        }
        for (field, id) in [("tokenizer.ggml.bos_token_id", bos_token_id), ("tokenizer.ggml.eos_token_id", eos_token_id)] {
            if id as usize >= vocab.len() {
                return Err(bad(field, format!("{id} is outside the {}-token vocabulary", vocab.len())));
            }
        }

        // Build reverse lookup
        let mut token_to_id = HashMap::with_capacity(vocab.len());
        for (id, token) in vocab.iter().enumerate() {
            token_to_id.insert(token.clone(), id as u32);
        }

        // Build byte fallback table
        let mut byte_to_token = [u32::MAX; 256];
        match mode {
            BpeMode::SentencePiece => {
                // SentencePiece uses "<0xHH>" format
                for byte_val in 0..=255u8 {
                    let hex_token = format!("<0x{:02X}>", byte_val);
                    if let Some(&id) = token_to_id.get(&hex_token) {
                        byte_to_token[byte_val as usize] = id;
                    }
                }
            }
            BpeMode::Gpt2 => {
                // GPT-2 uses Unicode-mapped single characters for each byte
                for byte_val in 0..=255u8 {
                    let ch = gpt2_byte_to_char(byte_val);
                    let s: String = std::iter::once(ch).collect();
                    if let Some(&id) = token_to_id.get(&s) {
                        byte_to_token[byte_val as usize] = id;
                    }
                }
            }
        }

        // Build merge rank table for GPT-2. A merge is exactly two
        // non-empty pieces; a duplicate pair keeps its FIRST (best) rank —
        // `insert` used to let a later duplicate overwrite it.
        let mut merge_ranks = HashMap::new();
        for (rank, merge_str) in merges.iter().enumerate() {
            let mut parts = merge_str.split(' ');
            if let (Some(left), Some(right), None) = (parts.next(), parts.next(), parts.next()) {
                if !left.is_empty() && !right.is_empty() {
                    merge_ranks
                        .entry(merge_key(left, right))
                        .or_insert(rank as u32);
                }
            }
        }

        // Collect control-type tokens for the encode pre-pass. Sorted
        // by length descending so when find_next_special compares matches
        // at the same starting position, the longest wins naturally.
        let mut special_tokens: Vec<(String, u32)> = vocab.iter().enumerate()
            .filter(|(id, s)| {
                !s.is_empty()
                    && token_types.get(*id).copied() == Some(TokenType::Control)
            })
            .map(|(id, s)| (s.clone(), id as u32))
            .collect();
        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        Ok(Self {
            vocab,
            scores,
            token_types,
            token_to_id,
            byte_to_token,
            bos_token_id,
            eos_token_id,
            mode,
            pre_type,
            merge_ranks,
            add_bos_default: true,
            special_tokens,
        })
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize { self.vocab.len() }

    /// Beginning-of-sequence token ID.
    pub fn bos_token_id(&self) -> u32 { self.bos_token_id }

    /// Whether the model wants BOS token prepended by default.
    pub fn add_bos_default(&self) -> bool { self.add_bos_default }

    /// End-of-sequence token ID.
    pub fn eos_token_id(&self) -> u32 { self.eos_token_id }

    /// Get token string by ID (empty for an id outside the vocabulary).
    pub fn token(&self, id: u32) -> &str {
        self.vocab.get(id as usize).map(String::as_str).unwrap_or("")
    }

    /// Get token ID by string.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Get token type (`Unknown` for an id outside the vocabulary).
    pub fn token_type(&self, id: u32) -> TokenType {
        self.token_types.get(id as usize).copied().unwrap_or(TokenType::Unknown)
    }

    /// Encode text to token IDs.
    ///
    /// Pre-pass: any occurrence of a known Control-type token string
    /// (e.g. `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`) is emitted
    /// as its single vocab ID rather than BPE-encoded into characters.
    /// Standard behavior matching HuggingFace tokenizers, OpenAI
    /// tiktoken, llama.cpp — required for chat templates to round-trip
    /// to the IDs the model was trained on.
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut tokens = Vec::new();
        if add_bos {
            tokens.push(self.bos_token_id);
        }
        if text.is_empty() {
            return tokens;
        }

        // Fast path: no special tokens in vocab (test fixtures, very old
        // models). Skip the scan entirely.
        if self.special_tokens.is_empty() {
            self.encode_segment(text, &mut tokens);
            return tokens;
        }

        // Split on special-token occurrences: leftmost match first, longest
        // on ties; BPE-encode the run before it; emit the special token as
        // a single id; repeat on the tail. Review #10: each token's next
        // occurrence is cached and re-searched only once the cursor has
        // passed it, so a text with k occurrences costs O(S·n) instead of
        // k searches of every token over the whole remainder (O(k·S·n)).
        let mut next_pos: Vec<Option<usize>> = self
            .special_tokens
            .iter()
            .map(|(s, _)| text.find(s.as_str()))
            .collect();
        let mut cursor = 0usize;
        loop {
            // `special_tokens` is sorted longest-first, so a strict `<`
            // keeps the longest token among equal positions.
            let mut best: Option<(usize, usize)> = None;
            for (idx, pos) in next_pos.iter().enumerate() {
                if let Some(pos) = *pos {
                    if best.map_or(true, |(bp, _)| pos < bp) {
                        best = Some((pos, idx));
                    }
                }
            }
            let Some((pos, idx)) = best else {
                if cursor < text.len() {
                    self.encode_segment(&text[cursor..], &mut tokens);
                }
                break;
            };
            if pos > cursor {
                self.encode_segment(&text[cursor..pos], &mut tokens);
            }
            let (tok_str, tok_id) = &self.special_tokens[idx];
            tokens.push(*tok_id);
            cursor = pos + tok_str.len();
            // Refresh only the cached matches the cursor has passed
            // (including an overlapping match inside the token just consumed).
            for (i, slot) in next_pos.iter_mut().enumerate() {
                if let Some(p) = *slot {
                    if p < cursor {
                        *slot = text[cursor..]
                            .find(self.special_tokens[i].0.as_str())
                            .map(|r| r + cursor);
                    }
                }
            }
        }
        tokens
    }

    /// Encode `text` as LITERAL content: special-token strings inside it
    /// are BPE-encoded like any other characters and are NEVER emitted as
    /// control ids. Use this for untrusted / client-supplied text placed
    /// inside a chat template, so a caller cannot forge a control token
    /// (e.g. `<|im_start|>system`) by putting its text in a message.
    /// No BOS is added. (Adversarial review 2026-09-02, #2.)
    pub fn encode_literal(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        if !text.is_empty() {
            self.encode_segment(text, &mut tokens);
        }
        tokens
    }

    /// Dispatch a (non-special) text segment to the appropriate BPE
    /// encoder. Shared between the top-level encode and the
    /// inter-special-token segments.
    fn encode_segment(&self, text: &str, tokens: &mut Vec<u32>) {
        match self.mode {
            BpeMode::SentencePiece => self.encode_sentencepiece(text, tokens),
            BpeMode::Gpt2 => self.encode_gpt2(text, tokens),
        }
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, tokens: &[u32]) -> String {
        match self.mode {
            BpeMode::SentencePiece => self.decode_sentencepiece(tokens),
            BpeMode::Gpt2 => self.decode_gpt2(tokens),
        }
    }

    // -----------------------------------------------------------------------
    // SentencePiece BPE
    // -----------------------------------------------------------------------

    fn encode_sentencepiece(&self, text: &str, tokens: &mut Vec<u32>) {
        // Prepend space, replace all spaces with ▁
        let text = format!(" {}", text).replace(' ', "\u{2581}");

        // Review #10: symbols are byte ranges into `text`; a character the
        // vocabulary lacks becomes one unmergeable range per UTF-8 byte
        // (its `<0xHH>` byte token). The old loop also tried to merge a
        // byte token's `<0xHH>` *text* with its neighbour — a pair no real
        // SentencePiece vocabulary contains, so none is lost.
        let mut pieces: Vec<(usize, usize, u32, bool)> = Vec::with_capacity(text.len());
        let mut off = 0usize;
        for ch in text.chars() {
            let len = ch.len_utf8();
            if let Some(&id) = self.token_to_id.get(&text[off..off + len]) {
                pieces.push((off, off + len, id, true));
            } else {
                for k in 0..len {
                    let byte_id = self.byte_to_token[text.as_bytes()[off + k] as usize];
                    if byte_id != u32::MAX {
                        pieces.push((off + k, off + k + 1, byte_id, false));
                    }
                }
            }
            off += len;
        }
        let mut key = String::new();
        let merged = bpe_merge_ranges(&text, &pieces, |left, right| {
            key.clear();
            key.push_str(left);
            key.push_str(right);
            let &id = self.token_to_id.get(key.as_str())?;
            let score = self.scores[id as usize];
            // A NaN score never won the old `score < best` comparison.
            if score.is_nan() { None } else { Some((OrdF32(score), id)) }
        });
        tokens.extend(merged.iter().map(|&(_, _, id)| id));
    }

    fn decode_sentencepiece(&self, tokens: &[u32]) -> String {
        let mut text = String::new();
        // Review #21: byte-fallback tokens are the UTF-8 bytes of a
        // character the vocabulary lacks; they must be accumulated and
        // decoded as UTF-8 (the GPT-2 path already does), not pushed as
        // `u8 as char`, which is Latin-1 and turned "é" into "Ã©".
        let mut pending: Vec<u8> = Vec::new();
        let flush = |pending: &mut Vec<u8>, text: &mut String| {
            if !pending.is_empty() {
                text.push_str(&String::from_utf8_lossy(pending));
                pending.clear();
            }
        };

        for &id in tokens {
            if id == self.bos_token_id || id == self.eos_token_id {
                continue;
            }
            // An id outside the vocabulary (a corrupt client id, or a
            // metadata array that lied) renders as U+FFFD — never an
            // index panic on the decode path.
            let (Some(token_str), Some(&token_type)) =
                (self.vocab.get(id as usize), self.token_types.get(id as usize))
            else {
                flush(&mut pending, &mut text);
                text.push('\u{FFFD}');
                continue;
            };

            match token_type {
                TokenType::Byte => {
                    if let Some(byte_val) = parse_byte_token(token_str) {
                        pending.push(byte_val);
                    }
                }
                TokenType::Control => {
                    flush(&mut pending, &mut text);
                }
                _ => {
                    flush(&mut pending, &mut text);
                    text.push_str(token_str);
                }
            }
        }
        flush(&mut pending, &mut text);

        text = text.replace('\u{2581}', " ");
        if text.starts_with(' ') {
            text = text[1..].to_string();
        }
        text
    }


    // -----------------------------------------------------------------------
    // GPT-2 BPE
    // -----------------------------------------------------------------------

    fn encode_gpt2(&self, text: &str, tokens: &mut Vec<u32>) {
        // Pre-tokenize: split into words using the appropriate regex pattern
        let words = match self.pre_type {
            PreTokenizerType::Gpt2 => gpt2_pre_tokenize(text),
            PreTokenizerType::Llama3 => llama3_pre_tokenize(text),
        };

        let mut key = String::new();
        for word in &words {
            // Convert each byte to its GPT-2 Unicode character; the word
            // buffer is the merge arena (review #10: ranges, not strings).
            let mapped: String = word.bytes().map(gpt2_byte_to_char).collect();
            let mut pieces: Vec<(usize, usize, u32, bool)> = Vec::with_capacity(word.len());
            let mut off = 0usize;
            for ch in mapped.chars() {
                let len = ch.len_utf8();
                pieces.push((off, off + len, u32::MAX, true));
                off += len;
            }

            // Apply BPE merges: lowest rank first, leftmost on ties.
            let merged = bpe_merge_ranges(&mapped, &pieces, |left, right| {
                key.clear();
                key.push_str(left);
                key.push('\0');
                key.push_str(right);
                self.merge_ranks.get(key.as_str()).map(|&rank| (rank, u32::MAX))
            });

            // Look up each merged piece in vocabulary
            for &(start, end, _) in &merged {
                let piece = &mapped[start..end];
                if let Some(&id) = self.token_to_id.get(piece) {
                    tokens.push(id);
                } else {
                    // Byte-level fallback: shouldn't happen in GPT-2 (every byte is in vocab)
                    for b in piece.bytes() {
                        let byte_id = self.byte_to_token[b as usize];
                        if byte_id != u32::MAX {
                            tokens.push(byte_id);
                        }
                    }
                }
            }
        }
    }

    fn decode_gpt2(&self, tokens: &[u32]) -> String {
        let mut bytes = Vec::new();

        for &id in tokens {
            if id == self.bos_token_id || id == self.eos_token_id {
                continue;
            }
            // Review #21: never index the decode path with a client id.
            let (Some(token_str), Some(&token_type)) =
                (self.vocab.get(id as usize), self.token_types.get(id as usize))
            else {
                bytes.extend_from_slice("\u{FFFD}".as_bytes());
                continue;
            };
            if token_type == TokenType::Control {
                continue;
            }

            // Map GPT-2 Unicode characters back to bytes
            for ch in token_str.chars() {
                bytes.push(gpt2_char_to_byte(ch));
            }
        }

        String::from_utf8_lossy(&bytes).to_string()
    }

}

/// The `merge_ranks` key for an adjacent pair: `left`, NUL, `right`. NUL
/// never occurs inside a merge entry, so the key is unambiguous.
fn merge_key(left: &str, right: &str) -> String {
    let mut key = String::with_capacity(left.len() + right.len() + 1);
    key.push_str(left);
    key.push('\0');
    key.push_str(right);
    key
}

/// A SentencePiece merge score with a total order (lower merges first).
#[derive(Clone, Copy, PartialEq)]
struct OrdF32(f32);
impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// One symbol of a BPE merge pass: a byte range into the word buffer,
/// linked to its live neighbours.
#[derive(Clone, Copy)]
struct BpeSym {
    start: usize,
    end: usize,
    prev: usize,
    next: usize,
    alive: bool,
    mergeable: bool,
    token_id: u32,
}

/// Review #10: one BPE merge pass in `O(m log m)` — the shape of
/// llama.cpp's `llm_bigram` queue. `pieces` are the initial symbols as
/// `(start, end, token_id, mergeable)` byte ranges tiling `word` in order;
/// `rank_of(left, right)` gives two adjacent pieces' merge priority (lower
/// merges first) and the merged token id, or `None`. Every adjacent pair
/// is a heap candidate keyed `(priority, left index)`; after a merge the
/// two new neighbour pairs are pushed; a popped candidate whose sides have
/// since changed (dead, re-linked or grown) is stale and skipped. That is
/// the same choice the previous loop made by rescanning every pair each
/// iteration — lowest priority, leftmost on ties — without its `O(m²)`
/// scans and its allocation per pair (a 2 MB word was minutes of CPU per
/// request). Unmergeable pieces (byte-fallback bytes) never enter a pair.
/// Returns the surviving `(start, end, token_id)` ranges in order.
fn bpe_merge_ranges<P: Ord + Copy>(
    word: &str,
    pieces: &[(usize, usize, u32, bool)],
    mut rank_of: impl FnMut(&str, &str) -> Option<(P, u32)>,
) -> Vec<(usize, usize, u32)> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let n = pieces.len();
    let mut syms: Vec<BpeSym> = pieces
        .iter()
        .enumerate()
        .map(|(i, &(start, end, token_id, mergeable))| BpeSym {
            start,
            end,
            prev: if i == 0 { usize::MAX } else { i - 1 },
            next: i + 1,
            alive: true,
            mergeable,
            token_id,
        })
        .collect();
    // (priority, left, right, left len, right len, merged id)
    let mut heap: BinaryHeap<Reverse<(P, usize, usize, usize, usize, u32)>> = BinaryHeap::new();
    let mut push = |heap: &mut BinaryHeap<Reverse<(P, usize, usize, usize, usize, u32)>>,
                    syms: &[BpeSym],
                    l: usize,
                    r: usize| {
        let (sl, sr) = (syms[l], syms[r]);
        if !sl.mergeable || !sr.mergeable {
            return;
        }
        if let Some((p, id)) = rank_of(&word[sl.start..sl.end], &word[sr.start..sr.end]) {
            heap.push(Reverse((p, l, r, sl.end - sl.start, sr.end - sr.start, id)));
        }
    };
    for i in 0..n.saturating_sub(1) {
        push(&mut heap, &syms, i, i + 1);
    }
    while let Some(Reverse((_, l, r, left_len, right_len, id))) = heap.pop() {
        let (sl, sr) = (syms[l], syms[r]);
        if !sl.alive
            || !sr.alive
            || sl.next != r
            || sl.end - sl.start != left_len
            || sr.end - sr.start != right_len
        {
            continue; // stale: one side merged since this candidate was pushed
        }
        syms[l].end = sr.end;
        syms[l].token_id = id;
        syms[l].next = sr.next;
        syms[r].alive = false;
        if sr.next < n {
            syms[sr.next].prev = l;
        }
        if sl.prev != usize::MAX {
            push(&mut heap, &syms, sl.prev, l);
        }
        if syms[l].next < n {
            push(&mut heap, &syms, l, syms[l].next);
        }
    }
    let mut out = Vec::with_capacity(n);
    let mut i = 0usize;
    while i < n {
        let s = syms[i];
        debug_assert!(s.alive);
        out.push((s.start, s.end, s.token_id));
        i = s.next;
    }
    out
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tokenizer(vocab={}, bos={}, eos={}, mode={:?})",
            self.vocab.len(),
            self.bos_token_id,
            self.eos_token_id,
            self.mode,
        )
    }
}

/// Parse a byte fallback token like "<0x41>" → Some(0x41).
fn parse_byte_token(s: &str) -> Option<u8> {
    if s.len() == 6 && s.starts_with("<0x") && s.ends_with('>') {
        u8::from_str_radix(&s[3..5], 16).ok()
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// GPT-2 byte ↔ Unicode mapping
// ---------------------------------------------------------------------------

/// GPT-2's byte→Unicode mapping. Printable ASCII and Latin-1 supplement map
/// to themselves; control chars and gaps map to U+0100 onwards.
fn gpt2_byte_to_char(b: u8) -> char {
    // Build the mapping: printable ranges stay put, gaps fill from U+0100
    let b = b as u32;
    let ch = match b {
        // ASCII printable: ! (33) through ~ (126)
        33..=126 => b,
        // Latin-1 supplement: ¡ (161) through ¬ (172)
        161..=172 => b,
        // Latin-1 supplement: ® (174) through ÿ (255)
        174..=255 => b,
        // Everything else: map to U+0100+
        _ => {
            // Count how many "direct" bytes come before this one
            let mut offset = 0u32;
            for i in 0..=255u32 {
                let is_direct = matches!(i, 33..=126 | 161..=172 | 174..=255);
                if i == b {
                    return char::from_u32(256 + offset).unwrap();
                }
                if !is_direct {
                    offset += 1;
                }
            }
            unreachable!()
        }
    };
    char::from_u32(ch).unwrap()
}

/// Reverse GPT-2 Unicode→byte mapping.
fn gpt2_char_to_byte(ch: char) -> u8 {
    let cp = ch as u32;
    // Direct mappings: if the codepoint falls in a "direct" range, it IS the byte
    if matches!(cp, 33..=126 | 161..=172 | 174..=255) {
        return cp as u8;
    }
    // Indirect: codepoints U+0100..U+013F map to the "gap" bytes
    if cp >= 256 {
        let offset = (cp - 256) as usize;
        let gap_bytes: Vec<u8> = (0..=255u8)
            .filter(|&b| !matches!(b as u32, 33..=126 | 161..=172 | 174..=255))
            .collect();
        if offset < gap_bytes.len() {
            return gap_bytes[offset];
        }
    }
    // Fallback: shouldn't happen with valid GPT-2 tokens
    b'?'
}

// ---------------------------------------------------------------------------
// GPT-2 pre-tokenization
// ---------------------------------------------------------------------------

/// Split text into words using a simplified GPT-2 regex pattern.
///
/// GPT-2 pattern: `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
///
/// We implement this without a regex crate via manual character-class matching.
fn gpt2_pre_tokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut result = Vec::new();
    let mut i = 0;

    while i < n {
        // Try contractions first: 's, 't, 're, 've, 'm, 'll, 'd
        if chars[i] == '\'' && i + 1 < n {
            if let Some(contraction) = try_contraction(&chars, i) {
                result.push(contraction.0.to_string());
                i += contraction.1;
                continue;
            }
        }

        // Optional leading space + letters
        if (chars[i] == ' ' && i + 1 < n && chars[i + 1].is_alphabetic())
            || chars[i].is_alphabetic()
        {
            let start = i;
            if chars[i] == ' ' {
                i += 1;
            }
            while i < n && chars[i].is_alphabetic() {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            result.push(word);
            continue;
        }

        // Optional leading space + digits
        if (chars[i] == ' ' && i + 1 < n && chars[i + 1].is_ascii_digit())
            || chars[i].is_ascii_digit()
        {
            let start = i;
            if chars[i] == ' ' {
                i += 1;
            }
            while i < n && chars[i].is_ascii_digit() {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            result.push(word);
            continue;
        }

        // Optional leading space + other (non-whitespace, non-letter, non-digit)
        if (chars[i] == ' ' && i + 1 < n && is_other(chars[i + 1]))
            || (chars[i] != ' ' && is_other(chars[i]))
        {
            let start = i;
            if chars[i] == ' ' {
                i += 1;
            }
            while i < n && is_other(chars[i]) {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            result.push(word);
            continue;
        }

        // Whitespace (including lone spaces not followed by letter/digit/other)
        if chars[i].is_whitespace() {
            let start = i;
            while i < n && chars[i].is_whitespace() {
                i += 1;
            }
            // If followed by non-whitespace, emit all but last whitespace separately
            // (GPT-2: `\s+(?!\S)|\s+` — trailing whitespace as one group, else split)
            let word: String = chars[start..i].iter().collect();
            result.push(word);
            continue;
        }

        // Catch-all: single character
        result.push(chars[i].to_string());
        i += 1;
    }

    result
}

fn try_contraction(chars: &[char], i: usize) -> Option<(&'static str, usize)> {
    let remaining = chars.len() - i;
    if remaining >= 3 {
        let two: String = chars[i..i + 3].iter().collect();
        match two.to_lowercase().as_str() {
            "'re" => return Some(("'re", 3)),
            "'ve" => return Some(("'ve", 3)),
            "'ll" => return Some(("'ll", 3)),
            _ => {}
        }
    }
    if remaining >= 2 {
        let one: String = chars[i..i + 2].iter().collect();
        match one.to_lowercase().as_str() {
            "'s" => return Some(("'s", 2)),
            "'t" => return Some(("'t", 2)),
            "'m" => return Some(("'m", 2)),
            "'d" => return Some(("'d", 2)),
            _ => {}
        }
    }
    None
}

fn is_other(ch: char) -> bool {
    !ch.is_whitespace() && !ch.is_alphabetic() && !ch.is_ascii_digit()
}

// ---------------------------------------------------------------------------
// LLaMA3 / Falcon3 pre-tokenization
// ---------------------------------------------------------------------------

/// Split text using the LLaMA3 regex pattern (used by Falcon3, LLaMA3, etc.).
///
/// Pattern: `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
///
/// Key differences from GPT-2:
/// - Case-insensitive contractions (`'S`, `'T`, `'Re`, etc.)
/// - Numbers split into groups of 1-3 digits
/// - Unicode letter class (`\p{L}`) instead of just ASCII alphabetic
/// - Explicit `\r\n` handling in whitespace rules
fn llama3_pre_tokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut result = Vec::new();
    let mut i = 0;

    while i < n {
        // 1. Case-insensitive contractions: 's|'t|'re|'ve|'m|'ll|'d
        if (chars[i] == '\'' || chars[i] == '\u{2019}') && i + 1 < n {
            if let Some(contraction) = try_contraction_ci(&chars, i) {
                result.push(contraction.0);
                i += contraction.1;
                continue;
            }
        }

        // 2. [^\r\n\p{L}\p{N}]?\p{L}+ — optional non-letter-digit-newline char + letters
        if chars[i].is_alphabetic() {
            let start = i;
            while i < n && chars[i].is_alphabetic() {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            result.push(word);
            continue;
        }
        if !chars[i].is_alphanumeric() && chars[i] != '\r' && chars[i] != '\n'
            && !chars[i].is_whitespace()
            && i + 1 < n && chars[i + 1].is_alphabetic()
        {
            let start = i;
            i += 1; // skip the non-letter-digit prefix char
            while i < n && chars[i].is_alphabetic() {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            result.push(word);
            continue;
        }

        // 3. \p{N}{1,3} — digits in groups of 1-3
        if chars[i].is_numeric() {
            let start = i;
            let mut count = 0;
            while i < n && chars[i].is_numeric() && count < 3 {
                i += 1;
                count += 1;
            }
            let word: String = chars[start..i].iter().collect();
            result.push(word);
            continue;
        }

        // 4. \s*[\r\n]+ — whitespace followed by newlines
        if chars[i] == '\r' || chars[i] == '\n' {
            let start = i;
            while i < n && (chars[i] == '\r' || chars[i] == '\n') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            result.push(word);
            continue;
        }
        if chars[i].is_whitespace() {
            // Look ahead: is there a \r\n coming?
            let start = i;
            let mut j = i;
            while j < n && chars[j].is_whitespace() && chars[j] != '\r' && chars[j] != '\n' {
                j += 1;
            }
            if j < n && (chars[j] == '\r' || chars[j] == '\n') {
                // \s*[\r\n]+
                i = j;
                while i < n && (chars[i] == '\r' || chars[i] == '\n') {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                result.push(word);
                continue;
            }

            // 6. \s+(?!\S)|\s+ — trailing whitespace or whitespace before more content
            while i < n && chars[i].is_whitespace() && chars[i] != '\r' && chars[i] != '\n' {
                i += 1;
            }
            // If followed by non-whitespace, this is the \s+ branch
            let word: String = chars[start..i].iter().collect();
            result.push(word);
            continue;
        }

        // 5. " ?[^\s\p{L}\p{N}]+[\r\n]*" — optional space + non-alphanum-non-whitespace + optional newlines
        if is_other(chars[i]) {
            let start = i;
            while i < n && is_other(chars[i]) {
                i += 1;
            }
            // Consume trailing \r\n
            while i < n && (chars[i] == '\r' || chars[i] == '\n') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            result.push(word);
            continue;
        }

        // Catch-all: single character
        result.push(chars[i].to_string());
        i += 1;
    }

    result
}

/// Case-insensitive contraction matching for LLaMA3.
fn try_contraction_ci(chars: &[char], i: usize) -> Option<(String, usize)> {
    let remaining = chars.len() - i;
    if remaining >= 3 {
        let c1 = chars[i + 1].to_ascii_lowercase();
        let c2 = chars[i + 2].to_ascii_lowercase();
        match (c1, c2) {
            ('r', 'e') | ('v', 'e') | ('l', 'l') => {
                let s: String = chars[i..i + 3].iter().collect();
                return Some((s, 3));
            }
            _ => {}
        }
    }
    if remaining >= 2 {
        let c1 = chars[i + 1].to_ascii_lowercase();
        match c1 {
            's' | 't' | 'm' | 'd' => {
                let s: String = chars[i..i + 2].iter().collect();
                return Some((s, 2));
            }
            _ => {}
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // =======================================================================
    // Review #21: metadata validation, UTF-8 byte fallback, merges, decode
    // =======================================================================

    fn tiny_parts() -> (Vec<String>, Vec<f32>, Vec<TokenType>) {
        let vocab: Vec<String> = ["<unk>", "<s>", "</s>", "a", "<|im_start|>"]
            .iter().map(|s| s.to_string()).collect();
        let scores = vec![0.0; vocab.len()];
        let types = vec![
            TokenType::Unknown, TokenType::Control, TokenType::Control,
            TokenType::Normal, TokenType::Control,
        ];
        (vocab, scores, types)
    }

    fn field_of(r: Result<Tokenizer, GgufError>) -> &'static str {
        match r {
            Err(GgufError::InvalidTokenizer { field, .. }) => field,
            Err(other) => panic!("expected InvalidTokenizer, got {other}"),
            Ok(_) => panic!("expected InvalidTokenizer, got Ok"),
        }
    }

    #[test]
    fn short_token_type_array_is_rejected() {
        // A short array used to silently drop the trailing specials — for
        // Qwen 2.5 exactly <|im_start|> / <|im_end|> — and index-panic in
        // decode.
        let (vocab, scores, mut types) = tiny_parts();
        types.pop();
        assert_eq!(field_of(Tokenizer::from_parts(vocab, scores, types, 1, 2)), "tokenizer.ggml.token_type");
    }

    #[test]
    fn short_scores_array_is_rejected() {
        let (vocab, mut scores, types) = tiny_parts();
        scores.pop();
        assert_eq!(field_of(Tokenizer::from_parts(vocab, scores, types, 1, 2)), "tokenizer.ggml.scores");
    }

    #[test]
    fn out_of_vocab_bos_or_eos_is_rejected() {
        let (vocab, scores, types) = tiny_parts();
        assert_eq!(field_of(Tokenizer::from_parts(vocab.clone(), scores.clone(), types.clone(), 9999, 2)), "tokenizer.ggml.bos_token_id");
        assert_eq!(field_of(Tokenizer::from_parts(vocab, scores, types, 1, 5)), "tokenizer.ggml.eos_token_id");
    }

    #[test]
    fn empty_vocab_is_rejected() {
        assert_eq!(field_of(Tokenizer::from_parts(vec![], vec![], vec![], 0, 0)), "tokenizer.ggml.tokens");
    }

    #[test]
    fn from_gguf_accepts_any_integer_width_for_token_types_and_ids() {
        use crate::gguf::tests::GgufBuilder;
        use crate::gguf::GgufFile;
        // token_type as a U32 array and ids as U64 used to fall back
        // silently (all Normal, eos = 2): the chat markers stopped being
        // special and got BPE-split.
        let mut b = GgufBuilder::new();
        b.add_metadata_string("tokenizer.ggml.model", "llama");
        b.add_metadata_array_string("tokenizer.ggml.tokens", &["<unk>", "<s>", "</s>", "a", "<|im_start|>"]);
        b.add_metadata_array_u32("tokenizer.ggml.token_type", &[2, 3, 3, 1, 3]);
        b.add_raw_metadata("tokenizer.ggml.eos_token_id", 10, 4u64.to_le_bytes().to_vec());
        let gguf = GgufFile::open_reader(std::io::Cursor::new(b.build())).unwrap();
        let tok = Tokenizer::from_gguf(&gguf).unwrap();
        assert_eq!(tok.eos_token_id(), 4);
        assert_eq!(tok.token_type(4), TokenType::Control);
        assert_eq!(tok.encode("<|im_start|>", false), vec![4]);

        // A present-but-short token_type array is an error, not a default.
        let mut b = GgufBuilder::new();
        b.add_metadata_string("tokenizer.ggml.model", "llama");
        b.add_metadata_array_string("tokenizer.ggml.tokens", &["<unk>", "<s>", "</s>", "a", "<|im_start|>"]);
        b.add_metadata_array_i32("tokenizer.ggml.token_type", &[2, 3, 3, 1]);
        let gguf = GgufFile::open_reader(std::io::Cursor::new(b.build())).unwrap();
        assert_eq!(field_of(Tokenizer::from_gguf(&gguf)), "tokenizer.ggml.token_type");
    }

    #[test]
    fn sentencepiece_byte_fallback_decodes_utf8_not_latin1() {
        // make_test_tokenizer: byte tokens <0xNN> live at id 3 + NN.
        let tok = make_test_tokenizer();
        let id = |b: u8| 3 + b as u32;
        // "é" = C3 A9; "日本" = E6 97 A5 E6 9C AC. The old `u8 as char`
        // path produced "Ã©".
        assert_eq!(tok.decode(&[id(0xC3), id(0xA9)]), "é");
        assert_eq!(tok.decode(&[id(0xE6), id(0x97), id(0xA5), id(0xE6), id(0x9C), id(0xAC)]), "日本");
        // Bytes flush at a normal token boundary, and a lone lead byte is
        // replaced rather than corrupting the following text.
        let a = tok.token_id("h").unwrap();
        assert_eq!(tok.decode(&[id(0xC3), id(0xA9), a]), "éh");
        assert_eq!(tok.decode(&[id(0xC3), a]), "\u{FFFD}h");
    }

    #[test]
    fn decode_of_an_out_of_vocab_id_is_a_replacement_char_not_a_panic() {
        let tok = make_test_tokenizer();
        let a = tok.token_id("h").unwrap();
        assert_eq!(tok.decode(&[a, 999_999, a]), "h\u{FFFD}h");
        assert_eq!(tok.token(999_999), "");
        assert_eq!(tok.token_type(999_999), TokenType::Unknown);
        let g = make_gpt2_tokenizer();
        assert!(g.decode(&[999_999]).contains('\u{FFFD}'));
    }

    #[test]
    fn duplicate_merge_keeps_its_first_rank_and_malformed_lines_are_skipped() {
        let vocab: Vec<String> = ["<|endoftext|>", "h", "e", "he"].iter().map(|s| s.to_string()).collect();
        let types = vec![TokenType::Control, TokenType::Normal, TokenType::Normal, TokenType::Normal];
        let merges = vec![
            "h e".to_string(),      // rank 0
            "no-space".to_string(), // skipped
            "a b c".to_string(),    // skipped (three pieces)
            "h e".to_string(),      // duplicate: must not overwrite rank 0
        ];
        let tok = Tokenizer::from_parts_gpt2(vocab, types, 0, 0, &merges).unwrap();
        assert_eq!(tok.merge_ranks.get(&merge_key("h", "e")), Some(&0));
        assert_eq!(tok.merge_ranks.len(), 1);
    }

    // =======================================================================
    // SentencePiece tests (existing)
    // =======================================================================

    fn make_test_tokenizer() -> Tokenizer {
        let mut vocab = Vec::new();
        let mut scores = Vec::new();
        let mut token_types = Vec::new();

        // 0: unknown
        vocab.push("<unk>".to_string());
        scores.push(0.0);
        token_types.push(TokenType::Unknown);

        // 1: BOS
        vocab.push("<s>".to_string());
        scores.push(0.0);
        token_types.push(TokenType::Control);

        // 2: EOS
        vocab.push("</s>".to_string());
        scores.push(0.0);
        token_types.push(TokenType::Control);

        // 3..258: byte tokens
        for b in 0..=255u8 {
            vocab.push(format!("<0x{:02X}>", b));
            scores.push(0.0);
            token_types.push(TokenType::Byte);
        }

        let extra_tokens: Vec<(&str, f32)> = vec![
            ("\u{2581}", 0.0),  // 259
            ("h", 0.0),         // 260
            ("e", 0.0),         // 261
            ("l", 0.0),         // 262
            ("o", 0.0),         // 263
            ("w", 0.0),         // 264
            ("r", 0.0),         // 265
            ("d", 0.0),         // 266
            ("he", 1.0),        // 267
            ("ll", 2.0),        // 268
            ("lo", 3.0),        // 269
            ("hel", 4.0),       // 270
            ("hell", 5.0),      // 271
            ("hello", 6.0),     // 272
            ("\u{2581}he", 7.0), // 273
            ("wo", 8.0),        // 274
            ("wor", 9.0),       // 275
            ("worl", 10.0),     // 276
            ("world", 11.0),    // 277
            ("\u{2581}world", 12.0), // 278
        ];

        for (text, score) in extra_tokens {
            vocab.push(text.to_string());
            scores.push(score);
            token_types.push(TokenType::Normal);
        }

        Tokenizer::from_parts(vocab, scores, token_types, 1, 2).unwrap()
    }

    /// Review #2: `encode_literal` must never emit a control id, even when
    /// the text contains a special-token string that `encode` WOULD
    /// recognize.
    #[test]
    fn encode_literal_never_emits_control_tokens() {
        let tok = make_test_tokenizer();
        let text = "hello</s>world";
        // Sanity: the special-parsing path recognizes `</s>` (EOS, id 2).
        assert!(
            tok.encode(text, false).contains(&2),
            "encode() should parse </s> as a control token"
        );
        // The literal path must not — and must add no BOS.
        let lit = tok.encode_literal(text);
        assert!(!lit.contains(&2), "encode_literal leaked a control id: {lit:?}");
        assert!(!lit.contains(&1), "encode_literal leaked BOS: {lit:?}");
        assert!(!lit.is_empty());
        assert!(tok.encode_literal("").is_empty());
    }

    #[test]
    fn construction() {
        let tok = make_test_tokenizer();
        assert_eq!(tok.vocab_size(), 279);
        assert_eq!(tok.bos_token_id(), 1);
        assert_eq!(tok.eos_token_id(), 2);
    }

    #[test]
    fn token_lookup() {
        let tok = make_test_tokenizer();
        assert_eq!(tok.token(0), "<unk>");
        assert_eq!(tok.token(1), "<s>");
        assert_eq!(tok.token(272), "hello");
    }

    #[test]
    fn token_id_lookup() {
        let tok = make_test_tokenizer();
        assert_eq!(tok.token_id("hello"), Some(272));
        assert_eq!(tok.token_id("<s>"), Some(1));
        assert_eq!(tok.token_id("nonexistent"), None);
    }

    #[test]
    fn token_types() {
        let tok = make_test_tokenizer();
        assert_eq!(tok.token_type(0), TokenType::Unknown);
        assert_eq!(tok.token_type(1), TokenType::Control);
        assert_eq!(tok.token_type(3), TokenType::Byte);
        assert_eq!(tok.token_type(272), TokenType::Normal);
    }

    #[test]
    fn byte_fallback_table() {
        let tok = make_test_tokenizer();
        assert_eq!(tok.byte_to_token[0x41], 68);
        assert_eq!(tok.byte_to_token[0x00], 3);
        assert_eq!(tok.byte_to_token[0xFF], 258);
    }

    #[test]
    fn encode_hello() {
        let tok = make_test_tokenizer();
        let tokens = tok.encode("hello", false);
        assert_eq!(tokens, vec![259, 272]);
    }

    #[test]
    fn encode_with_bos() {
        let tok = make_test_tokenizer();
        let tokens = tok.encode("hello", true);
        assert_eq!(tokens[0], 1);
        assert_eq!(tokens.len(), 3);
    }

    #[test]
    fn encode_empty() {
        let tok = make_test_tokenizer();
        assert_eq!(tok.encode("", false), Vec::<u32>::new());
        assert_eq!(tok.encode("", true), vec![1]);
    }

    #[test]
    fn encode_hello_world() {
        let tok = make_test_tokenizer();
        let tokens = tok.encode("hello world", false);
        assert_eq!(tokens, vec![259, 272, 278]);
    }

    /// Build a SentencePiece tokenizer that includes Qwen-style
    /// chat-marker control tokens (`<|im_start|>`, `<|im_end|>`,
    /// `<|endoftext|>`). Used to verify the encode pre-pass emits them
    /// as single IDs rather than character soup.
    fn make_test_tokenizer_with_specials() -> Tokenizer {
        let mut vocab = Vec::new();
        let mut scores = Vec::new();
        let mut token_types = Vec::new();

        // 0..258: same scaffold as make_test_tokenizer.
        vocab.push("<unk>".to_string()); scores.push(0.0); token_types.push(TokenType::Unknown);
        vocab.push("<s>".to_string());   scores.push(0.0); token_types.push(TokenType::Control);
        vocab.push("</s>".to_string());  scores.push(0.0); token_types.push(TokenType::Control);
        for b in 0..=255u8 {
            vocab.push(format!("<0x{:02X}>", b));
            scores.push(0.0);
            token_types.push(TokenType::Byte);
        }

        // 259..278: "hello"/"world" merge ladder (same as base fixture).
        let extra_tokens: Vec<(&str, f32)> = vec![
            ("\u{2581}", 0.0), ("h", 0.0), ("e", 0.0), ("l", 0.0), ("o", 0.0),
            ("w", 0.0), ("r", 0.0), ("d", 0.0),
            ("he", 1.0), ("ll", 2.0), ("lo", 3.0),
            ("hel", 4.0), ("hell", 5.0), ("hello", 6.0),
            ("\u{2581}he", 7.0), ("wo", 8.0), ("wor", 9.0),
            ("worl", 10.0), ("world", 11.0), ("\u{2581}world", 12.0),
        ];
        for (text, score) in extra_tokens {
            vocab.push(text.to_string());
            scores.push(score);
            token_types.push(TokenType::Normal);
        }

        // 279, 280, 281: chat-marker control tokens.
        vocab.push("<|im_start|>".to_string());   scores.push(0.0); token_types.push(TokenType::Control);
        vocab.push("<|im_end|>".to_string());     scores.push(0.0); token_types.push(TokenType::Control);
        vocab.push("<|endoftext|>".to_string());  scores.push(0.0); token_types.push(TokenType::Control);

        Tokenizer::from_parts(vocab, scores, token_types, 1, 2).unwrap()
    }

    #[test]
    fn special_token_at_start() {
        let tok = make_test_tokenizer_with_specials();
        let toks = tok.encode("<|im_start|>hello", false);
        // Expect: [279 = <|im_start|>, ...hello-encoded tokens].
        assert_eq!(toks[0], 279, "first token should be the special-token ID");
        // The "hello" suffix should match what plain `encode("hello")` produces.
        let hello_only = tok.encode("hello", false);
        assert_eq!(&toks[1..], &hello_only[..]);
    }

    #[test]
    fn special_token_at_end() {
        let tok = make_test_tokenizer_with_specials();
        let toks = tok.encode("hello<|im_end|>", false);
        // Expect: [...hello-encoded..., 280 = <|im_end|>].
        let hello_only = tok.encode("hello", false);
        assert_eq!(&toks[..toks.len() - 1], &hello_only[..]);
        assert_eq!(*toks.last().unwrap(), 280);
    }

    #[test]
    fn special_token_in_middle() {
        let tok = make_test_tokenizer_with_specials();
        let toks = tok.encode("hello<|im_start|>world", false);
        let hello_only = tok.encode("hello", false);
        let world_only = tok.encode("world", false);
        // [hello..., 279, world...]
        assert_eq!(&toks[..hello_only.len()], &hello_only[..]);
        assert_eq!(toks[hello_only.len()], 279);
        assert_eq!(&toks[hello_only.len() + 1..], &world_only[..]);
    }

    #[test]
    fn adjacent_special_tokens() {
        let tok = make_test_tokenizer_with_specials();
        let toks = tok.encode("<|im_start|><|im_end|>", false);
        // Two specials back-to-back with nothing in between.
        assert_eq!(toks, vec![279, 280]);
    }

    #[test]
    fn three_special_tokens() {
        let tok = make_test_tokenizer_with_specials();
        let toks = tok.encode("<|im_start|>hello<|im_end|><|endoftext|>", false);
        let hello_only = tok.encode("hello", false);
        let mut expected = vec![279u32];
        expected.extend_from_slice(&hello_only);
        expected.push(280);
        expected.push(281);
        assert_eq!(toks, expected);
    }

    #[test]
    fn no_special_tokens_unchanged() {
        // Regression: plain BPE behavior must be identical to the
        // base tokenizer fixture (which has no <|im_*|> tokens in vocab).
        let plain = make_test_tokenizer();
        let with_specials = make_test_tokenizer_with_specials();
        let plain_out = plain.encode("hello world", false);
        let specials_out = with_specials.encode("hello world", false);
        assert_eq!(plain_out, specials_out,
            "encode of text with no special tokens must match the no-specials fixture exactly");
    }

    #[test]
    fn longest_special_wins_on_tie() {
        // Make a fixture where two control tokens share a prefix; verify
        // the longer one is preferred when both could match at the same
        // starting position.
        let mut vocab = Vec::new();
        let mut scores = Vec::new();
        let mut types = Vec::new();
        vocab.push("<unk>".to_string());     scores.push(0.0); types.push(TokenType::Unknown);
        vocab.push("<s>".to_string());       scores.push(0.0); types.push(TokenType::Control);
        vocab.push("</s>".to_string());      scores.push(0.0); types.push(TokenType::Control);
        for b in 0..=255u8 {
            vocab.push(format!("<0x{:02X}>", b));
            scores.push(0.0);
            types.push(TokenType::Byte);
        }
        // 259, 260: short vs long with shared prefix.
        vocab.push("<|x|>".to_string());        scores.push(0.0); types.push(TokenType::Control);
        vocab.push("<|x|>extended".to_string());scores.push(0.0); types.push(TokenType::Control);
        let tok = Tokenizer::from_parts(vocab, scores, types, 1, 2).unwrap();

        // Text contains the LONGER token; the encoder must pick id 260
        // (longer match) over id 259 (shorter prefix match).
        let toks = tok.encode("<|x|>extended", false);
        assert_eq!(toks, vec![260]);
    }

    #[test]
    fn special_token_empty_string_ignored() {
        // Empty-string control tokens (test-fixture pathology) must not
        // crash the encoder — they'd otherwise match at every position
        // and produce an infinite loop.
        let mut vocab = Vec::new();
        let mut scores = Vec::new();
        let mut types = Vec::new();
        vocab.push("<unk>".to_string()); scores.push(0.0); types.push(TokenType::Unknown);
        vocab.push("<s>".to_string());   scores.push(0.0); types.push(TokenType::Control);
        vocab.push("</s>".to_string());  scores.push(0.0); types.push(TokenType::Control);
        for b in 0..=255u8 {
            vocab.push(format!("<0x{:02X}>", b));
            scores.push(0.0);
            types.push(TokenType::Byte);
        }
        // 259: empty string with Control type (should be filtered out).
        vocab.push("".to_string());      scores.push(0.0); types.push(TokenType::Control);
        let tok = Tokenizer::from_parts(vocab, scores, types, 1, 2).unwrap();
        // Should not hang or panic.
        let _ = tok.encode("hello", false);
    }

    #[test]
    fn decode_hello() {
        let tok = make_test_tokenizer();
        let text = tok.decode(&[259, 272]);
        assert_eq!(text, "hello");
    }

    #[test]
    fn decode_skips_bos_eos() {
        let tok = make_test_tokenizer();
        let text = tok.decode(&[1, 259, 272, 2]);
        assert_eq!(text, "hello");
    }

    #[test]
    fn decode_byte_fallback() {
        let tok = make_test_tokenizer();
        let text = tok.decode(&[68]);
        assert_eq!(text, "A");
    }

    #[test]
    fn roundtrip_hello() {
        let tok = make_test_tokenizer();
        let tokens = tok.encode("hello", false);
        let decoded = tok.decode(&tokens);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn roundtrip_hello_world() {
        let tok = make_test_tokenizer();
        let tokens = tok.encode("hello world", false);
        let decoded = tok.decode(&tokens);
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn parse_byte_token_valid() {
        assert_eq!(parse_byte_token("<0x00>"), Some(0x00));
        assert_eq!(parse_byte_token("<0x41>"), Some(0x41));
        assert_eq!(parse_byte_token("<0xFF>"), Some(0xFF));
    }

    #[test]
    fn parse_byte_token_invalid() {
        assert_eq!(parse_byte_token("hello"), None);
        assert_eq!(parse_byte_token("<0xGG>"), None);
        assert_eq!(parse_byte_token("<0x0>"), None);
    }

    #[test]
    fn debug_format() {
        let tok = make_test_tokenizer();
        let debug = format!("{:?}", tok);
        assert!(debug.contains("Tokenizer"));
        assert!(debug.contains("vocab=279"));
    }

    // =======================================================================
    // GPT-2 byte mapping tests
    // =======================================================================

    #[test]
    fn gpt2_byte_char_roundtrip() {
        for b in 0..=255u8 {
            let ch = gpt2_byte_to_char(b);
            let back = gpt2_char_to_byte(ch);
            assert_eq!(b, back, "roundtrip failed for byte {b}: char={ch}");
        }
    }

    #[test]
    fn gpt2_printable_ascii_identity() {
        // Printable ASCII (33-126) maps to itself
        for b in 33..=126u8 {
            let ch = gpt2_byte_to_char(b);
            assert_eq!(ch as u32, b as u32, "byte {b} should map to itself");
        }
    }

    #[test]
    fn gpt2_space_maps_to_unicode() {
        // Space (32) is NOT in the direct range, should map to U+0100+
        let ch = gpt2_byte_to_char(b' ');
        assert!(ch as u32 >= 256, "space should map above U+00FF, got U+{:04X}", ch as u32);
    }

    #[test]
    fn gpt2_all_chars_unique() {
        let mut seen = std::collections::HashSet::new();
        for b in 0..=255u8 {
            let ch = gpt2_byte_to_char(b);
            assert!(seen.insert(ch), "duplicate char for byte {b}");
        }
    }

    // =======================================================================
    // GPT-2 pre-tokenization tests
    // =======================================================================

    #[test]
    fn gpt2_pretokenize_simple() {
        let words = gpt2_pre_tokenize("hello world");
        assert_eq!(words, vec!["hello", " world"]);
    }

    #[test]
    fn gpt2_pretokenize_contractions() {
        let words = gpt2_pre_tokenize("I'm don't");
        // "I" "'m" " don" "'t"
        assert_eq!(words.len(), 4);
        assert_eq!(words[1], "'m");
        assert_eq!(words[3], "'t");
    }

    #[test]
    fn gpt2_pretokenize_numbers() {
        let words = gpt2_pre_tokenize("test 123 abc");
        assert!(words.contains(&" 123".to_string()));
    }

    #[test]
    fn gpt2_pretokenize_punctuation() {
        let words = gpt2_pre_tokenize("hello, world!");
        // "hello" "," " world" "!"
        assert!(words.contains(&",".to_string()));
        assert!(words.contains(&"!".to_string()));
    }

    // =======================================================================
    // GPT-2 BPE tests
    // =======================================================================

    /// Build a minimal GPT-2 tokenizer for testing.
    fn make_gpt2_tokenizer() -> Tokenizer {
        let mut vocab = Vec::new();
        let mut token_types = Vec::new();

        // Token 0: BOS (control)
        vocab.push("<|endoftext|>".to_string());
        token_types.push(TokenType::Control);

        // Tokens 1-256: single-byte tokens (GPT-2 Unicode mapping)
        for b in 0..=255u8 {
            let ch = gpt2_byte_to_char(b);
            vocab.push(ch.to_string());
            token_types.push(TokenType::Normal);
        }

        // Token 257: "he" (merge of "h" + "e")
        vocab.push("he".to_string());
        token_types.push(TokenType::Normal);

        // Token 258: "ll" (merge of "l" + "l")
        vocab.push("ll".to_string());
        token_types.push(TokenType::Normal);

        // Token 259: "lo" (merge of "l" + "o")
        vocab.push("lo".to_string());
        token_types.push(TokenType::Normal);

        // Token 260: "hel" (merge of "he" + "l")
        vocab.push("hel".to_string());
        token_types.push(TokenType::Normal);

        // Token 261: "hell" (merge of "hel" + "l")
        vocab.push("hell".to_string());
        token_types.push(TokenType::Normal);

        // Token 262: "hello" (merge of "hell" + "o")
        vocab.push("hello".to_string());
        token_types.push(TokenType::Normal);

        // Merge list (order = priority)
        let merges = vec![
            "h e".to_string(),     // rank 0: h+e → he
            "l l".to_string(),     // rank 1: l+l → ll
            "l o".to_string(),     // rank 2: l+o → lo
            "he l".to_string(),    // rank 3: he+l → hel
            "hel l".to_string(),   // rank 4: hel+l → hell
            "hell o".to_string(),  // rank 5: hell+o → hello
        ];

        // BOS=0, EOS=0 (simplified)
        Tokenizer::from_parts_gpt2(vocab, token_types, 0, 0, &merges).unwrap()
    }

    #[test]
    fn gpt2_encode_hello() {
        let tok = make_gpt2_tokenizer();
        let tokens = tok.encode("hello", false);
        // "hello" → pre-tokenize → ["hello"]
        // bytes: h(104) e(101) l(108) l(108) o(111) → GPT-2 chars: same (all printable ASCII)
        // BPE merges: h+e→he, l+l→ll, he+l→hel, hel+l→hell... wait.
        // Initial: ["h", "e", "l", "l", "o"]
        // Rank 0 (h+e): ["he", "l", "l", "o"]
        // Rank 1 (l+l): ["he", "ll", "o"]
        // No l+o pair (it's "ll" now). Rank 3 (he+l): need "he"+"l", but we have "he"+"ll", not a match.
        // Actually rank 3 is "he"+"l". The symbols are ["he", "ll", "o"]. The pair (he, ll) != (he, l).
        // So no more merges! Result: ["he", "ll", "o"]
        // That's token IDs for "he"(257), "ll"(258), "o"(1 + 'o' offset)
        // 'o' is byte 111, and since 111 is in 33..=126, gpt2_byte_to_char(111) = 'o'
        // The single-byte tokens start at index 1. So 'o' is at index 1 + 111 = 112? No.
        // Token index = 1 + byte_value. Token for byte 0 is index 1, byte 1 is index 2, etc.
        // Wait, I iterate 0..=255 starting at token 1. So byte 0 → token 1, byte 111 → token 112.
        // Actually no. Let me re-check. We push byte 0's char at index 1, byte 1 at index 2, ..., byte 255 at index 256.
        // So byte 111 ('o') → token 112.
        let o_token = 1 + 111; // = 112
        assert_eq!(tokens, vec![257, 258, o_token]);
    }

    #[test]
    fn gpt2_encode_with_space() {
        let tok = make_gpt2_tokenizer();
        let tokens = tok.encode("hi there", false);
        // Pre-tokenize: ["hi", " there"]
        // Both words get byte-encoded, and since we have limited merges, most stay as bytes
        assert!(!tokens.is_empty());
        // Decode should roundtrip
        let decoded = tok.decode(&tokens);
        assert_eq!(decoded, "hi there");
    }

    #[test]
    fn gpt2_roundtrip_ascii() {
        let tok = make_gpt2_tokenizer();
        for text in &["hello", "test", "a b c", "123"] {
            let tokens = tok.encode(text, false);
            let decoded = tok.decode(&tokens);
            assert_eq!(&decoded, text, "roundtrip failed for {:?}", text);
        }
    }

    // =======================================================================
    // Review #10: the bigram-heap merge is the old rescan loop, faster
    // =======================================================================

    /// The pre-#10 GPT-2 merge loop, kept as the oracle: rescan every
    /// adjacent pair, take the lowest rank (leftmost on ties), repeat.
    fn reference_gpt2_merge(tok: &Tokenizer, mut symbols: Vec<String>) -> Vec<String> {
        loop {
            if symbols.len() < 2 {
                break;
            }
            let mut best_rank = u32::MAX;
            let mut best_idx = usize::MAX;
            for i in 0..symbols.len() - 1 {
                if let Some(&rank) = tok.merge_ranks.get(&merge_key(&symbols[i], &symbols[i + 1])) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }
            if best_idx == usize::MAX {
                break;
            }
            let merged = format!("{}{}", symbols[best_idx], symbols[best_idx + 1]);
            symbols[best_idx] = merged;
            symbols.remove(best_idx + 1);
        }
        symbols
    }

    fn reference_encode_gpt2(tok: &Tokenizer, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        for word in gpt2_pre_tokenize(text) {
            let chars: Vec<String> = word.bytes().map(|b| gpt2_byte_to_char(b).to_string()).collect();
            for piece in reference_gpt2_merge(tok, chars) {
                if let Some(&id) = tok.token_to_id.get(&piece) {
                    tokens.push(id);
                } else {
                    for b in piece.bytes() {
                        let byte_id = tok.byte_to_token[b as usize];
                        if byte_id != u32::MAX {
                            tokens.push(byte_id);
                        }
                    }
                }
            }
        }
        tokens
    }

    /// The pre-#10 SentencePiece loop: rescan, lowest score first,
    /// leftmost on ties, byte-fallback pieces carry their `<0xHH>` text.
    fn reference_encode_sp(tok: &Tokenizer, text: &str) -> Vec<u32> {
        let text = format!(" {}", text).replace(' ', "\u{2581}");
        let mut symbols: Vec<(String, u32)> = Vec::new();
        for ch in text.chars() {
            let ch_str = ch.to_string();
            if let Some(&id) = tok.token_to_id.get(&ch_str) {
                symbols.push((ch_str, id));
            } else {
                let mut buf = [0u8; 4];
                for b in ch.encode_utf8(&mut buf).bytes() {
                    let byte_id = tok.byte_to_token[b as usize];
                    if byte_id != u32::MAX {
                        symbols.push((format!("<0x{:02X}>", b), byte_id));
                    }
                }
            }
        }
        loop {
            if symbols.len() < 2 {
                break;
            }
            let mut best_score = f32::INFINITY;
            let mut best_idx = usize::MAX;
            let mut best_id = 0u32;
            for i in 0..symbols.len() - 1 {
                let merged = format!("{}{}", symbols[i].0, symbols[i + 1].0);
                if let Some(&id) = tok.token_to_id.get(&merged) {
                    let score = tok.scores[id as usize];
                    if score < best_score {
                        best_score = score;
                        best_idx = i;
                        best_id = id;
                    }
                }
            }
            if best_idx == usize::MAX {
                break;
            }
            let merged_text = format!("{}{}", symbols[best_idx].0, symbols[best_idx + 1].0);
            symbols[best_idx] = (merged_text, best_id);
            symbols.remove(best_idx + 1);
        }
        symbols.iter().map(|s| s.1).collect()
    }

    /// Deterministic xorshift so the differential corpus is reproducible.
    fn random_words(seed: u64, n: usize, alphabet: &[&str], max_len: usize) -> Vec<String> {
        let mut x = seed | 1;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        (0..n)
            .map(|_| {
                let len = 1 + (next() as usize) % max_len;
                (0..len).map(|_| alphabet[(next() as usize) % alphabet.len()]).collect::<String>()
            })
            .collect()
    }

    /// A SentencePiece fixture with a real merge ladder: scores are
    /// "lower merges first", with ties to exercise leftmost-wins.
    fn make_sp_tokenizer() -> Tokenizer {
        let pieces: &[(&str, f32, TokenType)] = &[
            ("<unk>", 0.0, TokenType::Unknown),
            ("<s>", 0.0, TokenType::Control),
            ("</s>", 0.0, TokenType::Control),
            ("\u{2581}", -1.0, TokenType::Normal),
            ("h", -1.0, TokenType::Normal),
            ("e", -1.0, TokenType::Normal),
            ("l", -1.0, TokenType::Normal),
            ("o", -1.0, TokenType::Normal),
            ("w", -1.0, TokenType::Normal),
            ("he", -10.0, TokenType::Normal),
            ("ll", -10.0, TokenType::Normal),   // ties with "he": leftmost wins
            ("lo", -9.0, TokenType::Normal),
            ("hel", -8.0, TokenType::Normal),
            ("hell", -7.0, TokenType::Normal),
            ("hello", -6.0, TokenType::Normal),
            ("\u{2581}h", -5.0, TokenType::Normal),
            ("\u{2581}he", -4.5, TokenType::Normal),
            ("\u{2581}hello", -4.0, TokenType::Normal),
            ("ow", -3.0, TokenType::Normal),
            ("low", -2.0, TokenType::Normal),
            ("<0xC3>", 0.0, TokenType::Byte),
            ("<0xA9>", 0.0, TokenType::Byte),
        ];
        let vocab = pieces.iter().map(|p| p.0.to_string()).collect();
        let scores = pieces.iter().map(|p| p.1).collect();
        let types = pieces.iter().map(|p| p.2).collect();
        Tokenizer::from_parts(vocab, scores, types, 1, 2).unwrap()
    }

    #[test]
    fn bigram_merge_matches_the_rescan_loop_gpt2() {
        let tok = make_gpt2_tokenizer();
        let words = random_words(0x5eed, 500, &["h", "e", "l", "o", " ", "w", "'s", "1", "."], 24);
        for w in &words {
            assert_eq!(tok.encode(w, false), reference_encode_gpt2(&tok, w), "word {w:?}");
        }
        for w in ["hello", "hell", "hello hello", "helloworld", "ll", "lo", "hellohello", "he ll o"] {
            assert_eq!(tok.encode(w, false), reference_encode_gpt2(&tok, w), "word {w:?}");
        }
    }

    #[test]
    fn bigram_merge_matches_the_rescan_loop_sentencepiece() {
        let tok = make_sp_tokenizer();
        let words = random_words(0xbeef, 500, &["h", "e", "l", "o", "w", " ", "é"], 20);
        for w in &words {
            assert_eq!(tok.encode(w, false), reference_encode_sp(&tok, w), "word {w:?}");
        }
        for w in ["hello", "hello world", "low", "hellow", "héllo", "llll", "hehe"] {
            assert_eq!(tok.encode(w, false), reference_encode_sp(&tok, w), "word {w:?}");
        }
    }

    #[test]
    fn bigram_merge_ties_break_leftmost() {
        // "he" and "ll" share a score; in "hell" the leftmost pair merges
        // first either way, but in "llhe" it is "ll" — the position, not
        // the vocabulary order, decides.
        let tok = make_sp_tokenizer();
        assert_eq!(tok.encode("llhe", false), reference_encode_sp(&tok, "llhe"));
        let gpt2 = make_gpt2_tokenizer();
        assert_eq!(gpt2.encode("hell", false), vec![257, 258]);
    }

    #[test]
    fn long_inputs_encode_in_bounded_time() {
        // Review #10: an unbroken 512 KB word was O(m²) rescans with two
        // allocations per pair — effectively never. Bound: seconds in debug.
        let tok = make_gpt2_tokenizer();
        let t = std::time::Instant::now();
        let word_a = "a".repeat(512 * 1024);
        assert_eq!(tok.encode(&word_a, false).len(), 512 * 1024);
        // One 500 KB word of "hello"s: with this fixture "ll" (rank 1)
        // outranks "hel" (rank 3), so each "hello" is [he, ll, o] — the
        // same answer `gpt2_encode_hello` pins for a single word.
        let word_hello = "hello".repeat(100 * 1024);
        let toks = tok.encode(&word_hello, false);
        assert_eq!(toks.len(), 3 * 100 * 1024);
        let o_id = tok.token_id("o").unwrap();
        assert!(toks.chunks_exact(3).all(|c| c == [257, 258, o_id]));
        let alternating = "ab".repeat(256 * 1024);
        assert_eq!(tok.encode(&alternating, false).len(), 512 * 1024);
        let sp = make_sp_tokenizer();
        let sp_toks = sp.encode(&"hello".repeat(100 * 1024), false);
        assert!(sp_toks.len() <= 100 * 1024 + 1);
        assert!(t.elapsed().as_secs() < 20, "long inputs took {:?}", t.elapsed());
    }

    #[test]
    fn repeated_special_tokens_scan_in_linear_time() {
        let (vocab, scores, types) = tiny_parts();
        let tok = Tokenizer::from_parts(vocab, scores, types, 1, 2).unwrap();
        let text = "<|im_start|>".repeat(17_000); // ~200 KB
        let t = std::time::Instant::now();
        let toks = tok.encode(&text, false);
        assert_eq!(toks.len(), 17_000);
        assert!(toks.iter().all(|&t| t == 4));
        assert!(t.elapsed().as_secs() < 5, "special scan took {:?}", t.elapsed());
        // Tie-breaking and the run between specials are unchanged.
        assert_eq!(tok.encode("a<|im_start|>a", false), vec![3, 4, 3]);
    }

    #[test]
    fn gpt2_merge_priority() {
        let tok = make_gpt2_tokenizer();
        // "hell" should merge: h+e→he, l+l→ll. Then he+ll doesn't match any merge.
        // So result should be ["he", "ll"]
        let tokens = tok.encode("hell", false);
        assert_eq!(tokens, vec![257, 258]); // he=257, ll=258
    }

    #[test]
    fn gpt2_debug_format() {
        let tok = make_gpt2_tokenizer();
        let debug = format!("{:?}", tok);
        assert!(debug.contains("Gpt2"));
    }

    // =======================================================================
    // LLaMA3 pre-tokenization tests
    // =======================================================================

    #[test]
    fn llama3_pretokenize_simple() {
        let words = llama3_pre_tokenize("hello world");
        assert_eq!(words, vec!["hello", " ", "world"]);
    }

    #[test]
    fn llama3_pretokenize_contractions_case_insensitive() {
        let words = llama3_pre_tokenize("I'M DON'T");
        // Should split: "I" "'M" " " "DON" "'T"
        assert!(words.contains(&"'M".to_string()), "got: {:?}", words);
        assert!(words.contains(&"'T".to_string()), "got: {:?}", words);
    }

    #[test]
    fn llama3_pretokenize_numbers_grouped() {
        let words = llama3_pre_tokenize("test 123456 abc");
        // Numbers split into groups of 1-3: "123", "456"
        assert!(words.contains(&"123".to_string()), "got: {:?}", words);
        assert!(words.contains(&"456".to_string()), "got: {:?}", words);
    }

    #[test]
    fn llama3_pretokenize_punctuation() {
        let words = llama3_pre_tokenize("hello, world!");
        assert!(words.contains(&",".to_string()), "got: {:?}", words);
        assert!(words.contains(&"!".to_string()), "got: {:?}", words);
    }

    #[test]
    fn llama3_pretokenize_newlines() {
        let words = llama3_pre_tokenize("hello\nworld");
        // Newline should be its own token
        assert!(words.contains(&"\n".to_string()), "got: {:?}", words);
    }

    #[test]
    fn llama3_pretokenize_whitespace_separate() {
        // Unlike GPT-2, LLaMA3 does NOT attach leading space to words
        let words = llama3_pre_tokenize("hello world");
        assert_eq!(words[0], "hello");
        // Space should be separate from "world"
        assert!(words.iter().any(|w| w == "world"), "got: {:?}", words);
    }
}
