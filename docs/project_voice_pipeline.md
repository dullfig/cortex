---
name: voice-pipeline
description: "Architectural decision for Bob's voice conversation pipeline. Path A (chosen): streaming STT → cortex incremental prefill → token emission → streaming TTS (Kokoro / StyleTTS2 / Piper) → 78-record DSP filter → audio. Path B (rejected): Sesame CSM or Moshi end-to-end speech-language-model replacing cortex on the voice path. Path A' (also rejected): cortex + Sesame as TTS-only, which wastes Sesame's main value. Path A is chosen because it preserves substrate ownership, keeps the shim architecture load-bearing, and composes cleanly with the 78-record audio register (`project_bob_audio_register.md`) — we deliberately strip natural prosody, so Sesame's main advantage is something we'd filter away anyway. Streaming TTS is the key optimization (chunk by sentence, not waiting for full LLM response). Composes with incremental prefill (`project_incremental_prefill.md`) on the input side. Surfaced 2026-06-04 evening via Daniel's question about Sesame's role; clarified that Sesame is a speech-language model, not a TTS, which made the architectural choice between paths explicit."
metadata:
  node_type: memory
  type: project
  originSessionId: 0c8cbcaa-2244-408f-a2b3-9c792cfc2a2a
---

Articulated 2026-06-04 evening during conversation about how voice-Bob would actually work. Daniel sketched the basic pipeline (STT → cortex → emit tokens → Sesame → audio), then realized Sesame is a speech-language model rather than a TTS step. That clarification made the architectural choice explicit and the pin worth writing.

## The chosen pipeline (Path A)

```
[user audio]
  → Streaming STT (Whisper-streaming / Distil-Whisper / Riva)
  → token stream
  → Cortex incremental prefill (per `project_incremental_prefill.md`)
  → [user pauses / utterance ends]
  → Cortex generates response token stream
  → Streaming TTS (Kokoro / StyleTTS2 / Piper), sentence-bounded chunks
  → 78-record DSP filter (per `project_bob_audio_register.md`)
  → [audio out]
```

Key properties:

- **Substrate-owned**: cortex remains the model surface; shim architecture (`project_cortex_v1_shim_api.md`, `project_cortex_ffn_shims.md`, the per-layer auxiliary pin) stays load-bearing for voice the same way it is for text
- **Incremental prefill on input**: by the time the user stops talking, the KV cache is already mostly built; first response token streams within milliseconds
- **Streaming TTS on output**: TTS starts emitting audio after the first sentence is ready, not after the full LLM response completes; time-to-first-audio drops from "full response generation time" to "first-sentence latency"
- **78-record DSP filter on output**: deliberate stylization that disclosure-by-construction signals Bob is AI

Estimated end-to-end latency from user-stops-speaking to first-audio: plausibly <500ms with warm cache, modest concurrent load, and tuned chunk boundaries.

## The streaming TTS optimization

The most-skipped optimization in voice-LLM pipelines. Naive version: wait for the entire LLM response to be generated, then call TTS. Time-to-first-audio = full generation time, which is too long.

Streaming version:
1. Emit cortex's response tokens to a buffer
2. When the buffer contains a complete sentence (or a punctuation boundary with enough look-ahead), flush it to TTS
3. TTS begins synthesizing that chunk immediately
4. As subsequent sentences arrive, TTS queues and synthesizes them
5. Audio output streams to the user; the first audio chunk plays while later chunks are still being generated

**Look-ahead requirement**: TTS needs ~1 sentence of look-ahead to handle prosody (intonation depends on whether the sentence is a question, what's emphasized, etc.). Pure token-by-token streaming produces robotic output; sentence-chunked streaming preserves natural-enough prosody at low latency.

**78-record filter consideration**: since we're applying a DSP filter that strips prosody and bass anyway, the prosody quality requirement is actually *lower* than for naturalistic TTS pipelines. Aggressive chunking (shorter look-ahead, more chunks) is more tolerable for Bob than it would be for a high-fidelity voice assistant. **The 78-record register relaxes the streaming-TTS constraint.**

## The rejected alternatives

### Path B: Sesame CSM / Moshi end-to-end speech-language model

Sesame CSM-1B (Conversational Speech Model) and Moshi (Kyutai) are speech-language models — they emit audio tokens directly, no text round-trip. Audio in, audio out, one model end-to-end.

Real advantages over Path A:
- Lower latency (no STT→text→TTS round-trip)
- Better natural prosody (model expresses tone end-to-end)
- Tighter conversational dynamics (interruption, backchanneling)

Why we reject:
- **Replaces cortex with someone else's model.** Substrate ownership (`project_five_systems.md`) breaks. The project's whole architectural endgame (BitNet weights + TurboQuant cache + per-layer auxiliary + memex retrieval + persistent KV cache) doesn't extend to a non-cortex multimodal model. We'd be running two parallel architectures for text and voice.
- **Shim architecture only works on text-token-based models.** Bob's voice register (per `project_bob_voice.md`) is shaped by shims at the text layer. Speech-language models bypass that surface entirely; the wry-uncertain register goes away or has to be re-engineered into a different model.
- **We don't want natural prosody.** Per `project_bob_audio_register.md`, the 78-record DSP filter strips prosody deliberately as disclosure-by-construction. Sesame's main advantage is something we'd filter away.
- **No-camouflage principle** (`feedback_no_camouflage.md`): a natural-prosody Bob crosses toward camouflage-as-human territory. The 78-filtered TTS-of-text-LLM stays cleanly on the right side of that line.
- **Retrieval integration is harder.** Cortex composes with memex (`project_unified_memory_architecture.md`) for retrieval-as-attention-bias. Speech-language models don't have that retrieval substrate; we'd have to bolt it on or accept worse retrieval grounding.

Sesame and Moshi are *genuinely interesting* architectures — they're the right answer for a different project (one that wants natural prosody, doesn't have shim architecture, doesn't care about substrate ownership, doesn't have a 78-record register commitment). For this project, Path A is the architecturally consistent choice.

### The corpus-asymmetry argument (load-bearing reinforcement, added 2026-06-04)

Beyond architecture compatibility, there's a structural reason text-LLM-with-voice-edges will dominate for the foreseeable future: **the spoken-knowledge corpus gap is enormous and isn't closing.**

| Corpus | Approximate scale |
|---|---|
| **Text** (Common Crawl, books, Wikipedia, code, news archives, academic papers) | Trillions of tokens |
| **Speech** (LibriSpeech ~1K hours, LibriLight ~60K, CommonVoice ~20K, VoxPopuli ~400K, scrapeable podcasts/audiobooks/broadcasts) | Hundreds of thousands of hours; orders of magnitude smaller in tokens-equivalent |

The asymmetry is structural, not fixable by more scraping. Most human knowledge exists in text. Recorded speech of the kind useful for LLM training (clean, transcribable, content-rich, large-scale) is a fraction of what's been written down.

**What speech-language models actually do**: they don't train from scratch on audio. They text-pretrain a language backbone, then fine-tune a speech head. Moshi, Sesame CSM, GPT-4o-realtime, Gemini Live — all of them are *text-trained-then-adapted*. The "speech-language model" framing slightly obscures that they're inheriting text knowledge through adaptation.

**Adaptation has costs**:
- Text-pretrained knowledge gets partially overwritten during speech tuning
- Pure-text completion competence is stronger than speech-language competence in the same model family
- Domain-specific knowledge transfers unevenly through the speech adaptation
- Long-form structured reasoning is consistently weaker in speech-language models than text-only siblings of equivalent size

**For Bob specifically, this composes hard**:
- Bob's historian role is **knowledge-dominated**; the wry-uncertain register is **conversational-dominated**
- Knowledge-dominated tasks favor maximum leverage of text pretraining → text-LLM-in-the-middle wins
- The barbershop corpus we ingest via memex is **overwhelmingly text** (Harmonizer articles, district newsletters, transcribed interviews) — text-LLM consumption fits the corpus shape directly
- Even our own oral history captures (Pete interviews, wire recordings, convention audio) get transcribed to text for ingestion; the audio is artifact/preservation, not training substrate

**The mission-specific nuance** worth holding: barbershop oral tradition is one of the rare niche domains where the audio corpus is the *primary* source for some material — Pete's anecdotes, the wire recordings, convention performance audio. Our domain has a *smaller* text-vs-audio asymmetry than the general internet because some knowledge only exists in spoken form. Even so, text-LLM + memex retrieval over text-transcribed audio still beats audio-LLM + audio-retrieval because: (a) we can use any text model that exists, not just speech-language models; (b) memex already works in text-token space; (c) retrieval grounding is much more developed for text than for audio.

**Strategic claim**: Path B isn't just architecturally incompatible — it's *corpus-incompatible*. Speech-language models are catching up on conversational fluency; they're not catching up on the underlying knowledge corpus asymmetry. Bob is a knowledge-dense character (`project_bob_voice.md` "historian is my day job"); choosing Path A is choosing to honor that character with the architecture that can deliver it.

This argument makes the Path B rejection structural rather than provisional. "Path B might catch up eventually" was a plausible hedge; "Path B is bounded by a corpus gap that isn't closing" is a stronger claim.

### Path A': cortex + Sesame as TTS-only

Use Sesame's text-input mode as the TTS step: cortex emits tokens; text is passed to Sesame; Sesame produces audio.

Why we reject: this wastes Sesame's main value (it's not using the conversational speech context that makes Sesame special). A smaller dedicated TTS (Kokoro, Piper) gives equivalent results for the constrained use case at lower compute cost. Sesame in text-input mode is over-engineered TTS.

## TTS choice within Path A

| TTS | Profile | Verdict for Bob |
|---|---|---|
| **Kokoro** | Small, fast, decent quality, very low latency, open weights | **Probably v1 voice choice.** Compose-friendly, low latency, voice can be selected from preset speakers — appropriate quality for what the 78-record filter will then transform |
| **StyleTTS2** | Higher quality, somewhat slower, open weights | Better if quality dominates latency; second choice |
| **XTTS-v2 (Coqui)** | Open, voice-cloning-capable | Quality good without cloning; the cloning feature is irrelevant for us (would violate no-camouflage if we tried to clone a real member) |
| **Piper** | Local-first, lightweight, lower quality | Ships easily; good fallback for resource-constrained deployments (e.g., chapter-appliance v3+ vision) |
| **ElevenLabs** | Proprietary, very high quality | Rejected: external API, breaks substrate ownership, cost scales with usage |
| **OpenAI TTS** | Proprietary, good quality | Rejected: same reasons as ElevenLabs |
| **Sesame text-input** | Higher quality, conversational-aware (mode-limited) | Path A'; wastes Sesame's value; covered above |

Default recommendation: **Kokoro for v1 voice**, with the 78-record DSP filter doing the stylization. Re-evaluate at v2 if quality complaints surface (probably won't — the register is what the audience expects given the visual stylization).

## STT choice on the input side

| STT | Profile | Verdict |
|---|---|---|
| **Whisper-streaming** (faster-whisper streaming) | Open, strong quality, supports streaming | Likely v1 choice |
| **Distil-Whisper** | Faster, slightly lower quality | Good fallback if compute-constrained |
| **NVIDIA Riva** | Proprietary, very fast | Rejected: vendor lock-in |
| **Deepgram / AssemblyAI streaming** | Proprietary APIs | Rejected: external dependency, member-data-flow violates pseudonymization boundary |

Default recommendation: **faster-whisper streaming** for v1 voice. Composes with local deployment; no member audio leaves AgentOS.

## The 78-record DSP filter

Per `project_bob_audio_register.md`: any TTS output should be filtered to sound like an old 78 RPM record (tinny, no bass, period-stylized). Implementation: standard audio DSP applied to TTS output before delivery.

Recipe (approximate; tune empirically):
- Low-pass filter: ~5-6 kHz cutoff (78s had limited high-frequency response)
- High-pass filter: ~200 Hz cutoff (78s had almost no bass)
- Compression: gentle, to flatten dynamic range like an old recording
- Optional: subtle vinyl crackle / wow & flutter (period authenticity); use sparingly
- Optional: very mild reverb/space (parlor / radio-booth ambience)

The filter is **deterministic and CPU-cheap**. Applies to any TTS output; doesn't require model retraining. Can be tuned at deployment without touching the TTS layer.

## Privacy composition

The voice path must compose with the existing privacy commitments:

- **STT runs locally on AgentOS** — member audio never goes to cloud STT (Deepgram / AssemblyAI rejected for this reason); per `project_dm_privacy_structural.md` extended to voice
- **Streaming partial-utterance tokens** to cortex are conversation-local and ephemeral (per `project_incremental_prefill.md` privacy composition); not logged, not entering Bob-memory until committed
- **TTS runs locally on AgentOS** — Bob's response audio is generated locally, not via cloud TTS (ElevenLabs, OpenAI TTS rejected)
- **Audio doesn't enter training corpus** — per `project_training_vs_retrieval_substrate.md`; if member audio surfaces in retrieval, it's in audio shards with per-user provenance, not aggregated into training data

The pseudonymization boundary (`project_hybrid_serving_pseudonymization.md`) applies on the agentic path if/when voice-Bob makes agentic tool calls. Voice-mode chat remains local.

## v1 vs later timing

| Capability | Phase | Note |
|---|---|---|
| Text incremental prefill | v2 polish | Per `project_incremental_prefill.md` |
| Streaming STT integration | v3+/v4 voice | Coupled to overall voice work |
| Streaming TTS integration | v3+/v4 voice | Same |
| 78-record DSP filter | v3+/v4 voice | Per `project_bob_audio_register.md` |
| Full voice-Bob | v3+/v4 | Big enough to be its own phase |
| Voice-first AgentOS tenant (FreePBX-style) | When tenant ships first | Could pull this forward; would compose with `project_incremental_prefill.md`'s voice-first prioritization |

v1 ships text-only. Voice work is downstream. But the **architectural decision** (Path A, not Path B) should be made now so cortex/AgentOS work doesn't drift toward incompatibility with the voice plan. Specifically:

- Cortex's shim API should stay text-token-based; don't expand it to handle audio tokens as inputs/outputs (that path leads toward Path B by accident)
- AgentOS API contract should anticipate audio streams as a parallel channel to text streams (per `project_incremental_prefill.md`'s `prompt_extend` ops, which generalize from text tokens to STT-emitted tokens)
- The 78-record filter recipe should be documented and tested in `project_bob_audio_register.md` ahead of voice integration so it's ready when needed

## How to apply

### When evaluating voice-AI research papers

Filter by which path:
- Speech-language model papers (Moshi, Sesame CSM, GPT-4o, Gemini Live): interesting for understanding what's possible, but not our path; deprioritize implementation reading
- Streaming TTS papers (chunking strategies, look-ahead minimization, prosody preservation): directly relevant to Path A
- Streaming STT papers: directly relevant
- DSP filter / audio register papers: directly relevant to 78-record filter design

### When evaluating TTS vendor pitches

External vendors (ElevenLabs, OpenAI, Deepgram) are structurally rejected for member data flow; only relevant for non-Bob tenants. For Bob: open-weights local TTS only.

### When designing FreePBX or similar voice-first tenant

Voice-first AgentOS tenants get the same Path A pipeline: STT → text-LLM → TTS → (optional tenant-specific DSP). The 78-record filter is Bob-specific; other tenants choose their own stylization (or none). The **architecture** (streaming STT, streaming TTS, incremental prefill) is shared platform infrastructure; the *content* (which LLM, which voice, which filter) is tenant-specific. Per `project_multi_tenant_readiness.md`.

### When evaluating "should we just use Sesame end-to-end?"

The answer is "not for Bob; possibly for a different project." If a future direction emerges where substrate ownership is less load-bearing and natural prosody matters more — re-evaluate. For now, Path A is the structurally consistent choice and the rejection of Path B is principled, not provisional.

## Related pins

- `project_incremental_prefill.md` — input-side optimization; voice gets it nearly free (append-only stream)
- `project_bob_audio_register.md` — 78-record DSP filter; output-side stylization
- `project_bob_voice.md` — text-layer voice register; shim-shaped; only works with text-token-based models
- `project_cortex_v1_shim_api.md` — shim API that voice path keeps using; text-token-based
- `project_cortex_ffn_shims.md` — small FFN shims; same
- `project_per_layer_injection_auxiliary.md` — v3+ shim extension; also text-token-based
- `feedback_no_camouflage.md` — natural-prosody Bob would risk camouflage; 78-record filter is disclosure-by-construction
- `project_five_systems.md` — substrate ownership; the meta-reason Path A wins
- `project_hybrid_serving_pseudonymization.md` — privacy boundary; STT/TTS local-only to preserve it
- `project_dm_privacy_structural.md` — member data never leaves AgentOS; extends to voice
- `project_training_vs_retrieval_substrate.md` — audio retrieval separate from audio training
- `project_multi_tenant_readiness.md` — voice pipeline as shared platform infrastructure
- `project_operator_bob.md` — operator-Bob's voice channel benefits from this pipeline
- `lore_paris_etymology_era.md` — shims fix base-model behavior; same shims apply to voice via text layer

## The phrase to remember

> *Path A: streaming STT → cortex incremental prefill → token emission → streaming TTS (Kokoro / StyleTTS2 / Piper) → 78-record DSP filter → audio out. Substrate-owned, shim-compatible, low-latency. Path B (Sesame end-to-end speech-language model) rejected: would replace cortex, break the shim architecture, and waste its main advantage (natural prosody) on a register we deliberately filter away.*

Plus the latency story:

> *Time-from-user-stops-talking to first-audio could be under 500ms with warm cache, streaming STT and streaming TTS at sentence boundaries, and the incremental-prefill optimization on the input side. The 78-record filter relaxes the streaming-TTS constraint because we're not optimizing for natural prosody anyway.*

Plus the architectural principle:

> *The voice path uses the same text-token-based cortex and shim infrastructure as the chat path. Voice I/O is added at the edges (STT in, TTS out, DSP filter) rather than replacing the model. This preserves substrate ownership, keeps shims load-bearing, composes with retrieval and the 78-record register, and stays on the right side of the no-camouflage line.*
