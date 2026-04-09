# morsel-retrieval

Test fixtures for evaluating cross-document **morsel-level retrieval** — the
architectural claim made in `pinky/POSITION-addendum.md` section 11 ("The
cross-document morsel property") and grounded in the broader pipeline described
in `pinky/POSITION.md` ("The full pipeline").

## What this directory contains

- `papers/paper_01.txt` ... `paper_08.txt` — eight short synthetic research
  papers (200-400 words each) written to read as plausible research output
  across biomedicine, computer security, materials science, and atmospheric
  science.
- `connections.json` — ground-truth file documenting three deliberately planted
  cross-paper morsel connections. Each entry names the two papers, quotes the
  exact morsel sentences, gives a test query, and describes what a naive
  keyword/topic retriever would fail to do.
- `demo_analysis.py` — stdlib-only Python script that loads the corpus,
  verifies the morsels appear verbatim, and prints what a successful morsel
  retriever should surface for each test query. It does **not** run retrieval.
- This `README.md`.

## The architectural claim being tested

Standard retrieval-augmented generation (RAG) indexes documents (or chunks) by
their dominant topic. A query is embedded, compared against chunk embeddings,
and the top-k topically-similar chunks are returned. This works well when the
answer to a query lives in a document that is *about* the query.

The cortex architecture argues that the most valuable connections in a
scientific corpus are precisely the ones that are NOT about the query — where
paper A mentions a finding as a side observation in an unrelated study, and
paper B mentions a complementary finding as a side observation in a different
unrelated study, and the useful synthesis lives only in the juxtaposition.
Neither paper's topic vector is close to the query. A chunk-level embedding
retriever cannot find them. A human skimming either paper in isolation is
likely to gloss right over the morsel, because it isn't load-bearing for
either paper's own argument.

The proposed mechanism is to store a KV cache per document and, at query
time, run raw Q·K^T attention over the full stored key set. Attention
distributes its weight per-token, so a query can simultaneously attend
strongly to a single sentence in paper A and a single sentence in paper B
even when both papers' overall topic signatures are far from the query. The
retrieval pass returns ranked token spans (morsels), not whole documents.

## Why standard RAG can't find these connections

Each planted connection in `connections.json` is built so that:

1. The two papers belong to different fields or sub-fields.
2. Neither paper's title, abstract-level topic, or dominant vocabulary
   mentions the connection's subject.
3. The morsel sentences are *tangential* to each paper's main argument —
   they read as incidental observations, footnote-caliber findings, or
   "worth following up" remarks.
4. The test query, phrased naturally from the perspective of a researcher
   in one field, would retrieve one paper (the topically-obvious one) and
   miss the other entirely under any bag-of-words or sentence-embedding
   approach.

For example, "what could reduce sleep fragmentation in elderly residents"
retrieves the sleep study, but not a nutrition survey that happens to
contain one sentence linking low magnesium intake to nocturnal leg cramps
— even though leg cramps turn out to be the largest underrecognized cause
of awakenings in that sleep study. The connection is real; no retriever
that scores whole documents against a topic vector can see it.

## How the planted connections work

Three connections are planted across the eight papers, spanning three
distinct fields (biomedicine, cloud/endpoint security, materials science
crossed with atmospheric science). Two papers (`paper_07.txt` on melatonin
in shift workers, `paper_08.txt` on GNN malware classification) are
distractors: topical neighbors that a naive retriever is likely to return
for the connection queries, but that do NOT participate in any planted
connection. Their presence makes the evaluation more honest by ensuring
that a trivially correct answer isn't available via topic match alone.

See `connections.json` for the full ground truth: each entry gives the
`id`, description, the two (file, morsel_text) pairs, the `test_query`,
and the expected naive-retriever failure mode.

## How a future cortex retrieval pass would be evaluated

When cortex exposes the planned `project_qk()` method on `TransformerModel`
and wires in the engram KV-cache-backed retrieval pass, the evaluation loop
will look like:

1. For each paper in `papers/`, run a forward pass through the loaded model
   and store the full per-layer KV cache (keyed by document id and token
   offset).
2. For each `test_query` in `connections.json`, project the query tokens
   through the same model's Q projection to obtain query vectors.
3. Compute raw Q·K^T against the concatenated key set of all eight papers,
   without a causal mask, and take the top-k attended token spans across
   all documents.
4. For each top-k span, record which (paper, offset) it came from and
   whether its surrounding window contains the planted morsel.

The **success criterion** is simple and binary: for each planted
connection, the top-k morsel retriever should surface *both* planted
morsels — one from each paper — within its top-k results for that
connection's `test_query`. A baseline sentence-embedding RAG retriever run
against the same corpus should, for the same queries, surface at most one
of the two morsels (typically the topically-obvious paper) and in several
cases surface neither, instead returning the distractor papers.

The existence of a gap between those two behaviors on a controlled corpus
is what the cortex architecture predicts and what this fixture set exists
to let us measure.
