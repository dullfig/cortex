# synthetic/ — fixtures for the trainable boundary classifier

This directory generates synthetic training data for the `concept-boundaries`
trust-region transition classifier. The classifier is a small binary model
that learns to detect, from per-token attention features, the positions where
the *source* of the token stream changes (e.g. `system` -> `user`,
`user` -> `doc`). Those source changes are the "trust boundaries" the
classifier is trained to predict.

## Contents

- `snippets/system.json` — 31 system-prompt snippets
- `snippets/user.json` — 51 user-message snippets
- `snippets/doc.json` — 50 retrieved-document snippets
- `snippets/tool.json` — 32 tool-output snippets
- `generate_fixtures.py` — fixture generator (Python stdlib only)
- `fixtures/synth_000.txt` … `synth_049.txt` — 50 generated fixtures

## Regenerating the fixtures

From this directory:

```
python generate_fixtures.py --count 50 --seed 42
```

Both flags are optional. The default seed (42) makes runs reproducible. Pass
`--out some/dir` to write fixtures elsewhere. The script prints a summary
showing the number of fixtures, total regions, total trust boundaries
(positive examples), per-source distribution, and the distribution of
fixture lengths.

## Adding more snippets

Edit the appropriate `snippets/<source>.json` file and append strings to the
`"snippets"` list. Keep snippets between 30 and 150 characters, one to four
sentences, and stylistically authentic to the source type. Variety is
important: the classifier should learn the *source* signal rather than topic
shortcuts, so avoid repeating phrases or clustering on a single topic.

## Fixture format

Each fixture is plain text. Lines starting with `#` are comments; blank lines
are ignored. Every other line has the form:

```
<source>: <content>
```

where `<source>` is `system`, `user`, `doc`, or `tool`. Every generated
fixture has 2-5 regions and at least one source-tag change, guaranteeing at
least one positive training example. The parser lives in
`pinky/concept-boundaries/src/bin/dump_features.rs` and emits per-token
features to CSV; any position where consecutive tokens have different source
tags is labeled as a trust boundary.

## Purpose

This data feeds the trainable classifier experiment in
`pinky/concept-boundaries-train/`. The hand-crafted fixtures in
`pinky/concept-boundaries/fixtures/` are held out as the evaluation set and
are not touched by this generator.
