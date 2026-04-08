# concept-boundaries-train

Pinky experiment 1, training stage. Sibling to `concept-boundaries/`,
which produces the feature CSVs that this directory consumes.

## What this is

A small Python training pipeline that fits a classifier on the per-position
attention features captured by `concept-boundaries`'s `dump_features` binary.
Uses sklearn for the classifier (logistic regression first, MLP if needed,
gradient-boosted trees as the escape hatch). No PyTorch dependency at this
stage — sklearn is enough for the question we're answering tonight.

## What we're trying to learn tonight

**Can a logistic regression on per-layer attention features learn to predict
the trust-boundary positions in our existing fixtures?**

The labels are auto-generated from the fixture format (positions where
`tokens[i].source != tokens[i-1].source`), so no labeling work is needed.
The features come from cortex's `forward_traced` via `dump_features`.

If logistic regression learns *any* signal above chance, the feature
pipeline works and we can iterate (move to harder labels, expand to more
fixtures, switch to an MLP if the linear model is the bottleneck).

If logistic regression learns nothing, we know the feature pipeline is
broken or the features don't carry the signal we hoped, and we debug
that before adding model capacity.

## How to use it

1. From `pinky/concept-boundaries/`, run the feature dumper:

   ```bash
   ./target/release/dump_features.exe \
     --model /path/to/qwen2.5-0.5b-instruct-q8_0.gguf \
     --fixture fixtures/mixed-trust.txt \
     --fixture fixtures/multi-source.txt \
     --fixture fixtures/identical-content.txt \
     --fixture fixtures/dog-conversation.txt \
     --output features.csv
   ```

   This produces `features.csv` in the concept-boundaries directory.

2. From this directory, run the trainer:

   ```bash
   python train.py --features ../concept-boundaries/features.csv
   ```

   This fits a logistic regression and prints accuracy + per-feature
   weights + per-position predictions on the training set itself
   (since we have so few positives, leave-one-fixture-out cross
   validation is the more honest evaluation).

## Dependencies

- numpy
- pandas
- scikit-learn

If these aren't installed:

```bash
pip install numpy pandas scikit-learn
```
