"""train.py — fit a classifier to per-position attention features.

First pinky training experiment. Loads the CSV produced by
concept-boundaries' dump_features binary, fits a logistic regression
on the per-layer attention features, and reports accuracy + which
features carry signal.

Tonight's question: can a linear classifier on per-layer attention
features predict trust-boundary positions? If yes, the feature pipeline
is sound and we can iterate. If no, we debug the pipeline before
adding model capacity.

Usage:
    python train.py --features ../concept-boundaries/features.csv
    python train.py --features ../concept-boundaries/features.csv --loo

The --loo flag enables leave-one-fixture-out cross-validation, which is
the more honest evaluation given the small dataset (~200 rows, 7
positives total across 4 fixtures).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


METADATA_COLS = {"fixture", "position", "token_id", "token_text", "source", "label"}


def load_features(csv_path: Path):
    """Load the feature CSV using stdlib csv (no pandas dependency).

    Returns:
        rows: list of dicts (metadata only) for inspection
        X: feature matrix [N, D]
        y: label vector [N]
        feature_names: list of column names corresponding to X columns
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_cols = reader.fieldnames or []
        feature_cols = [c for c in all_cols if c not in METADATA_COLS]

        rows = []
        X_list = []
        y_list = []
        for r in reader:
            rows.append({
                "fixture": r["fixture"],
                "position": int(r["position"]),
                "token_id": int(r["token_id"]),
                "token_text": r["token_text"],
                "source": r["source"],
                "label": int(r["label"]),
            })
            X_list.append([float(r[c]) for c in feature_cols])
            y_list.append(int(r["label"]))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return rows, X, y, feature_cols


def fit_and_report(X_train, y_train, X_test, y_test, label):
    """Fit logistic regression and report key metrics."""
    if y_train.sum() == 0:
        print(f"  [{label}] no positive examples in training set, skipping")
        return {}

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    p, r, f, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print(f"  [{label}] accuracy={acc:.3f}  precision={p:.3f}  recall={r:.3f}  f1={f:.3f}")
    print(f"  [{label}] confusion: TN={cm[0, 0]}  FP={cm[0, 1]}  FN={cm[1, 0]}  TP={cm[1, 1]}")

    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f,
        "confusion": cm.tolist(),
        "model": model,
        "y_proba": y_proba,
    }


def in_sample_run(X, y, feature_names, rows):
    """Train on everything, evaluate on the same data."""
    print("=" * 70)
    print("IN-SAMPLE FIT (training and testing on the same data)")
    print("=" * 70)
    print(f"  total rows: {len(y)}, positives: {y.sum()} ({100 * y.mean():.1f}%)")
    print()

    result = fit_and_report(X, y, X, y, "in-sample")
    if not result:
        return

    print()
    print("most-confident positive predictions (top 12):")
    proba = result["y_proba"]
    order = np.argsort(-proba)[:12]
    print(f"  {'fixture':<28} {'pos':>4} {'true':>5} {'proba':>7}  text")
    print(f"  {'-' * 60}")
    for idx in order:
        row = rows[idx]
        marker = "*" if row["label"] == 1 else " "
        text = str(row["token_text"])[:25]
        print(
            f"  {row['fixture']:<28} {row['position']:>4} {row['label']:>4}{marker} {proba[idx]:>6.3f}  {text!r}"
        )
    print()

    print("all true positives (the labeled trust boundaries):")
    print(f"  {'fixture':<28} {'pos':>4} {'proba':>7}  {'pred':>4}  text")
    print(f"  {'-' * 60}")
    pos_idx = np.where(y == 1)[0]
    for idx in pos_idx:
        row = rows[idx]
        text = str(row["token_text"])[:25]
        pred = int(result["model"].predict(X[idx:idx + 1])[0])
        print(
            f"  {row['fixture']:<28} {row['position']:>4} {proba[idx]:>6.3f}  {pred:>4}  {text!r}"
        )
    print()

    coef = result["model"].coef_[0]
    feature_importance = sorted(
        [(name, w) for name, w in zip(feature_names, coef)],
        key=lambda x: -abs(x[1]),
    )
    print("top 16 features by |weight| (positive weight = pushes toward boundary):")
    print(f"  {'feature':<25} {'weight':>10}")
    print(f"  {'-' * 40}")
    for name, w in feature_importance[:16]:
        sign = "+" if w >= 0 else "-"
        print(f"  {name:<25} {sign}{abs(w):>9.4f}")
    print()


def leave_one_fixture_out(X, y, feature_names, rows):
    """Hold out one fixture at a time, train on the rest."""
    print("=" * 70)
    print("LEAVE-ONE-FIXTURE-OUT CROSS-VALIDATION")
    print("=" * 70)

    fixture_per_row = np.array([r["fixture"] for r in rows])
    fixtures = sorted(set(fixture_per_row.tolist()))
    print(f"  fixtures: {fixtures}")
    print()

    fold_results = []
    for held_out in fixtures:
        train_mask = fixture_per_row != held_out
        test_mask = fixture_per_row == held_out

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        n_train_pos = int(y_train.sum())
        n_test_pos = int(y_test.sum())
        print(
            f"hold out: {held_out:<28} "
            f"train={len(y_train)} ({n_train_pos} pos)  "
            f"test={len(y_test)} ({n_test_pos} pos)"
        )

        result = fit_and_report(X_train, y_train, X_test, y_test, held_out)
        if result:
            fold_results.append(result)
        print()

    if not fold_results:
        print("no folds had positive training examples; can't compute mean metrics")
        return

    mean_acc = float(np.mean([r["accuracy"] for r in fold_results]))
    mean_p = float(np.mean([r["precision"] for r in fold_results]))
    mean_r = float(np.mean([r["recall"] for r in fold_results]))
    mean_f = float(np.mean([r["f1"] for r in fold_results]))
    print(
        f"MEAN over folds:  accuracy={mean_acc:.3f}  precision={mean_p:.3f}  "
        f"recall={mean_r:.3f}  f1={mean_f:.3f}"
    )
    print()


def main():
    parser = argparse.ArgumentParser(description="Train a classifier on attention features.")
    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="Path to features CSV from dump_features binary",
    )
    parser.add_argument(
        "--loo",
        action="store_true",
        help="Run leave-one-fixture-out cross-validation in addition to in-sample",
    )
    args = parser.parse_args()

    if not args.features.exists():
        print(f"error: features file not found: {args.features}", file=sys.stderr)
        sys.exit(1)

    print(f"loading features from {args.features}")
    rows, X, y, feature_names = load_features(args.features)
    print(f"  {len(rows)} rows, {X.shape[1]} features")
    classes, counts = np.unique(y, return_counts=True)
    print(f"  classes: {dict(zip(classes.tolist(), counts.tolist()))}")
    print()

    in_sample_run(X, y, feature_names, rows)

    if args.loo:
        leave_one_fixture_out(X, y, feature_names, rows)


if __name__ == "__main__":
    main()
