"""Synthetic fixture generator for the concept-boundaries trainable classifier.

Loads per-source snippet libraries and emits text fixtures in the format
consumed by pinky/concept-boundaries/src/bin/dump_features.rs:

    <source>: <content>

where <source> is one of system, user, doc, tool. Every fixture contains at
least one source-tag change (so at least one positive trust-boundary example)
and avoids degenerate single-region outputs.

Usage:
    python generate_fixtures.py --count 50 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

SOURCES = ["system", "user", "doc", "tool"]

HERE = Path(__file__).resolve().parent
SNIPPETS_DIR = HERE / "snippets"
OUT_DIR = HERE / "fixtures"


def load_snippets() -> dict[str, list[str]]:
    libs: dict[str, list[str]] = {}
    for src in SOURCES:
        path = SNIPPETS_DIR / f"{src}.json"
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        snippets = data["snippets"]
        if not snippets:
            raise ValueError(f"snippet library {path} is empty")
        libs[src] = list(snippets)
    return libs


def pick_region_sequence(rng: random.Random) -> list[str]:
    """Pick 2-5 source tags with at least one change. Mix up ordering."""
    n = rng.randint(2, 5)
    while True:
        seq = [rng.choice(SOURCES) for _ in range(n)]
        # Must have at least one source-tag change between consecutive regions.
        if any(seq[i] != seq[i + 1] for i in range(len(seq) - 1)):
            return seq


def generate_fixture(
    idx: int, libs: dict[str, list[str]], rng: random.Random
) -> tuple[str, list[str]]:
    seq = pick_region_sequence(rng)
    lines = [
        f"# synth_{idx:03d}: auto-generated fixture with {len(seq)} regions.",
        "# Format: <source>: <content>. Source changes are trust boundaries.",
        "",
    ]
    for src in seq:
        snippet = rng.choice(libs[src])
        lines.append(f"{src}: {snippet}")
    lines.append("")
    return "\n".join(lines), seq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=50, help="number of fixtures to emit")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR,
        help="output directory for fixture files",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    libs = load_snippets()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    total_regions = 0
    total_boundaries = 0
    source_counter: Counter[str] = Counter()
    length_counter: Counter[int] = Counter()

    for i in range(args.count):
        text, seq = generate_fixture(i, libs, rng)
        path = out_dir / f"synth_{i:03d}.txt"
        path.write_text(text, encoding="utf-8")

        total_regions += len(seq)
        total_boundaries += sum(1 for j in range(len(seq) - 1) if seq[j] != seq[j + 1])
        source_counter.update(seq)
        length_counter[len(seq)] += 1

    print(f"Wrote {args.count} fixtures to {out_dir}")
    print(f"  total regions:           {total_regions}")
    print(f"  total trust boundaries:  {total_boundaries} (positive examples)")
    print("  source distribution:")
    for src in SOURCES:
        print(f"    {src:<6} {source_counter[src]}")
    print("  fixture length distribution (regions -> count):")
    for length in sorted(length_counter):
        print(f"    {length} regions: {length_counter[length]}")


if __name__ == "__main__":
    main()
