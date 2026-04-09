"""Synthetic fixture generator with adversarial doc-region content.

Like ../synthetic/generate_fixtures.py, but every fixture has at least one
doc region whose content is a real prompt-injection attack drawn from the
Lakera Gandalf dataset (gandalf_ignore_instructions, train split, MIT
license). About 20% of fixtures additionally include a normal doc region
so the classifier learns that "doc" != "always injection".

Usage:
    python generate_injection_fixtures.py --count 50 --seed 43
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

SOURCES = ["system", "user", "doc", "tool"]

HERE = Path(__file__).resolve().parent
SNIPPETS_DIR = HERE.parent / "synthetic" / "snippets"
GANDALF_PATH = HERE.parent.parent / "datasets" / "gandalf" / "gandalf_all.json"
OUT_DIR = HERE / "fixtures"

MAX_INJECTION_LEN = 400
BOTH_DOC_FRACTION = 0.20


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


def clean_injection(raw: str) -> str | None:
    """Normalize a Gandalf attack string for use in a single-line fixture region."""
    s = raw.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    # collapse runs of whitespace
    s = " ".join(s.split())
    s = s.strip()
    if not s:
        return None
    if s.startswith("#"):
        return None
    if len(s) > MAX_INJECTION_LEN:
        cut = s[:MAX_INJECTION_LEN]
        # truncate at last word boundary
        sp = cut.rfind(" ")
        if sp > MAX_INJECTION_LEN // 2:
            cut = cut[:sp]
        s = cut.rstrip() + "..."
    return s


def load_injections() -> list[str]:
    with GANDALF_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out: list[str] = []
    skipped = 0
    for entry in data["train"]:
        cleaned = clean_injection(entry.get("text", ""))
        if cleaned is None:
            skipped += 1
            continue
        out.append(cleaned)
    if not out:
        raise ValueError("no usable Gandalf injections after cleaning")
    print(f"  loaded {len(out)} Gandalf injections ({skipped} skipped)")
    return out


def pick_region_sequence(rng: random.Random, include_extra_doc: bool) -> list[str]:
    """Pick 2-5 source tags. Guarantee at least one doc region (the injection slot)
    and at least one source-tag transition. If include_extra_doc, ensure two doc slots."""
    while True:
        n = rng.randint(2, 5)
        seq = [rng.choice(SOURCES) for _ in range(n)]
        doc_count = sum(1 for s in seq if s == "doc")
        need_doc = 2 if include_extra_doc else 1
        if doc_count < need_doc:
            # force-place docs at random positions
            positions = list(range(n))
            rng.shuffle(positions)
            for p in positions[: need_doc - doc_count]:
                seq[p] = "doc"
        if not any(seq[i] != seq[i + 1] for i in range(len(seq) - 1)):
            continue
        return seq


def generate_fixture(
    idx: int,
    libs: dict[str, list[str]],
    injections: list[str],
    rng: random.Random,
) -> tuple[str, list[str], bool]:
    include_extra_doc = rng.random() < BOTH_DOC_FRACTION
    seq = pick_region_sequence(rng, include_extra_doc)

    # Choose which doc index gets the injection (exactly one).
    doc_indices = [i for i, s in enumerate(seq) if s == "doc"]
    inj_idx = rng.choice(doc_indices)

    lines = [
        f"# inj_{idx:03d}: auto-generated fixture with {len(seq)} regions.",
        "# Format: <source>: <content>. Source changes are trust boundaries.",
        "# At least one doc region contains a real Gandalf prompt-injection attack.",
        "",
    ]
    for i, src in enumerate(seq):
        if src == "doc" and i == inj_idx:
            content = rng.choice(injections)
        else:
            content = rng.choice(libs[src])
        lines.append(f"{src}: {content}")
    lines.append("")
    return "\n".join(lines), seq, include_extra_doc and len(doc_indices) >= 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    libs = load_snippets()
    injections = load_injections()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    total_regions = 0
    total_boundaries = 0
    source_counter: Counter[str] = Counter()
    length_counter: Counter[int] = Counter()
    both_doc_count = 0

    for i in range(args.count):
        text, seq, both = generate_fixture(i, libs, injections, rng)
        path = out_dir / f"inj_{i:03d}.txt"
        path.write_text(text, encoding="utf-8")

        total_regions += len(seq)
        total_boundaries += sum(1 for j in range(len(seq) - 1) if seq[j] != seq[j + 1])
        source_counter.update(seq)
        length_counter[len(seq)] += 1
        if both:
            both_doc_count += 1

    print(f"Wrote {args.count} fixtures to {out_dir}")
    print(f"  total regions:           {total_regions}")
    print(f"  total trust boundaries:  {total_boundaries} (positive examples)")
    print(f"  fixtures w/ injection:   {args.count} (every fixture)")
    print(f"  fixtures w/ both normal+injection doc: {both_doc_count}")
    print("  source distribution:")
    for src in SOURCES:
        print(f"    {src:<6} {source_counter[src]}")
    print("  fixture length distribution (regions -> count):")
    for length in sorted(length_counter):
        print(f"    {length} regions: {length_counter[length]}")


if __name__ == "__main__":
    main()
