"""Demonstration analysis script for the morsel-retrieval test fixture corpus.

This script does NOT run retrieval. It loads the planted corpus and the
ground-truth connections file, prints a summary, and shows what a future
morsel-level retriever is expected to surface for each test query.

Usage:
    python demo_analysis.py
    python demo_analysis.py --papers-dir papers --connections connections.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_papers(papers_dir: Path) -> dict[str, dict]:
    papers: dict[str, dict] = {}
    for p in sorted(papers_dir.glob("*.txt")):
        text = p.read_text(encoding="utf-8").strip()
        lines = text.splitlines()
        title = lines[0].strip() if len(lines) > 0 else "(no title)"
        authors = lines[1].strip() if len(lines) > 1 else "(no authors)"
        body = "\n".join(lines[2:]).strip()
        word_count = len(text.split())
        papers[p.name] = {
            "path": p,
            "title": title,
            "authors": authors,
            "body": body,
            "word_count": word_count,
        }
    return papers


def load_connections(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("connections", [])


def verify_morsels(papers: dict[str, dict], connections: list[dict]) -> list[str]:
    """Check that every planted morsel actually appears verbatim in its paper."""
    problems: list[str] = []
    for conn in connections:
        for ref in conn["papers"]:
            fname = ref["file"]
            morsel = ref["morsel_text"]
            if fname not in papers:
                problems.append(f"connection {conn['id']}: paper {fname} not found")
                continue
            if morsel not in papers[fname]["body"] and morsel not in (
                papers[fname]["title"] + "\n" + papers[fname]["authors"] + "\n" + papers[fname]["body"]
            ):
                problems.append(
                    f"connection {conn['id']}: morsel not found verbatim in {fname}"
                )
    return problems


def print_corpus_summary(papers: dict[str, dict]) -> None:
    total_words = sum(p["word_count"] for p in papers.values())
    print("=" * 72)
    print("CORPUS SUMMARY")
    print("=" * 72)
    print(f"Papers:      {len(papers)}")
    print(f"Total words: {total_words}")
    print(f"Mean words:  {total_words // max(1, len(papers))}")
    print()
    print("Titles:")
    for name, p in papers.items():
        print(f"  [{name}] ({p['word_count']} words)")
        print(f"      {p['title']}")
    print()


def print_connection(idx: int, conn: dict, papers: dict[str, dict]) -> None:
    print("-" * 72)
    print(f"CONNECTION {idx}: {conn['id']}")
    print("-" * 72)
    print(f"Description:")
    print(f"  {conn['description']}")
    print()
    print(f"Test query:")
    print(f"  \"{conn['test_query']}\"")
    print()
    print("Planted morsels (these are what a morsel retriever should surface):")
    for ref in conn["papers"]:
        fname = ref["file"]
        title = papers.get(fname, {}).get("title", "(unknown)")
        print(f"  - {fname}")
        print(f"    Paper title: {title}")
        print(f"    Morsel: \"{ref['morsel_text']}\"")
        print()
    print("Naive retriever failure mode:")
    print(f"  {conn['naive_query_result']}")
    print()
    print("Success criterion:")
    print("  A morsel-level retriever using raw Q.K^T attention over stored KV")
    print("  cache should surface BOTH morsels above in its top-k results for")
    print("  the test query, while a conventional topic/keyword retriever will")
    print("  return at most one of the two papers.")
    print()


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--papers-dir",
        type=Path,
        default=here / "papers",
        help="Directory containing paper .txt files",
    )
    parser.add_argument(
        "--connections",
        type=Path,
        default=here / "connections.json",
        help="Path to connections.json ground-truth file",
    )
    args = parser.parse_args()

    if not args.papers_dir.is_dir():
        print(f"error: papers directory not found: {args.papers_dir}")
        return 2
    if not args.connections.is_file():
        print(f"error: connections file not found: {args.connections}")
        return 2

    papers = load_papers(args.papers_dir)
    connections = load_connections(args.connections)

    print_corpus_summary(papers)

    print("=" * 72)
    print(f"PLANTED CONNECTIONS ({len(connections)})")
    print("=" * 72)
    print()
    for i, conn in enumerate(connections, 1):
        print_connection(i, conn, papers)

    problems = verify_morsels(papers, connections)
    print("=" * 72)
    print("MORSEL VERIFICATION")
    print("=" * 72)
    if problems:
        print("Problems found:")
        for prob in problems:
            print(f"  ! {prob}")
        return 1
    print("All planted morsels verified verbatim in their source papers.")
    print()
    print("This fixture is ready for retrieval evaluation. No retrieval was")
    print("run by this script — wiring cortex's Q.K^T retrieval pass against")
    print("this corpus is future work.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
