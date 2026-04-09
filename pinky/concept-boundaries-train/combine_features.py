"""combine_features.py — concatenate two feature CSVs into one.

The training pipeline takes a single training CSV. This script combines
two CSVs (verifying they have the same header) into a single output.

Usage:
    python combine_features.py \
        --inputs ../concept-boundaries/features_synthetic.csv \
                 ../concept-boundaries/features_injection.csv \
        --output ../concept-boundaries/features_combined.csv

The header must match between input files (same number and order of
feature columns). The output preserves the header from the first file
and concatenates all data rows.
"""

import argparse
import csv
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Concatenate feature CSVs.")
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="Two or more input CSVs to concatenate (in order).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    args = parser.parse_args()

    if len(args.inputs) < 2:
        print("error: need at least two input files", file=sys.stderr)
        sys.exit(1)

    for p in args.inputs:
        if not p.exists():
            print(f"error: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    # Read first file to get the header
    with open(args.inputs[0], "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

    # Verify all other files have the same header
    for p in args.inputs[1:]:
        with open(p, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            other_header = next(reader)
            if other_header != header:
                print(
                    f"error: header mismatch between {args.inputs[0]} and {p}",
                    file=sys.stderr,
                )
                sys.exit(1)

    # Concatenate
    total_rows = 0
    total_positives = 0
    label_idx = header.index("label")
    with open(args.output, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f, lineterminator="\n")
        writer.writerow(header)
        for p in args.inputs:
            with open(p, "r", encoding="utf-8", newline="") as in_f:
                reader = csv.reader(in_f)
                next(reader)  # skip header
                file_rows = 0
                file_positives = 0
                for row in reader:
                    writer.writerow(row)
                    file_rows += 1
                    if row[label_idx] == "1":
                        file_positives += 1
                print(
                    f"  {p.name}: {file_rows} rows, {file_positives} positives"
                )
                total_rows += file_rows
                total_positives += file_positives

    print(
        f"\nwrote {total_rows} total rows ({total_positives} positives, "
        f"{100 * total_positives / total_rows:.1f}%) to {args.output}"
    )


if __name__ == "__main__":
    main()
