"""fetch_via_api.py — fetch the Lakera/gandalf_ignore_instructions dataset
via HuggingFace's datasets-server JSON API.

This avoids needing pyarrow or pandas for parquet reading. The dataset is
small enough (1000 entries total) that paginating through the JSON API
is faster than installing parquet support.

Output: gandalf_all.json with the structure:
  {
    "train": [{"text": "...", "similarity": 0.85}, ...],
    "validation": [...],
    "test": [...]
  }
"""

import json
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "Lakera/gandalf_ignore_instructions"
SPLITS = ["train", "validation", "test"]
PAGE_SIZE = 100


def fetch_split(split):
    rows = []
    offset = 0
    while True:
        url = (
            "https://datasets-server.huggingface.co/rows"
            f"?dataset={urllib.parse.quote(DATASET)}"
            "&config=default"
            f"&split={split}"
            f"&offset={offset}"
            f"&length={PAGE_SIZE}"
        )
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        new_rows = [r["row"] for r in data["rows"]]
        rows.extend(new_rows)
        total = data["num_rows_total"]
        if offset + len(new_rows) >= total or not new_rows:
            break
        offset += len(new_rows)
    return rows


def main():
    out = {}
    for split in SPLITS:
        print(f"fetching {split}...")
        rows = fetch_split(split)
        out[split] = rows
        print(f"  got {len(rows)} rows")

    output_path = Path(__file__).parent / "gandalf_all.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in out.values())
    print(f"\nwrote {total} total rows to {output_path}")


if __name__ == "__main__":
    main()
