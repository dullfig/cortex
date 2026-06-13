# Validate a retrieval Q/D set (Phase P.3).
#
# For every query: gold_text must appear VERBATIM in its shard file, and
# (warn) ideally exactly once — the resolver uses first-occurrence, so a
# multi-occurrence gold is only OK if the first hit is the intended span.
# Also reports the gold span's token-position as a % of shard length, so
# we can confirm a spread of near/far golds (position-bias coverage), and
# the train/test balance.
#
# Server-based (uses /v1/tokenize, same as the harness).
#   python validate_qd.py --qd C:\Users\danu\polar-recall-qd-50.json \
#       --server http://127.0.0.1:8100 --corpus C:\src\bhs-corpus
import argparse, json, os, sys, urllib.request

def post(base, path, body, timeout=120):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qd", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:8100")
    ap.add_argument("--corpus", default=r"C:\src\bhs-corpus")
    args = ap.parse_args()

    qd = json.load(open(args.qd, encoding="utf-8"))
    shards = {s["id"]: s for s in qd["shards"]}
    texts, lens = {}, {}
    for sid, s in shards.items():
        texts[sid] = open(os.path.join(args.corpus, s["file"]), encoding="utf-8").read()
        lens[sid] = post(args.server, "/v1/tokenize", {"text": texts[sid], "add_bos": False})["count"]

    n_problems = 0
    per_shard = {}
    print(f"{'qid':>5} {'shard':<18} {'occ':>3} {'pos%':>5}  status")
    for q in qd["queries"]:
        sid = q["gold_shard"]
        split = shards[sid].get("split", "?")
        per_shard.setdefault(sid, {"split": split, "n": 0})["n"] += 1
        text = texts[sid]
        gt = q["gold_text"]
        occ = text.count(gt)
        if occ == 0:
            print(f"{q['id']:>5} {sid:<18} {0:>3} {'--':>5}  *** GOLD NOT FOUND: {gt!r}")
            n_problems += 1
            continue
        idx = text.find(gt)
        start_tok = post(args.server, "/v1/tokenize", {"text": text[:idx], "add_bos": False})["count"]
        pos_pct = 100.0 * start_tok / max(1, lens[sid])
        # occ==0 is a hard failure (typo / stale gold). occ>1 is only a
        # WARNING: acceptable when the first occurrence is the intended
        # span (e.g. a fact repeated verbatim in the same doc).
        status = "ok" if occ == 1 else f"WARN multi x{occ} (first-occ used)"
        print(f"{q['id']:>5} {sid:<18} {occ:>3} {pos_pct:>4.0f}%  {status}")

    print("\n=== shard summary ===")
    tr = sum(1 for s in shards.values() if s.get("split") == "train")
    te = sum(1 for s in shards.values() if s.get("split") == "test")
    print(f"shards: {len(shards)}  (train {tr} / test {te})")
    qtr = sum(v["n"] for v in per_shard.values() if v["split"] == "train")
    qte = sum(v["n"] for v in per_shard.values() if v["split"] == "test")
    print(f"queries: {len(qd['queries'])}  (train {qtr} / test {qte})")
    for sid, v in sorted(per_shard.items(), key=lambda kv: kv[1]["split"]):
        print(f"  {sid:<18} {v['split']:<6} {v['n']} queries  ({lens[sid]} tok)")

    print(f"\n{'CLEAN' if n_problems == 0 else f'{n_problems} PROBLEM(S)'}")
    sys.exit(1 if n_problems else 0)

if __name__ == "__main__":
    main()
