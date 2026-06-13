# Per-head retrieval sweep (Phase P.1).
#
# Drives the 10-query Q/D set against a cortex-server started with
# CORTEX_RETRIEVE_HEAD_DUMP=<dir>, then computes per-(layer, head)
# differential R@1/5/10 from the head-resolved dumps. Identifies
# retrieval heads (Wu et al. 2404.15574): the hypothesis is that
# aggregate attention mass is plumbing and recall lives in a sparse
# subset of heads.
#
# Usage:
#   1. start cortex-server with --enable-retrieve [--enable-polar-cache]
#      and CORTEX_RETRIEVE_HEAD_DUMP=<dumpdir> in its environment
#   2. python sweep.py --server http://127.0.0.1:8100 --dumps <dumpdir> \
#        [--polar] [--label polar]
#
# Output: heat table per metric, ranked head list, greedy head subset,
# and a sanity check that all-heads-MAX reproduces the production
# aggregate.
import argparse, json, os, sys, urllib.request

QD = r"C:\Users\danu\polar-recall-qd.json"
CORPUS = r"C:\src\bhs-corpus"
TOL = 3

def post(base, path, body, timeout=600):
    req = urllib.request.Request(
        base + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def delete(base, cid):
    try:
        req = urllib.request.Request(f"{base}/v1/cache/{cid}", method="DELETE")
        urllib.request.urlopen(req, timeout=60)
    except Exception:
        pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8100")
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--polar", action="store_true",
                    help="load shards polar_only (server must have --enable-polar-cache)")
    ap.add_argument("--label", default="run")
    ap.add_argument("--qd", default=QD, help="Q/D json path")
    ap.add_argument("--holdout", choices=["shard"], default=None,
                    help="by-shard train/test: greedy-select heads on train-split "
                         "queries, validate on test-split (unseen-document) queries")
    ap.add_argument("--skip-drive", action="store_true",
                    help="analyze existing dumps only (records.json must exist)")
    args = ap.parse_args()
    os.makedirs(args.dumps, exist_ok=True)
    records_path = os.path.join(args.dumps, "records.json")

    qd = json.load(open(args.qd, encoding="utf-8"))
    by_shard = {}
    for q in qd["queries"]:
        by_shard.setdefault(q["gold_shard"], []).append(q)
    # qid -> split, from each query's gold_shard's split (default "train")
    shard_split = {s["id"]: s.get("split", "train") for s in qd["shards"]}
    split_of = {q["id"]: shard_split.get(q["gold_shard"], "train") for q in qd["queries"]}

    # ---- drive phase: fire queries, server writes dumps in order ----
    if not args.skip_drive:
        records = []  # parallel to dump seq order
        try:
            drive(args, qd, by_shard, records)
        except Exception as e:
            print(f"WARN: drive aborted early ({type(e).__name__}: {e}); "
                  f"analyzing the {len(records)} completed queries", file=sys.stderr)
        json.dump(records, open(records_path, "w"))
        print(f"drove {len(records)} queries; dumps + records in {args.dumps}")
    records = json.load(open(records_path))
    analyze(args, records, split_of)

def drive(args, qd, by_shard, records):
        for shard in qd["shards"]:
            sid = shard["id"]
            if sid not in by_shard:
                continue
            text = open(os.path.join(CORPUS, shard["file"]), encoding="utf-8").read()
            tok = post(args.server, "/v1/tokenize", {"text": text, "add_bos": False})
            delete(args.server, sid)
            post(args.server, "/v1/cache/load", {
                "cache_id": sid, "tokens": tok["tokens"],
                "polar_only": args.polar, "polar_chat": False, "qjl": False,
            })
            for q in by_shard[sid]:
                # gold span resolution — token-prefix counting, same as the ps1
                idx = text.find(q["gold_text"])
                if idx < 0:
                    print(f"WARN: gold text not found for {q['id']}", file=sys.stderr)
                    continue
                start_tok = post(args.server, "/v1/tokenize",
                                 {"text": text[:idx], "add_bos": False})["count"]
                end_tok = post(args.server, "/v1/tokenize",
                               {"text": text[:idx + len(q["gold_text"])], "add_bos": False})["count"]
                r = post(args.server, "/v1/chat/completions", {
                    "model": "cortex",
                    "messages": [{"role": "user", "content": q["query"]}],
                    "temperature": 0, "max_tokens": 1,
                    "mode": "retrieve", "top_k": 10, "cache_shards": [sid],
                })
                prod_hits = [(h["offset"], h["score"]) for h in r.get("hits", [])]
                records.append({
                    "qid": q["id"], "shard": sid,
                    "gold_start": start_tok, "gold_end": end_tok,
                    "prod_hits": prod_hits,
                })
            delete(args.server, sid)

def analyze(args, records, split_of=None):
    # ---- analysis phase ----
    dumps = sorted(f for f in os.listdir(args.dumps) if f.startswith("headdump-"))
    assert len(dumps) == len(records), f"{len(dumps)} dumps vs {len(records)} records"

    loaded = []
    for fname, rec in zip(dumps, records):
        d = json.load(open(os.path.join(args.dumps, fname)))
        loaded.append((d, rec))

    layers = loaded[0][0]["layers"]
    n_heads = loaded[0][0]["n_heads"]
    KS = (1, 5, 10)

    def rank_hits(diff, sink, gold_start, gold_end):
        """positions ranked by diff desc, skipping sinks; return content offsets"""
        order = sorted(
            ((k, s) for k, s in enumerate(diff) if k >= sink),
            key=lambda t: -t[1])
        out = []
        for k, s in order[:10]:
            out.append(k - sink)
        return out

    def is_hit(offsets, gold_start, gold_end, topk):
        return any(gold_start - TOL <= o <= gold_end + TOL for o in offsets[:topk])

    # per-(layer, head) recall
    head_recall = {}  # (layer, head) -> [r1, r5, r10]
    per_query_diff = []  # [(rec, {(li,h): diff vector})]
    for d, rec in loaded:
        sink = d["sink_tokens"]
        diffs = {}
        for li in range(len(layers)):
            for h in range(n_heads):
                qrow = d["query"][li][h]
                brow = d["baseline"][li][h]
                diffs[(li, h)] = [a - b for a, b in zip(qrow, brow)]
        per_query_diff.append((d, rec, diffs))
        for (li, h), diff in diffs.items():
            offs = rank_hits(diff, sink, rec["gold_start"], rec["gold_end"])
            cell = head_recall.setdefault((layers[li], h), [0, 0, 0])
            for i, kk in enumerate(KS):
                cell[i] += is_hit(offs, rec["gold_start"], rec["gold_end"], kk)

    n = len(loaded)
    print(f"\n=== per-head differential R@K over {n} queries ({args.label}) ===")
    print(f"{'layer':>6} {'head':>4}   R@1   R@5  R@10")
    ranked_heads = sorted(head_recall.items(), key=lambda kv: (-kv[1][2], -kv[1][1], -kv[1][0]))
    for (l, h), (r1, r5, r10) in ranked_heads[:20]:
        print(f"{l:>6} {h:>4}  {r1/n:.2f}  {r5/n:.2f}  {r10/n:.2f}")
    print(f"  ... ({len(ranked_heads)} heads total; showing top 20 by R@10)")

    # heat table (R@10)
    print(f"\n=== R@10 heat table (rows = layers, cols = heads) ===")
    print("      " + "".join(f"{h:>5}" for h in range(n_heads)))
    for l in layers:
        row = "".join(f"{head_recall[(l, h)][2]:>5}" for h in range(n_heads))
        print(f"L{l:<4} {row}")

    # all-heads MAX sanity vs production. `subset` = list of (d, rec, diffs)
    # to evaluate over; defaults to all queries.
    def combined_recall(head_set, subset=None):
        """elementwise max of diffs over the head set (matches production MAX agg)"""
        items = subset if subset is not None else per_query_diff
        if not items:
            return [0.0, 0.0, 0.0]
        counts = [0, 0, 0]
        for d, rec, diffs in items:
            sink = d["sink_tokens"]
            clen = d["corpus_len"]
            combo = [max(diffs[(li, h)][k]
                         for li in range(len(layers)) for h in range(n_heads)
                         if (layers[li], h) in head_set)
                     for k in range(clen)]
            offs = rank_hits(combo, sink, rec["gold_start"], rec["gold_end"])
            for i, kk in enumerate(KS):
                counts[i] += is_hit(offs, rec["gold_start"], rec["gold_end"], kk)
        return [c / len(items) for c in counts]

    all_heads = {(l, h) for l in layers for h in range(n_heads)}
    allr = combined_recall(all_heads)
    print(f"\nall-heads MAX-combined (≈ production w/ MAX-differential): "
          f"R@1={allr[0]:.2f} R@5={allr[1]:.2f} R@10={allr[2]:.2f}")

    def greedy_select(train_items, steps=8):
        """greedy head-subset maximizing (R@10,R@5,R@1) on train_items.
        Returns list of (chosen_set_copy, train_r) per step."""
        chosen, best = set(), (-1.0, -1.0, -1.0)
        history = []
        for _ in range(steps):
            best_h, best_r = None, best
            for cand in all_heads - chosen:
                r = combined_recall(chosen | {cand}, train_items)
                key = (r[2], r[1], r[0])
                if key > best_r:
                    best_r, best_h = key, cand
            if best_h is None:
                break
            chosen.add(best_h); best = best_r
            history.append((set(chosen), best_r))
        return history

    # ---- whole-set greedy (selection-biased; for description only) ----
    print("\n=== greedy head-subset selection (WHOLE set — selection-biased) ===")
    for chosen, r in greedy_select(per_query_diff):
        spec = ",".join(f"{l}:{h}" for l, h in sorted(chosen))
        last = sorted(chosen)[-1]
        print(f"  +L{last[0]}:H{last[1]}  ->  R@1={r[2]:.2f} R@5={r[1]:.2f} "
              f"R@10={r[0]:.2f}   ({len(chosen)} heads)  CORTEX_RETRIEVE_HEADS={spec}")

    # ---- by-shard holdout: the trustworthy number ----
    if args.holdout == "shard" and split_of is not None:
        train_items = [(d, rec, df) for (d, rec, df) in per_query_diff
                       if split_of.get(rec["qid"]) == "train"]
        test_items = [(d, rec, df) for (d, rec, df) in per_query_diff
                      if split_of.get(rec["qid"]) == "test"]
        print(f"\n=== BY-SHARD HOLDOUT (train {len(train_items)} q / "
              f"test {len(test_items)} q, unseen documents) ===")
        all_test = combined_recall(all_heads, test_items)
        print(f"all-heads on TEST: R@1={all_test[0]:.2f} R@5={all_test[1]:.2f} R@10={all_test[2]:.2f}")
        print(f"{'#heads':>6} {'trainR@10':>10} {'testR@10':>9} {'gap':>6}  heads")
        for chosen, train_r in greedy_select(train_items):
            test_r = combined_recall(chosen, test_items)
            spec = ",".join(f"{l}:{h}" for l, h in sorted(chosen))
            gap = train_r[0] - test_r[0]
            print(f"{len(chosen):>6} {train_r[0]:>10.2f} {test_r[0]:>9.2f} {gap:>+6.2f}  {spec}")
        print("\nThe TEST R@10 column is the trustworthy 'generalizes to unseen "
              "documents' number; train-test gap = overfitting.")

if __name__ == "__main__":
    main()
