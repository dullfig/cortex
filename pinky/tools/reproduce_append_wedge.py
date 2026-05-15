"""Repro driver for memex's incremental cache_append wedge bug.

Hammers POST /v1/cache/append with synthetic chunks of ~750 tokens each
until cumulative seq_len crosses ~10K. Times each call. Probes /health
between calls.

Expected (from memex's report): first 10 calls succeed in seconds; the
11th (or whichever crosses ~9.7K) hangs and /health stops responding.

Usage:
    python reproduce_append_wedge.py [--port 8181] [--chunks 12]
"""
import argparse, json, time, urllib.request, urllib.error, threading, random

def post_json(url, payload, timeout=120):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode("utf-8")
    return time.time() - t0, json.loads(body)

def get(url, timeout=5):
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=timeout) as r:
        body = r.read().decode("utf-8")
    return time.time() - t0, body

def probe_health(port, results):
    try:
        dt, _ = get(f"http://localhost:{port}/health", timeout=5)
        results.append(("ok", dt))
    except Exception as e:
        results.append(("fail", str(e)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8181)
    ap.add_argument("--chunks", type=int, default=15)
    ap.add_argument("--chunk_tokens", type=int, default=750)
    ap.add_argument("--cache_id", default="wedge_repro")
    args = ap.parse_args()

    base = f"http://localhost:{args.port}"
    rng = random.Random(0xCAFE)

    # 1. Create empty cache.
    print(f"=== load empty cache: {args.cache_id} ===")
    dt, resp = post_json(f"{base}/v1/cache/load", {"cache_id": args.cache_id, "tokens": []})
    print(f"  load: dt={dt:.2f}s  resp={resp}")

    # 2. Append in chunks of ~chunk_tokens. Use random token IDs in valid Qwen vocab.
    for i in range(1, args.chunks + 1):
        toks = [rng.randint(1, 100000) for _ in range(args.chunk_tokens)]

        # Probe health from a separate thread (background) so we can see
        # if /health hangs while append is running.
        health_results = []
        t = threading.Thread(target=probe_health, args=(args.port, health_results))

        print(f"\n=== chunk {i:2d}: appending {len(toks)} tokens ===")
        try:
            t0 = time.time()
            t.start()
            dt, resp = post_json(f"{base}/v1/cache/append",
                {"cache_id": args.cache_id, "tokens": toks},
                timeout=900,
            )
            t.join(timeout=10)
            print(f"  append: dt={dt:.2f}s  seq_len={resp.get('seq_len','?')}")
            if health_results:
                hr = health_results[0]
                print(f"  health during append: {hr[0]} ({hr[1] if hr[0]=='fail' else f'{hr[1]:.2f}s'})")
        except Exception as e:
            print(f"  append FAILED after {time.time()-t0:.2f}s: {e}")
            if health_results:
                hr = health_results[0]
                print(f"  health during append: {hr[0]} ({hr[1] if hr[0]=='fail' else f'{hr[1]:.2f}s'})")
            print("  WEDGE DETECTED — stopping repro driver")
            return

if __name__ == "__main__":
    main()
