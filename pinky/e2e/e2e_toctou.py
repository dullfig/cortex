"""End-to-end for review #9 — lock re-acquisition is never trusted.

Two deterministic races (tokio's mutex is FIFO, so a writer queued while
an append chunk holds the lock lands exactly between chunks) and one
best-effort race. The append must span THREE chunks: a same-id load takes
the lock once for its pre-flight check (queued behind chunk 1), prefills
during chunk 2, and its insert then lands before chunk 3.
Expects cortex-server on :8124 started with
  --enable-cache --enable-retrieve --max-seq-len 16384
"""
import json
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8124"


def req(method, path, body=None, timeout=1200):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"raw": raw.decode(errors="replace")[:300]}


def etype(r):
    try:
        return r["error"]["type"]
    except Exception:
        return None


def tokens_of(n):
    s, t = req("POST", "/v1/tokenize", {"text": "The quick brown fox jumps over the lazy dog. "})
    unit = t["tokens"] if isinstance(t, dict) and "tokens" in t else t
    return (unit * (n // len(unit) + 1))[:n]


class Call(threading.Thread):
    def __init__(self, *args):
        super().__init__(daemon=True)
        self.args, self.result = args, None

    def run(self):
        self.result = req(*self.args)


for _ in range(150):
    try:
        if req("GET", "/health")[0] == 200:
            break
    except Exception:
        pass
    time.sleep(2)
else:
    print("SERVER NOT READY"); sys.exit(1)
print("server ready")

big = tokens_of(12000)  # >= three append chunks at Qwen 3B (dispatch cap 4095)
small = tokens_of(100)

# The advance-only forward SUBMITS GPU work without waiting, so an append
# holds the pool lock only for the CPU-side submit of each chunk (~40 ms)
# and the whole 12000-token append returns in ~150 ms. The racing request
# must be in flight inside that window; tokio's FIFO mutex then lands it
# between two chunks. Loop and require the race to trigger at least once;
# every outcome must be a clean status and the pool must end consistent.

# ---- A: same-id reload lands between append chunks -> append 409 ----------
hits_a = 0
for i in range(6):
    s, r = req("POST", "/v1/cache/load", {"cache_id": "s", "tokens": small})
    assert s in (200, 201), r
    t = Call("POST", "/v1/cache/append", {"cache_id": "s", "tokens": big})
    t.start()
    time.sleep(0.02 + 0.01 * i)
    s2, r2 = req("POST", "/v1/cache/load", {"cache_id": "s", "tokens": small})
    t.join()
    sa, ra = t.result
    assert s2 in (200, 201), r2
    assert sa in (200, 409), ra
    s, r = req("GET", "/v1/cache/s")
    # Three consistent outcomes: the reload landed mid-append (409, the new
    # entry stays at 104), after the append (200, then replaced -> 104), or
    # before the append took its witness (200, the append extended the NEW
    # entry -> 12104). A split would show 104 + a chunk prefix (3914, 8009).
    if sa == 409:
        assert etype(ra) == "cache_changed", ra
        assert r.get("seq_len") == 104, ("chunks leaked into the reloaded entry", r)
        hits_a += 1
    else:
        assert r.get("seq_len") in (104, 12104), ("append split across entries", r)
    print(f"A[{i}]: append -> {sa} {etype(ra) or ''}; reload -> {s2}; final seq_len {r.get('seq_len')}")
req("DELETE", "/v1/cache/s")
# Since review #7 (gpu_gate) a cache/load queues behind the whole append
# for the GPU, so its insert can no longer land between chunks: expect 0
# hits here. The witness path is exercised by B (DELETE takes no gate)
# and C, and by the unit tests. Pre-#7 this scenario hit 6/6.
print(f"A: same-id reload landed mid-append {hits_a}/6 times (0 expected with the gpu_gate)")

# ---- B: DELETE lands between append chunks -> append 404 ------------------
hits_b = 0
for i in range(6):
    s, r = req("POST", "/v1/cache/load", {"cache_id": "d", "tokens": small})
    assert s in (200, 201), r
    t = Call("POST", "/v1/cache/append", {"cache_id": "d", "tokens": big})
    t.start()
    time.sleep(0.02 + 0.01 * i)
    sd, _ = req("DELETE", "/v1/cache/d")
    t.join()
    sb, rb = t.result
    assert sd == 204, sd
    assert sb in (200, 404), rb
    s, r = req("GET", "/v1/cache/d")
    assert s == 404, ("shard survived its DELETE", r)
    if sb == 404:
        assert etype(rb) == "cache_not_found", rb
        hits_b += 1
    print(f"B[{i}]: append -> {sb} {etype(rb) or ''}; DELETE -> {sd}")
assert hits_b >= 1, "DELETE never landed between chunks; widen the window"

# ---- C: retrieve vs DELETE, best effort (window is microseconds) ----------
counts = {}
for i in range(10):
    s, r = req("POST", "/v1/cache/load", {"cache_id": "r", "tokens": tokens_of(500)})
    assert s in (200, 201), r
    t = Call("POST", "/v1/chat/completions", {
        "mode": "retrieve", "cache_shards": ["r"], "top_k": 3,
        "messages": [{"role": "user", "content": "word " * 200}],
    })
    t.start()
    time.sleep(0.002 * i)
    req("DELETE", "/v1/cache/r")
    t.join()
    sc, rc = t.result
    key = f"{sc}/{etype(rc) or 'ok'}"
    counts[key] = counts.get(key, 0) + 1
    assert sc in (200, 404, 409), rc
print("C: retrieve-vs-DELETE outcomes:", counts)
req("DELETE", "/v1/cache/r")

print("E2E TOCTOU OK")
