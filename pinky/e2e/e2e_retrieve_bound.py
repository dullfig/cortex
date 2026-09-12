"""End-to-end for review #12 — retrieve queries are bounded by lane B, the
storage-binding cap and (polar) host_readback_heap summed over the four
captured layers, as a 400, instead of `.expect("host_readback_heap
capacity")` panicking a worker.

Run twice: `python e2e_retrieve_bound.py polar` against a server started
with --enable-cache --enable-retrieve --enable-polar-cache --max-seq-len 8192
and `python e2e_retrieve_bound.py f32` against one without --enable-polar-cache.
"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8124"
MODE = sys.argv[1] if len(sys.argv) > 1 else "polar"


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
    except Exception as e:
        return -1, {"raw": repr(e)}


def etype(r):
    try:
        return r["error"]["type"]
    except Exception:
        return None


def tokens_of(n):
    s, t = req("POST", "/v1/tokenize", {"text": "The quick brown fox jumps over the lazy dog. "})
    unit = t["tokens"] if isinstance(t, dict) and "tokens" in t else t
    return (unit * (n // len(unit) + 1))[:n]


def retrieve(shards, n_words):
    return req("POST", "/v1/chat/completions", {
        "mode": "retrieve", "cache_shards": shards, "top_k": 3,
        "messages": [{"role": "user", "content": "word " * n_words}],
    })


for _ in range(150):
    try:
        if req("GET", "/health")[0] == 200:
            break
    except Exception:
        pass
    time.sleep(2)
else:
    print("SERVER NOT READY"); sys.exit(1)
print(f"server ready ({MODE})")

s, r = req("POST", "/v1/cache/load", {"cache_id": "big", "tokens": tokens_of(4000)})
assert s in (200, 201), r

if MODE == "polar":
    # ~300-token query against a 4004-token shard: 4 layers x 300 x 16 x
    # 4304 x 4 B = 330 MB of readback stagings > 256 MiB heap. Before: panic.
    s, r = retrieve(["big"], 300)
    print("polar: 300-token query ->", s, etype(r), str(r.get("error", {}).get("message", ""))[:110] if isinstance(r, dict) else "")
    assert s == 400 and etype(r) == "context_length_exceeded", r
    s, r = retrieve(["big"], 100)
    print("polar: 100-token query ->", s, etype(r))
    assert s == 200, r
else:
    # Review #23: the f32 traced path's captures and stagings are budgeted
    # like the polar path's (4 captures on lane B, one readback span), so
    # it is readback-bound at the same ~238 tokens against a 4004-token
    # shard: a 300-token query is a 400, a 100-token query runs. (Before
    # #23 the f32 stagings were raw, unbudgeted buffers and 300 passed.)
    s, r = retrieve(["big"], 300)
    print("f32: 300-token query ->", s, etype(r), str(r.get("error", {}).get("message", ""))[:110] if isinstance(r, dict) else "")
    assert s == 400 and etype(r) == "context_length_exceeded", r
    s, r = retrieve(["big"], 100)
    print("f32: 100-token query ->", s, etype(r))
    assert s == 200, r
    s, r = retrieve(["big"], 4300)
    print("f32: 4300-token query (4004 + 4300 > 8192 window / 4095 dispatch) ->", s, etype(r))
    assert s == 400 and etype(r) == "context_length_exceeded", r

req("DELETE", "/v1/cache/big")
print(f"E2E RETRIEVE BOUND ({MODE}) OK")
