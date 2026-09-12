"""End-to-end for the boundary-validation PR (review #3, #5, #6, #11).

Expects cortex-server on :8124 started with:
  --enable-cache --enable-retrieve --max-seq-len 4096 --max-cache-shards 3
Every case below was a panic (500 / bare [DONE]) before the fix.
"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8124"


def req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(r, timeout=900) as resp:
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


def chat(**kw):
    body = {"max_tokens": 8, "messages": [{"role": "user", "content": "hello"}]}
    body.update(kw)
    return req("POST", "/v1/chat/completions", body)


def tokens_of(n):
    s, t = req("POST", "/v1/tokenize", {"text": "The quick brown fox jumps over the lazy dog. "})
    unit = t["tokens"] if isinstance(t, dict) and "tokens" in t else t
    return (unit * (n // len(unit) + 1))[:n]


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

# ---- #3 temperature ------------------------------------------------------
s, r = chat(temperature=1e-40)
print("temperature 1e-40 ->", s); assert s == 200, r
s, r = chat(temperature=-1)
print("temperature -1 ->", s, etype(r)); assert s == 400 and etype(r) == "invalid_temperature", r
s, r = chat(temperature=1e30)
print("temperature 1e30 ->", s); assert s == 200, r

# ---- #5 prompt length (stateless + streaming share the one check) --------
s, r = chat(messages=[{"role": "user", "content": "word " * 5000}])
print("5000-token prompt ->", s, etype(r)); assert s == 400 and etype(r) == "context_length_exceeded", r
s, r = chat(messages=[{"role": "user", "content": "word " * 2200}], max_tokens=4)
print("2200-token prompt (chunked prelude) ->", s); assert s == 200, r
s, r = chat(messages=[{"role": "user", "content": "word " * 5000}], stream=True)
print("5000-token prompt, stream=true ->", s, etype(r)); assert s == 400, r

# ---- #11 token ids -------------------------------------------------------
s, r = req("POST", "/v1/detokenize", {"tokens": [4294967295]})
print("detokenize bad id ->", s, etype(r)); assert s == 400 and etype(r) == "invalid_token_id", r
s, r = req("POST", "/v1/cache/load", {"cache_id": "bad", "tokens": [999999999]})
print("cache/load bad id ->", s, etype(r)); assert s == 400 and etype(r) == "invalid_token_id", r

# ---- #6 pool cap (3) -----------------------------------------------------
small = tokens_of(50)
for name in ("s1", "s2", "s3"):
    s, r = req("POST", "/v1/cache/load", {"cache_id": name, "tokens": small})
    assert s in (200, 201), (name, s, r)
s, r = req("POST", "/v1/cache/load", {"cache_id": "s4", "tokens": small})
print("4th shard with cap 3 ->", s, etype(r)); assert s == 507 and etype(r) == "cache_pool_full", r
s, r = req("POST", "/v1/cache/load", {"cache_id": "s1", "tokens": small})
print("replace existing s1 at cap ->", s); assert s in (200, 201), r
for name in ("s1", "s2", "s3"):
    req("DELETE", "/v1/cache/" + name)

# ---- #5 composition sum check (chat + retrieve) ---------------------------
big = tokens_of(3000)
for name in ("A", "B"):
    s, r = req("POST", "/v1/cache/load", {"cache_id": name, "tokens": big})
    assert s in (200, 201), (name, s, r)
s, r = chat(cache_shards=["A", "B"])
print("compose A+B (6000 tok) chat ->", s, etype(r)); assert s == 400 and etype(r) == "context_length_exceeded", r
s, r = chat(cache_shards=["A", "B"], mode="retrieve", top_k=3)
print("compose A+B retrieve ->", s, etype(r)); assert s == 400 and etype(r) == "context_length_exceeded", r
# a composition that DOES fit must still work (chunked composition prefill)
s, r = req("POST", "/v1/cache/load", {"cache_id": "C", "tokens": tokens_of(400)})
assert s in (200, 201), r
req("DELETE", "/v1/cache/B")
s, r = chat(cache_shards=["A", "C"], max_tokens=4)
print("compose A+C (3400 tok) chat ->", s); assert s == 200, r
for name in ("A", "C"):
    req("DELETE", "/v1/cache/" + name)

print("E2E BOUNDARY OK")
