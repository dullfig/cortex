"""End-to-end for review #4 — the wgpu 65535 dispatch-dimension class.

The memex-reported crash: one-shot POST /v1/cache/load of ~6K tokens on a
12 GB card picked a 4178-token first chunk; softmax dispatches
n_tokens * n_heads = 16 * 4178 = 66848 > 65535 -> wgpu validation error ->
panic. Expects cortex-server on :8124 started with
  --enable-cache --enable-retrieve --enable-shims --max-seq-len 8192
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
        with urllib.request.urlopen(r, timeout=1200) as resp:
            raw = resp.read()
            try:
                return resp.status, (json.loads(raw) if raw else None)
            except Exception:  # SSE bodies are not JSON; keep the whole stream
                return resp.status, {"raw": raw.decode(errors="replace")}
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

# ---- THE memex scenario: one-shot load of a large history ----------------
s, r = req("POST", "/v1/cache/load", {"cache_id": "history", "tokens": tokens_of(6000)})
print("one-shot cache/load 6000 tokens ->", s, str(r)[:120])
assert s in (200, 201) and r.get("seq_len") == 6004, r
req("DELETE", "/v1/cache/history")

# ---- unchunked paths are bounded by the dispatch limit (4095 on Qwen 3B) --
big = "word " * 4300  # ~4300 tokens: < max_seq_len 8192, > dispatch bound 4095
s, r = req("POST", "/v1/shims/embed", {"text": big, "layer": "final"})
print("shims/embed 4300 tokens ->", s, etype(r))
assert s == 400 and etype(r) == "context_length_exceeded", r

s, r = req("POST", "/v1/cache/load", {"cache_id": "s", "tokens": tokens_of(50)})
assert s in (200, 201), r
s, r = req("POST", "/v1/chat/completions", {
    "mode": "retrieve", "cache_shards": ["s"], "top_k": 3,
    "messages": [{"role": "user", "content": big}],
})
print("retrieve query 4300 tokens ->", s, etype(r))
assert s == 400 and etype(r) == "context_length_exceeded", r
req("DELETE", "/v1/cache/s")

# ---- a chunked path at the same size still works --------------------------
s, r = req("POST", "/v1/chat/completions", {
    "max_tokens": 4, "messages": [{"role": "user", "content": big}],
})
print("stateless 4300-token prompt (chunked) ->", s)
assert s == 200, r

s, r = req("POST", "/v1/chat/completions", {
    "max_tokens": 4, "stream": True, "messages": [{"role": "user", "content": big}],
})
print("streaming 4300-token prompt (chunked) ->", s, "[DONE]" in str(r))
assert s == 200 and "[DONE]" in str(r), r

print("E2E MEMEX OK")
