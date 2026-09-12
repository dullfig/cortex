"""End-to-end for review #24 (panics -> structured errors).

Expects cortex-server on :8124 booted with
  --enable-cache --enable-test-hooks --max-seq-len 4096
and the env var CORTEX_TEST_PANIC_AFTER_TOKENS=3 (the generation loops
panic on purpose once 3 tokens exist).

Asserts:
  1. streaming: the SSE stream carries an `internal_generation_error`
     event and does NOT end with `[DONE]` (before #24: a bare `[DONE]`,
     indistinguishable from an empty completion);
  2. the server keeps serving (/health 200) and the GPU gate is free (a
     2-token request, which never reaches the hook, completes promptly);
  3. non-streaming stateless: 500 `internal_generation_error` (before
     #24: the connection was dropped with no response);
  4. non-streaming cached: 500, then the shard reports 409
     `cache_desynced` (the review #8 backstop for a half-updated entry)
     and can be deleted;
  5. /metrics counts the failures under status="err".
"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8124"


def req(method, path, body=None, timeout=300):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read()
            try:
                return resp.status, (json.loads(raw) if raw else None)
            except Exception:
                return resp.status, {"raw": raw.decode(errors="replace")}
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"raw": raw.decode(errors="replace")[:200]}
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

msgs = [{"role": "user", "content": "Count slowly from one to twenty."}]

# 1. streaming
s, r = req("POST", "/v1/chat/completions", {"stream": True, "max_tokens": 20, "messages": msgs})
assert s == 200, (s, r)
raw = r["raw"] if isinstance(r, dict) and "raw" in r else json.dumps(r)
datas = [ln[6:].strip() for ln in raw.splitlines() if ln.startswith("data: ")]
print(f"stream: {len(datas)} data events; last = {datas[-1][:100] if datas else None}")
errs = [d for d in datas if '"internal_generation_error"' in d]
assert errs, f"no error event in stream: {datas}"
assert "[DONE]" not in datas, f"[DONE] must not follow an error: {datas}"
assert datas[-1] == errs[-1], "the error event must be the last event"
print("stream: error event present, no [DONE]  OK")

# 2. server alive, gate free
s, _ = req("GET", "/health", timeout=10)
assert s == 200, s
t0 = time.time()
s, r = req("POST", "/v1/chat/completions", {"max_tokens": 2, "messages": msgs})
dt = time.time() - t0
print(f"2-token request after the stream panic -> {s} in {dt:.1f}s")
assert s == 200, (s, r)
assert dt < 30, f"gate not released? {dt:.1f}s"

# 3. non-streaming stateless
s, r = req("POST", "/v1/chat/completions", {"max_tokens": 20, "messages": msgs})
print(f"stateless non-stream -> {s} {etype(r)} {str(r)[:120]}")
assert s == 500 and etype(r) == "internal_generation_error", (s, r)

# 4. cached
s, r = req("POST", "/v1/cache/load", {"cache_id": "p", "tokens": tokens_of(300)})
assert s in (200, 201), r
s, r = req("POST", "/v1/chat/completions", {"cache_shards": ["p"], "max_tokens": 20, "messages": msgs})
print(f"cached non-stream -> {s} {etype(r)}")
assert s == 500 and etype(r) == "internal_generation_error", (s, r)
s, r = req("POST", "/v1/chat/completions", {"cache_shards": ["p"], "max_tokens": 2, "messages": msgs})
print(f"cached follow-up on the half-updated shard -> {s} {etype(r)}")
assert s == 409 and etype(r) == "cache_desynced", (s, r)
s, r = req("DELETE", "/v1/cache/p")
print(f"DELETE the drifted shard -> {s}")
assert s in (200, 204), (s, r)

# 5. metrics
s, r = req("GET", "/metrics")
text = r["raw"] if isinstance(r, dict) and "raw" in r else str(r)
err_lines = [ln for ln in text.splitlines() if ln.startswith("cortex_requests_total") and 'endpoint="chat_completions"' in ln and 'status="err"' in ln]
print("metrics:", err_lines)
assert err_lines and float(err_lines[0].split()[-1]) >= 3, err_lines

print("E2E STREAM PANIC OK")
