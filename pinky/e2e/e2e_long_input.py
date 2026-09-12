"""End-to-end for review #10 (tokenizer cost bounded, inputs capped).

Expects cortex-server on :8124 booted with --max-seq-len 4096 (input cap =
max(64 KiB, 16 x 4096) = 64 KiB).

  1. /v1/tokenize with 60 KB of unbroken letters returns 200 in seconds
     (before #10: O(n^2) merges with two allocations per pair — minutes);
  2. /v1/tokenize with 200 KB -> 400 input_too_long, before any encoding;
  3. a chat whose messages total 200 KB -> 400 input_too_long;
  4. a 3 MB JSON body -> 413 (explicit DefaultBodyLimit);
  5. the server is responsive right after (health 200, a 2-token chat).
"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8124"


def req(method, path, body=None, timeout=600, raw_body=None):
    data = raw_body if raw_body is not None else (json.dumps(body).encode() if body is not None else None)
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
                return resp.status, {"raw": raw.decode(errors="replace")[:200]}
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

# 1. 60 KB unbroken word (under the 64 KiB cap)
t0 = time.time()
s, r = req("POST", "/v1/tokenize", {"text": "a" * (60 * 1024)})
dt = time.time() - t0
print(f"tokenize 60 KB of 'a' -> {s} count={r.get('count') if isinstance(r, dict) else None} in {dt:.2f}s")
assert s == 200 and r["count"] >= 1, (s, r)
assert dt < 20, f"tokenize took {dt:.1f}s"
t0 = time.time()
s, r = req("POST", "/v1/tokenize", {"text": "hello" * (12 * 1024)})
dt = time.time() - t0
print(f"tokenize 60 KB of 'hello' -> {s} count={r.get('count') if isinstance(r, dict) else None} in {dt:.2f}s")
assert s == 200 and 1 <= r["count"] <= 12 * 1024 + 1, (s, r)
assert dt < 20, f"tokenize took {dt:.1f}s"

# 2. over the cap
s, r = req("POST", "/v1/tokenize", {"text": "a" * (200 * 1024)})
print(f"tokenize 200 KB -> {s} {etype(r)}")
assert s == 400 and etype(r) == "input_too_long", (s, r)

# 3. chat over the cap
s, r = req("POST", "/v1/chat/completions", {"max_tokens": 2, "messages": [{"role": "user", "content": "a" * (200 * 1024)}]})
print(f"chat with 200 KB content -> {s} {etype(r)}")
assert s == 400 and etype(r) == "input_too_long", (s, r)

# 4. body limit
big = b'{"text": "' + b"a" * (3 * 1024 * 1024) + b'"}'
s, r = req("POST", "/v1/tokenize", raw_body=big)
print(f"3 MB body -> {s}")
assert s == 413, (s, r)

# 5. alive
s, _ = req("GET", "/health", timeout=10)
assert s == 200
s, r = req("POST", "/v1/chat/completions", {"max_tokens": 2, "messages": [{"role": "user", "content": "Say hi."}]})
assert s == 200, (s, r)
print("E2E LONG INPUT OK")
