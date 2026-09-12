"""End-to-end for review #24 (client disconnect cancels what can be cancelled).

Expects cortex-server on :8124 booted with --enable-cache --max-seq-len 4096.

A. stateless, non-streaming: a 400-token generation whose socket is closed
   after 1.5 s must stop within one decode step; a 2-token request sent
   right after completes well before the abandoned one would have
   (the GPU gate was released). The server log shows
   "generation cancelled: client disconnected".
B. stateless, streaming: same, via the closed SSE body
   ("stream cancelled: client disconnected").
C. cached (single shard): NOT cancellable today (block_in_place holds the
   pool entry — filed as #32); the follow-up request waits for the bounded
   generation and succeeds. Documented, not asserted on time.
Exit 0 = all assertions hold.
"""
import json
import socket
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8124"
HOST, PORT = "127.0.0.1", 8124


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
                return resp.status, {"raw": raw.decode(errors="replace")[:200]}
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"raw": raw.decode(errors="replace")[:200]}
    except Exception as e:
        return -1, {"raw": repr(e)}


def fire_and_drop(body, hold_s):
    """POST and close the socket after hold_s seconds without reading."""
    payload = json.dumps(body).encode()
    head = (
        f"POST /v1/chat/completions HTTP/1.1\r\nHost: {HOST}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n\r\n"
    ).encode()
    s = socket.create_connection((HOST, PORT))
    s.sendall(head + payload)
    time.sleep(hold_s)
    s.close()


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

long_msgs = [{"role": "user", "content": "Write a very long story about a fox, a river and a bridge."}]
short = {"max_tokens": 2, "messages": [{"role": "user", "content": "Say hi."}]}

# Calibrate: how long does a 400-token generation take here?
t0 = time.time()
s, r = req("POST", "/v1/chat/completions", {"max_tokens": 400, "messages": long_msgs})
full = time.time() - t0
assert s == 200, (s, r)
print(f"calibration: 400-token generation = {full:.1f}s")
budget = max(6.0, full * 0.6)

# A. stateless non-streaming
fire_and_drop({"max_tokens": 400, "messages": long_msgs}, 1.5)
t0 = time.time()
s, r = req("POST", "/v1/chat/completions", short)
dt = time.time() - t0
print(f"A stateless: follow-up after dropping a 400-token request -> {s} in {dt:.1f}s (budget {budget:.1f}s)")
assert s == 200, (s, r)
assert dt < budget, f"not cancelled: follow-up waited {dt:.1f}s"

# B. streaming
fire_and_drop({"max_tokens": 400, "stream": True, "messages": long_msgs}, 1.5)
t0 = time.time()
s, r = req("POST", "/v1/chat/completions", short)
dt = time.time() - t0
print(f"B streaming: follow-up after dropping a 400-token stream -> {s} in {dt:.1f}s (budget {budget:.1f}s)")
assert s == 200, (s, r)
assert dt < budget, f"stream not cancelled: follow-up waited {dt:.1f}s"

# C. cached: bounded, not cancelled (review #32)
s, r = req("POST", "/v1/cache/load", {"cache_id": "d", "tokens": tokens_of(200)})
assert s in (200, 201), r
fire_and_drop({"max_tokens": 120, "cache_shards": ["d"], "messages": long_msgs}, 1.0)
t0 = time.time()
s, r = req("POST", "/v1/chat/completions", short)
dt = time.time() - t0
print(f"C cached: follow-up after dropping a 120-token cached chat -> {s} in {dt:.1f}s (waits for the bounded generation; #32)")
assert s == 200, (s, r)
s, r = req("POST", "/v1/chat/completions", {"cache_shards": ["d"], "max_tokens": 2, "messages": short["messages"]})
print(f"C cached: the shard is consistent afterwards -> {s}")
assert s == 200, (s, r)
req("DELETE", "/v1/cache/d")

print("E2E DISCONNECT OK")
