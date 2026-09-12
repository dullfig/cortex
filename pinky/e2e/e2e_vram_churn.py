"""Stress driver for review #23/#31 — VRAM churn under the shard cap.

Loops the two flows that produced the intermittent
  "Buffer with 'forward_advance_only.hidden' label is invalid"
worker panic: (a) three shards resident at cap 3, a same-id reload at the
cap, a composed retrieve and a single-shard chat; (b) append + same-id
reload + DELETE. A stateless 1500-token chat runs in a parallel thread
throughout so the lanes are busy. Expects cortex-server on :8124 with
  --enable-cache --enable-retrieve --max-seq-len 4096 --max-cache-shards 3
Exit 0 = every response was a clean status (2xx/4xx/503/507) and the
server stayed up; exit 2 = at least one dropped connection.
Usage: python e2e_vram_churn.py [loops]
"""
import json
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8124"
LOOPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20


def req(method, path, body=None, timeout=1200):
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
    except Exception as e:  # connection dropped = worker panicked
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
print(f"server ready; {LOOPS} loops")

t1500 = tokens_of(1500)
t300 = tokens_of(300)
stop = threading.Event()
bg = {"n": 0, "bad": 0}


def background_chats():
    while not stop.is_set():
        s, r = req("POST", "/v1/chat/completions",
                   {"max_tokens": 4, "messages": [{"role": "user", "content": "word " * 1500}]})
        bg["n"] += 1
        if s not in (200, 400, 503):
            bg["bad"] += 1
            print(f"  bg chat -> {s} {str(r)[:100]}")


bgt = threading.Thread(target=background_chats, daemon=True)
bgt.start()

dropped = 0
statuses = {}


def note(label, s, r):
    global dropped
    key = f"{s}/{etype(r) or 'ok'}"
    statuses[key] = statuses.get(key, 0) + 1
    if s == -1:
        dropped += 1
        print(f"  {label} -> DROPPED {str(r)[:120]}")
    elif s >= 500 and s not in (503, 507):
        dropped += 1
        print(f"  {label} -> {s} {str(r)[:120]}")


t0 = time.time()
for i in range(LOOPS):
    for cid in ("a", "b", "c"):
        note(f"load {cid}", *req("POST", "/v1/cache/load", {"cache_id": cid, "tokens": t1500}))
    # same-id reload at the cap (transient 2x for shard a)
    note("reload a", *req("POST", "/v1/cache/load", {"cache_id": "a", "tokens": t1500}))
    note("retrieve a+b", *req("POST", "/v1/chat/completions", {
        "mode": "retrieve", "cache_shards": ["a", "b"], "top_k": 3,
        "messages": [{"role": "user", "content": "fox"}]}))
    note("chat c", *req("POST", "/v1/chat/completions", {
        "cache_shards": ["c"], "max_tokens": 4,
        "messages": [{"role": "user", "content": "Say hi."}]}))
    note("append b", *req("POST", "/v1/cache/append", {"cache_id": "b", "tokens": t300}))
    note("reload b", *req("POST", "/v1/cache/load", {"cache_id": "b", "tokens": t1500}))
    for cid in ("a", "b", "c"):
        note(f"delete {cid}", *req("DELETE", f"/v1/cache/{cid}"))
    s, _ = req("GET", "/health", timeout=10)
    if s != 200:
        print(f"  loop {i}: health -> {s}")
        dropped += 1
        break
    if (i + 1) % 5 == 0:
        print(f"  loop {i + 1}/{LOOPS} ok so far; dropped={dropped}; bg chats={bg['n']} bad={bg['bad']}")

stop.set()
bgt.join(timeout=60)
for cid in ("a", "b", "c"):
    req("DELETE", f"/v1/cache/{cid}")
print(f"wall {time.time() - t0:.0f}s; statuses={statuses}; bg chats={bg['n']} bad={bg['bad']}")
if dropped or bg["bad"]:
    print(f"E2E VRAM CHURN: {dropped} dropped/5xx, {bg['bad']} bad background chats")
    sys.exit(2)
print("E2E VRAM CHURN OK")
