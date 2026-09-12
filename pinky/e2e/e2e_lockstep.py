"""End-to-end for review #8 / #20 — entry.tokens and the resident KV caches
stay in lockstep across every mutation path, and retrieve against a
mutated shard is a 200, not an out-of-bounds panic.

`tokens.len()` is observed through the /metrics gauge
cortex_cache_pool_tokens_total (sum over shards, sampled at 1 Hz) and
compared with GET /v1/cache/{id}.seq_len. Expects cortex-server on :8124
started with
  --enable-cache --enable-retrieve --enable-polar-cache --max-seq-len 4096
"""
import json
import re
import sys
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
            try:
                return resp.status, (json.loads(raw) if raw else None)
            except Exception:
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


def pool_tokens_total():
    time.sleep(1.6)  # sampler period is 1 s
    s, r = req("GET", "/metrics")
    m = re.search(r"^cortex_cache_pool_tokens\s+(\d+)", r["raw"], re.M)
    assert m, "gauge not found"
    return int(m.group(1))


def seq_len(cid):
    s, r = req("GET", f"/v1/cache/{cid}")
    assert s == 200, r
    return r["seq_len"]


def chat(shards, max_tokens, temperature=None, content="Say something short."):
    body = {"cache_shards": shards, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}]}
    if temperature is not None:
        body["temperature"] = temperature
    s, r = req("POST", "/v1/chat/completions", body)
    assert s == 200, r
    return r


def retrieve(shards, content="What does the fox do?"):
    s, r = req("POST", "/v1/chat/completions", {
        "mode": "retrieve", "cache_shards": shards, "top_k": 3,
        "messages": [{"role": "user", "content": content}],
    })
    return s, r


def clear():
    for cid in ("s", "p", "a", "b", "r"):
        req("DELETE", f"/v1/cache/{cid}")


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
clear()

# ---- 1. f32 shard: max_tokens exit used to leave tokens one ahead -------
s, r = req("POST", "/v1/cache/load", {"cache_id": "s", "tokens": tokens_of(200)})
assert s in (200, 201), r
r = chat(["s"], 5)
fr = r["choices"][0]["finish_reason"]
sl, tt = seq_len("s"), pool_tokens_total()
print(f"1: f32 chat max_tokens=5 finish={fr}: seq_len={sl} tokens={tt}")
assert sl == tt, "f32 shard desynced after chat"
s, rr = retrieve(["s"])
print("1: retrieve after chat ->", s, etype(rr))
assert s == 200, rr
req("DELETE", "/v1/cache/s")

# ---- 2. polar_chat shard: greedy (polar drives) then non-greedy (f32) ---
s, r = req("POST", "/v1/cache/load", {"cache_id": "p", "tokens": tokens_of(200), "polar_chat": True})
assert s in (200, 201), r
chat(["p"], 5)
sl, tt = seq_len("p"), pool_tokens_total()
print(f"2a: polar greedy chat: seq_len={sl} tokens={tt}")
assert sl == tt, "polar_chat shard desynced after greedy chat"
chat(["p"], 5, temperature=1.0)
sl, tt = seq_len("p"), pool_tokens_total()
print(f"2b: polar_chat non-greedy (f32 fallback) chat: seq_len={sl} tokens={tt}")
assert sl == tt, "polar_chat shard desynced after f32 fallback"
s, rr = retrieve(["p"])
print("2: retrieve after both ->", s, etype(rr))
assert s == 200, rr
s, rr = req("POST", "/v1/cache/append", {"cache_id": "p", "tokens": tokens_of(20)})
print("2: append after both ->", s, etype(rr))
assert s == 200, rr
req("DELETE", "/v1/cache/p")

# ---- 3. multi-shard chat: last shard absorbs the turn, first untouched --
for cid in ("a", "b"):
    s, r = req("POST", "/v1/cache/load", {"cache_id": cid, "tokens": tokens_of(100)})
    assert s in (200, 201), r
a0, b0 = seq_len("a"), seq_len("b")
chat(["a", "b"], 4)
a1, b1, tt = seq_len("a"), seq_len("b"), pool_tokens_total()
print(f"3: multi-shard chat: a {a0}->{a1}, b {b0}->{b1}, tokens_total={tt}")
assert a1 == a0 and b1 > b0 and tt == a1 + b1, "multi-shard write-back desynced"
s, rr = retrieve(["b"])
print("3: retrieve against the mutated last shard ->", s, etype(rr))
assert s == 200, rr
s, rr = retrieve(["a", "b"])
print("3: composed retrieve ->", s, etype(rr))
assert s == 200, rr
req("DELETE", "/v1/cache/a"); req("DELETE", "/v1/cache/b")

# ---- 4. room exit: cache fills mid-decode; the trailing token is dropped -
s, r = req("POST", "/v1/cache/load", {"cache_id": "r", "tokens": tokens_of(4000)})
assert s in (200, 201), r
r = chat(["r"], 300, content="Continue the story.")
fr = r["choices"][0]["finish_reason"]
sl, tt = seq_len("r"), pool_tokens_total()
print(f"4: room-exit chat finish={fr}: seq_len={sl} tokens={tt}")
assert fr == "length" and sl == 4096 and tt == sl, "room-exit left tokens ahead of the cache"
s, rr = retrieve(["r"], content="fox")
print("4: retrieve against the full shard ->", s, etype(rr))
# The shard fills its window exactly; no query fits. Before: engine assert
# "cache overflow: 4096 + 9 > 4096" panicked the worker.
assert s == 400 and etype(rr) == "context_length_exceeded", rr
req("DELETE", "/v1/cache/r")

# ---- 5. the same bound leaves room-fitting queries alone ----------------
s, r = req("POST", "/v1/cache/load", {"cache_id": "r", "tokens": tokens_of(4000)})
assert s in (200, 201), r
s, rr = retrieve(["r"], content="fox")
print("5: retrieve against a 4004-token shard (92 tokens of room) ->", s, etype(rr))
assert s == 200, rr
req("DELETE", "/v1/cache/r")

# ---- 6. review #26: clamp-to-zero is an empty completion, not one token --
# (batch 5 Item 4; run with LOCKSTEP_ITEM4=1 once #26 is in the binary)
import os
if os.environ.get("LOCKSTEP_ITEM4") != "1":
    print("E2E LOCKSTEP OK (scenario 6 skipped: LOCKSTEP_ITEM4 unset)")
    sys.exit(0)
# Measure the templated prompt length on a fresh shard: seq_len after a
# 1-token chat = 4004 + prompt_len + 1.
s, r = req("POST", "/v1/cache/load", {"cache_id": "z", "tokens": tokens_of(4000)})
assert s in (200, 201), r
chat(["z"], 1, content="Say something short.")
prompt_len = seq_len("z") - 4004 - 1
req("DELETE", "/v1/cache/z")
print(f"6: templated prompt is {prompt_len} tokens")
# A shard with room for exactly the prompt: 4096 - 4 sinks - prompt_len.
s, r = req("POST", "/v1/cache/load", {"cache_id": "z", "tokens": tokens_of(4096 - 4 - prompt_len)})
assert s in (200, 201), r
r = chat(["z"], 50, content="Say something short.")
content = r["choices"][0]["message"]["content"]
fr = r["choices"][0]["finish_reason"]
sl, tt = seq_len("z"), pool_tokens_total()
print(f"6: exact-fit chat -> finish={fr} content={content!r} seq_len={sl} tokens={tt}")
assert fr == "length" and content == "" and sl == 4096 and tt == sl, "clamp-to-zero must be an empty completion in lockstep"
s, rr = req("POST", "/v1/chat/completions", {"cache_shards": ["z"], "max_tokens": 0, "messages": [{"role": "user", "content": "x"}]})
print("6: max_tokens 0 ->", s, etype(rr))
assert s == 400 and etype(rr) == "invalid_request", rr
req("DELETE", "/v1/cache/z")

print("E2E LOCKSTEP OK")
