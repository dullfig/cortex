"""End-to-end for review #7 — concurrent GPU regions.

Fires a burst of 8 GPU-heavy requests within ~100 ms: 4 stateless chats
with ~1500-token prompts, 2 cache/loads of 1500 tokens, 2 shims/embed of
~1000 tokens. Without admission control the transient lanes are shared
between overlapping prefills and the second BlockScratch::allocate panics
(worker dies, connection closed). With the gate every request completes
with a clean status and /metrics shows cortex_gpu_gate_waiting > 0 during
the burst. Expects cortex-server on :8124 started with
  --enable-cache --enable-shims --max-seq-len 8192
Exit code 0 = all clean; 2 = burst produced connection drops / 5xx.
"""
import json
import re
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
    except Exception as e:  # connection dropped = worker panicked
        return -1, {"raw": repr(e)}


def tokens_of(n):
    s, t = req("POST", "/v1/tokenize", {"text": "The quick brown fox jumps over the lazy dog. "})
    unit = t["tokens"] if isinstance(t, dict) and "tokens" in t else t
    return (unit * (n // len(unit) + 1))[:n]


class Call(threading.Thread):
    def __init__(self, label, *args):
        super().__init__(daemon=True)
        self.label, self.args, self.result = label, args, None

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

prompt = "word " * 1500
embed_text = "word " * 1000
load_tokens = tokens_of(1500)

calls = []
for i in range(4):
    calls.append(Call(f"chat{i}", "POST", "/v1/chat/completions",
                      {"max_tokens": 4, "messages": [{"role": "user", "content": prompt}]}))
for i in range(2):
    calls.append(Call(f"load{i}", "POST", "/v1/cache/load",
                      {"cache_id": f"c{i}", "tokens": load_tokens}))
for i in range(2):
    calls.append(Call(f"embed{i}", "POST", "/v1/shims/embed",
                      {"text": embed_text, "layer": "final", "pooling": "last_token"}))

# 9th thread: sample the gate-depth gauge while the burst runs.
peak = {"waiting": 0}
stop = threading.Event()


def sampler():
    while not stop.is_set():
        s, r = req("GET", "/metrics", timeout=10)
        if s == 200 and isinstance(r, dict) and "raw" in r:
            m = re.search(r"^cortex_gpu_gate_waiting\s+(\d+)", r["raw"], re.M)
            if m:
                peak["waiting"] = max(peak["waiting"], int(m.group(1)))
        time.sleep(0.05)


smp = threading.Thread(target=sampler, daemon=True)
smp.start()
t0 = time.time()
for c in calls:
    c.start()
    time.sleep(0.012)
for c in calls:
    c.join()
stop.set()
smp.join(timeout=2)

bad = 0
for c in calls:
    s, r = c.result
    ok = s in (200, 201)
    bad += 0 if ok else 1
    print(f"  {c.label:7s} -> {s} {'' if ok else str(r)[:120]}")
print(f"burst wall {time.time() - t0:.1f}s; peak cortex_gpu_gate_waiting = {peak['waiting']}")
for i in range(2):
    req("DELETE", f"/v1/cache/c{i}")

if bad:
    print(f"E2E CONCURRENCY: {bad} of {len(calls)} requests failed")
    sys.exit(2)
print("E2E CONCURRENCY OK")
