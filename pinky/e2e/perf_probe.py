"""Prefill / decode timing probe for the release-profile change (batch 4,
Item 5: overflow-checks in release). Same prompt, same server flags, run
before and after; report medians. Expects cortex-server on :8124.
"""
import json
import statistics
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8124"
LABEL = sys.argv[1] if len(sys.argv) > 1 else "run"


def req(path, body):
    r = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(r, timeout=600) as resp:
        return json.loads(resp.read())


for _ in range(150):
    try:
        urllib.request.urlopen(BASE + "/health", timeout=5)
        break
    except Exception:
        time.sleep(2)

prompt_500 = ("The quick brown fox jumps over the lazy dog near the riverbank while the sun sets. " * 42).strip()
n_prompt = req("/v1/tokenize", {"text": prompt_500})
n_prompt = len(n_prompt["tokens"] if isinstance(n_prompt, dict) else n_prompt)

# warm-up
req("/v1/chat/completions", {"max_tokens": 1, "messages": [{"role": "user", "content": prompt_500}]})

prefill = []
for _ in range(5):
    t0 = time.perf_counter()
    req("/v1/chat/completions", {"max_tokens": 1, "messages": [{"role": "user", "content": prompt_500}]})
    prefill.append(time.perf_counter() - t0)

decode = []
for _ in range(3):
    t0 = time.perf_counter()
    r = req("/v1/chat/completions", {"max_tokens": 64, "temperature": 0,
                                     "messages": [{"role": "user", "content": "Write a short paragraph about rivers."}]})
    dt = time.perf_counter() - t0
    n = r["usage"]["completion_tokens"] if "usage" in r else 64
    decode.append(n / dt)

print(f"{LABEL}: prompt={n_prompt} tokens; prefill median {statistics.median(prefill)*1000:.0f} ms "
      f"(min {min(prefill)*1000:.0f}); decode median {statistics.median(decode):.1f} tok/s (max {max(decode):.1f})")
