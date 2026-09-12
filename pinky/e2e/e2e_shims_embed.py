"""End-to-end for review #27 — /v1/shims/embed returned garbage.

Before the fix the final post-norm hidden was f32-misread packed f16 with
the tail zeroed. After it the vectors must be (a) fully non-zero, (b)
semantically ordered: two paraphrases closer than an unrelated sentence.
Expects cortex-server on :8124 started with --enable-shims.
"""
import json
import math
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
        with urllib.request.urlopen(r, timeout=600) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"raw": raw.decode(errors="replace")[:300]}


def embed(text, layer="final", pooling="last_token"):
    s, r = req("POST", "/v1/shims/embed", {"text": text, "layer": layer, "pooling": pooling})
    assert s == 200, (s, r)
    v = r.get("embedding")
    assert isinstance(v, list) and v, r
    return v


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb)


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

a = embed("The cat sat on the mat.")
b = embed("A cat was sitting on a mat.")
c = embed("Quarterly revenue guidance was revised downward by the CFO.")

dim = len(a)
zeros_a = sum(1 for x in a if x == 0.0)
tail_zero = all(x == 0.0 for x in a[dim // 2:])
print(f"dim={dim} zeros={zeros_a} tail_half_all_zero={tail_zero}")
assert dim > 0 and not tail_zero, "second half of the vector is zero: the #27 symptom"
assert zeros_a < dim // 10, f"{zeros_a} exact zeros of {dim}"
assert all(math.isfinite(x) for x in a + b + c)

cab, cac, cbc = cos(a, b), cos(a, c), cos(b, c)
print(f"cos(paraphrase)={cab:.4f} cos(a,unrelated)={cac:.4f} cos(b,unrelated)={cbc:.4f}")
assert cab > cac and cab > cbc, "paraphrases should be closer than the unrelated sentence"

# Per-layer capture path (shims/infer uses it) — mean pooling on an early layer.
m = embed("The cat sat on the mat.", layer="entrance:2", pooling="mean")
assert len(m) == dim and all(math.isfinite(x) for x in m) and any(x != 0.0 for x in m[dim // 2:])
print("layer-2 mean-pooled capture non-degenerate")

print("E2E SHIMS EMBED OK")
