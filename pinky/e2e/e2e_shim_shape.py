"""End-to-end for review #25 (shim ONNX graphs validated at registration)
and #26a (max_tokens: 0 is a 400).

Expects cortex-server on :8124 booted with --enable-shims --max-seq-len 4096
and the `onnx` python package on this box. The graphs carry no weights
(Concat / Slice / Mul-by-zero / ReduceMean), so every PUT is a few KB —
a real 2048-wide linear steer is ~16 MB of ONNX and must be shipped some
other way than a 2 MiB JSON body (unchanged by this batch).

  1. a steer shim whose ONNX declares [?, 512] while the model is 2048-wide
     -> 400 shape_mismatch at PUT time (before #25: registered, then a
     panic in the decode loop after a full prefill);
  2. a steer shim with a wrong output width -> 400 shape_mismatch;
  3. a correct [?, 2048] -> [?, 2048] steer shim -> 201, and a chat that
     uses it -> 200;
  4. a gate shim declared scalar whose graph emits [?, 3] -> 400; the
     scalar and category:3 shapes register;
  5. max_tokens: 0 -> 400 invalid_request.
"""
import base64
import json
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

BASE = "http://127.0.0.1:8124"


def req(method, path, body=None, timeout=600):
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


def etype(r):
    try:
        return r["error"]["type"]
    except Exception:
        return None


def graph_onnx(kind, in_dim, out_dim):
    """A weight-free graph from [batch, in_dim] to [batch, out_dim]."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", in_dim])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", out_dim])
    inits = []
    if kind == "zero":            # y = x * 0  (a no-op steer)
        assert in_dim == out_dim
        inits.append(numpy_helper.from_array(np.zeros((1,), dtype=np.float32), name="s"))
        nodes = [helper.make_node("Mul", ["x", "s"], ["y"])]
    elif kind == "concat":        # y = [x, x, ...]  (widen)
        assert out_dim % in_dim == 0
        nodes = [helper.make_node("Concat", ["x"] * (out_dim // in_dim), ["y"], axis=1)]
    elif kind == "slice":         # y = x[:, :out_dim]  (narrow)
        inits += [
            numpy_helper.from_array(np.array([0], dtype=np.int64), name="st"),
            numpy_helper.from_array(np.array([out_dim], dtype=np.int64), name="en"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), name="ax"),
        ]
        nodes = [helper.make_node("Slice", ["x", "st", "en", "ax"], ["y"])]
    elif kind == "mean":          # y = mean(x, axis 1) -> [batch, 1]
        assert out_dim == 1
        nodes = [helper.make_node("ReduceMean", ["x"], ["y"], axes=[1], keepdims=1)]
    else:
        raise ValueError(kind)
    graph = helper.make_graph(nodes, "shim", [x], [y], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    b = model.SerializeToString()
    assert len(b) < 64 * 1024, len(b)
    return base64.b64encode(b).decode()


def manifest(id_, phase, kind, hidden_dim=2048, layer="final", pooling="last_token"):
    return {
        "id": id_, "version": "1", "phase": phase,
        "attachment": {"layer": layer, "pooling": pooling},
        "input_shape": {"hidden_dim": hidden_dim},
        "output_shape": {"kind": kind},
        "description": "e2e #25",
    }


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

# 1. wrong input width
s, r = req("PUT", "/v1/shims/bad_in", {"manifest": manifest("bad_in", "steer", "hidden_delta"), "onnx_base64": graph_onnx("concat", 512, 2048)})
print(f"steer with [?,512] input -> {s} {etype(r)} {str(r.get('error', {}).get('message', ''))[:100] if isinstance(r, dict) else ''}")
assert s == 400 and etype(r) == "shape_mismatch", (s, r)
s, _ = req("GET", "/v1/shims/bad_in")
assert s == 404, "a refused shim must not be registered"

# 2. wrong output width
s, r = req("PUT", "/v1/shims/bad_out", {"manifest": manifest("bad_out", "steer", "hidden_delta"), "onnx_base64": graph_onnx("slice", 2048, 512)})
print(f"steer with [?,512] output -> {s} {etype(r)}")
assert s == 400 and etype(r) == "shape_mismatch", (s, r)

# 3. correct
s, r = req("PUT", "/v1/shims/ok_steer", {"manifest": manifest("ok_steer", "steer", "hidden_delta"), "onnx_base64": graph_onnx("zero", 2048, 2048)})
print(f"steer with [?,2048] -> [?,2048] -> {s}")
assert s in (200, 201), (s, r)
s, r = req("POST", "/v1/chat/completions", {"max_tokens": 3, "steer_shims": ["ok_steer"], "messages": [{"role": "user", "content": "Say hi."}]})
print(f"chat with the steer -> {s} {etype(r)}")
assert s == 200, (s, r)

# 4. gate declared scalar, graph emits [?, 3]
s, r = req("PUT", "/v1/shims/bad_gate", {"manifest": manifest("bad_gate", "gate", "scalar"), "onnx_base64": graph_onnx("slice", 2048, 3)})
print(f"gate scalar with [?,3] output -> {s} {etype(r)}")
assert s == 400 and etype(r) == "shape_mismatch", (s, r)
s, r = req("PUT", "/v1/shims/ok_gate", {"manifest": manifest("ok_gate", "gate", "scalar"), "onnx_base64": graph_onnx("mean", 2048, 1)})
print(f"gate scalar with [?,1] output -> {s}")
assert s in (200, 201), (s, r)
s, r = req("PUT", "/v1/shims/ok_cat", {"manifest": manifest("ok_cat", "gate", "category:3"), "onnx_base64": graph_onnx("slice", 2048, 3)})
print(f"gate category:3 with [?,3] output -> {s}")
assert s in (200, 201), (s, r)

# 5. max_tokens 0
s, r = req("POST", "/v1/chat/completions", {"max_tokens": 0, "messages": [{"role": "user", "content": "Say hi."}]})
print(f"max_tokens 0 -> {s} {etype(r)}")
assert s == 400 and etype(r) == "invalid_request", (s, r)

for sid in ("ok_steer", "ok_gate", "ok_cat"):
    req("DELETE", f"/v1/shims/{sid}")
print("E2E SHIM SHAPE OK")
