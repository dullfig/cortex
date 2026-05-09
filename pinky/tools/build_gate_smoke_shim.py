"""Build a tiny ONNX gate shim used to validate cortex's #6a gate
phase end-to-end. The shim takes pooled hidden state (shape [1,
EMBED_DIM]) and emits a single scalar: the squared L2 norm of the
input. Always positive and substantial — lets the smoke test write
deterministic shim_rules around extreme thresholds (e.g. gt: 0 = always,
gt: 1e30 = never) without depending on the actual hidden values.

Usage:
    python build_gate_smoke_shim.py [embed_dim] [out_path]

Defaults to embed_dim=2048 (Qwen 2.5-3B) and pinky/tools/gate_smoke_shim.onnx.
"""
import sys
from pathlib import Path

import onnx
from onnx import helper, TensorProto


def build(embed_dim: int) -> onnx.ModelProto:
    input_t = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, embed_dim])
    output_t = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    # opset 13: ReduceSum takes `axes` as a second input tensor (not attr)
    axes_init = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
    sq = helper.make_node("Mul", inputs=["x", "x"], outputs=["x_sq"])
    norm = helper.make_node(
        "ReduceSum", inputs=["x_sq", "axes"], outputs=["y"],
        keepdims=0,
    )
    graph = helper.make_graph(
        [sq, norm], "gate_smoke",
        inputs=[input_t], outputs=[output_t],
        initializer=[axes_init],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=8,
    )
    onnx.checker.check_model(model)
    return model


def main():
    embed_dim = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        Path(__file__).parent / "gate_smoke_shim.onnx"
    )
    model = build(embed_dim)
    onnx.save(model, str(out_path))
    print(f"wrote {out_path} (embed_dim={embed_dim}, {out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
