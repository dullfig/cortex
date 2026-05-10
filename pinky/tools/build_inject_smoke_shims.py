"""Build two ONNX injection shims for #6c end-to-end validation.

  identity_inject.onnx — output = zeros([1, embed_dim])
    Adding zero deltas at any block entrance is a no-op. Generated
    tokens MUST match the no-injection baseline. Proves the inject
    plumbing is wired correctly without testing modification.

  noise_inject.onnx — output = +0.1 * input
    A small fraction of the input projected back as a delta. The
    delta scales with the model's natural activation magnitude, so
    each forward step's hidden gets a 10% perturbation at the chosen
    block entrance. Should shift generation noticeably without
    sending the model into a degenerate loop.

Usage:
    python build_inject_smoke_shims.py [embed_dim]

Defaults to embed_dim=2048 (Qwen 2.5-3B).
"""
import sys
from pathlib import Path

import onnx
from onnx import helper, TensorProto


def build_identity(embed_dim: int) -> onnx.ModelProto:
    """Output = input * 0. Always zeros of input shape."""
    input_t = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, embed_dim])
    output_t = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, embed_dim])
    zero_init = helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0])
    node = helper.make_node("Mul", inputs=["x", "zero"], outputs=["y"])
    graph = helper.make_graph(
        [node], "identity_inject",
        inputs=[input_t], outputs=[output_t],
        initializer=[zero_init],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=8,
    )
    onnx.checker.check_model(model)
    return model


def build_noise(embed_dim: int) -> onnx.ModelProto:
    """Output = input * 0.1. Scales naturally with prompt activations."""
    input_t = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, embed_dim])
    output_t = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, embed_dim])
    scale_init = helper.make_tensor("scale", TensorProto.FLOAT, [], [0.1])
    node = helper.make_node("Mul", inputs=["x", "scale"], outputs=["y"])
    graph = helper.make_graph(
        [node], "noise_inject",
        inputs=[input_t], outputs=[output_t],
        initializer=[scale_init],
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
    out_dir = Path(__file__).parent
    for name, build_fn in [
        ("identity_inject", build_identity),
        ("noise_inject", build_noise),
    ]:
        p = out_dir / f"{name}.onnx"
        onnx.save(build_fn(embed_dim), str(p))
        print(f"wrote {p} (embed_dim={embed_dim}, {p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
