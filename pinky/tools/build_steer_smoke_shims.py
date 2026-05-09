"""Build two ONNX steer shims for #6b end-to-end validation:

  identity_steer.onnx — output = zeros([1, embed_dim])
    Adding zeros to hidden leaves it unchanged. With this steer active,
    generated tokens MUST match the no-steer baseline byte-for-byte.
    Proves the plumbing without testing modification semantics.

  noise_steer.onnx — output[0]=0.5, output[1..]=0
    A tiny perturbation on dim 0. Logits will shift slightly. Used to
    confirm that an active steer actually changes the output token
    distribution (different generated text vs baseline).

Usage:
    python build_steer_smoke_shims.py [embed_dim]

Defaults to embed_dim=2048 (Qwen 2.5-3B).
"""
import sys
from pathlib import Path

import onnx
from onnx import helper, TensorProto


def build_identity(embed_dim: int) -> onnx.ModelProto:
    """Output = input * 0 (broadcast). Always returns zeros of input shape."""
    input_t = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, embed_dim])
    output_t = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, embed_dim])
    zero_init = helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0])
    node = helper.make_node("Mul", inputs=["x", "zero"], outputs=["y"])
    graph = helper.make_graph(
        [node], "identity_steer",
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
    """Output = constant [0.5, 0, 0, ..., 0] regardless of input.

    Computed as Mul(x, 0) + delta. The Mul keeps `x` referenced by the
    graph (some ORT versions are picky about unused inputs).
    """
    input_t = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, embed_dim])
    output_t = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, embed_dim])
    zero_init = helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0])
    # Substantial perturbation across all dims. Per-dim magnitude in
    # Qwen 2.5-3B's final hidden state is ~2.6, so +5 across all dims
    # is a clear signal — argmax should flip on at least some tokens.
    delta_vec = [5.0] * embed_dim
    delta_init = helper.make_tensor(
        "delta", TensorProto.FLOAT, [1, embed_dim], delta_vec,
    )
    mul = helper.make_node("Mul", inputs=["x", "zero"], outputs=["x_zero"])
    add = helper.make_node("Add", inputs=["x_zero", "delta"], outputs=["y"])
    graph = helper.make_graph(
        [mul, add], "noise_steer",
        inputs=[input_t], outputs=[output_t],
        initializer=[zero_init, delta_init],
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
        ("identity_steer", build_identity),
        ("noise_steer", build_noise),
    ]:
        p = out_dir / f"{name}.onnx"
        onnx.save(build_fn(embed_dim), str(p))
        print(f"wrote {p} (embed_dim={embed_dim}, {p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
