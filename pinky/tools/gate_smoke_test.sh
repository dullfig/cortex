#!/usr/bin/env bash
# End-to-end smoke for the #6a gate phase. Assumes:
#   - cortex-server is running on localhost:$PORT with --enable-shims
#   - gate_smoke_shim.onnx is at $SHIM_PATH (built via build_gate_smoke_shim.py)
#
# Validates three things:
#   1. shim PUT registers the gate shim with phase=gate
#   2. shim_rules silent path: rule never matches → finish_reason=silent,
#      no content, signals=["test_silent"]
#   3. shim_rules proceed path: rule always matches → finish_reason=stop|length,
#      content non-empty, metadata.active_steers=["follow_instructions"]

set -euo pipefail

PORT=${PORT:-8181}
SHIM_PATH=${SHIM_PATH:-$(dirname "$0")/gate_smoke_shim.onnx}

if [ ! -f "$SHIM_PATH" ]; then
    echo "missing shim: $SHIM_PATH (run build_gate_smoke_shim.py first)" >&2
    exit 1
fi

ONNX_B64=$(base64 -w0 < "$SHIM_PATH")

echo "=== 1. Register gate_smoke shim ==="
curl -sS -X PUT "http://localhost:$PORT/v1/shims/gate_smoke" \
    -H "Content-Type: application/json" \
    -d "{
        \"manifest\": {
            \"id\": \"gate_smoke\",
            \"version\": \"0.1.0\",
            \"phase\": \"gate\",
            \"attachment\": {\"layer\": \"final\", \"pooling\": \"last_token\"},
            \"input_shape\": {\"hidden_dim\": 2048},
            \"output_shape\": {\"kind\": \"scalar\"},
            \"description\": \"Squared L2 norm gate for #6a smoke\"
        },
        \"onnx_base64\": \"$ONNX_B64\"
    }" | python -m json.tool
echo

echo "=== 2. Silent path: gt=1e30 should never match → silent ==="
curl -sS -X POST "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen-2.5-3b",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 16,
        "temperature": 0.0,
        "gate_shims": ["gate_smoke"],
        "shim_rules": [
            {"if": {"gate": "gate_smoke", "gt": 1e30}, "then": {"activate": []}},
            {"else": {"silent": true, "signal": "test_silent"}}
        ]
    }' | python -m json.tool
echo

echo "=== 3. Proceed path: gt=0 always matches → activate steers, generate ==="
curl -sS -X POST "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen-2.5-3b",
        "messages": [{"role": "user", "content": "Say hi briefly."}],
        "max_tokens": 16,
        "temperature": 0.0,
        "gate_shims": ["gate_smoke"],
        "shim_rules": [
            {"if": {"gate": "gate_smoke", "gt": 0.0},
             "then": {"activate": ["follow_instructions"], "signal": "test_proceed"}},
            {"else": {"silent": true}}
        ]
    }' | python -m json.tool
echo

echo "=== 4. Streaming silent path ==="
curl -sN -X POST "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen-2.5-3b",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": true,
        "gate_shims": ["gate_smoke"],
        "shim_rules": [
            {"else": {"silent": true, "signal": "test_silent_stream"}}
        ]
    }'
echo
echo

echo "=== 5. Streaming proceed path ==="
curl -sN -X POST "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen-2.5-3b",
        "messages": [{"role": "user", "content": "Say hi briefly."}],
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": true,
        "gate_shims": ["gate_smoke"],
        "shim_rules": [
            {"if": {"gate": "gate_smoke", "gt": 0.0},
             "then": {"activate": ["voice_bob"], "signal": "test_proceed_stream"}}
        ]
    }'
echo
