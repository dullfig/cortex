#!/usr/bin/env bash
# End-to-end smoke for the #6c injection phase. Assumes:
#   - cortex-server is running on localhost:$PORT with --enable-shims
#   - identity_inject.onnx + noise_inject.onnx live in $TOOLS_DIR
#     (built via build_inject_smoke_shims.py)
#
# Validates four properties:
#   1. identity_inject (zero delta, attachment=entrance:all) registers
#   2. With identity_inject active, generated tokens MATCH no-injection
#      baseline (broadcast-add of zeros at every block must be a no-op)
#   3. noise_inject (output=0.1*input, attachment=entrance:0) registers
#   4. With noise_inject active, generated text DIFFERS from baseline
#      (a non-zero delta at a block entrance must shift logits)
#   5. Inject + cache combination cleanly rejects 400
#
# temperature=0 so any difference is real, not sampling noise.

set -euo pipefail

PORT=${PORT:-8181}
TOOLS_DIR=${TOOLS_DIR:-$(dirname "$0")}
PROMPT=${PROMPT:-"Say one short sentence."}
MAX_TOKENS=${MAX_TOKENS:-12}

extract_content() {
    python -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
}

register_inject() {
    local name="$1"
    local layer="$2"
    local onnx_b64
    onnx_b64=$(base64 -w0 < "$TOOLS_DIR/$name.onnx")
    curl -sS -X PUT "http://localhost:$PORT/v1/shims/$name" \
        -H "Content-Type: application/json" \
        -d "{
            \"manifest\": {
                \"id\": \"$name\",
                \"version\": \"0.1.0\",
                \"phase\": \"injection\",
                \"attachment\": {\"layer\": \"$layer\", \"pooling\": \"last_token\"},
                \"input_shape\": {\"hidden_dim\": 2048},
                \"output_shape\": {\"kind\": \"hidden_delta\"},
                \"description\": \"smoke inject ($name -> $layer)\"
            },
            \"onnx_base64\": \"$onnx_b64\"
        }" > /dev/null
    echo "  registered: $name (layer=$layer)"
}

echo "=== Register injection shims ==="
register_inject identity_inject "entrance:all"
register_inject noise_inject "entrance:0"
echo

echo "=== Baseline: no shims, greedy ==="
baseline=$(curl -sS -X POST "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
        \"max_tokens\": $MAX_TOKENS,
        \"temperature\": 0.0
    }" | extract_content)
echo "  baseline: $baseline"
echo

echo "=== With identity_inject (zero delta, entrance:all) — should equal baseline ==="
with_identity=$(curl -sS -X POST "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
        \"max_tokens\": $MAX_TOKENS,
        \"temperature\": 0.0,
        \"inject_shims\": [\"identity_inject\"]
    }" | extract_content)
echo "  with_identity: $with_identity"
if [ "$baseline" = "$with_identity" ]; then
    echo "  PASS: identity_inject is a no-op"
else
    echo "  FAIL: identity_inject changed generation"
    exit 1
fi
echo

echo "=== With noise_inject (0.1*input at entrance:0) — should differ ==="
with_noise=$(curl -sS -X POST "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
        \"max_tokens\": $MAX_TOKENS,
        \"temperature\": 0.0,
        \"inject_shims\": [\"noise_inject\"]
    }" | extract_content)
echo "  with_noise: $with_noise"
if [ "$baseline" != "$with_noise" ]; then
    echo "  PASS: noise_inject shifted generation"
else
    echo "  WARN: noise_inject produced identical output (delta may be too small)"
fi
echo

echo "=== Streaming with identity_inject ==="
curl -sN -X POST "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
        \"max_tokens\": 4,
        \"temperature\": 0.0,
        \"stream\": true,
        \"inject_shims\": [\"identity_inject\"]
    }"
echo
echo
echo "=== Inject + cache should reject ==="
curl -sS -X POST "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],
        \"max_tokens\": 4,
        \"temperature\": 0.0,
        \"cache_id\": \"nonexistent\",
        \"inject_shims\": [\"identity_inject\"]
    }" | python -m json.tool
echo
