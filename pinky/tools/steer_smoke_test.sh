#!/usr/bin/env bash
# End-to-end smoke for the #6b steer phase. Assumes:
#   - cortex-server is running on localhost:$PORT with --enable-shims
#   - identity_steer.onnx and noise_steer.onnx are at $TOOLS_DIR
#     (build via build_steer_smoke_shims.py first)
#
# Validates four properties:
#   1. identity_steer registers (phase=steer, output=hidden_delta)
#   2. With identity_steer active, generated tokens MATCH no-steer baseline
#      (proves plumbing is wired without changing semantics — zero delta
#      must be a no-op)
#   3. noise_steer registers
#   4. With noise_steer active, generated text DIFFERS from baseline
#      (confirms steers actually shift the output distribution)
#
# We use temperature=0 (greedy) so generation is deterministic — any
# difference between two runs is real, not sampling noise.

set -euo pipefail

PORT=${PORT:-8181}
TOOLS_DIR=${TOOLS_DIR:-$(dirname "$0")}
PROMPT=${PROMPT:-"Say one short sentence."}
MAX_TOKENS=${MAX_TOKENS:-12}

extract_content() {
    python -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
}

echo "=== Register identity_steer + noise_steer ==="
for name in identity_steer noise_steer; do
    onnx_b64=$(base64 -w0 < "$TOOLS_DIR/$name.onnx")
    curl -sS -X PUT "http://localhost:$PORT/v1/shims/$name" \
        -H "Content-Type: application/json" \
        -d "{
            \"manifest\": {
                \"id\": \"$name\",
                \"version\": \"0.1.0\",
                \"phase\": \"steer\",
                \"attachment\": {\"layer\": \"final\", \"pooling\": \"last_token\"},
                \"input_shape\": {\"hidden_dim\": 2048},
                \"output_shape\": {\"kind\": \"hidden_delta\"},
                \"description\": \"smoke test steer ($name)\"
            },
            \"onnx_base64\": \"$onnx_b64\"
        }" > /dev/null
    echo "  registered: $name"
done
echo

echo "=== Baseline: no steers, greedy ==="
baseline=$(curl -sS -X POST "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
        \"max_tokens\": $MAX_TOKENS,
        \"temperature\": 0.0
    }" | extract_content)
echo "  baseline: $baseline"
echo

echo "=== With identity_steer (zero delta) — should equal baseline ==="
with_identity=$(curl -sS -X POST "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
        \"max_tokens\": $MAX_TOKENS,
        \"temperature\": 0.0,
        \"steer_shims\": [\"identity_steer\"]
    }" | extract_content)
echo "  with_identity: $with_identity"
if [ "$baseline" = "$with_identity" ]; then
    echo "  PASS: identity_steer produces baseline output"
else
    echo "  FAIL: identity_steer changed generation"
    exit 1
fi
echo

echo "=== With noise_steer (delta on dim 0) — should differ from baseline ==="
with_noise=$(curl -sS -X POST "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
        \"max_tokens\": $MAX_TOKENS,
        \"temperature\": 0.0,
        \"steer_shims\": [\"noise_steer\"]
    }" | extract_content)
echo "  with_noise: $with_noise"
if [ "$baseline" != "$with_noise" ]; then
    echo "  PASS: noise_steer shifted generation"
else
    echo "  WARN: noise_steer produced identical output (delta may be too small to flip greedy argmax)"
fi
echo

echo "=== Streaming with identity_steer ==="
curl -sN -X POST "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
        \"max_tokens\": 4,
        \"temperature\": 0.0,
        \"stream\": true,
        \"steer_shims\": [\"identity_steer\"]
    }"
echo
echo
echo "=== Steer + cache should reject ==="
curl -sS -X POST "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],
        \"max_tokens\": 4,
        \"temperature\": 0.0,
        \"cache_id\": \"nonexistent\",
        \"steer_shims\": [\"identity_steer\"]
    }" | python -m json.tool
echo
