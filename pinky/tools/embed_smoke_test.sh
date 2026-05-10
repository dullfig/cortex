#!/usr/bin/env bash
# End-to-end smoke for /v1/shims/embed. Assumes:
#   - cortex-server is running on localhost:$PORT with --enable-shims
#
# Validates seven properties:
#   1. final + last_token returns a 2048-dim vector
#   2. final + mean returns a 2048-dim vector that DIFFERS from last_token
#      (mean and last_token integrate over different token spans)
#   3. entrance:5 + last_token returns a 2048-dim vector
#   4. entrance:0 rejects with 400 (embedding-lookup output not captured in v1)
#   5. entrance:9999 rejects with 400 (out of range)
#   6. empty text rejects with 400
#   7. unknown pooling rejects with 400

set -euo pipefail

PORT=${PORT:-8181}

PROMPT_A="The 1941 Bluejacket Manual covers naval procedures."
PROMPT_B="Paris is the capital of France."

call_embed() {
    curl -sS -X POST "http://localhost:$PORT/v1/shims/embed" \
        -H "Content-Type: application/json" \
        -d "$1"
}

embed_dim() {
    python -c "import json,sys; d=json.load(sys.stdin); print(len(d.get('embedding', [])))"
}

first_n() {
    # Print the first $1 elements of the embedding so we can compare runs.
    python -c "import json,sys; d=json.load(sys.stdin); print(d['embedding'][:int($1)])"
}

error_type() {
    python -c "import json,sys; d=json.load(sys.stdin); print(d.get('error', {}).get('type', '<no-error>'))"
}

echo "=== 1. final + last_token ==="
resp1=$(call_embed "{\"text\": \"$PROMPT_A\", \"layer\": \"final\", \"pooling\": \"last_token\"}")
dim=$(echo "$resp1" | embed_dim)
echo "  embedding dim: $dim"
if [ "$dim" -ne 2048 ]; then echo "  FAIL: expected 2048"; exit 1; fi
echo "  first 4: $(echo "$resp1" | first_n 4)"
echo "  PASS"
echo

echo "=== 2. final + mean differs from last_token ==="
resp2=$(call_embed "{\"text\": \"$PROMPT_A\", \"layer\": \"final\", \"pooling\": \"mean\"}")
dim=$(echo "$resp2" | embed_dim)
echo "  embedding dim: $dim"
if [ "$dim" -ne 2048 ]; then echo "  FAIL: expected 2048"; exit 1; fi
v1=$(echo "$resp1" | first_n 4)
v2=$(echo "$resp2" | first_n 4)
if [ "$v1" != "$v2" ]; then
    echo "  PASS: mean ($v2) != last_token ($v1)"
else
    echo "  FAIL: mean and last_token returned identical first-4 elements"
    exit 1
fi
echo

echo "=== 3. entrance:5 + last_token ==="
resp3=$(call_embed "{\"text\": \"$PROMPT_B\", \"layer\": \"entrance:5\", \"pooling\": \"last_token\"}")
dim=$(echo "$resp3" | embed_dim)
echo "  embedding dim: $dim"
if [ "$dim" -ne 2048 ]; then echo "  FAIL: expected 2048"; exit 1; fi
echo "  first 4: $(echo "$resp3" | first_n 4)"
echo "  PASS"
echo

echo "=== 4. entrance:0 rejects ==="
err=$(call_embed "{\"text\": \"hi\", \"layer\": \"entrance:0\"}" | error_type)
if [ "$err" = "unsupported" ]; then echo "  PASS: $err"; else echo "  FAIL: got '$err'"; exit 1; fi
echo

echo "=== 5. entrance:9999 rejects ==="
err=$(call_embed "{\"text\": \"hi\", \"layer\": \"entrance:9999\"}" | error_type)
if [ "$err" = "unsupported" ]; then echo "  PASS: $err"; else echo "  FAIL: got '$err'"; exit 1; fi
echo

echo "=== 6. empty text rejects ==="
err=$(call_embed "{\"text\": \"\"}" | error_type)
if [ "$err" = "invalid_request" ]; then echo "  PASS: $err"; else echo "  FAIL: got '$err'"; exit 1; fi
echo

echo "=== 7. unknown pooling rejects ==="
err=$(call_embed "{\"text\": \"hi\", \"pooling\": \"attention\"}" | error_type)
if [ "$err" = "unsupported" ]; then echo "  PASS: $err"; else echo "  FAIL: got '$err'"; exit 1; fi
echo

echo "=== ALL PASSED ==="
