#!/bin/bash
# Probe the chat_completions prefill stage timing at several prompt sizes.
# Server must be running with RUST_LOG including cortex=info so the
# "fwd_cache stage timings" lines land in its log.
#
# Usage: bash probe_ttft_stages.sh PORT SERVER_LOG_FILE
set -u
PORT=${1:-8181}
LOG=${2:-/tmp/cortex.log}

probe() {
  local label="$1"; local words="$2"
  local prompt
  prompt=$(python -c "print(('Word ' * ${words}).strip())")
  local payload
  payload=$(python -c "import json; print(json.dumps({'model':'cortex','messages':[{'role':'user','content':'$prompt'}],'max_tokens':3,'temperature':0,'stream':False}))")
  local t0; t0=$(date +%s.%N)
  curl -s -o /dev/null -w "  http_code=%{http_code} time_total=%{time_total}s\n" \
       -X POST "http://localhost:${PORT}/v1/chat/completions" \
       -H "Content-Type: application/json" \
       --max-time 120 --data-binary "$payload"
  local rc=$?
  local t1; t1=$(date +%s.%N)
  echo "  [${label}: ${words}w] wall=$(printf '%.2f' $(echo "$t1 - $t0" | bc -l))s rc=${rc}"
}

echo "=== probe ==="
for label in run1 run2; do
  for words in 50 500 2000; do
    probe "$label" "$words"
  done
done
