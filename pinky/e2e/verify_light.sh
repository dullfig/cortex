#!/usr/bin/env bash
# Light verification bar for load-path (non-engine) commits:
# workspace tests, the real-Qwen ignored test, release boot + brick/boundary/shims e2e.
set -u
S="$(cd "$(dirname "$0")" && pwd)"   # this directory (pinky/e2e)
cd "$S/../.." || exit 1   # repo root
LABEL="${1:-run}"
LOG="$S/verify_light_$LABEL.log"
: > "$LOG"
say() { echo "$*" | tee -a "$LOG"; }
BOOT_N=0
boot() {
  local port="$1"; shift
  BOOT_N=$((BOOT_N + 1))
  SRV_LOG="$S/srv_${LABEL}_boot${BOOT_N}.log"
  ./target/release/cortex-server.exe --model models/Qwen2.5-3B-Q4_K_M.gguf --port "$port" "$@" > "$SRV_LOG" 2>&1 &
  for i in $(seq 1 150); do
    curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$port/health" 2>/dev/null | grep -q 200 && break
    sleep 2
  done
  WPID=$(netstat -ano | grep -E ":$port " | grep LISTEN | awk '{print $5}' | head -1)
}
halt() {
  taskkill //PID "$WPID" //F >/dev/null 2>&1; sleep 1
  local n; n=$(grep -icE 'panicked|Validation Error|Out of Memory' "$SRV_LOG")
  say "  panics: $n (log: $(basename "$SRV_LOG"))"
  if [ "$n" != "0" ]; then sed -E 's/\x1b\[[0-9;]*m//g' "$SRV_LOG" | grep -A4 "panicked" | cut -c1-200 | head -12 | tee -a "$LOG"; fi
}

say "== cargo test --workspace -- --test-threads=1 =="
cargo test --workspace -- --test-threads=1 > "$S/ws_test_$LABEL.log" 2>&1
say "  exit=$?"
grep -E "^test result" "$S/ws_test_$LABEL.log" | awk '{p+=$4; f+=$6; i+=$8} END {print "  passed="p" failed="f" ignored="i}' | tee -a "$LOG"
grep -E "FAILED|panicked" "$S/ws_test_$LABEL.log" | head -5 | tee -a "$LOG"

say "== real Qwen 3B through the loader (ignored test) =="
cargo test -p cortex --lib -- --ignored --test-threads=1 forward_full_gpu_real_qwen3b_no_crash 2>&1 | grep -E "^test |test result|panicked" | tee -a "$LOG"

say "== release build =="
cargo build --release -p cortex-cloud 2>&1 | grep -E "^error|Finished" | tail -1 | tee -a "$LOG"

say "== e2e: boundary (4096, cap 3) =="
boot 8124 --enable-cache --enable-retrieve --max-seq-len 4096 --max-cache-shards 3
say "  e2e_boundary: $(python -u "$S/e2e_boundary.py" 2>&1 | tail -1)"
halt

say "== e2e: brick + shims_embed (4096, cache+shims) =="
boot 8124 --enable-cache --enable-shims --max-seq-len 4096
say "  e2e_brick: $(python -u "$S/e2e_brick.py" 2>&1 | tail -1)"
say "  e2e_shims_embed: $(python -u "$S/e2e_shims_embed.py" 2>&1 | tail -1)"
halt

say "live servers: $(tasklist | grep -ci cortex-server)"
say "== DONE =="
