#!/usr/bin/env bash
# Remainder of the Items 3+4 bar after the low-memory kill: workspace tests
# minus the three ~12 GB Qwen-shaped no-crash tests, then toctou again.
set -u
S="$(cd "$(dirname "$0")" && pwd)"   # this directory (pinky/e2e)
cd "$S/../.." || exit 1   # repo root
LABEL="item4b"
LOG="$S/verify_$LABEL.log"
: > "$LOG"
say() { echo "$*" | tee -a "$LOG"; }

say "== cargo test --workspace -- --test-threads=1 --skip qwen_shape =="
cargo test --workspace -- --test-threads=1 --skip qwen_shape > "$S/ws_test_$LABEL.log" 2>&1
say "  exit=$?"
grep -E "^test result" "$S/ws_test_$LABEL.log" | awk '{p+=$4; f+=$6; i+=$8} END {print "  passed="p" failed="f" ignored="i}' | tee -a "$LOG"
grep -E "FAILED|panicked" "$S/ws_test_$LABEL.log" | head -5 | tee -a "$LOG"

say "== e2e: toctou (16384), healthy machine =="
SRV_LOG="$S/srv_${LABEL}_toctou.log"
./target/release/cortex-server.exe --model models/Qwen2.5-3B-Q4_K_M.gguf --port 8124 --enable-cache --enable-retrieve --max-seq-len 16384 > "$SRV_LOG" 2>&1 &
for i in $(seq 1 150); do
  curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8124/health 2>/dev/null | grep -q 200 && break
  sleep 2
done
WPID=$(netstat -ano | grep -E ":8124 " | grep LISTEN | awk '{print $5}' | head -1)
python -u "$S/e2e_toctou.py" > "$S/toctou_${LABEL}.txt" 2>&1
say "  e2e_toctou: $(tail -1 "$S/toctou_${LABEL}.txt")"
taskkill //PID "$WPID" //F >/dev/null 2>&1; sleep 1
n=$(grep -icE 'panicked|Validation Error|Out of Memory' "$SRV_LOG")
say "  panics: $n"
if [ "$n" != "0" ]; then sed -E 's/\x1b\[[0-9;]*m//g' "$SRV_LOG" | grep -A4 "panicked" | cut -c1-200 | head -12 | tee -a "$LOG"; fi
say "live servers: $(tasklist | grep -ci cortex-server)"
say "== DONE =="
