#!/usr/bin/env bash
# Item 4 (#12) verification: the two retrieve-bound boots, then the full bar.
set -u
S="$(cd "$(dirname "$0")" && pwd)"   # this directory (pinky/e2e)
cd "$S/../.." || exit 1   # repo root
LOG="$S/verify_item4_pre.log"
: > "$LOG"
say() { echo "$*" | tee -a "$LOG"; }
boot() {
  local port="$1"; shift
  ./target/release/cortex-server.exe --model models/Qwen2.5-3B-Q4_K_M.gguf --port "$port" "$@" > "$S/srv_verify.log" 2>&1 &
  for i in $(seq 1 150); do
    curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$port/health" 2>/dev/null | grep -q 200 && break
    sleep 2
  done
  WPID=$(netstat -ano | grep -E ":$port " | grep LISTEN | awk '{print $5}' | head -1)
}
halt() {
  taskkill //PID "$WPID" //F >/dev/null 2>&1; sleep 1
  say "  panics: $(grep -icE 'panicked|Validation Error|Out of Memory' "$S/srv_verify.log")"
}

say "== retrieve bound, polar (8192) =="
boot 8124 --enable-cache --enable-retrieve --enable-polar-cache --max-seq-len 8192
python -u "$S/e2e_retrieve_bound.py" polar 2>&1 | tee -a "$LOG" | tail -0
halt

say "== retrieve bound, f32 (8192) =="
boot 8124 --enable-cache --enable-retrieve --max-seq-len 8192
python -u "$S/e2e_retrieve_bound.py" f32 2>&1 | tee -a "$LOG" | tail -0
halt

say "== full bar =="
bash "$S/verify_batch3.sh" item4
