#!/usr/bin/env bash
# Item 2 follow-up (lockstep check on the chat path): cortex-cloud tests,
# release build, then every cache-path driver — the engine is untouched.
set -u
S="$(cd "$(dirname "$0")" && pwd)"   # this directory (pinky/e2e)
cd "$S/../.." || exit 1   # repo root
LABEL="item2b"
LOG="$S/verify_${LABEL}.log"
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
  local n; n=$(grep -icE 'panicked|Validation Error|Out of Memory|is invalid' "$SRV_LOG")
  say "  panics: $n (expected ${1:-0}) (log: $(basename "$SRV_LOG"))"
  if [ "$n" != "${1:-0}" ]; then sed -E 's/\x1b\[[0-9;]*m//g' "$SRV_LOG" | grep -B2 -A6 "panicked\|is invalid" | cut -c1-220 | head -30 | tee -a "$LOG"; fi
}

say "== cortex-cloud tests + release build =="
cargo test -p cortex-cloud -- --test-threads=1 2>&1 | grep -E "test result|FAILED" | tee -a "$LOG"
cargo build --release 2>&1 | grep -E "^error|Finished" | tail -1 | tee -a "$LOG"

say "== #24: stream panic (test hook armed, 4096, cache) =="
CORTEX_TEST_PANIC_AFTER_TOKENS=3 boot 8124 --enable-cache --enable-test-hooks --max-seq-len 4096
python -u "$S/e2e_stream_panic.py" > "$S/stream_panic_out.txt" 2>&1
say "  e2e_stream_panic: rc=$? $(tail -1 "$S/stream_panic_out.txt")"
grep -E "->|stream:" "$S/stream_panic_out.txt" | cut -c1-140 | tee -a "$LOG"
# 3 hook panics + the supervisor's "streaming generation panicked" line
halt 4

say "== #24: disconnect (4096, cache) =="
boot 8124 --enable-cache --max-seq-len 4096
python -u "$S/e2e_disconnect.py" > "$S/disconnect_out.txt" 2>&1
say "  e2e_disconnect: rc=$? $(tail -1 "$S/disconnect_out.txt")"
grep -E "A stateless|B streaming|C cached" "$S/disconnect_out.txt" | tee -a "$LOG"
halt

say "== e2e: memex + shims_embed + brick (8192, cache+retrieve+shims) =="
boot 8124 --enable-cache --enable-retrieve --enable-shims --max-seq-len 8192
for f in e2e_memex e2e_shims_embed e2e_brick; do say "  $f: $(python -u "$S/$f.py" 2>&1 | tail -1)"; done
halt

say "== e2e: boundary (4096, cap 3) =="
boot 8124 --enable-cache --enable-retrieve --max-seq-len 4096 --max-cache-shards 3
say "  e2e_boundary: $(python -u "$S/e2e_boundary.py" 2>&1 | tail -1)"
halt

say "== e2e: lockstep (4096, polar) =="
boot 8124 --enable-cache --enable-retrieve --enable-polar-cache --max-seq-len 4096
say "  e2e_lockstep: $(python -u "$S/e2e_lockstep.py" 2>&1 | tail -1)"
halt

say "== e2e: toctou (16384) =="
boot 8124 --enable-cache --enable-retrieve --max-seq-len 16384
say "  e2e_toctou: $(python -u "$S/e2e_toctou.py" 2>&1 | tail -1)"
halt

say "== e2e: concurrency (8192, cache+shims) =="
boot 8124 --enable-cache --enable-shims --max-seq-len 8192
say "  e2e_concurrency: $(python -u "$S/e2e_concurrency.py" 2>&1 | tail -2 | tr '\n' ' ')"
halt

say "live servers: $(tasklist | grep -ci cortex-server)"
say "== DONE =="
