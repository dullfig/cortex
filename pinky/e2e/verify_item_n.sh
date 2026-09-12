#!/usr/bin/env bash
# Batch 5 Items 2-4 verification. Usage: verify_item_n.sh <item2|item3|item4>
# Runs that item's own e2e boots, then the full engine bar (verify_batch3.sh).
set -u
S="$(cd "$(dirname "$0")" && pwd)"   # this directory (pinky/e2e)
cd "$S/../.." || exit 1   # repo root
LABEL="${1:-itemN}"
LOG="$S/verify_${LABEL}_pre.log"
: > "$LOG"
say() { echo "$*" | tee -a "$LOG"; }
BOOT_N=0
boot() {
  local port="$1"; shift
  BOOT_N=$((BOOT_N + 1))
  SRV_LOG="$S/srv_${LABEL}_pre_boot${BOOT_N}.log"
  ./target/release/cortex-server.exe --model models/Qwen2.5-3B-Q4_K_M.gguf --port "$port" "$@" > "$SRV_LOG" 2>&1 &
  for i in $(seq 1 150); do
    curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$port/health" 2>/dev/null | grep -q 200 && break
    sleep 2
  done
  WPID=$(netstat -ano | grep -E ":$port " | grep LISTEN | awk '{print $5}' | head -1)
}
halt() {  # halt [expected_panics]
  taskkill //PID "$WPID" //F >/dev/null 2>&1; sleep 1
  local n; n=$(grep -icE 'panicked|Validation Error|Out of Memory|is invalid' "$SRV_LOG")
  say "  panics: $n (expected ${1:-0}) (log: $(basename "$SRV_LOG"))"
  if [ "$n" != "${1:-0}" ]; then sed -E 's/\x1b\[[0-9;]*m//g' "$SRV_LOG" | grep -B2 -A6 "panicked\|is invalid" | cut -c1-220 | head -30 | tee -a "$LOG"; fi
}

case "$LABEL" in
  item2)
    say "== #24: stream panic (test hook armed, 4096, cache) =="
    CORTEX_TEST_PANIC_AFTER_TOKENS=3 boot 8124 --enable-cache --enable-test-hooks --max-seq-len 4096
    python -u "$S/e2e_stream_panic.py" > "$S/stream_panic_out.txt" 2>&1
    say "  e2e_stream_panic: rc=$? $(tail -1 "$S/stream_panic_out.txt")"
    say "  hook panics logged: $(grep -c 'test hook: CORTEX_TEST_PANIC_AFTER_TOKENS' "$SRV_LOG")"
    halt 3
    say "== #24: disconnect (4096, cache) =="
    boot 8124 --enable-cache --max-seq-len 4096
    python -u "$S/e2e_disconnect.py" > "$S/disconnect_out.txt" 2>&1
    say "  e2e_disconnect: rc=$? $(tail -1 "$S/disconnect_out.txt")"
    grep -E "A stateless|B streaming|C cached|calibration" "$S/disconnect_out.txt" | tee -a "$LOG"
    say "  cancellations logged: $(grep -c 'cancelled: client disconnected' "$SRV_LOG")"
    halt
    ;;
  item3)
    say "== #10: long input (4096) =="
    boot 8124 --max-seq-len 4096
    python -u "$S/e2e_long_input.py" > "$S/long_input_out.txt" 2>&1
    say "  e2e_long_input: rc=$? $(tail -1 "$S/long_input_out.txt")"
    grep -E "tokenize|chat|body" "$S/long_input_out.txt" | tee -a "$LOG"
    halt
    ;;
  item4)
    say "== #25/#26: shim shapes + max_tokens 0 (shims, 4096) =="
    boot 8124 --enable-shims --max-seq-len 4096
    python -u "$S/e2e_shim_shape.py" > "$S/shim_shape_out.txt" 2>&1
    say "  e2e_shim_shape: rc=$? $(tail -1 "$S/shim_shape_out.txt")"
    grep -E "->" "$S/shim_shape_out.txt" | tee -a "$LOG"
    halt
    say "== #26: lockstep incl. the clamp-to-zero scenario (4096, polar) =="
    boot 8124 --enable-cache --enable-retrieve --enable-polar-cache --max-seq-len 4096
    LOCKSTEP_ITEM4=1 python -u "$S/e2e_lockstep.py" > "$S/lockstep_item4.txt" 2>&1
    say "  e2e_lockstep(item4): rc=$? $(tail -1 "$S/lockstep_item4.txt")"
    grep -E "^6:" "$S/lockstep_item4.txt" | tee -a "$LOG"
    halt
    ;;
esac

say "live servers: $(tasklist | grep -ci cortex-server)"
say "free RAM GB: $(powershell -NoProfile -Command '(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB')"
say "== full bar =="
bash "$S/verify_batch3.sh" "$LABEL"
