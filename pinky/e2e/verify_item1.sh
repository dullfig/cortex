#!/usr/bin/env bash
# Batch 5 Item 1 (#23/#31) verification: churn "after", tiny-lane chunking,
# retrieve bounds, then the full engine bar (sweep + e2e + workspace tests).
set -u
S="$(cd "$(dirname "$0")" && pwd)"   # this directory (pinky/e2e)
cd "$S/../.." || exit 1   # repo root
LABEL="item1"
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
halt() {
  taskkill //PID "$WPID" //F >/dev/null 2>&1; sleep 1
  local n; n=$(grep -icE 'panicked|Validation Error|Out of Memory|is invalid' "$SRV_LOG")
  say "  panics: $n (log: $(basename "$SRV_LOG"))"
  if [ "$n" != "0" ]; then sed -E 's/\x1b\[[0-9;]*m//g' "$SRV_LOG" | grep -B2 -A6 "panicked\|is invalid" | cut -c1-220 | head -24 | tee -a "$LOG"; fi
}

say "== churn AFTER (4096, cap 3, 20 loops) =="
boot 8124 --enable-cache --enable-retrieve --max-seq-len 4096 --max-cache-shards 3
python -u "$S/e2e_vram_churn.py" 20 > "$S/churn_after.txt" 2>&1
say "  rc=$? $(tail -1 "$S/churn_after.txt")"
grep -E "statuses=" "$S/churn_after.txt" | cut -c1-220 | tee -a "$LOG"
halt

say "== tiny lanes (A 24 / B 96 / C 24 MiB, readback 32 MiB): chunking must absorb the per-forward buffers =="
CORTEX_VRAM_HEAP_A_MB=24 CORTEX_VRAM_HEAP_B_MB=96 CORTEX_VRAM_HEAP_C_MB=24 CORTEX_VRAM_HEAP_READBACK_MB=32 \
  boot 8124 --enable-cache --enable-retrieve --enable-shims --max-seq-len 4096 --max-cache-shards 3
say "  e2e_boundary: $(python -u "$S/e2e_boundary.py" 2>&1 | tail -1)"
say "  e2e_shims_embed: $(python -u "$S/e2e_shims_embed.py" 2>&1 | tail -1)"
halt

say "== retrieve bound, polar (8192) =="
boot 8124 --enable-cache --enable-retrieve --enable-polar-cache --max-seq-len 8192
python -u "$S/e2e_retrieve_bound.py" polar 2>&1 | tail -4 | tee -a "$LOG"
halt

say "== retrieve bound, f32 (8192) =="
boot 8124 --enable-cache --enable-retrieve --max-seq-len 8192
python -u "$S/e2e_retrieve_bound.py" f32 2>&1 | tail -4 | tee -a "$LOG"
halt

say "live servers: $(tasklist | grep -ci cortex-server)"
say "free RAM GB: $(powershell -NoProfile -Command '(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB')"

say "== full bar =="
bash "$S/verify_batch3.sh" "$LABEL"
