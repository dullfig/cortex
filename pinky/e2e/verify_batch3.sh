#!/usr/bin/env bash
# Full verification bar for an engine-touching cortex-cloud commit.
# Usage: verify_batch3.sh <label>   (release binary must already be built)
set -u
S="$(cd "$(dirname "$0")" && pwd)"   # this directory (pinky/e2e)
cd "$S/../.." || exit 1   # repo root
LABEL="${1:-run}"
LOG="$S/verify_$LABEL.log"
: > "$LOG"
say() { echo "$*" | tee -a "$LOG"; }

BOOT_N=0
boot() {  # boot <port> <flags...> ; sets WPID, SRV_LOG
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

say "== polar sweep vs dumps_base (5744f51) =="
rm -rf "$S/dumps_new"
CORTEX_RETRIEVE_HEAD_DUMP="$S/dumps_new" boot 8100 --enable-cache --enable-retrieve --enable-polar-cache
PYTHONIOENCODING=utf-8 python pinky/retrieval-heads/sweep.py --server http://127.0.0.1:8100 --dumps "$S/dumps_new" --polar --label new 2>&1 | grep -E "all-heads|Traceback" | tee -a "$LOG"
halt
python - "$S" <<'EOF' 2>&1 | tee -a "$LOG"
import sys, os, json
S = sys.argv[1]
def load(d):
    fs = sorted(f for f in os.listdir(os.path.join(S, d)) if f.startswith("headdump-"))[:10]
    return [json.load(open(os.path.join(S, d, f))) for f in fs]
a, b = load("dumps_base"), load("dumps_new")
ident = sum(1 for x, y in zip(a, b) if json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True))
print(f"  dumps identical: {ident} of {min(len(a), len(b))} (base {len(a)}, new {len(b)})")
EOF

say "== e2e: memex + shims_embed + brick (8192, cache+retrieve+shims) =="
boot 8124 --enable-cache --enable-retrieve --enable-shims --max-seq-len 8192
for f in e2e_memex e2e_shims_embed e2e_brick; do say "  $f: $(python "$S/$f.py" 2>&1 | tail -1)"; done
halt

say "== e2e: boundary (4096, cap 3) =="
boot 8124 --enable-cache --enable-retrieve --max-seq-len 4096 --max-cache-shards 3
say "  e2e_boundary: $(python "$S/e2e_boundary.py" 2>&1 | tail -1)"
halt

say "== e2e: lockstep (4096, polar) =="
boot 8124 --enable-cache --enable-retrieve --enable-polar-cache --max-seq-len 4096
say "  e2e_lockstep: $(python "$S/e2e_lockstep.py" 2>&1 | tail -1)"
halt

say "== e2e: toctou (16384) =="
boot 8124 --enable-cache --enable-retrieve --max-seq-len 16384
say "  e2e_toctou: $(python "$S/e2e_toctou.py" 2>&1 | tail -1)"
halt

say "== e2e: concurrency (8192, cache+shims) =="
boot 8124 --enable-cache --enable-shims --max-seq-len 8192
say "  e2e_concurrency: $(python "$S/e2e_concurrency.py" 2>&1 | tail -2 | tr '\n' ' ')"
halt

say "live servers: $(tasklist | grep -ci cortex-server)"

say "== cargo test --workspace -- --test-threads=1 =="
cargo test --workspace -- --test-threads=1 > "$S/ws_test_$LABEL.log" 2>&1
say "  exit=$?"
grep -E "^test result" "$S/ws_test_$LABEL.log" | awk '{p+=$4; f+=$6; i+=$8} END {print "  passed="p" failed="f" ignored="i}' | tee -a "$LOG"
grep -E "FAILED|panicked" "$S/ws_test_$LABEL.log" | head -5 | tee -a "$LOG"
say "== DONE =="
