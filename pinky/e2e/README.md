# pinky/e2e — end-to-end drivers and verification bars

The scripts that verified the adversarial-review hardening pass
(`docs/adversarial-review-2026-09-02.md`, 2026-09-04 .. 09-11). Each driver
boots nothing itself: it expects a `cortex-server` already listening on
`:8124` (`:8100` for the sweep), fires HTTP requests the way memex / Bob
would, and asserts on the responses. The `verify_*.sh` bars boot the release
binary with the right flags per driver, run it, kill the server by Windows
PID, and count panics in the per-boot server log.

Loose standards apply (this is pinky): the drivers are deliberately blunt,
and their expectations are the *current* contract — when a contract changes
on purpose, change the driver in the same commit.

## Prerequisites

- `cargo build --release` (the bars run `./target/release/cortex-server.exe`)
  and `models/Qwen2.5-3B-Q4_K_M.gguf` at the repo root.
- Python 3 with `urllib` (stdlib). `e2e_shim_shape.py` also needs `onnx`
  and `numpy` (it builds real ONNX graphs).
- Git Bash on Windows for the bars. Two quirks the bars already handle:
  kill the server by the Windows PID from `netstat -ano` (never `$!`, which
  is the MSYS pid), and write long driver output to files (the harness
  loses piped output on backgrounded commands).
- ≥ 16 GB free RAM before `cargo test --workspace` (three Qwen-shaped tests
  take ~12 GB each; `--skip qwen_shape` otherwise).

## Drivers

| Driver | Boot flags | Review # | What it provokes |
|---|---|---|---|
| `e2e_boundary.py` | `--enable-cache --enable-retrieve --max-seq-len 4096 --max-cache-shards 3` | #1-#6, #11 | window / room / token-id / pool-cap 400s and 507s; compose at the cap |
| `e2e_brick.py` | `--enable-cache --enable-shims` | #1 | the max_tokens-overflow "bricked shard" |
| `e2e_shims_embed.py` | `--enable-cache --enable-shims` | #27 | `/v1/shims/embed` returns finite, non-zero, layer-distinct vectors |
| `e2e_memex.py` | `--enable-cache --enable-retrieve --enable-shims --max-seq-len 8192` | #4 | the memex one-shot 6000-token `cache/load` |
| `e2e_toctou.py` | `--enable-cache --enable-retrieve --max-seq-len 16384` | #9, #30 | lock re-acquisition races: append vs same-id reload vs delete |
| `e2e_lockstep.py` | `--enable-cache --enable-retrieve --enable-polar-cache --max-seq-len 4096` | #8, #20, #26 | `tokens` == f32 seq == polar seq after every mutation; `LOCKSTEP_ITEM4=1` adds the clamp-to-zero scenario |
| `e2e_concurrency.py` | `--enable-cache --enable-shims --max-seq-len 8192` | #7 | an 8-request burst across paths; reads `cortex_gpu_gate_waiting` |
| `e2e_retrieve_bound.py polar` / `f32` | `... --enable-polar-cache --max-seq-len 8192` / `... --max-seq-len 8192` | #12, #23 | the traced-query bound (readback-heap-limited) is a 400 |
| `e2e_vram_churn.py [loops]` | `--enable-cache --enable-retrieve --max-seq-len 4096 --max-cache-shards 3` | #23, #31 | load ×3 / same-id reload at cap / compose / chat / append / reload / delete, with a stateless chat alongside |
| `e2e_stream_panic.py` | `--enable-cache --enable-test-hooks --max-seq-len 4096` + env `CORTEX_TEST_PANIC_AFTER_TOKENS=3` | #24 | a panicking generation: SSE error event (no `[DONE]`), 500s, 409 on the drifted shard, `/metrics` |
| `e2e_disconnect.py` | `--enable-cache --max-seq-len 4096` | #24, #32 | hang up mid-generation; stateless/streaming follow-ups are fast, cached waits (bounded) |
| `e2e_long_input.py` | `--max-seq-len 4096` | #10 | 60 KB tokenizes in ms; 200 KB → 400 `input_too_long`; 3 MB → 413 |
| `e2e_shim_shape.py` | `--enable-shims --max-seq-len 4096` | #25, #26 | wrong-shape ONNX refused at PUT; `max_tokens: 0` → 400 |
| `perf_probe.py` | any | — | TTFT / decode timing probe |

## Bars

- `verify_batch3.sh <label>` — the full engine bar: polar sweep dumps
  byte-compared against `dumps_base/`, then memex + shims_embed + brick,
  boundary, lockstep, toctou, concurrency, then `cargo test --workspace`.
  Run it for any commit that touches the engine or the cache paths.
- `verify_light.sh <label>` — workspace tests + boot + boundary/brick/shims
  for load-path-only commits.
- `verify_item1.sh`, `verify_item_n.sh item2|item3|item4`, `verify_item2b.sh`,
  `verify_item4.sh`, `verify_rest.sh` — the batch-5 / batch-3 item bars;
  each runs its item's own boots, then `verify_batch3.sh`.

Logs land next to the scripts: `verify_<label>.log`, `srv_<label>_bootN.log`
(one per boot), `ws_test_<label>.log`.

## `dumps_base/`

The polar sweep head dumps (`pinky/retrieval-heads/sweep.py --polar`,
all-heads R@10 = 0.10) produced by commit 5744f51. The bar asserts the
current build's dumps are byte-identical — the oracle that the engine
changes since (packed readbacks, lane moves, chunking, the tokenizer's
merge rewrite) did not change a single score. To regenerate from scratch:
build 5744f51 in a worktree, boot it with
`CORTEX_RETRIEVE_HEAD_DUMP=<dir> --enable-cache --enable-retrieve --enable-polar-cache`
on `:8100`, and run the sweep with `--dumps <dir> --polar`.
