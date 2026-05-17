"""Cortex baseline benchmark — TTFT and decode rate at varying prompt lengths.

Hits POST /v1/chat/completions with stream=true and measures:
  - TTFT (time-to-first-token): seconds from request submit to first chunk
    that carries content
  - Prefill rate: prompt_tokens / TTFT
  - Decode rate: completion_tokens / (total - TTFT)
  - End-to-end wall time

Runs each prompt-length point N times and reports min/median/max.

Usage:
  python bench_baseline.py [--port 8181] [--max_tokens 128] [--repeats 3]
"""
import argparse, json, time, urllib.request, statistics, sys
sys.stdout.reconfigure(line_buffering=True)

PROMPT_LENGTHS_TARGET = [50, 500, 2000]  # rough token targets

# Build prompts of approximately these token counts. Qwen's tokenizer is
# roughly ~1.3 tokens per word for English; we pad with filler.
FILLER = (
    "The quick brown fox jumps over the lazy dog. "
    "She sells sea shells by the sea shore. "
    "Peter piper picked a peck of pickled peppers. "
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood. "
)

def build_prompt(target_tokens):
    # ~1 token per word approx; pad with filler until close to target.
    words_per_repeat = len(FILLER.split())
    repeats = max(1, target_tokens // words_per_repeat)
    body = (FILLER * repeats).strip()
    return body + "\n\nIn one sentence, summarize the above."

def stream_chat(base_url, prompt, max_tokens, timeout=600):
    payload = {
        "model": "cortex",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )

    t_submit = time.time()
    t_first = None
    last_chunk = None
    n_chunks = 0
    n_content_chunks = 0
    usage = None

    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.strip()
            if not line:
                continue
            if not line.startswith(b"data:"):
                continue
            body = line[len(b"data:"):].strip()
            if body == b"[DONE]":
                break
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                continue
            n_chunks += 1
            choices = obj.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    if t_first is None:
                        t_first = time.time()
                    n_content_chunks += 1
            if "usage" in obj and obj["usage"]:
                usage = obj["usage"]
            last_chunk = obj

    t_end = time.time()
    if t_first is None:
        t_first = t_end
    return {
        "ttft_s": t_first - t_submit,
        "total_s": t_end - t_submit,
        "decode_s": t_end - t_first,
        "n_chunks": n_chunks,
        "n_content_chunks": n_content_chunks,
        "usage": usage,
        "last_chunk": last_chunk,
    }

def summarize(samples, key):
    vals = [s[key] for s in samples]
    return {
        "min": min(vals),
        "median": statistics.median(vals),
        "max": max(vals),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8181)
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--prompt_lengths", type=int, nargs="*",
                    default=PROMPT_LENGTHS_TARGET,
                    help="approximate prompt token targets")
    args = ap.parse_args()

    base = f"http://localhost:{args.port}"

    # Sanity check health.
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
            r.read()
        print(f"  health: ok")
    except Exception as e:
        print(f"  health FAILED: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== cortex baseline benchmark ===")
    print(f"  endpoint:    {base}/v1/chat/completions")
    print(f"  max_tokens:  {args.max_tokens}")
    print(f"  repeats:     {args.repeats}")
    print(f"  prompt_lens: {args.prompt_lengths}")

    # Warm-up run (one round per prompt length, results discarded).
    print("\n--- warm-up ---", flush=True)
    for target in args.prompt_lengths:
        prompt = build_prompt(target)
        try:
            r = stream_chat(base, prompt, args.max_tokens)
            ct = r['n_content_chunks']
            print(f"  warm-up target={target}: total={r['total_s']:.2f}s "
                  f"ttft={r['ttft_s']:.2f}s content_chunks={ct}", flush=True)
        except Exception as e:
            print(f"  warm-up target={target} FAILED: {e}", file=sys.stderr, flush=True)

    # Timed runs. completion_tokens estimated from content chunks (cortex
    # streams 1 token per chunk). prompt_tokens not exposed by the wire
    # protocol, so prefill_rate column shows the rough target word count.
    print("\n--- measured runs ---", flush=True)
    print(f"{'target':>8} {'words':>6} {'ct':>5} "
          f"{'TTFT (s)':>10} {'prefill w/s':>12} "
          f"{'decode t/s':>11} {'total (s)':>10}", flush=True)
    print("-" * 76, flush=True)

    all_results = {}
    for target in args.prompt_lengths:
        prompt = build_prompt(target)
        prompt_words = len(prompt.split())
        samples = []
        for i in range(args.repeats):
            try:
                samples.append(stream_chat(base, prompt, args.max_tokens))
            except Exception as e:
                print(f"  iter target={target} #{i+1} FAILED: {e}", file=sys.stderr, flush=True)
                continue
        if not samples:
            continue
        all_results[target] = samples

        ct = samples[0]['n_content_chunks']
        ttft = summarize(samples, 'ttft_s')
        total = summarize(samples, 'total_s')

        decode_rates = [s['n_content_chunks'] / max(s['decode_s'], 1e-6) for s in samples]
        prefill_rates = [prompt_words / max(s['ttft_s'], 1e-6) for s in samples]

        print(f"{target:>8} {prompt_words:>6} {ct:>5} "
              f"{ttft['median']:>10.3f} "
              f"{statistics.median(prefill_rates):>12.1f} "
              f"{statistics.median(decode_rates):>11.2f} "
              f"{total['median']:>10.3f}", flush=True)

    # Pretty min/max table for headlines.
    print("\n--- min..max across repeats ---", flush=True)
    print(f"{'target':>8} {'TTFT':>20} {'decode t/s':>20} {'total':>20}", flush=True)
    print("-" * 72, flush=True)
    for target, samples in all_results.items():
        ttft = summarize(samples, 'ttft_s')
        total = summarize(samples, 'total_s')
        decode_rates = [s['n_content_chunks'] / max(s['decode_s'], 1e-6) for s in samples]
        dr_min, dr_max = min(decode_rates), max(decode_rates)
        print(f"{target:>8} "
              f"{ttft['min']:>7.3f}..{ttft['max']:<7.3f}s   "
              f"{dr_min:>6.2f}..{dr_max:<6.2f}     "
              f"{total['min']:>7.3f}..{total['max']:<7.3f}s", flush=True)

if __name__ == "__main__":
    main()
