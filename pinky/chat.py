# Minimal interactive chat REPL for a running cortex-server.
# Talks to the OpenAI-compatible /v1/chat/completions endpoint, keeps
# conversation history. Run it in your own terminal:
#     python C:\src\cortex\pinky\chat.py
# (or with --shard <id> to chat against a loaded retrieval cache).
# Ctrl-C or "/quit" to exit; "/reset" clears history.
import argparse, json, sys, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--server", default="http://127.0.0.1:8100")
ap.add_argument("--temp", type=float, default=0.7)
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--shard", default=None, help="cache_id to ground chat against (must be loaded)")
args = ap.parse_args()

def post(body):
    req = urllib.request.Request(args.server + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))

try:
    name = json.load(urllib.request.urlopen(args.server + "/health", timeout=5))["model"]
except Exception as e:
    print(f"can't reach cortex at {args.server}: {e}", file=sys.stderr); sys.exit(1)

print(f"cortex chat — {name}  (temp={args.temp}"
      + (f", shard={args.shard}" if args.shard else "") + ")")
print("type /quit to exit, /reset to clear history\n")

history = []
while True:
    try:
        user = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        print(); break
    if not user:
        continue
    if user == "/quit":
        break
    if user == "/reset":
        history = []; print("(history cleared)\n"); continue
    history.append({"role": "user", "content": user})
    body = {"model": "cortex", "messages": history,
            "temperature": args.temp, "max_tokens": args.max_tokens}
    if args.shard:
        body["cache_shards"] = [args.shard]
    try:
        reply = post(body)["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[error: {e}]\n"); history.pop(); continue
    print(f"bot> {reply.strip()}\n")
    history.append({"role": "assistant", "content": reply})
