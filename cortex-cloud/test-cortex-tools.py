#!/usr/bin/env python3
"""End-to-end test for cortex tool-calling round trip.

From agentos-Claude, 2026-04-10. Tests the full OpenAI-format tool
calling round trip: user asks → model invokes tool → tool result
returned → model produces final answer incorporating result.

Assumes cortex-server is running at localhost:8090.
"""
import json
import urllib.request

CORTEX_URL = "http://localhost:8090/v1/chat/completions"
MODEL = "qwen2.5-0.5b-instruct"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_events",
        "description": "Search ringhub barbershop events by query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"},
            },
            "required": ["query"],
        },
    },
}]

def post(payload):
    req = urllib.request.Request(
        CORTEX_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())

# -- Turn 1: user asks, model should invoke the tool --
messages = [
    {"role": "system", "content": "You are the ringhub concierge. Use search_events when the user asks about upcoming events."},
    {"role": "user", "content": "What barbershop events are happening this weekend?"},
]

print("=== Turn 1: expecting tool_calls ===")
resp1 = post({"model": MODEL, "messages": messages, "tools": TOOLS, "max_tokens": 200})
print(json.dumps(resp1, indent=2))

choice = resp1["choices"][0]
assert choice["finish_reason"] == "tool_calls", f"expected tool_calls, got {choice['finish_reason']!r}"
tool_call = choice["message"]["tool_calls"][0]
print(f"\n-> Model invoked: {tool_call['function']['name']}({tool_call['function']['arguments']})")

# -- Turn 2: append assistant + tool result, expect final answer --
messages.append(choice["message"])
messages.append({
    "role": "tool",
    "tool_call_id": tool_call["id"],
    "content": json.dumps([
        {"name": "Saturday Open Sing", "venue": "Torrance Cultural Arts Center", "time": "Sat 7pm"},
        {"name": "Sunday Coaching", "venue": "Long Beach", "time": "Sun 2pm"},
    ]),
})

print("\n=== Turn 2: expecting final stop ===")
resp2 = post({"model": MODEL, "messages": messages, "tools": TOOLS, "max_tokens": 300})
print(json.dumps(resp2, indent=2))

choice2 = resp2["choices"][0]
assert choice2["finish_reason"] == "stop", f"expected stop, got {choice2['finish_reason']!r}"
print(f"\n-> Final response: {choice2['message']['content']}")
print("\nTool-calling round trip complete.")
