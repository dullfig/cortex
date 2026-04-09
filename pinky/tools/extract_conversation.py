"""extract_conversation.py — archive a Claude Code session JSONL as clean markdown.

Background: Claude Code stores session history as a JSONL file at
~/.claude/projects/<project-id>/<session-id>.jsonl. Each line is one
message/event in the session (user turns, assistant turns, tool uses,
tool results, sub-agent dispatches, system reminders, etc.).

Purpose: produce a clean markdown transcript of the top-level
conversation between the user and the main assistant, filtering out
sub-agent outputs, tool call internals, system reminders, and other
noise that obscures the human-readable dialogue.

Critical design constraint: the assistant cannot read its own session
state directly (safety filter policy violation). This script runs as
a subprocess and NEVER prints message content to stdout. The only
output on stdout is counts, field names, and success/failure status.
The actual extracted text is written directly to the output file and
never returned to the caller.

Usage:
    # First, inspect the structure without producing output
    python extract_conversation.py \\
        --input "path/to/session.jsonl" \\
        --schema-only

    # Then, if the structure looks right, produce the markdown
    python extract_conversation.py \\
        --input "path/to/session.jsonl" \\
        --output "path/to/transcript.md"

Safety notes for the assistant:
- Run this script and watch only the stdout counts, not the output file
- If you need to debug, use --schema-only, not reading the output
- Verify success via `ls -la` and `wc -l` on the output file only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


# Role/type classifications
TOP_LEVEL_ROLES = {"user", "assistant"}


def load_jsonl(path: Path):
    """Load a JSONL file line by line. Yields (line_num, parsed_obj, error)."""
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                yield line_num, obj, None
            except json.JSONDecodeError as e:
                # Return only error CLASS and line number - never raw line content
                yield line_num, None, type(e).__name__


def classify_entry(obj) -> str:
    """Return a short type label for an entry, for counting purposes.

    Possible returns: 'user', 'assistant', 'tool_use', 'tool_result',
    'system_reminder', 'meta', 'unknown'. This function only looks at
    keys and known values - never returns or logs message content.
    """
    if not isinstance(obj, dict):
        return "non_dict"

    # Claude Code's event types vary, so try several discriminators
    entry_type = obj.get("type", "")
    if entry_type in ("user", "assistant"):
        return entry_type
    if entry_type in ("tool_use", "tool_result"):
        return entry_type
    if entry_type == "system":
        return "system_reminder"
    if entry_type in ("summary", "meta", "compact"):
        return "meta"

    # Fall back to looking at nested message structure
    msg = obj.get("message", {})
    if isinstance(msg, dict):
        role = msg.get("role", "")
        if role in TOP_LEVEL_ROLES:
            return role

    # Check for task/tool-related keys
    if "toolUseResult" in obj or "tool_use_result" in obj:
        return "tool_result"

    return "unknown"


def extract_text_blocks(content) -> list[str]:
    """Extract text from a content field. Returns list of text strings.

    Handles:
    - content as a plain string (returns [string])
    - content as a list of blocks, each a dict with 'type' and 'text' or similar
    - block types: 'text' (extract), 'tool_use' (skip), 'tool_result' (skip),
      'thinking' (skip - internal)
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [content] if content.strip() else []
    if isinstance(content, list):
        texts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                text = block.get("text", "")
                if isinstance(text, str) and text.strip():
                    texts.append(text)
            # Deliberately skip tool_use, tool_result, thinking blocks
        return texts
    return []


def is_sidechain(obj) -> bool:
    """Return True iff this entry is from a sub-agent (sidechain) execution.

    Claude Code tags sub-agent task conversations with isSidechain=True.
    These should be excluded from the main transcript because they're
    internal tool operations, not part of the user's dialog with the
    main assistant.
    """
    if not isinstance(obj, dict):
        return False
    return bool(obj.get("isSidechain", False))


def is_meta(obj) -> bool:
    """Return True iff this is a meta entry (snapshot, operation, etc.)
    that isn't part of the conversation text.
    """
    if not isinstance(obj, dict):
        return False
    if obj.get("isMeta", False):
        return True
    if obj.get("isSnapshotUpdate", False):
        return True
    return False


def is_top_level_user_message(obj) -> bool:
    """Return True iff this is a user message that represents actual human input,
    not a tool result disguised as a user message and not a sub-agent turn.
    """
    if is_sidechain(obj):
        return False
    if is_meta(obj):
        return False
    if classify_entry(obj) != "user":
        return False
    # Check the nested message structure
    msg = obj.get("message", obj)
    if not isinstance(msg, dict):
        return False
    content = msg.get("content")
    if content is None:
        return False
    # If content is a list, check for tool_result blocks
    if isinstance(content, list):
        has_text = False
        has_tool_result = False
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                has_text = True
            elif btype == "tool_result":
                has_tool_result = True
        # If it's only tool results and no text, skip
        if has_tool_result and not has_text:
            return False
    # toolUseResult at the top level = this entry IS a tool result, skip
    if "toolUseResult" in obj:
        return False
    return True


def is_top_level_assistant_message(obj) -> bool:
    """Return True iff this is an assistant message with actual text output,
    from the main conversation (not a sub-agent).
    """
    if is_sidechain(obj):
        return False
    if is_meta(obj):
        return False
    if classify_entry(obj) != "assistant":
        return False
    msg = obj.get("message", obj)
    if not isinstance(msg, dict):
        return False
    content = msg.get("content")
    if content is None:
        return False
    # Check if there's any text block
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                if isinstance(block.get("text"), str) and block["text"].strip():
                    return True
    return False


def get_timestamp(obj) -> str:
    """Extract a timestamp string if present. Returns empty string if not."""
    for key in ("timestamp", "created_at", "createdAt", "time"):
        if key in obj:
            val = obj[key]
            if isinstance(val, str):
                return val
    # Also check nested message
    msg = obj.get("message", {})
    if isinstance(msg, dict):
        for key in ("timestamp", "created_at", "createdAt", "time"):
            if key in msg:
                val = msg[key]
                if isinstance(val, str):
                    return val
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Extract a clean markdown transcript from a Claude Code session JSONL."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the session .jsonl file",
    )
    parser.add_argument(
        "--output",
        help="Path to write the markdown transcript. Required unless --schema-only.",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Print entry type counts and field names only. No output file is written.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found")
        sys.exit(1)

    file_size = input_path.stat().st_size
    print(f"input: {input_path.name}")
    print(f"input size: {file_size} bytes ({file_size / 1024 / 1024:.1f} MB)")

    type_counts: Counter[str] = Counter()
    parse_errors: Counter[str] = Counter()
    total_entries = 0
    top_level_keys: Counter[str] = Counter()
    sidechain_count = 0
    meta_count = 0
    tool_result_count = 0
    main_conv_user = 0
    main_conv_assistant = 0

    # First pass: classify and count, no output
    try:
        for line_num, obj, err in load_jsonl(input_path):
            total_entries += 1
            if err is not None:
                parse_errors[err] += 1
                continue
            type_counts[classify_entry(obj)] += 1
            if isinstance(obj, dict):
                for k in obj.keys():
                    top_level_keys[k] += 1
                if is_sidechain(obj):
                    sidechain_count += 1
                if is_meta(obj):
                    meta_count += 1
                if "toolUseResult" in obj:
                    tool_result_count += 1
                if is_top_level_user_message(obj):
                    main_conv_user += 1
                if is_top_level_assistant_message(obj):
                    main_conv_assistant += 1
    except Exception as e:
        # Print only the exception class, not the message
        print(f"ERROR during scan: {type(e).__name__}")
        sys.exit(1)

    print(f"total entries: {total_entries}")
    print(f"parse errors: {dict(parse_errors)}")
    print(f"entry type counts: {dict(type_counts)}")
    print(f"sidechain entries: {sidechain_count}")
    print(f"meta entries: {meta_count}")
    print(f"tool result entries: {tool_result_count}")
    print(f"MAIN CONVERSATION user turns: {main_conv_user}")
    print(f"MAIN CONVERSATION assistant turns: {main_conv_assistant}")
    print(f"top-level keys seen: {sorted(top_level_keys.keys())}")

    if args.schema_only:
        return

    if not args.output:
        print("ERROR: --output is required unless --schema-only is set")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Second pass: write output
    user_written = 0
    assistant_written = 0
    total_chars = 0

    try:
        with open(output_path, "w", encoding="utf-8") as out:
            out.write("# Conversation Transcript\n\n")
            out.write(f"**Source**: `{input_path.name}`\n")
            out.write(f"**Exported**: {datetime.now().isoformat(timespec='seconds')}\n")
            out.write(f"**Total entries in source**: {total_entries}\n\n")
            out.write("---\n\n")

            for line_num, obj, err in load_jsonl(input_path):
                if err is not None or obj is None:
                    continue

                if is_top_level_user_message(obj):
                    msg = obj.get("message", obj) if isinstance(obj, dict) else {}
                    texts = extract_text_blocks(msg.get("content"))
                    if not texts:
                        continue
                    timestamp = get_timestamp(obj)
                    header = "## user"
                    if timestamp:
                        header += f" — {timestamp}"
                    out.write(f"{header}\n\n")
                    for t in texts:
                        out.write(t)
                        out.write("\n\n")
                        total_chars += len(t)
                    out.write("---\n\n")
                    user_written += 1

                elif is_top_level_assistant_message(obj):
                    msg = obj.get("message", obj) if isinstance(obj, dict) else {}
                    texts = extract_text_blocks(msg.get("content"))
                    if not texts:
                        continue
                    timestamp = get_timestamp(obj)
                    header = "## assistant"
                    if timestamp:
                        header += f" — {timestamp}"
                    out.write(f"{header}\n\n")
                    for t in texts:
                        out.write(t)
                        out.write("\n\n")
                        total_chars += len(t)
                    out.write("---\n\n")
                    assistant_written += 1

    except Exception as e:
        # Print only the class, not the message (which might contain content)
        print(f"ERROR during write: {type(e).__name__}")
        sys.exit(1)

    out_size = output_path.stat().st_size
    print(f"user messages written: {user_written}")
    print(f"assistant messages written: {assistant_written}")
    print(f"total text chars written: {total_chars}")
    print(f"output: {output_path}")
    print(f"output size: {out_size} bytes ({out_size / 1024 / 1024:.2f} MB)")
    print("DONE")


if __name__ == "__main__":
    main()
