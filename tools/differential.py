#!/usr/bin/env python3
"""Differential probe: requests the corpus does NOT contain, through the
reference and this port, every difference reported (playbooks/port.md
§ Reviewing a port, step 5).

Both shims speak harness/PROTOCOL.md on stdin/stdout. For every probe and
every provider it names, this script asks both for `build_request` and
diffs the wire request (method, url, params, headers, body). With
`--bodies <dir>` it also feeds each recorded live body (a receipt written
by `examples/live_smoke.rs`) to both shims' `parse_response` and diffs the
canonical `Response`. Same input, two implementations; a difference is a
finding, never "close enough".

    python3 tools/differential.py [--bodies receipts/<folder>] [--out report.json]

Run from lm15-rs with the sibling lm15-python (its .venv) and a release
build of lm15-vet.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
PYTHON_SHIM = [str(HERE.parent / "lm15-python/.venv/bin/python"), "-m", "lm15.vet"]
PYTHON_CWD = HERE.parent / "lm15-python"
RUST_SHIM = [str(HERE / "target/release/lm15-vet")]

USER = lambda text: {"role": "user", "parts": [{"type": "text", "text": text}]}  # noqa: E731
WEATHER = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}
ADD = {
    "type": "function",
    "name": "add",
    "description": "Add two numbers",
    "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}},
}
ALL = ["openai", "openai_chat", "anthropic", "gemini"]

# Requests outside the corpus: combinations of fields and part kinds the
# fixtures do not pin together. Each is (name, providers, canonical request).
PROBES: list[tuple[str, list[str], dict]] = [
    (
        "system_plus_tools_plus_sampling",
        ALL,
        {
            "model": "m",
            "system": "You are terse.",
            "messages": [USER("Weather in Montreal?")],
            "tools": [WEATHER, ADD],
            "config": {"max_tokens": 64, "temperature": 0.3, "top_p": 0.9, "stop": ["END"]},
        },
    ),
    (
        "developer_message_and_forced_tool",
        ALL,
        {
            "model": "m",
            "messages": [{"role": "developer", "parts": [{"type": "text", "text": "Always call a tool."}]}, USER("hi")],
            "tools": [WEATHER],
            "config": {"tool_choice": {"mode": "required"}},
        },
    ),
    (
        "required_one_allowed_tool_no_parallel",
        ALL,
        {
            "model": "m",
            "messages": [USER("Add 2 and 3.")],
            "tools": [WEATHER, ADD],
            "config": {"tool_choice": {"mode": "required", "allowed": ["add"], "parallel": False}},
        },
    ),
    (
        "tool_choice_none_with_tools_declared",
        ALL,
        {
            "model": "m",
            "messages": [USER("Just chat.")],
            "tools": [WEATHER],
            "config": {"tool_choice": {"mode": "none"}},
        },
    ),
    (
        "two_tool_calls_then_two_results_in_one_turn",
        ALL,
        {
            "model": "m",
            "messages": [
                USER("Weather in Montreal and Paris?"),
                {
                    "role": "assistant",
                    "parts": [
                        {"type": "text", "text": "Checking both."},
                        {"type": "tool_call", "id": "c1", "name": "get_weather", "input": {"city": "Montreal"}},
                        {"type": "tool_call", "id": "c2", "name": "get_weather", "input": {"city": "Paris"}},
                    ],
                },
                {
                    "role": "tool",
                    "parts": [
                        {"type": "tool_result", "id": "c1", "content": [{"type": "text", "text": "12 C"}]},
                        {"type": "tool_result", "id": "c2", "content": [{"type": "text", "text": "18 C"}], "is_error": True},
                    ],
                },
            ],
            "tools": [WEATHER],
        },
    ),
    (
        "image_url_and_text_in_two_user_turns",
        ALL,
        {
            "model": "m",
            "messages": [
                USER("First."),
                {"role": "assistant", "parts": [{"type": "text", "text": "Yes?"}]},
                {
                    "role": "user",
                    "parts": [
                        {"type": "text", "text": "Describe:"},
                        {"type": "image", "media_type": "image/png", "url": "https://example.com/a.png"},
                        {"type": "text", "text": "in five words."},
                    ],
                },
            ],
            "config": {"max_tokens": 32},
        },
    ),
    (
        "image_base64_with_detail_hint_in_extensions",
        ALL,
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "parts": [
                        {"type": "image", "media_type": "image/jpeg", "data": "/9j/4AAQSkZJRg=="},
                        {"type": "text", "text": "What is this?"},
                    ],
                }
            ],
        },
    ),
    (
        "json_schema_with_tools",
        ALL,
        {
            "model": "m",
            "messages": [USER("Cookie recipe.")],
            "tools": [ADD],
            "config": {
                "response_format": {
                    "type": "json_schema",
                    "name": "recipe",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                        "additionalProperties": False,
                    },
                },
                "max_tokens": 200,
            },
        },
    ),
    (
        "json_object_mode",
        ["openai", "openai_chat", "gemini"],
        {"model": "m", "messages": [USER("Give JSON.")], "config": {"response_format": {"type": "json_object"}}},
    ),
    (
        "reasoning_effort_with_summary_and_budget",
        ["anthropic", "gemini"],
        {
            "model": "m",
            "messages": [USER("143 * 27?")],
            "config": {"reasoning": {"effort": "high", "thinking_budget": 2048, "summary": "auto"}, "max_tokens": 4096},
        },
    ),
    (
        "reasoning_effort_low_with_summary_detailed",
        ["openai"],
        {"model": "m", "messages": [USER("143 * 27?")], "config": {"reasoning": {"effort": "low", "summary": "detailed"}}},
    ),
    (
        "thinking_part_without_state_replays_as_text",
        ALL,
        {
            "model": "m",
            "messages": [
                USER("Weather?"),
                {
                    "role": "assistant",
                    "parts": [
                        {"type": "thinking", "text": "I should check."},
                        {"type": "tool_call", "id": "c1", "name": "get_weather", "input": {"city": "Oslo"}},
                    ],
                },
                {"role": "tool", "parts": [{"type": "tool_result", "id": "c1", "content": [{"type": "text", "text": "3 C"}]}]},
            ],
            "tools": [WEATHER],
        },
    ),
    (
        "signed_thinking_replayed_natively_on_anthropic",
        ["anthropic"],
        {
            "model": "m",
            "messages": [
                USER("Weather?"),
                {
                    "role": "assistant",
                    "parts": [
                        {
                            "type": "thinking",
                            "text": "I should check.",
                            "continuation": [
                                {"provider": "anthropic", "kind": "thinking_signature", "data": {"signature": "sig-1"}}
                            ],
                        },
                        {"type": "tool_call", "id": "c1", "name": "get_weather", "input": {"city": "Oslo"}},
                    ],
                },
                {"role": "tool", "parts": [{"type": "tool_result", "id": "c1", "content": [{"type": "text", "text": "3 C"}]}]},
            ],
            "tools": [WEATHER],
            "config": {"reasoning": {"effort": "medium"}, "max_tokens": 4096},
        },
    ),
    (
        "gemini_tool_call_with_thought_signature_and_second_turn",
        ["gemini"],
        {
            "model": "m",
            "messages": [
                USER("Weather?"),
                {
                    "role": "assistant",
                    "parts": [
                        {
                            "type": "tool_call",
                            "id": "c1",
                            "name": "get_weather",
                            "input": {"city": "Oslo"},
                            "continuation": [{"provider": "gemini", "kind": "thought_signature", "data": {"value": "c2ln"}}],
                        }
                    ],
                },
                {"role": "tool", "parts": [{"type": "tool_result", "id": "c1", "content": [{"type": "text", "text": "3 C"}]}]},
                {"role": "assistant", "parts": [{"type": "text", "text": "It is 3 C."}]},
                USER("And Paris?"),
            ],
            "tools": [WEATHER],
        },
    ),
    (
        "openai_reasoning_item_replayed_then_tool_result",
        ["openai"],
        {
            "model": "m",
            "messages": [
                USER("Weather?"),
                {
                    "role": "assistant",
                    "parts": [
                        {
                            "type": "thinking",
                            "text": "",
                            "continuation": [
                                {"provider": "openai", "kind": "reasoning_item", "data": {"id": "rs_1", "encrypted_content": "enc"}}
                            ],
                        },
                        {"type": "tool_call", "id": "c1", "name": "get_weather", "input": {"city": "Oslo"}},
                    ],
                },
                {"role": "tool", "parts": [{"type": "tool_result", "id": "c1", "content": [{"type": "text", "text": "3 C"}]}]},
            ],
            "tools": [WEATHER],
            "config": {"reasoning": {"effort": "low"}},
        },
    ),
    (
        "cache_stable_with_long_retention_and_user_id",
        ["openai", "anthropic"],
        {
            "model": "m",
            "system": "Long system prompt " * 10,
            "messages": [USER("hi")],
            "config": {"cache": {"mode": "auto", "prefix": "stable", "retention": "long"}, "user_id": "u-1"},
        },
    ),
    (
        "logprobs_zero_with_service_tier",
        ["openai", "openai_chat", "gemini"],
        {"model": "m", "messages": [USER("hi")], "config": {"logprobs": 0, "service_tier": "default"}},
    ),
    (
        "store_false_with_stop_string",
        ["openai", "gemini"],
        {"model": "m", "messages": [USER("hi")], "config": {"store": False, "stop": "STOP"}},
    ),
    (
        "top_k_and_max_tokens_stream",
        ["anthropic", "gemini"],
        {"model": "m", "messages": [USER("hi")], "config": {"top_k": 40, "max_tokens": 10}},
    ),
    (
        "document_part_in_user_message",
        ["anthropic", "gemini", "openai"],
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "parts": [
                        {"type": "document", "media_type": "application/pdf", "data": "JVBERi0xLjQK", "name": "a.pdf"},
                        {"type": "text", "text": "Summarize."},
                    ],
                }
            ],
        },
    ),
    (
        "extensions_passthrough",
        ALL,
        {"model": "m", "messages": [USER("hi")], "config": {"extensions": {"x_custom": {"k": 1}}}},
    ),
    (
        "assistant_prefill_last",
        ["anthropic", "openai_chat"],
        {
            "model": "m",
            "messages": [USER("Answer in JSON."), {"role": "assistant", "parts": [{"type": "text", "text": "{"}]}],
        },
    ),
]

MODELS = {"openai": "gpt-4.1-mini", "openai_chat": "gpt-4.1-mini", "anthropic": "claude-haiku-4-5", "gemini": "gemini-2.5-flash"}


class Shim:
    def __init__(self, argv: list[str], cwd: Path):
        self.proc = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.n = 0

    def call(self, op: str, **fields) -> dict:
        self.n += 1
        line = json.dumps({"op": op, "id": str(self.n), **fields})
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        out = self.proc.stdout.readline()
        if not out:
            raise RuntimeError(f"shim {self.proc.args} died on {op}")
        return json.loads(out)

    def close(self) -> None:
        assert self.proc.stdin
        self.proc.stdin.close()
        self.proc.wait(timeout=30)


def normalize(reply: dict) -> dict:
    """What we compare: the result, or the refusal's class and code."""
    if reply.get("ok"):
        result = reply["result"]
        if "headers" in result:
            result = dict(result)
            # The transport adds these; the wire request the dialect built is what is compared.
            result["headers"] = {k.lower(): v for k, v in result["headers"].items()}
        return {"ok": True, "result": result}
    err = reply.get("error", {})
    return {"ok": False, "type": err.get("type"), "code": err.get("code")}


def diff(a, b, path="") -> list[str]:
    if type(a) is not type(b):
        return [f"{path or '$'}: python {json.dumps(a)[:200]} vs rust {json.dumps(b)[:200]}"]
    if isinstance(a, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            if key not in a:
                out.append(f"{path}/{key}: only in rust: {json.dumps(b[key])[:200]}")
            elif key not in b:
                out.append(f"{path}/{key}: only in python: {json.dumps(a[key])[:200]}")
            else:
                out.extend(diff(a[key], b[key], f"{path}/{key}"))
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(diff(x, y, f"{path}[{i}]"))
        return out
    if a != b:
        return [f"{path or '$'}: python {json.dumps(a)[:200]} vs rust {json.dumps(b)[:200]}"]
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", type=Path, help="a live_smoke receipts folder: parse_response through both")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    py = Shim(PYTHON_SHIM, PYTHON_CWD)
    rs = Shim(RUST_SHIM, HERE)
    report = {"build_request": [], "parse_response": []}
    findings = 0

    for name, providers, request in PROBES:
        for provider in providers:
            for stream in (False, True):
                req = dict(request, model=MODELS[provider])
                fields = {"provider": provider, "canonical_request": req, "stream": stream, "api_key": "test-key-123"}
                a = normalize(py.call("build_request", **fields))
                b = normalize(rs.call("build_request", **fields))
                problems = diff(a, b)
                entry = {"probe": name, "provider": provider, "stream": stream, "python": a, "rust": b, "diff": problems}
                report["build_request"].append(entry)
                tag = "OK  " if not problems else "DIFF"
                refused = "" if a["ok"] else f" (python refuses: {a['type']})"
                print(f"{tag} build_request {name:<48} {provider:<12} stream={str(stream):<5}{refused}")
                for p in problems:
                    print(f"       {p}")
                findings += bool(problems)

    if args.bodies:
        for path in sorted(args.bodies.glob("*-complete.json")):
            receipt = json.loads(path.read_text())
            provider = path.name.split("-complete")[0]
            shim_provider = {"groq": "openai_chat"}.get(provider, provider)
            body = receipt["body"]
            body_bytes = json.dumps(body).encode() if not isinstance(body, str) else body.encode()
            # Gemini names the model in the URL, not the body.
            sent = receipt["sent"]
            model = sent["body"].get("model") or sent["url"].rsplit("/models/", 1)[1].split(":")[0]
            request = {"model": model, "messages": [USER("x")]}
            fields = {
                "provider": shim_provider,
                "canonical_request": request,
                "status": receipt["status"],
                "body_b64": base64.b64encode(body_bytes).decode(),
                "api_key": "test-key-123",
            }
            if provider == "groq":
                fields["base_url"] = "https://api.groq.com/openai/v1"
            a = normalize(py.call("parse_response", **fields))
            b = normalize(rs.call("parse_response", **fields))
            problems = diff(a, b)
            report["parse_response"].append({"receipt": path.name, "python": a, "rust": b, "diff": problems})
            print(f"{'OK  ' if not problems else 'DIFF'} parse_response {path.name}")
            for p in problems:
                print(f"       {p}")
            findings += bool(problems)

    py.close()
    rs.close()
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
    total = len(report["build_request"]) + len(report["parse_response"])
    print(f"\n{total} comparisons, {findings} with differences")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
