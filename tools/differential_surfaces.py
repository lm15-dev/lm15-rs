#!/usr/bin/env python3
"""Differential probe for modules 7–9 (files, batch, cache, generation,
video, live): inputs the corpus does NOT contain, through the reference
and this port, every difference reported (playbooks/port.md § Reviewing
a port, step 5). The companion of tools/differential.py for the request
side.

Both shims speak harness/PROTOCOL.md. For every probe and every provider
it names, this script asks both for the build op and diffs the wire
request(s); for the parse probes it feeds hand-mutated bodies (a corpus
body with a field dropped, a status word the fixtures do not pin, an
unknown key) and diffs the canonical result. A refusal is compared by
class and code. Same input, two implementations; a difference is a
finding, never "close enough".

    python3 tools/differential_surfaces.py [--out report.json]

Run from lm15-rs with the sibling lm15-python (its .venv) and a release
build of lm15-vet.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from differential import PYTHON_CWD, PYTHON_SHIM, RUST_SHIM, Shim, diff  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
USER = lambda text: {"role": "user", "parts": [{"type": "text", "text": text}]}  # noqa: E731
B64 = lambda s: base64.b64encode(s.encode() if isinstance(s, str) else s).decode()  # noqa: E731
WEATHER = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}
KEY = "test-key-123"
BOUNDARY = re.compile(rb"lm15-[0-9a-f]{32}")


def normalize(reply: dict) -> dict:
    """The result with the multipart boundary (the one legitimately random
    byte) normalized, or the refusal's class and code."""
    if reply.get("ok"):
        return {"ok": True, "result": scrub(reply["result"])}
    err = reply.get("error", {})
    return {"ok": False, "type": err.get("type"), "code": err.get("code")}


def scrub(value):
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k == "body_b64" and isinstance(v, str):
                raw = base64.b64decode(v)
                out[k] = B64(BOUNDARY.sub(b"lm15-BOUNDARY", raw))
            elif k == "headers" and isinstance(v, dict):
                out[k] = {hk.lower(): BOUNDARY.sub(b"lm15-BOUNDARY", hv.encode()).decode() for hk, hv in v.items()}
            else:
                out[k] = scrub(v)
        return out
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value


# ─── probes: (op, name, providers, fields) ──────────────────────────

FILES = ["openai", "anthropic", "gemini"]
BATCH = ["openai", "anthropic", "gemini"]
GEN = ["openai", "gemini", "xai"]
VIDEO = ["openai", "gemini", "xai"]
ALL_SURFACES = ["openai", "openai_chat", "anthropic", "gemini", "xai"]

PROBES: list[tuple[str, str, list[str], dict]] = []


def probe(op: str, name: str, providers: list[str], **fields) -> None:
    PROBES.append((op, name, providers, fields))


# Files — ids with reserved characters, media types with parameters, a
# filename with spaces and unicode, list paging edge values, the surface
# on a provider that has none.
probe("file_op_build", "upload_pdf_with_unicode_filename", FILES, file_op="upload",
      upload_request={"filename": "résumé 2026.pdf", "media_type": "application/pdf", "bytes_data": B64("%PDF-1.4 fake")})
probe("file_op_build", "upload_parameterized_media_type", FILES, file_op="upload",
      upload_request={"filename": "a.csv", "media_type": "text/csv; charset=utf-8", "bytes_data": B64("a,b\n1,2\n")})
probe("file_op_build", "upload_default_media_type", FILES, file_op="upload",
      upload_request={"filename": "blob.bin", "bytes_data": B64(bytes(range(256)))})
probe("file_op_build", "upload_with_extensions", FILES, file_op="upload",
      upload_request={"filename": "x.txt", "media_type": "text/plain", "bytes_data": B64("x"), "extensions": {"purpose": "assistants"}})
probe("file_op_build", "get_id_with_reserved_chars", FILES, file_op="get", file_id="file/with?odd=chars&x#1")
probe("file_op_build", "delete_id_with_space", FILES, file_op="delete", file_id="file 123")
probe("file_op_build", "download_plain", FILES, file_op="download", file_id="file-abc")
probe("file_op_build", "list_limit_1_no_cursor", FILES, file_op="list", limit=1, cursor=None)
probe("file_op_build", "list_with_cursor", FILES, file_op="list", limit=50, cursor="page-token/with=chars")
probe("file_op_build", "list_limit_zero", FILES, file_op="list", limit=0)
probe("file_op_build", "files_on_a_provider_without_the_surface", ["openai_chat", "xai"], file_op="list", limit=10)

# Batch — several entries, entries with tools/system/config, a label, an
# empty request list, list limits, result fetches from unusual status bodies.
REQS3 = [
    {"model": "m", "messages": [USER("one")]},
    {"model": "m", "system": "terse", "messages": [USER("two")], "tools": [WEATHER],
     "config": {"max_tokens": 32, "temperature": 0.1}},
    {"model": "m", "messages": [USER("three"), {"role": "assistant", "parts": [{"type": "text", "text": "3"}]}, USER("more")]},
]
probe("batch_op_build", "upload_three_entries_with_tools", BATCH, action="upload",
      batch_request={"model": "m", "requests": REQS3})
probe("batch_op_build", "submit_three_entries_with_tools", BATCH, action="submit",
      batch_request={"model": "m", "requests": REQS3},
      upload_body={"object": "file", "id": "file-up", "purpose": "batch", "filename": "input.jsonl", "bytes": 1, "created_at": 1, "status": "processed"})
probe("batch_op_build", "submit_labeled", BATCH, action="submit",
      batch_request={"model": "m", "requests": [REQS3[0]], "label": "nightly-eval"},
      upload_body={"object": "file", "id": "file-up", "purpose": "batch", "filename": "i.jsonl", "bytes": 1, "created_at": 1, "status": "processed"})
probe("batch_op_build", "submit_with_extensions", BATCH, action="submit",
      batch_request={"model": "m", "requests": [REQS3[0]], "extensions": {"completion_window": "24h", "note": 1}},
      upload_body={"object": "file", "id": "file-up", "purpose": "batch", "filename": "i.jsonl", "bytes": 1, "created_at": 1, "status": "processed"})
probe("batch_op_build", "status_odd_id", BATCH, action="status", batch_id="batch/odd id?x")
probe("batch_op_build", "cancel_plain", BATCH, action="cancel", batch_id="batch_1")
probe("batch_op_build", "list_limit_100", BATCH, action="list", limit=100)
probe("batch_op_build", "list_limit_1", BATCH, action="list", limit=1)
probe("batch_op_build", "result_fetches_openai_both_files", ["openai"], action="result_fetches",
      batch_id="batch_1", status_body={"id": "batch_1", "object": "batch", "status": "completed",
                                       "output_file_id": "file-out", "error_file_id": "file-err"})
probe("batch_op_build", "result_fetches_openai_no_error_file", ["openai"], action="result_fetches",
      batch_id="batch_1", status_body={"id": "batch_1", "object": "batch", "status": "completed",
                                       "output_file_id": "file-out", "error_file_id": None})
probe("batch_op_build", "result_fetches_anthropic_results_url", ["anthropic"], action="result_fetches",
      batch_id="msgbatch_1", status_body={"id": "msgbatch_1", "type": "message_batch", "processing_status": "ended",
                                          "results_url": "https://api.anthropic.com/v1/messages/batches/msgbatch_1/results"})
probe("batch_op_build", "result_fetches_gemini_inlined", ["gemini"], action="result_fetches",
      batch_id="batches/1", status_body={"name": "batches/1", "done": True, "response": {"inlinedResponses": {"inlinedResponses": []}}})
probe("batch_op_build", "batch_on_a_provider_without_the_surface", ["openai_chat", "xai"], action="list", limit=10)

# Cache — the Gemini tier and every provider without one.
PREFIX = {"model": "gemini-2.5-flash", "system": "You are a cache test.", "messages": [USER("Context " * 50)], "tools": [WEATHER]}
probe("cache_op_build", "create_system_tools_ttl_label", ["gemini"], cache_op="create", prefix_request=PREFIX, ttl_seconds=600, label="probe")
probe("cache_op_build", "create_no_ttl_no_label", ["gemini"], cache_op="create", prefix_request={"model": "gemini-2.5-flash", "messages": [USER("hello")]})
probe("cache_op_build", "create_multi_turn_prefix", ["gemini"], cache_op="create",
      prefix_request={"model": "gemini-2.5-flash", "messages": [USER("a"), {"role": "assistant", "parts": [{"type": "text", "text": "b"}]}, USER("c")]})
probe("cache_op_build", "get_odd_id", ["gemini"], cache_op="get", cache_id="cachedContents/abc-123")
probe("cache_op_build", "update_ttl_only", ["gemini"], cache_op="update", cache_id="cachedContents/abc", ttl_seconds=30)
probe("cache_op_build", "list_with_cursor", ["gemini"], cache_op="list", limit=5, cursor="tok")
probe("cache_op_build", "delete_plain", ["gemini"], cache_op="delete", cache_id="cachedContents/abc")
probe("cache_op_build", "cache_on_a_provider_without_the_tier", ["openai", "openai_chat", "anthropic", "xai"], cache_op="list", limit=5)

# Generation — image knobs, edits with each addressing mode, speech knobs,
# and the refusals (fields with no wire slot).
IMG_URL = {"type": "image", "url": "https://example.com/in.png", "media_type": "image/png"}
IMG_B64 = {"type": "image", "data": B64(b"\x89PNG fake"), "media_type": "image/png"}
IMG_FILE = {"type": "image", "file_id": "file-img-1", "media_type": "image/png"}
probe("generation_build", "image_size_only", GEN, kind="image", generation_request={"model": "m", "prompt": "a cat", "size": "1024x1024"})
probe("generation_build", "image_aspect_ratio_size", GEN, kind="image", generation_request={"model": "m", "prompt": "a cat", "size": "16:9"})
probe("generation_build", "image_with_extensions", GEN, kind="image", generation_request={"model": "m", "prompt": "a cat", "extensions": {"n": 2, "quality": "high"}})
probe("generation_build", "image_edit_from_url", GEN, kind="image", generation_request={"model": "m", "prompt": "make it blue", "images": [IMG_URL]})
probe("generation_build", "image_edit_from_bytes", GEN, kind="image", generation_request={"model": "m", "prompt": "make it blue", "images": [IMG_B64]})
probe("generation_build", "image_edit_from_file_id", GEN, kind="image", generation_request={"model": "m", "prompt": "make it blue", "images": [IMG_FILE]})
probe("generation_build", "image_edit_two_inputs", GEN, kind="image", generation_request={"model": "m", "prompt": "merge", "images": [IMG_B64, IMG_URL]})
probe("generation_build", "speech_voice_format", GEN, kind="speech", generation_request={"model": "m", "prompt": "Hello there.", "voice": "nova", "format": "wav"})
probe("generation_build", "speech_voice_only", GEN, kind="speech", generation_request={"model": "m", "prompt": "Hello.", "voice": "Kore"})
probe("generation_build", "speech_with_extensions", GEN, kind="speech", generation_request={"model": "m", "prompt": "Hi.", "extensions": {"speed": 1.2}})
probe("generation_build", "generation_on_a_provider_without_it", ["anthropic", "openai_chat"], kind="image", generation_request={"model": "m", "prompt": "a cat"})

# Video — duration, input frames, status ids, result fetches, list with and without model.
probe("video_op_build", "submit_with_seconds", VIDEO, action="submit", video_request={"model": "m", "prompt": "a ball", "seconds": 8})
probe("video_op_build", "submit_with_image_frame", VIDEO, action="submit", video_request={"model": "m", "prompt": "animate", "images": [IMG_B64]})
probe("video_op_build", "submit_with_extensions", VIDEO, action="submit", video_request={"model": "m", "prompt": "a ball", "extensions": {"size": "1280x720"}})
probe("video_op_build", "status_odd_id", VIDEO, action="status", video_id="video/odd id")
probe("video_op_build", "list_no_model", VIDEO, action="list", limit=5, model=None)
probe("video_op_build", "list_with_model", VIDEO, action="list", limit=5, model="veo-3.1")
probe("video_op_build", "result_fetch_openai", ["openai"], action="result_fetch", video_id="video_1",
      status_body={"id": "video_1", "object": "video", "status": "completed", "progress": 100, "created_at": 1})
probe("video_op_build", "result_fetch_gemini_uri", ["gemini"], action="result_fetch", video_id="operations/1",
      status_body={"name": "operations/1", "done": True, "response": {"generateVideoResponse": {"generatedSamples": [{"video": {"uri": "https://generativelanguage.googleapis.com/v1beta/files/abc:download?alt=media"}}]}}})
probe("video_op_build", "result_fetch_xai_url", ["xai"], action="result_fetch", video_id="v1",
      status_body={"status": "done", "video": {"url": "https://cdn.x.ai/v.mp4"}})
probe("video_op_build", "video_on_a_provider_without_it", ["anthropic", "openai_chat"], action="submit", video_request={"model": "m", "prompt": "a ball"})

# Live — configs and client events the transcripts do not pin together.
LIVE = ["openai", "gemini"]
probe("replay_live", "config_voice_formats_no_tools", LIVE,
      live_config={"model": "m", "voice": "alloy", "input_format": {"encoding": "pcm16", "sample_rate": 16000, "channels": 1},
                   "output_format": {"encoding": "pcm16", "sample_rate": 24000, "channels": 1}},
      client_events=[{"type": "text", "text": "hi"}], server_frames_b64=[])
probe("replay_live", "config_system_parts_and_tools", LIVE,
      live_config={"model": "m", "system": [{"type": "text", "text": "Be brief."}], "tools": [WEATHER]},
      client_events=[{"type": "turn", "parts": [{"type": "text", "text": "hello"}], "turn_complete": True}], server_frames_b64=[])
probe("replay_live", "audio_chunks_then_end", LIVE, live_config={"model": "m"},
      client_events=[{"type": "audio", "data": B64(b"\x00\x01"), "media_type": "audio/pcm;rate=16000"},
                     {"type": "audio", "data": B64(b"\x02\x03"), "media_type": "audio/pcm;rate=16000"},
                     {"type": "end_audio"}], server_frames_b64=[])
probe("replay_live", "image_event", LIVE, live_config={"model": "m"},
      client_events=[{"type": "image", "data": B64(b"\xff\xd8 fake"), "media_type": "image/jpeg"}], server_frames_b64=[])
probe("replay_live", "tool_result_then_interrupt", LIVE, live_config={"model": "m", "tools": [WEATHER]},
      client_events=[{"type": "tool_result", "id": "call_1", "content": [{"type": "text", "text": "sunny"}]}, {"type": "interrupt"}],
      server_frames_b64=[])
probe("replay_live", "turn_not_complete", LIVE, live_config={"model": "m"},
      client_events=[{"type": "turn", "parts": [{"type": "text", "text": "partial"}], "turn_complete": False}], server_frames_b64=[])
probe("replay_live", "config_extensions_passthrough", LIVE, live_config={"model": "m", "extensions": {"turn_detection": False, "custom": {"x": 1}}},
      client_events=[], server_frames_b64=[])

# ─── parse probes: mutated bodies ───────────────────────────────────

PARSE: list[tuple[str, str, str, dict]] = []


def parse(op: str, name: str, provider: str, **fields) -> None:
    PARSE.append((op, name, provider, fields))


def body(obj) -> str:
    return B64(json.dumps(obj))


# Files: missing optional fields, unknown status words, extra keys.
parse("file_op_parse", "openai_info_processing_unknown_status", "openai", kind="info", status=200,
      body_b64=body({"object": "file", "id": "file-1", "purpose": "user_data", "filename": "a.txt", "bytes": 3, "created_at": 1700000000, "status": "weird-new-word", "extra": {"k": 1}}))
parse("file_op_parse", "openai_info_missing_bytes_and_created", "openai", kind="info", status=200,
      body_b64=body({"object": "file", "id": "file-1", "filename": "a.txt"}))
parse("file_op_parse", "openai_page_empty", "openai", kind="page", status=200, body_b64=body({"object": "list", "data": [], "has_more": False}))
parse("file_op_parse", "openai_page_has_more_with_cursor", "openai", kind="page", status=200,
      body_b64=body({"object": "list", "data": [{"object": "file", "id": "f1", "filename": "x", "bytes": 1, "created_at": 1}], "has_more": True, "last_id": "f1", "first_id": "f1"}))
parse("file_op_parse", "anthropic_info_downloadable_false", "anthropic", kind="info", status=200,
      body_b64=body({"type": "file", "id": "file_1", "filename": "a.txt", "mime_type": "text/plain", "size_bytes": 3, "created_at": "2026-01-01T00:00:00Z", "downloadable": False}))
parse("file_op_parse", "gemini_info_state_failed", "gemini", kind="info", status=200,
      body_b64=body({"name": "files/abc", "displayName": "a.txt", "mimeType": "text/plain", "sizeBytes": "3", "createTime": "2026-01-01T00:00:00Z", "state": "FAILED", "error": {"message": "bad"}}))
parse("file_op_parse", "gemini_info_state_processing", "gemini", kind="info", status=200,
      body_b64=body({"name": "files/abc", "mimeType": "video/mp4", "sizeBytes": "1000", "state": "PROCESSING"}))
parse("file_op_parse", "gemini_page_next_token", "gemini", kind="page", status=200,
      body_b64=body({"files": [{"name": "files/1", "mimeType": "text/plain", "sizeBytes": "1", "state": "ACTIVE"}], "nextPageToken": "tok"}))
parse("file_op_parse", "openai_info_404_error", "openai", kind="info", status=404,
      body_b64=body({"error": {"message": "No such file", "type": "invalid_request_error", "code": None}}))

# Batch: status words outside the pinned set, missing counts, out-of-order entries.
parse("batch_op_parse", "openai_job_validating", "openai", kind="job", status=200,
      body_b64=body({"id": "batch_1", "object": "batch", "status": "validating", "created_at": 1700000000, "metadata": {"label": "x"}}))
parse("batch_op_parse", "openai_job_expired", "openai", kind="job", status=200,
      body_b64=body({"id": "batch_1", "object": "batch", "status": "expired", "created_at": 1700000000}))
parse("batch_op_parse", "openai_job_cancelling", "openai", kind="job", status=200,
      body_b64=body({"id": "batch_1", "object": "batch", "status": "cancelling", "created_at": 1700000000}))
parse("batch_op_parse", "openai_job_unknown_status", "openai", kind="job", status=200,
      body_b64=body({"id": "batch_1", "object": "batch", "status": "brand_new", "created_at": 1700000000}))
parse("batch_op_parse", "anthropic_job_canceling", "anthropic", kind="job", status=200,
      body_b64=body({"id": "msgbatch_1", "type": "message_batch", "processing_status": "canceling", "created_at": "2026-01-01T00:00:00Z",
                     "request_counts": {"processing": 1, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0}}))
parse("batch_op_parse", "gemini_job_pending_no_metadata", "gemini", kind="job", status=200,
      body_b64=body({"name": "batches/1", "done": False}))
parse("batch_op_parse", "gemini_job_error", "gemini", kind="job", status=200,
      body_b64=body({"name": "batches/1", "done": True, "error": {"code": 3, "message": "bad"}, "metadata": {"state": "BATCH_STATE_FAILED"}}))
parse("batch_op_parse", "openai_list_empty", "openai", kind="list", status=200, body_b64=body({"object": "list", "data": [], "has_more": False}))
parse("batch_op_parse", "openai_entries_out_of_order_with_error", "openai", kind="entries",
      status_body={"id": "batch_1", "object": "batch", "status": "completed", "output_file_id": "file-out", "error_file_id": "file-err"},
      fetched_b64=[
          B64("\n".join([
              json.dumps({"id": "b1", "custom_id": "1", "response": {"status_code": 200, "body": {"id": "resp_2", "object": "response", "model": "m", "status": "completed", "output": [{"type": "message", "id": "m2", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "two", "annotations": []}]}], "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}}}),
              json.dumps({"id": "b0", "custom_id": "0", "response": {"status_code": 200, "body": {"id": "resp_1", "object": "response", "model": "m", "status": "completed", "output": [{"type": "message", "id": "m1", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "one", "annotations": []}]}], "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}}}),
          ]) + "\n"),
          B64(json.dumps({"id": "b2", "custom_id": "2", "response": {"status_code": 429, "body": {"error": {"message": "slow down", "type": "rate_limit_error"}}}}) + "\n"),
      ])
parse("batch_op_parse", "anthropic_entries_expired_and_errored", "anthropic", kind="entries",
      status_body={"id": "msgbatch_1", "type": "message_batch", "processing_status": "ended", "results_url": "https://api.anthropic.com/v1/messages/batches/msgbatch_1/results"},
      fetched_b64=[B64("\n".join([
          json.dumps({"custom_id": "1", "result": {"type": "expired"}}),
          json.dumps({"custom_id": "0", "result": {"type": "errored", "error": {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}}}),
      ]) + "\n")])

# Video: statuses outside the pinned set, progress edge values, parts.
parse("video_op_parse", "openai_job_in_progress_progress_0", "openai", kind="job", status=200,
      body_b64=body({"id": "video_1", "object": "video", "status": "in_progress", "progress": 0, "created_at": 1700000000, "model": "sora-2"}))
parse("video_op_parse", "openai_job_failed_with_error", "openai", kind="job", status=200,
      body_b64=body({"id": "video_1", "object": "video", "status": "failed", "error": {"message": "moderation"}, "created_at": 1700000000}))
parse("video_op_parse", "openai_job_unknown_status", "openai", kind="job", status=200,
      body_b64=body({"id": "video_1", "object": "video", "status": "brand_new", "created_at": 1700000000}))
parse("video_op_parse", "gemini_job_not_done_no_metadata", "gemini", kind="job", status=200, body_b64=body({"name": "operations/1", "done": False}))
parse("video_op_parse", "gemini_job_done_error", "gemini", kind="job", status=200,
      body_b64=body({"name": "operations/1", "done": True, "error": {"code": 8, "message": "quota"}}))
parse("video_op_parse", "xai_job_pending_no_id_in_body", "xai", kind="job", status=200, video_id="v1", body_b64=body({"status": "pending"}))
parse("video_op_parse", "xai_job_unknown_status", "xai", kind="job", status=200, video_id="v1", body_b64=body({"status": "something_else"}))
parse("video_op_parse", "xai_part_from_status_body", "xai", kind="part", video_id="v1",
      status_body={"status": "done", "video": {"url": "https://cdn.x.ai/v.mp4", "duration": 5}})
parse("video_op_parse", "openai_part_from_fetched_bytes", "openai", kind="part", video_id="video_1",
      status_body={"id": "video_1", "object": "video", "status": "completed"},
      fetched_b64=B64(b"\x00\x00\x00\x18ftypmp42"), headers={"content-type": "video/mp4"})
parse("video_op_parse", "openai_list_empty", "openai", kind="list", status=200, body_b64=body({"object": "list", "data": []}))

# Generation: parse edge shapes.
parse("generation_parse", "openai_image_b64_two_images_with_revised_prompt", "openai", kind="image",
      generation_request={"model": "gpt-image-1", "prompt": "cats"}, status=200,
      body_b64=body({"created": 1700000000, "data": [{"b64_json": B64(b"\x89PNG one"), "revised_prompt": "two cats"}, {"b64_json": B64(b"\x89PNG two")}],
                     "usage": {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15}}))
parse("generation_parse", "openai_image_url_delivery", "openai", kind="image",
      generation_request={"model": "dall-e-3", "prompt": "cats"}, status=200,
      body_b64=body({"created": 1700000000, "data": [{"url": "https://cdn.example/a.png"}]}))
parse("generation_parse", "gemini_image_text_only_no_image", "gemini", kind="image",
      generation_request={"model": "gemini-2.5-flash-image", "prompt": "cats"}, status=200,
      body_b64=body({"candidates": [{"content": {"parts": [{"text": "I cannot draw that."}], "role": "model"}, "finishReason": "STOP"}],
                     "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 5, "totalTokenCount": 8}}))
parse("generation_parse", "openai_speech_wav_header", "openai", kind="speech",
      generation_request={"model": "gpt-4o-mini-tts", "prompt": "hi", "format": "wav"}, status=200,
      headers={"content-type": "audio/wav"}, body_b64=B64(b"RIFF\x00\x00\x00\x00WAVEfmt "))
parse("generation_parse", "openai_speech_no_content_type", "openai", kind="speech",
      generation_request={"model": "gpt-4o-mini-tts", "prompt": "hi"}, status=200, body_b64=B64(b"\xff\xf3 mp3"))
parse("generation_parse", "gemini_speech_inline_l16", "gemini", kind="speech",
      generation_request={"model": "gemini-2.5-flash-preview-tts", "prompt": "hi"}, status=200,
      body_b64=body({"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "audio/L16;codec=pcm;rate=24000", "data": B64(b"\x00\x01")}}], "role": "model"}}]}))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    py = Shim(PYTHON_SHIM, PYTHON_CWD)
    rs = Shim(RUST_SHIM, HERE)
    report = {"build": [], "parse": []}
    findings = 0

    for op, name, providers, fields in PROBES:
        for provider in providers:
            call = dict(fields, provider=provider)
            if op != "replay_live":
                call["api_key"] = KEY
            a = normalize(py.call(op, **call))
            b = normalize(rs.call(op, **call))
            problems = diff(a, b)
            report["build"].append({"op": op, "probe": name, "provider": provider, "python": a, "rust": b, "diff": problems})
            tag = "OK  " if not problems else "DIFF"
            refused = "" if a["ok"] else f" (python refuses: {a['type']})"
            print(f"{tag} {op:<17} {name:<46} {provider:<12}{refused}")
            for p in problems:
                print(f"       {p}")
            findings += bool(problems)

    for op, name, provider, fields in PARSE:
        call = dict(fields, provider=provider)
        a = normalize(py.call(op, **call))
        b = normalize(rs.call(op, **call))
        problems = diff(a, b)
        report["parse"].append({"op": op, "probe": name, "provider": provider, "python": a, "rust": b, "diff": problems})
        tag = "OK  " if not problems else "DIFF"
        refused = "" if a["ok"] else f" (python refuses: {a['type']})"
        print(f"{tag} {op:<17} {name:<46} {provider:<12}{refused}")
        for p in problems:
            print(f"       {p}")
        findings += bool(problems)

    py.close()
    rs.close()
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
    total = len(report["build"]) + len(report["parse"])
    print(f"\n{total} comparisons, {findings} with differences")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
