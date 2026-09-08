# 2026-09-08 — modules 3b, 7, 8, 9 live

Two env-gated examples, one run each, keys from the environment.

## `cargo run --example surfaces_smoke`

| Surface | Provider | Result |
|---|---|---|
| files: upload (multipart/form-data), wait_ready, list, delete | openai | `file-AWC8TzqcvRY5VXkiZPuvtL`, 45 bytes, ready, listed, deleted |
| files: upload, wait_ready, list, delete | anthropic | `file_01Dv34isHbgTfwDwAcGZHnW5`, 45 bytes, ready, listed, deleted |
| files: upload (multipart/related on the `/upload` host), wait_ready, list, delete | gemini | `https://generativelanguage.googleapis.com/v1beta/files/c2d6uqcgkusg`, 45 bytes, ready, listed, deleted |
| cache: create (ttl 300 s, label), get, update (ttl 600 s), delete | gemini | `cachedContents/4i6g3xt6nfaz5uo84e5w99jzj34ca2e5u2vk5exh`, 1808 tokens, the expiry moved from 12:01:14Z to 12:06:14Z |
| speech: `gpt-4o-mini-tts`, voice `alloy` | openai | `audio/mpeg`, 10752 base64 chars |

## `cargo run --example live_session`

| Provider | Model | Result |
|---|---|---|
| openai | gpt-realtime-mini | one text turn over wss: `Live hello.`, 5 output tokens, 2.5 s |
| gemini | gemini-3.1-flash-live-preview | one text turn over wss (audio-native, transcript): `live hello`, 50 output tokens, 2.8 s |

Not exercised live (cost, or no account): batch (jobs take up to 24 h;
the harness pins the recorded lifecycles), image and video generation
(the harness pins the recorded bodies), the cloud credential chains
(no AWS / Azure / GCP account on this machine; the harness pins the
token vectors and the eleven doctor cases).
