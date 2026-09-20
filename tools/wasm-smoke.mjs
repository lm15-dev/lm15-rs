// Offline smoke of the actual browser ABI; no network or provider credentials.
// node tools/wasm-smoke.mjs [path/to/lm15.wasm]
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const path = process.argv[2] ?? "target/wasm32-unknown-unknown/wasm/lm15.wasm";
const module = await WebAssembly.compile(await readFile(path));
assert.deepEqual(WebAssembly.Module.imports(module), [], "codec must not need host/native imports");
const { exports: wasm } = await WebAssembly.instantiate(module);
const encoder = new TextEncoder();
const decoder = new TextDecoder();
function call(op, input = {}) {
  const operation = encoder.encode(op);
  const body = encoder.encode(JSON.stringify(input));
  const opPtr = wasm.lm15_alloc(operation.length);
  const inPtr = wasm.lm15_alloc(body.length);
  let outPtr;
  let outLen;
  try {
    new Uint8Array(wasm.memory.buffer, opPtr, operation.length).set(operation);
    new Uint8Array(wasm.memory.buffer, inPtr, body.length).set(body);
    outPtr = wasm.lm15_call(opPtr, operation.length, inPtr, body.length);
    outLen = new DataView(wasm.memory.buffer).getUint32(outPtr, true);
    return JSON.parse(decoder.decode(new Uint8Array(wasm.memory.buffer, outPtr + 4, outLen)));
  } finally {
    wasm.lm15_free(opPtr, operation.length);
    wasm.lm15_free(inPtr, body.length);
    if (outPtr !== undefined) wasm.lm15_free(outPtr, outLen + 4);
  }
}
function ok(op, input) {
  const reply = call(op, input);
  assert.equal(reply.error, undefined, `${op}: ${JSON.stringify(reply)}`);
  return reply;
}
const request = (model, config = {}) => ({
  model,
  messages: [{ role: "user", parts: [{ type: "text", text: "hello" }] }],
  config,
});
const envelope = (provider, model, config = {}) => ({
  provider, api_key: "offline-placeholder", canonical_request: request(model, config),
});
assert.equal(ok("version").language, "rust");
const providers = ok("providers").providers;
for (const provider of ["openai", "openai-chat", "anthropic", "gemini", "typesafe"]) {
  assert(providers.includes(provider), provider);
}
for (const [provider, model] of [["openai", "gpt-test"], ["openai-chat", "gpt-test"], ["anthropic", "claude-test"], ["gemini", "gemini-test"]]) {
  const built = ok("build_request", envelope(provider, model));
  assert.equal(built.method, "POST");
  assert(built.url.startsWith("https://"));
  assert(built.body);
}
const openai = envelope("openai", "gpt-test", { top_k: 3 });
const completed = { id: "r", model: "gpt-test", status: "completed", output: [{ type: "message", role: "assistant", content: [{ type: "output_text", text: "hello" }] }] };
const raw = ok("parse_response", { ...openai, body: completed }).canonical_response;
assert(!raw.adaptations?.length);
const prepared = ok("parse_response", { ...openai, body: completed, apply_request: true }).canonical_response;
assert(prepared.adaptations.some(a => a.field === "config.top_k"));
assert.deepEqual(prepared.message, raw.message);
const strict = call("parse_response", { ...openai, adaptations: "refuse", body: completed, apply_request: true });
assert.equal(strict.error.feature, "config.top_k");

const judge = envelope("typesafe", "jev-latest", {
  probabilities: "required",
  response_format: { type: "json_schema", schema: { type: "object", properties: { ok: { type: "boolean", description: "Is it fine?" } }, required: ["ok"], additionalProperties: false } },
});
judge.canonical_request.messages[0].parts = [{ type: "data", value: { note: "fine", empty: null } }];
const judgedWire = ok("build_request", judge);
assert.deepEqual(judgedWire.body.state, { note: "fine", empty: null });
assert.equal(judgedWire.body.questions.ok.type, "noul");
const judged = ok("parse_response", { ...judge, apply_request: true, headers: [["x-typesafe-request-id", "jev-offline"]], body: { answers: { ok: { type: "noul", noul: 0.75 } } } }).canonical_response;
assert.equal(judged.id, "jev-offline");
assert.equal(judged.message.parts[0].type, "data");
assert.equal(judged.message.parts[0].value.ok, true);
assert.equal(judged.message.parts[0].probabilities.ok.true, 0.75);

const stopped = envelope("openai", "gpt-test", { stop: ["STOP"] });
assert.equal(ok("build_request", stopped).requires_stream, true);
const { handle } = ok("stream_open", stopped);
const delta = text => `event: response.output_text.delta\ndata: ${JSON.stringify({ type: "response.output_text.delta", output_index: 0, content_index: 0, delta: text })}\n\n`;
const first = ok("stream_feed", { handle, body: delta("caféST") });
assert.equal(first.close_source, false);
const cut = ok("stream_feed", { handle, body: delta("OP ignored") + "event: invalid\ndata: not-json\n\n" });
assert.equal(cut.close_source, true);
const final = ok("stream_close", { handle }).canonical_response;
assert.equal(final.message.parts[0].text, "café");
assert(final.adaptations.some(a => a.field === "config.stop"));
assert(call("stream_feed", { handle, body: "" }).error);

const headers = [["Content-Type", "text/event-stream"], ["x-request-id", "offline-stream"], ["retry-after-ms", "1250"], ["x-ratelimit-remaining-requests", "-1"], ["x-ratelimit-remaining-requests", "0"]];
const broken = ok("stream_open", { ...openai, headers }).handle;
const fault = call("stream_feed", { handle: broken, body: "data: not-json\n\n" }).error;
assert.equal(fault.name, "ProviderError");
assert.equal(fault.status, undefined);
assert.equal(fault.http_response.request_id, "offline-stream");
assert.equal(fault.http_response.retry_after, 1.25);
assert.deepEqual(fault.http_response.rate_limit_headers["x-ratelimit-remaining-requests"], ["-1", "0"]);
ok("stream_abort", { handle: broken });
const bad = call("parse_response", { ...openai, body: "not-json", headers: [["content-type", "text/html"], ["apim-request-id", "offline-http"]] }).error;
assert.equal(bad.status, 200);
assert.equal(bad.http_response.request_id, "offline-http");
assert.equal(bad.body_excerpt, "not-json");
assert.equal(call("parse_response", { ...openai, body: completed, body_encoding: "gzip" }).error.name, "TransportError");
console.log(`PASS ${path}: import-free ABI, provider builds, prepared parse, Jev judgments, stop/close_source, diagnostics, decoded-byte guard`);
