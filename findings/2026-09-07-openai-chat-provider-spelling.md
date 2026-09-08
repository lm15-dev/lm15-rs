# Finding: `goldens/openai_chat/models.json` pins the provider as `openai_chat`; the canonical id is `openai-chat`

Found by: `harness/check.py --shim rust --direction models` at pin
`4ebdf7d059aabf7799aa89af3055c623fdab630a` — 33 pass, 1 fail:
`openai_chat.models[parse]`, `$.models[0].provider` expected `"openai_chat"`,
actual `"openai-chat"`.

## Evidence

- `spec/support-matrix.json` names the provider `openai-chat`.
  `lm15-python/lm15/registry.py:295` registers it as `"openai-chat"` with
  dialect `"openai-chat"`; `canonical_provider` maps `_` to `-`.
- `lm15-python/lm15/providers/openai_chat.py:206`:
  `provider: str = field(default="openai_chat", init=False)` — the adapter's
  self-name predates the registry and was never moved.
- The other chat-dialect goldens pin the canonical spelling:
  `goldens/meta-chat/models.json` → `meta-chat`, `goldens/azure-chat` →
  `azure-chat`, `goldens/bedrock-mantle-chat` → `bedrock-mantle-chat`.
  `openai_chat` is the one golden carrying an underscore provider value.
- In the reference itself the two spellings meet: `LMRouter().resolve(
  "openai_chat:m").provider == "openai-chat"` while
  `router.lm("openai_chat:m").provider == "openai_chat"`, so a
  `ModelInfo.provider` from `list_models()` does not equal the
  `Resolution.provider` that produced the adapter.

## Proposal (a `changes/` entry for the contract; not applied by the port)

Pin `openai-chat` in `goldens/openai_chat/models.json` and align
`OpenAIChatLM.provider` in the reference with its registry id. No other
fixture pins the underscore value (the error cases for this dialect are
keyed by the bound providers, which are hyphenated). Until decided, this
port answers the canonical `openai-chat` everywhere (`ProviderLM::provider`,
`ErrorMeta.provider`, `ModelInfo.provider`, `Resolution.provider`) and the
one case stays red, stated in the README.
