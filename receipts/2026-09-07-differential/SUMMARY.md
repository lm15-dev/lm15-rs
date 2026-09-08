# 2026-09-07 — differential probe, reference vs port (port.md § Reviewing a port, step 5)

`python3 tools/differential.py --bodies receipts/2026-09-07-router-live --out report.json`

126 build_request comparisons (120 built, 6 refused) and 4 parse_response comparisons; 0 with differences.

Every probe is a canonical request the corpus does not contain, sent as
`build_request` (complete and stream) to both shims with the same key
and diffed field by field: method, URL, params, headers, body. The four
`parse_response` rows feed the live bodies of
`receipts/2026-09-07-router-live` (captured by the port) to both shims
and diff the canonical `Response`.

| Probe | Providers | Outcome |
|---|---|---|
| system_plus_tools_plus_sampling | anthropic, gemini, openai, openai_chat | built identically; refused identically on openai (UnsupportedFeatureError) |
| developer_message_and_forced_tool | anthropic, gemini, openai, openai_chat | built identically |
| required_one_allowed_tool_no_parallel | anthropic, gemini, openai, openai_chat | built identically; refused identically on gemini (UnsupportedFeatureError) |
| tool_choice_none_with_tools_declared | anthropic, gemini, openai, openai_chat | built identically |
| two_tool_calls_then_two_results_in_one_turn | anthropic, gemini, openai, openai_chat | built identically |
| image_url_and_text_in_two_user_turns | anthropic, gemini, openai, openai_chat | built identically |
| image_base64_with_detail_hint_in_extensions | anthropic, gemini, openai, openai_chat | built identically |
| json_schema_with_tools | anthropic, gemini, openai, openai_chat | built identically |
| json_object_mode | gemini, openai, openai_chat | built identically |
| reasoning_effort_with_summary_and_budget | anthropic, gemini | built identically |
| reasoning_effort_low_with_summary_detailed | openai | built identically |
| thinking_part_without_state_replays_as_text | anthropic, gemini, openai, openai_chat | built identically |
| signed_thinking_replayed_natively_on_anthropic | anthropic | built identically |
| gemini_tool_call_with_thought_signature_and_second_turn | gemini | built identically |
| openai_reasoning_item_replayed_then_tool_result | openai | built identically |
| cache_stable_with_long_retention_and_user_id | anthropic, openai | built identically |
| logprobs_zero_with_service_tier | gemini, openai, openai_chat | built identically |
| store_false_with_stop_string | gemini, openai | built identically; refused identically on openai (UnsupportedFeatureError) |
| top_k_and_max_tokens_stream | anthropic, gemini | built identically |
| document_part_in_user_message | anthropic, gemini, openai | built identically |
| extensions_passthrough | anthropic, gemini, openai, openai_chat | built identically |
| assistant_prefill_last | anthropic, openai_chat | built identically |

Refusals are the same class in both (`UnsupportedFeatureError` for
`stop` on the Responses dialect, per MAP-8; the harness protocol carries
the class name). Nothing here amends the corpus: it is port evidence.
