# Differential probe, modules 7–9 — 2026-09-08

`tools/differential_surfaces.py` at lm15-python 228775a / lm15-rs (this
commit), contract 6880e17: 177 comparisons of inputs the corpus does not
contain (files, batch, cache, generation, video, live; build and parse),
through both shims. 7 with differences, in two clusters, neither a
port bug:

1. **`get_id_with_reserved_chars` / `status_odd_id` (6):** both
   implementations interpolate a file / batch id raw into the URL path
   (`.../files/{file_id}`); the difference is the reference's vet shim
   splitting the built URL at `?` into `params` for display. Same bytes,
   different rendering. The underlying gap — no percent-encoding of a
   path segment — is shared with the reference and filed as a contract
   finding: `findings/2026-09-08-id-path-escaping.md`.
2. **`openai_entries_out_of_order_with_error` (1):** the reference appends
   its "To fix:" guidance to a batch entry's `RateLimitError` message;
   this port carries the provider's message only. Messages are not
   pinned (spec/vocabularies.md); stated deviation in the README.

Every refusal (a surface a provider lacks, a knob with no wire slot)
matched by class and code on both sides; every live-codec probe, every
image-edit addressing mode, every mutated status word matched exactly.

| Op | Probe | Providers | Outcome |
|---|---|---|---|
| `file_op_build` | upload_pdf_with_unicode_filename | openai, anthropic, gemini | identical |
| `file_op_build` | upload_parameterized_media_type | openai, anthropic, gemini | identical |
| `file_op_build` | upload_default_media_type | openai, anthropic, gemini | identical |
| `file_op_build` | upload_with_extensions | openai, anthropic, gemini | identical |
| `file_op_build` | get_id_with_reserved_chars | openai, anthropic, gemini | 3 difference(s) |
| `file_op_build` | delete_id_with_space | openai, anthropic, gemini | identical |
| `file_op_build` | download_plain | openai, anthropic, gemini | identical |
| `file_op_build` | list_limit_1_no_cursor | openai, anthropic, gemini | identical |
| `file_op_build` | list_with_cursor | openai, anthropic, gemini | identical |
| `file_op_build` | list_limit_zero | openai, anthropic, gemini | identical |
| `file_op_build` | files_on_a_provider_without_the_surface | openai_chat, xai | identical; 2 refused on both |
| `batch_op_build` | upload_three_entries_with_tools | openai, anthropic, gemini | identical |
| `batch_op_build` | submit_three_entries_with_tools | openai, anthropic, gemini | identical |
| `batch_op_build` | submit_labeled | openai, anthropic, gemini | identical; 1 refused on both |
| `batch_op_build` | submit_with_extensions | openai, anthropic, gemini | identical |
| `batch_op_build` | status_odd_id | openai, anthropic, gemini | 3 difference(s) |
| `batch_op_build` | cancel_plain | openai, anthropic, gemini | identical |
| `batch_op_build` | list_limit_100 | openai, anthropic, gemini | identical |
| `batch_op_build` | list_limit_1 | openai, anthropic, gemini | identical |
| `batch_op_build` | result_fetches_openai_both_files | openai | identical |
| `batch_op_build` | result_fetches_openai_no_error_file | openai | identical |
| `batch_op_build` | result_fetches_anthropic_results_url | anthropic | identical |
| `batch_op_build` | result_fetches_gemini_inlined | gemini | identical |
| `batch_op_build` | batch_on_a_provider_without_the_surface | openai_chat, xai | identical; 2 refused on both |
| `cache_op_build` | create_system_tools_ttl_label | gemini | identical |
| `cache_op_build` | create_no_ttl_no_label | gemini | identical |
| `cache_op_build` | create_multi_turn_prefix | gemini | identical |
| `cache_op_build` | get_odd_id | gemini | identical |
| `cache_op_build` | update_ttl_only | gemini | identical |
| `cache_op_build` | list_with_cursor | gemini | identical |
| `cache_op_build` | delete_plain | gemini | identical |
| `cache_op_build` | cache_on_a_provider_without_the_tier | openai, openai_chat, anthropic, xai | identical; 4 refused on both |
| `generation_build` | image_size_only | openai, gemini, xai | identical; 1 refused on both |
| `generation_build` | image_aspect_ratio_size | openai, gemini, xai | identical; 1 refused on both |
| `generation_build` | image_with_extensions | openai, gemini, xai | identical |
| `generation_build` | image_edit_from_url | openai, gemini, xai | identical; 1 refused on both |
| `generation_build` | image_edit_from_bytes | openai, gemini, xai | identical |
| `generation_build` | image_edit_from_file_id | openai, gemini, xai | identical; 1 refused on both |
| `generation_build` | image_edit_two_inputs | openai, gemini, xai | identical; 2 refused on both |
| `generation_build` | speech_voice_format | openai, gemini, xai | identical; 2 refused on both |
| `generation_build` | speech_voice_only | openai, gemini, xai | identical; 1 refused on both |
| `generation_build` | speech_with_extensions | openai, gemini, xai | identical; 1 refused on both |
| `generation_build` | generation_on_a_provider_without_it | anthropic, openai_chat | identical; 2 refused on both |
| `video_op_build` | submit_with_seconds | openai, gemini, xai | identical; 1 refused on both |
| `video_op_build` | submit_with_image_frame | openai, gemini, xai | identical; 3 refused on both |
| `video_op_build` | submit_with_extensions | openai, gemini, xai | identical |
| `video_op_build` | status_odd_id | openai, gemini, xai | identical |
| `video_op_build` | list_no_model | openai, gemini, xai | identical; 2 refused on both |
| `video_op_build` | list_with_model | openai, gemini, xai | identical; 1 refused on both |
| `video_op_build` | result_fetch_openai | openai | identical |
| `video_op_build` | result_fetch_gemini_uri | gemini | identical |
| `video_op_build` | result_fetch_xai_url | xai | identical |
| `video_op_build` | video_on_a_provider_without_it | anthropic, openai_chat | identical; 2 refused on both |
| `replay_live` | config_voice_formats_no_tools | openai, gemini | identical |
| `replay_live` | config_system_parts_and_tools | openai, gemini | identical |
| `replay_live` | audio_chunks_then_end | openai, gemini | identical |
| `replay_live` | image_event | openai, gemini | identical |
| `replay_live` | tool_result_then_interrupt | openai, gemini | identical |
| `replay_live` | turn_not_complete | openai, gemini | identical |
| `replay_live` | config_extensions_passthrough | openai, gemini | identical |
| `file_op_parse` | openai_info_processing_unknown_status | openai | identical |
| `file_op_parse` | openai_info_missing_bytes_and_created | openai | identical |
| `file_op_parse` | openai_page_empty | openai | identical |
| `file_op_parse` | openai_page_has_more_with_cursor | openai | identical |
| `file_op_parse` | anthropic_info_downloadable_false | anthropic | identical |
| `file_op_parse` | gemini_info_state_failed | gemini | identical |
| `file_op_parse` | gemini_info_state_processing | gemini | identical |
| `file_op_parse` | gemini_page_next_token | gemini | identical |
| `file_op_parse` | openai_info_404_error | openai | identical; 1 refused on both |
| `batch_op_parse` | openai_job_validating | openai | identical |
| `batch_op_parse` | openai_job_expired | openai | identical |
| `batch_op_parse` | openai_job_cancelling | openai | identical |
| `batch_op_parse` | openai_job_unknown_status | openai | identical |
| `batch_op_parse` | anthropic_job_canceling | anthropic | identical |
| `batch_op_parse` | gemini_job_pending_no_metadata | gemini | identical |
| `batch_op_parse` | gemini_job_error | gemini | identical |
| `batch_op_parse` | openai_list_empty | openai | identical |
| `batch_op_parse` | openai_entries_out_of_order_with_error | openai | 1 difference(s) |
| `batch_op_parse` | anthropic_entries_expired_and_errored | anthropic | identical |
| `video_op_parse` | openai_job_in_progress_progress_0 | openai | identical |
| `video_op_parse` | openai_job_failed_with_error | openai | identical |
| `video_op_parse` | openai_job_unknown_status | openai | identical; 1 refused on both |
| `video_op_parse` | gemini_job_not_done_no_metadata | gemini | identical |
| `video_op_parse` | gemini_job_done_error | gemini | identical |
| `video_op_parse` | xai_job_pending_no_id_in_body | xai | identical |
| `video_op_parse` | xai_job_unknown_status | xai | identical; 1 refused on both |
| `video_op_parse` | xai_part_from_status_body | xai | identical |
| `video_op_parse` | openai_part_from_fetched_bytes | openai | identical |
| `video_op_parse` | openai_list_empty | openai | identical |
| `generation_parse` | openai_image_b64_two_images_with_revised_prompt | openai | identical |
| `generation_parse` | openai_image_url_delivery | openai | identical |
| `generation_parse` | gemini_image_text_only_no_image | gemini | identical; 1 refused on both |
| `generation_parse` | openai_speech_wav_header | openai | identical |
| `generation_parse` | openai_speech_no_content_type | openai | identical; 1 refused on both |
| `generation_parse` | gemini_speech_inline_l16 | gemini | identical |
