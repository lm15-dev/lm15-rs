# Finding: provider ids are interpolated raw into URL paths (reference and port alike)

Status: RESOLVED 2026-09-08 — ratified as MAP-11
(`lm15-contract/changes/2026-09-08-id-path-escaping.md`): percent-encode,
`/` kept only on the resource-name dialect; ten pinned cases; both
implementations fixed the same day.

Found by: `tools/differential_surfaces.py` (2026-09-08), probes
`file_op_build get_id_with_reserved_chars` and `batch_op_build
status_odd_id`: a file id `file/with?odd=chars&x#1` and a batch id
`batch/odd id?x`.

## Evidence

- `lm15-python/lm15/providers/openai.py:1703-1733` (`_file_get_request`,
  `_file_delete_request`, `_file_download_request`) and the batch/video
  siblings on every dialect build `f"{base_url}/files/{file_id}"` with the
  id verbatim; this port copies that (`src/dialects/*/files.rs`,
  `batch.rs`, `video.rs`).
- Both therefore emit `.../files/file/with?odd=chars&x#1`: the `?` starts
  a query string and the `#` a fragment on the wire. The reference's vet
  shim renders the same URL split into `url` + `params`; the bytes an HTTP
  client sends are the same.
- Nothing in the corpus pins an id with a reserved character; every live
  id seen (`file-…`, `batch_…`, `msgbatch_…`, `files/…`,
  `cachedContents/…`, `operations/…`, UUIDs) is path-safe. So this is a
  latent gap, not a live failure.

## Why it matters

An id is an opaque string the provider handed back. If one ever contains
a reserved character (a space, `?`, `#`, `%`), both implementations send
a request for a different resource than the one asked for — silently.
The rule is not obvious: Gemini resource names contain `/` that must NOT
be encoded, so "percent-encode the segment" is not the whole answer.

## Proposal (a `changes/` entry for the contract; not applied by the port)

State the escaping rule per id namespace (percent-encode RFC 3986
reserved characters except `/` in Gemini resource names; reject an id
containing `?`, `#` or whitespace with `InvalidRequestError` rather than
guess), fix the reference, and pin one request case per surface with a
reserved character in the id. Until decided, this port matches the
reference byte for byte, which is the rule for a port
(playbooks/port.md § Rules 1).
