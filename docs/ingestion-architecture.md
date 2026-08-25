# Ingestion architecture

The system separates durable files, conversation scope, and model-facing representations. They have different lifecycles and never share an identifier.

```text
Asset (immutable original in Railway Bucket)
   │
   ├── AssetIngestion 1..n ── versioned prepared evidence + status
   │       └── AssetIndexArtifact 1..n ── bucket object + provider file + provenance
   │                                      ↓
   │                              UserVectorStore (one per owner)
   │
   ├── ThreadAssetInclude (thread scope + user intent)
   │       │
   │       └── DerivedArtifact 0..n
   │              ├── profile/transcript/frame/page/chunks in our bucket
   │              └── replaceable provider-file reference
   │
   └── another ThreadAssetInclude (independent plan and lifecycle)

ChatKit Attachment = transport/UI metadata pointing at an include, never storage
```

## Invariants

1. The accepted original reaches our bucket before an agent or external provider processes it.
2. Any bytes uploaded to OpenAI also have a canonical or reproducible bucket copy. OpenAI IDs are never recovery paths.
3. An include is reversible and thread-scoped. Excluding a file does not delete the asset.
4. The agent's ingestion plan is provisional conversation guidance, not executable state.
5. Each tool call validates authorization and limits independently and returns its result through ChatKit.
6. Only ready artifacts from active includes owned by the current user enter a chat turn.
7. Browser results are bounded proposals. The backend verifies identity, size, hash, and authorization before persisting or trusting them.
8. Bucket objects are durable until explicitly deleted. Disposable OpenAI resources use the
   provider's expiration controls and are never treated as durable storage.
9. A vector store belongs to a user, never to a conversation. Every provider hit is resolved back
   through an owner-matching canonical asset before bytes or structured tools become available.
10. A collection is a logical partition, not another vector store. Uploads, ingestion attempts,
    provider attributes, search filters, and follow-up hydration must all match the user's global
    selected collection.

## Storage: Railway Buckets

Railway Buckets are S3-compatible and private. `S3BlobStore` uses boto3 with Railway's standard variables:

- `AWS_ENDPOINT_URL`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_S3_BUCKET_NAME`
- `AWS_DEFAULT_REGION`
- `AWS_S3_URL_STYLE`

The existing `OBJECT_STORE_*` names remain local-development aliases. Browser uploads use backend-issued presigned PUT URLs; no S3 credential reaches JavaScript. Reads use short-lived GET URLs or authenticated ranged proxy responses. For backend uploads, async chunks spool to disk and boto3 performs multipart transfer without accumulating the object in memory.

Presigned upload completion is not proof of safe ingestion. Finalization must `HEAD` the exact key, stream it through hashing/type-sniffing/malware checks, and only then promote the asset from quarantine to `STORED`.

## Interactive ingestion flow

```text
upload → quarantine → hash/sniff/scan → durable Asset
                                      ↓
prepare_ingestion → persisted modality evidence and artifacts
                                      ↓
               matching specialist produces a grounded description
                                      ↓
       commit_ingestion → replacement upload set → atomic activation
                                      ↓
              file_search returns ranked metadata only
                  ↓                 ↓                   ↓
              get_file          query_file        get_transcript
```

`prepare_ingestion` persists each completed stage under a versioned attempt. States are
`preparing`, `prepared`, `awaiting_guidance`, `indexing`, `ready`, and `failed`. The ingestion
strategist alone calls `commit_ingestion`. A replacement is uploaded completely before its database
records become active; failed replacements leave the previous ready attempt searchable. Retrying a
commit reuses already uploaded artifacts and does not repeat provider uploads.

## Structured-data tools

### CSV and JSON

The browser exposes one `query_structured_data` tool for included files. JSON is parsed directly;
CSV is parsed with its header row as object keys and dynamic JSON-compatible typing for values. The
resulting array of row objects is queried with JMESPath (`jmespath.js`). Results are capped at 100
top-level array entries and 256 KiB, and complete-file parsing has a 64 MiB browser ceiling.

The analyst starts with `[0]` for CSV or a focused object projection for JSON, then uses JMESPath
filters, projections, multiselects, and functions for subsequent questions. CSV and JSON content
never enters the prompt wholesale. Repeated calls should eventually use a materialized
DuckDB/Polars-style artifact rather than rescanning remote bytes. For a user-library result found by
`file_search`, the structured-data specialist uses `query_file` against the
owner-checked canonical bucket object.

`create_chart` applies a second bounded JMESPath expression to the canonical CSV/JSON asset and
supports line, grouped-bar, and scatter charts. It accepts at most 5,000 rows, 12 series, and 50 bar
categories, requires a numeric Y field, and emits an optimized 1200×675 PNG of at most 512 KiB.
Each saved chart records its source asset, collection, thread, expression, complete chart spec,
sample size, and plotted-series provenance. ChatKit receives an inline generated-image item while
the authenticated artifact endpoint lets the file panel restore the same bucket object. The tool
belongs only to the structured-data specialist, whose prompt requires sample-size and observational
data caveats.

For semantic search over a text column, do not create one OpenAI file per row. That creates provider-object and lifecycle overhead. First normalize rows into provenance-bearing shards with stable row IDs. Evaluate shard retrieval. If true row-level nearest-neighbor behavior is required, use a direct embeddings + pgvector path later; OpenAI vector stores ingest files and do not promise one vector per CSV row.

For JSON only, `json_chars` remains available for streaming a bounded character range before a
complete parse. Above the JMESPath input limit, a later server tool can provide streaming inspection
or a structural index.

The OpenAI custom-tool contract embeds `jmespath.lark`, a Lark translation of the official JMESPath
1.0 ABNF and binding order. The same grammar is validated server-side; grammar-constrained model
output is still not authorization.

Selected subtrees can later be normalized into structure-aware text shards with JMESPath provenance
before vector-store ingestion.

## PDF strategy

Browser utilities use PDF.js for sampled text extraction and page rendering, plus `pdf-lib` for bounded page extraction:

- `pdf_random_sample`: sample up to 10 pages inside a requested range. `text_content`
  returns bounded library-extracted text; `as_files` creates one compact PDF containing the sampled
  pages and records their original page numbers;
- `pdf_render_page`: render selected visual evidence to PNG;
- `pdf_extract_range`: create a page-range PDF, capped because `pdf-lib` loads source bytes in browser memory.

`as_files` never serializes PDF bytes into the tool result. The browser uploads the new `Blob`
directly to the application, and the resumed function output supplies the model a short-lived signed
`input_file` URL. This avoids base64 expansion and keeps bucket storage canonical.

For very large PDFs, browser rendering can still inspect local pages through PDF.js, but `pdf-lib`
is not a safe 1 GiB splitter. A bounded server-side splitting tool remains required. The browser is
an accelerator; durable assets remain available for later ChatKit turns after the tab closes.

The durable adaptive plan is:

1. Extract page-aware text, outlines, and a bounded set of non-decorative embedded images.
2. Propose outline-aware ranges of at most 20 pages and retain original page numbers.
3. Auto-continue for small, readable, structurally simple files; pause as `awaiting_guidance` for
   large, image-heavy, encrypted, weak-text, or ambiguous files.
4. Materialize confirmed page ranges, caption at most 20 selected images, and index page-aware text
   and caption documents alongside the range PDFs.
5. Hydrate the PDF range matching the search artifact by default; `get_file(original=true)` remains
   available when the complete original is required.

A ten-page paper will normally use the full provider file plus rendered figure pages. A 300-page textbook should probe the table of contents, index, and representative ranges, then ask a focused user question before choosing direct ranges versus retrieval.

Visual PDF samples resume the requesting tool call with a bounded `input_file`; they intentionally
join the current conversation context. A future isolated probe gateway can use an independent
Responses call when context isolation is required.

## Audio, video, and images

Audio uses `gpt-4o-transcribe-diarize` when speaker labeling is needed, producing timestamped labeled sections. Small transcripts enter bounded context; larger transcripts become structure-aware text artifacts for vector retrieval.

Supported MP4/WebM video containers are submitted directly to the same transcription endpoint; no
FFmpeg dependency is required. The indexed transcript explicitly warns that only the audio track
was analyzed. `get_transcript` returns a complete bounded transcript when possible and otherwise
paginates with timestamp continuity.

Images go directly to vision in bounded batches. If a thread includes too many images for one request, build small batches or contact sheets, summarize each batch with stable asset IDs, and perform a second synthesis pass. Never silently omit images; expose the batch/progress state and retain per-image provenance.

## Vector-store policy

Each user has one lazily created OpenAI vector store. `FileCollection` records and the persisted
`UserCollectionSelection` provide a global working scope. Assets snapshot the selected collection
at upload, ingestion attempts snapshot it again, and every uploaded provider file carries a
`collection_id` attribute.

CSV/JSON add bounded profiles; images add
specialist descriptions; audio/video add timestamped diarized transcript shards; text adds the
original with explicit provider chunking plus heading-aware shards; PDFs add range PDFs and
page/image-caption documents.

`file_search` preserves individual artifact hits and returns `assetId`, `artifactId`, score,
snippets, modality, artifact kind, provenance, and allowed follow-ups. It never attaches bytes.
Search sends an equality filter for the selected `collection_id` to OpenAI. Every returned hit must
also match the current owner, selected collection, active ingestion attempt, artifact row, and exact
provider file ID, so stale, cross-owner, and cross-collection hits are discarded. `get_file`, `query_file`, and
`get_transcript` hydrate canonical evidence only after discovery.

## Agent boundaries

- **Conversation manager:** discovers files, routes to modality specialists, and synthesizes results.
- **Ingestion strategist:** combines intent, metadata, and specialist evidence into a provisional
  descriptive approach. It does not create jobs or executable steps.
- **Document specialist:** interprets browser PDF probes, isolated provider probes, text extraction, and page/layout evidence.
- **Structured-data specialist:** interprets CSV head/statistic results plus bounded JSON characters and safe JSONPath results.
- **Media specialist:** plans diarization and bounded frame sampling/refinement as separate aligned evidence channels.
- **Image specialist:** reasons about explicitly supplied visual evidence and batching/contact-sheet needs.

Modality specialists own their inspection tools and hand control back to the root. They do not call
one another, own storage credentials, delete data, or mutate thread membership. Model-generated
tool input is validated again at the tool boundary.

Browser-local files retain opaque local IDs until saved. A PDF sample requested with `as_files` is
uploaded and attached to the conversation immediately; its tool result contains only compact
metadata and original-page provenance. Page renders and manually extracted ranges remain local
preview artifacts until explicitly saved.

## Next implementation slices

1. Add collection rename/delete/move operations and bulk import from URLs such as arXiv.
2. Improve PDF outline/image heuristics and add OCR for scanned documents.
3. Add quotas, cancellation, ingestion observability, and explicit deletion of provider artifacts.
4. Expand opt-in live coverage to embedded-image PDFs and diarized MP4/WebM files.
