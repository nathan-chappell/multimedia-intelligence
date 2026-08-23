# Ingestion architecture

The system separates durable files, conversation scope, and model-facing representations. They have different lifecycles and never share an identifier.

```text
Asset (immutable original in Railway Bucket)
   │
   ├── ThreadAssetInclude (thread scope + user intent)
   │       │
   │       └── DerivedArtifact 0..n
   │              ├── profile/transcript/frame/page/chunks in our bucket
   │              └── disposable OpenAI file/vector-store reference
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
8. Every bucket object and OpenAI file reference expires after 24 hours; provider IDs are deleted by the application rather than treated as durable storage.

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
include + intent → bounded ChatKit inspection tool
                                      ↓
                  specialist evidence + provisional approach
                                      ↓
                      next ChatKit tool call
                              ↓
                    result or new discovery
                       ↓                 ↓
                  continue          revise approach
```

The ingestion strategist returns a Pydantic-validated descriptive object with a short summary,
provisional approach, and conditions to watch for. It is part of the conversation and is not stored
as a database workflow. The active modality specialist chooses bounded inspection tools; the root
routes work and revises the overall approach. Tool schemas and backend policy—not plan prose—enforce
size, ownership, and expiration rules.

## Structured-data tools

### CSV

`CsvAnalyzer` implements the initial read-only tool surface:

- `Head()` returns headers, inferred types, nullability, and up to ten coerced rows.
- `Rows(start, count, columns?)` projects columns and enforces a per-call row limit.
- `Stats(columns?)` streams finite numeric values through Welford variance. Quantiles use a deterministic 10,000-value reservoir and report when approximate.
- `Plot({column,label?}, {column,label?})` produces a PNG. Numeric pairs become a sampled scatter plot; categorical-x/numeric-y becomes a top-category mean bar chart.

The analyst should inspect the head, request relevant statistics and rows, inspect at most a few plots, summarize evidence, and ask the user what to pursue next. Plot PNGs become bucket-backed `TABLE_PLOT` artifacts before they are supplied as model vision input.

CSV content never enters the prompt wholesale. Repeated calls should eventually use a materialized DuckDB/Polars-style artifact rather than rescanning remote bytes.

For semantic search over a text column, do not create one OpenAI file per row. That creates provider-object and lifecycle overhead. First normalize rows into provenance-bearing shards with stable row IDs. Evaluate shard retrieval. If true row-level nearest-neighbor behavior is required, use a direct embeddings + pgvector path later; OpenAI vector stores ingest files and do not promise one vector per CSV row.

### JSON

The client tool layer implements:

- `Chars(start,count)`, using streaming UTF-8 decoding so a character range does not load the whole file;
- `JsonPath(query|query...)`, using `jsonpath-plus`, capped at eight queries, 100 values per query, and a response-byte budget.

JSONPath requires parsing the complete JSON value and therefore has a browser file-size ceiling.
Above that limit, a later server tool can provide streaming inspection or a structural index.

The OpenAI custom-tool contract includes `json_inspection.lark`. It allows property, array-index, wildcard, and quoted-property selectors but intentionally excludes script expressions and filters. The same grammar is validated server-side; grammar-constrained model output is not authorization.

Selected JSON subtrees can later be normalized into structure-aware text shards with JSONPath provenance before vector-store ingestion.

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

The adaptive plan is:

1. Randomly sample a bounded range as text, retaining page numbers and truncation status.
2. For a text-heavy document, extract structure-aware text and use retrieval when it exceeds direct context.
3. For important figures, tables, formulas, or weak OCR, render selected pages and mark those images as preferred visual evidence.
4. Use bounded PDF ranges as clearly labeled `input_file` evidence.
5. Persist range manifests, page numbers, prompt, and compact result so every claim can be traced to the original.

A ten-page paper will normally use the full provider file plus rendered figure pages. A 300-page textbook should probe the table of contents, index, and representative ranges, then ask a focused user question before choosing direct ranges versus retrieval.

Visual PDF samples resume the requesting tool call with a bounded `input_file`; they intentionally
join the current conversation context. A future isolated probe gateway can use an independent
Responses call when context isolation is required.

## Audio, video, and images

Audio uses `gpt-4o-transcribe-diarize` when speaker labeling is needed, producing timestamped labeled sections. Small transcripts enter bounded context; larger transcripts become structure-aware text artifacts for vector retrieval.

Video keeps two aligned evidence channels:

- diarized transcript segments with timestamps;
- a retained frame manifest with sampled images and timestamps.

There is no active Cohere or visual-embedding path. The agent receives a bounded frame set through vision and can request denser extraction for a time interval against the retained original. Scene-change sampling can replace fixed intervals after the basic pipeline is reliable.

Images go directly to vision in bounded batches. If a thread includes too many images for one request, build small batches or contact sheets, summarize each batch with stable asset IDs, and perform a second synthesis pass. Never silently omit images; expose the batch/progress state and retain per-image provenance.

## Vector-store policy

Use OpenAI vector stores/file search only for large text, text-heavy PDFs, transcripts, and deliberately normalized JSON/CSV text artifacts. Preserve structural boundaries before upload: headings, pages, transcript timestamps/speakers, JSONPath, and row IDs. Keep our chunk/provenance manifest even when OpenAI performs the final chunking.

Use automatic provider chunking only when generic large text makes it reasonable. Retrieval results must retain OpenAI annotations/file references and map them back to our include and source artifact. A provider vector-store ID remains disposable state.

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

1. Add upload-ticket/finalization endpoints, quarantine promotion, hashes, and the SQLAlchemy asset repository.
2. Add include/exclude/list endpoints and connect them to the custom artifact-panel flow. ChatKit
   attachments remain disabled and are never part of ingestion.
3. Add the next bounded ingestion tools and render their calls/results as ChatKit activity.
4. Materialize bucket objects into controlled temporary files for CSV, media, and PDF tools.
5. Add isolated OpenAI PDF probe, Files, transcription/diarization, and vector-store gateways with expiry cleanup.
6. Add per-tool cancellation, idempotency, quotas, and concise observability.
7. Evaluate groundedness, page/timestamp provenance, table-stat correctness, isolation, and failure behavior.
