# Multimedia Intelligence

A collection-scoped multimedia assistant: each user has a durable file library and one semantic
reverse index, partitioned by user-created collections, while conversations retain explicit file
includes for focused work. The implementation combines a
Vite/React frontend, self-hosted OpenAI ChatKit, a FastAPI backend, the OpenAI Agents SDK, and a
SQLAlchemy-backed ChatKit store.

> Status: executable foundation. Chat streaming and persistence are wired. A request-scoped manager
> delegates to ingestion, document, structured-data, media, and image specialists. Modality-aware
> ingestion commits versioned artifact sets to one OpenAI vector store per user. Discovery-only file
> search resolves provider hits back to canonical assets; specialist tools then hydrate images/PDF
> ranges, query JSON/CSV with JMESPath, or page through diarized transcripts.

## Why this shape

The design sits between “chat with docs” and meeting intelligence: a user library may contain text,
tabular data, PDFs, images, or recorded meetings. Conversations can focus on explicit includes while
the per-user reverse index supports safe discovery across that user's prior conversations.

```text
Browser
  ├─ React application shell
  ├─ ChatKit conversation UI ───── POST /chatkit (streaming)
  └─ signed/ranged previews ───── /api/assets/*
                                      │
FastAPI                              │
  ├─ ChatKitServer ── Agents SDK ─── OpenAI Responses API
  ├─ asset service ──────────────── S3-compatible object store (canonical)
  ├─ ingestion tools ────────────── provider files, transcription, analysis
  ├─ retrieval adapters ─────────── OpenAI text/vector search
  └─ SQLAlchemy stores ──────────── SQLite now / Postgres later
```

## Repository layout

```text
backend/   FastAPI, ChatKit server, Agents SDK, storage, ingestion policy
frontend/  Vite + React + TypeScript, ChatKit and artifact panel
docs/      detailed ingestion architecture and implementation boundaries
Dockerfile multi-stage frontend build and single production server image
```

## Quick start

Requirements: Python 3.12+, Node 20+, npm 10+, OpenAI credentials, and a Clerk application.

```bash
cp .env.example .env
# Add OPENAI_API_KEY, CLERK_SECRET_KEY, VITE_CLERK_PUBLISHABLE_KEY,
# and the Clerk frontend origin to CLERK_AUTHORIZED_PARTIES.

python3.12 -m venv .venv
source .venv/bin/activate
pip install -e './backend[dev]'

npm --prefix frontend install
```

Run the backend and frontend in separate terminals:

```bash
./scripts/create-dev-certs.sh
set -a && source .env && set +a
uvicorn multimedia_intelligence.main:app --reload --host 0.0.0.0 --port 8000 \
  --ssl-keyfile certs/localhost-key.pem \
  --ssl-certfile certs/localhost.pem
```

```bash
npm --prefix frontend run dev
```

Vite reads frontend variables from the repository-root `.env` file and refuses to start or build
when `VITE_CLERK_PUBLISHABLE_KEY` is empty or missing. This prevents serving a bundle that can only
fail later in the browser.

When working inside WSL with NVM, load NVM before running frontend commands so `npm` does not
resolve to a Windows installation through the mounted PATH:

```bash
source "$NVM_DIR/nvm.sh"
nvm use
```

Open <http://localhost:5173>. The Vite development server proxies `/api` and `/chatkit` to the
HTTPS FastAPI server. The certificate is self-signed, so open <https://localhost:8000/api/health>
once and accept the local certificate warning if the Vite proxy cannot connect. Sign in through
Clerk. Set `public_metadata.role` to `admin` on the interviewer's Clerk user to expose the admin
console. Swagger remains available at
<https://localhost:8000/docs>.

For a LAN demo, rerun `./scripts/create-dev-certs.sh` after the machine receives its LAN address,
then open `https://<lan-ip>:8000` and accept the certificate once. ChatKit requires this HTTPS
origin because plain HTTP on a non-localhost address is not a secure browser context. Development
CORS accepts localhost and RFC1918 private-network origins; production still requires explicit
origins.

Use a long random `COUPON_CODE_PEPPER`; changing it invalidates outstanding coupon codes. Clerk
session tokens are resolved at request time and are never stored by the application.

To exercise the production-shaped build:

```bash
docker compose up --build
```

Then open <http://localhost:8000>.

## Current decisions

- **Self-hosted ChatKit:** preserves control over authentication, conversation-scoped context, tools, storage, and file lifecycle while retaining ChatKit's UI and streaming protocol.
- **Clerk plus event-sourced access:** Clerk owns identity and administrator metadata. Signed
  micro-USD ledger events are the sole source of truth for user balance; coupon redemptions,
  administrator corrections, and model/transcription charges all append to the same table.
- **Post-charge enforcement:** non-admin users need a positive balance to start billable work. The
  completed request records its actual marked-up cost even if that makes the balance negative, and
  later paid actions return HTTP 402. The default markup is `1.5`.
- **Agents SDK:** owns the model/tool loop and streams through ChatKit's `stream_agent_response` adapter.
- **Root plus specialists:** the root discovers files and hands content work to modality specialists.
  Specialists return control after gathering evidence. ChatKit's model picker applies to the entire
  graph.
- **Client tools:** the root can list staged files. Document and structured-data specialists own the
  relevant text, PDF, CSV, and JSON browser tools. Results are typed evidence, not proof of durable
  storage.
- **Separate dictation and file inspection:** ChatKit attachments stay disabled. Composer dictation
  forwards an ephemeral browser recording through the authenticated backend to
  `gpt-4o-mini-transcribe` without locally decoding or transforming it. Uploaded files are stored
  byte-for-byte and inspected with browser tools; only the explicit demo seeder prepares indexes.
- **SQLAlchemy store:** ChatKit thread/item payloads are stored as version-tolerant JSON with indexed relational identity and timestamps. Each thread owns one OpenAI conversation ID, reused for turns and client-tool continuations and deleted with the thread. Successful turns checkpoint the newest provider item. An interrupted or invalid turn removes only the uncommitted provider suffix, supplies those removed items to the retry as JSON playback, and preserves the earlier conversation history. SQLite keeps local setup small; the same boundary can move to Postgres.
- **History and isolation:** ChatKit history is backed by owner-filtered thread/item queries. Opening a
  previous thread restores its saved files into the artifact panel; both thread ownership and
  owner-matching include/asset rows are required. Bucket keys use
  `assets/users/{user_id}/files/{asset_id}/...`, and inaccessible threads return `404` rather than
  revealing whether another user's conversation exists.
- **File routing before inference:** text-like files, PDFs, images, and transcribable media are classified explicitly. Unknown formats are rejected instead of being silently sent to a model.
- **Bucket first:** every accepted original is durably written to our object store before inspection or provider upload. OpenAI file IDs and vector-store IDs are disposable references, not storage.
- **Include is not upload:** a thread include is a reversible relationship to an asset. The same asset can be included in multiple threads without duplicating the original.
- **Demo artifacts are replaceable:** demo transcripts, PDF ranges, chunks, profiles, and provider
  IDs can be regenerated without losing the original.
- **No runtime media processing:** the application server does not parse PDFs, decode or manipulate
  images, extract audio/video, parse canonical CSV/JSON files, or render charts. Browser tools own
  interactive file inspection. Demo-only preparation lives under `multimedia_intelligence.demo`.
- **Collections:** users create and globally select a collection in the file panel. New uploads and
  ingestion attempts inherit that selection. One lazily created OpenAI vector store still serves
  the user; `collection_id` attributes partition its files without multiplying stores.
- **Per-user retrieval:** each modality contributes tailored profiles, transcript shards,
  descriptions, source text, or page-aware PDF artifacts with provenance.
- **Search routing:** `file_search` returns ranked metadata only. `get_file` and `get_transcript`
  provide owner- and selected-collection-scoped read access to artifacts already created by the
  demo seeder.

## Asset and inclusion pipeline

The policy currently recognizes:

- text/data: `.txt`, `.md`, `.json`, `.csv`
- documents: `.pdf`
- images: `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`
- transcribable media: `.flac`, `.mp3`, `.mp4`, `.mpeg`, `.mpga`, `.m4a`, `.ogg`, `.wav`, `.webm`

The production asset path is deliberately small:

1. **Upload asset:** validate the envelope and stream the original bytes to immutable object storage.
2. **Include asset:** create a `ThreadAssetInclude` with user intent; this does not copy or re-upload the original.
3. **Inspect:** browser-side PDF.js, pdf-lib, and structured-data tools inspect explicit files.
4. **Preview:** issue short-lived signed URLs or stream stored bytes without parsing them.
5. **Exclude/delete:** excluding removes the relationship. Asset deletion is a separate lifecycle operation that checks references.

`multimedia-demo seed` has a separate offline preparation path for reproducible interview fixtures.
It may parse and transform media before the application starts; Pillow, pypdf, and pypdfium2 are
therefore development dependencies and are absent from the production server image.

See [the ingestion architecture](docs/ingestion-architecture.md) for strategy examples, agent roles, retrieval choices, and invariants.
See [the agent hierarchy](docs/agent-hierarchy.md) for root-agent delegation and the shared,
owner-scoped application context.

## Demo collections

The committed manifest under `demo/` defines three reproducible collections: Language Trends,
Type Systems, and ML Foundations. Downloads and generated files stay ignored under `tmp/demo`.
The language table is derived from official Stack Overflow survey files; Type Systems combines
selected TAPL page ranges with official TypeScript documentation; ML Foundations contains ten
manifest-pinned arXiv papers.

```bash
source .venv/bin/activate
multimedia-demo prepare
multimedia-demo seed
multimedia-demo verify --live-search
multimedia-demo rehearse
```

`prepare` is cache-aware, `seed` reuses matching ready checksums and resumable ingestion attempts,
and `verify` checks expected readiness and collection-scoped provider results. TAPL must remain at
`tmp/files/Pierce 2002 - Types and Programming Languages.pdf`; the existing Transformer PDF is
reused when available. The rehearsal prompts are committed in `demo/prompts/`.

Railway Bucket objects are durable and remain until an explicit asset-deletion operation removes
them. Preview links are short-lived signed credentials, but their expiry does not affect the stored
object. Disposable OpenAI resources may use provider-managed expiration. During this prototype phase
schema changes use a hard reset of the development database; migrations are intentionally deferred.

## Tests

The fast suite includes agent graph/tool-contract tests, storage adapter tests, collection isolation,
and a full modality integration matrix. The matrix ingests the real exchange-rate CSV and
Transformer PDF under `tmp/files`, derives bounded representative fixtures for the other modalities,
and completes a search plus modality-specific follow-up for every route.

```bash
.venv/bin/ruff check --no-cache backend/src backend/tests
.venv/bin/mypy --cache-dir=/tmp/multimedia-intelligence-mypy --config-file backend/pyproject.toml backend/src
.venv/bin/pyright
.venv/bin/pytest -q -s -p no:cacheprovider backend/tests
npm --prefix frontend run build
npm --prefix frontend run lint
npm --prefix frontend run test:e2e
```

Tests that spend OpenAI tokens or drive a complete browser/backend workflow remain opt-in. This
keeps deterministic file behavior separate from provider and UI behavior.

The default Playwright suite includes a deterministic ChatKit protocol round trip that stages a
file, executes `list_files` in the browser, posts the matching client-tool result, and requires a
rendered assistant response. It also verifies that a generated chart renders inline and is restored
as a saved collection artifact. An opt-in live browser test exercises the same path through the real
FastAPI and OpenAI agent stack. With the backend already running:

```bash
RUN_OPENAI_E2E=1 \
LIVE_E2E_BASE_URL=http://127.0.0.1:8000 \
ADMIN_USERNAME=admin ADMIN_PASSWORD=admin \
npm --prefix frontend run test:e2e -- e2e/live-agent.spec.ts --workers=1
```

Use the configured development credentials when they differ from the defaults.

After `multimedia-demo seed`, the opt-in browser rehearsal runs all three collection scenarios,
including the two inline language charts:

```bash
RUN_DEMO_E2E=1 \
LIVE_E2E_BASE_URL=http://127.0.0.1:8000 \
ADMIN_USERNAME=admin ADMIN_PASSWORD=admin \
npm --prefix frontend run test:e2e -- e2e/live-demo.spec.ts --workers=1
```

The demo-only live vector-store test creates a temporary per-user store, ingests the real CSV and Transformer
PDF, performs collection-filtered searches and follow-ups, and removes the OpenAI files/store in a
`finally` block:

```bash
set -a
source .env
set +a
RUN_OPENAI_INGESTION_LIVE=1 \
  .venv/bin/pytest -vv -s -p no:cacheprovider \
  backend/tests/live/test_vector_store_ingestion_live.py
```

The current live agent suite makes real `gpt-5.6` API calls for browser inspection of text, JSON,
CSV, PDF, image, audio, and video files. It stages a real representative file for each route instead
of inserting a prewritten inspection result into the prompt. Each scenario verifies the browser
tool requests, checks that the root calls the route-appropriate overview specialist, verifies that
no server-ingestion tool is used, and checks the resulting overview. Load the test key without printing it and
opt in explicitly:

```bash
set -a
source .env
set +a
TMPDIR=/tmp RUN_OPENAI_BEHAVIORAL=1 \
  .venv/bin/pytest -vv -s -p no:cacheprovider backend/tests/live
```

The live harness models the same pause/resume boundary as ChatKit:

1. Run the graph until the active agent requests a client tool and stops.
2. Execute that tool against fixture-backed browser files.
3. Validate and normalize the result through the production Pydantic result model.
4. Replace the waiting `function_call_output` for that exact call ID in `RunResult.to_input_list()`.
5. Resume the agent that issued the tool call and continue until the graph returns a final answer or
   reaches the bounded client-tool round limit.

Deterministic tests cover tool contracts and exact call-ID replacement without API calls.
The fixture client currently mirrors text, JSON, CSV, and PDF inspection. Image, audio, and video
scenarios can exercise metadata discovery and agent orchestration, but the frontend does not yet
expose client tools that return visual previews, audio probes, or video frame/transcript evidence;
those scenarios must report insufficient content evidence rather than pretending ingestion ran.

## Observability

Production ChatKit runs and live tests use the same Agents SDK tracing policy. Each workflow gets
an OpenAI trace ID, opaque conversation group ID, and opaque turn ID. Those identifiers accompany
every local agent/model/tool lifecycle event, including specialist handoffs. The SDK supplies the
agent, generation, function-tool, and handoff spans; the application does not duplicate them. Local
logs retain OpenAI request and response IDs and token counts. Prompts, model output,
tool arguments, tool results, and file contents are excluded by default. This preserves the request
IDs OpenAI support uses for API troubleshooting without turning application logs into another
content store.

Set these variables in Railway and local `.env` files as needed:

```dotenv
LOG_LEVEL=INFO
OPENAI_TRACING_ENABLED=true
OPENAI_TRACE_INCLUDE_SENSITIVE_DATA=false
```

Sensitive trace export should remain `false` outside a deliberately controlled debugging session.
The application also keeps the lower-level `openai` and `agents` Python loggers at warning level,
because debug HTTP logs may include request bodies.

## Quality and productionization backlog

- Add Clerk webhooks for identity lifecycle cleanup and an explicit production data-retention policy.
- Move SQLite to Postgres and connect the implemented `S3BlobStore` to upload-ticket/finalization routes, quarantine promotion, signed URLs, versioning, and lifecycle deletion.
- Add bounded ingestion/transcription tools with idempotency, cancellation, and progress returned through ChatKit.
- Add malware scanning, content-type sniffing, decompression limits, upload quotas, and image/media safety validation.
- Add prompt-injection defenses around untrusted file content and make citations/source boundaries visible.
- Extend the existing ChatKit/thread/Agents/OpenAI trace correlation through ingestion tool calls and add latency, quality, and failure metrics.
- Evaluate with a curated set covering groundedness, citation correctness, action-item extraction, speaker/timestamp fidelity, isolation, and refusal when evidence is absent.
- Add CI for Ruff, mypy, pytest, ESLint, TypeScript, frontend tests, container build, and dependency/security scanning.
- Add screenshots/video and replace this README's placeholder trade-off language with the author's own final reflections before submission.

## AI-assisted development note

This skeleton was AI-assisted. Before submission, keep a short human-authored record of what was generated, what was reviewed or changed, which commands/tests were run, and where AI output was deliberately not trusted (security boundaries, SDK/API assumptions, and architecture rationale).
