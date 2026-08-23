# Multimedia Intelligence

A conversation-scoped multimedia assistant: each chat is grounded only in the files attached to that chat. The first-pass implementation combines a Vite/React frontend, self-hosted OpenAI ChatKit, a FastAPI backend, the OpenAI Agents SDK, and a SQLAlchemy-backed ChatKit store.

> Status: executable foundation. Chat streaming and persistence are wired. A request-scoped manager delegates to ingestion, document, structured-data, media, and image specialists. The artifact panel stages local files, can save originals to the configured Railway/S3 bucket and active conversation, and ChatKit can pause for bounded text, JSON, CSV, and PDF browser tools. The asset/include/artifact domain, descriptive ingestion strategy, typed object-store adapter, server CSV/PDF analysis, and transient PDF previews are implemented. Provider gateways, additional ingestion tools, and final artifact-to-agent input conversion remain milestones.

## Why this shape

The assignment asks for document-grounded conversational AI and values engineering decisions as much as breadth. This design sits between “chat with docs” and meeting intelligence: a conversation may contain text, tabular data, PDFs, images, or recorded meetings, but there is no global document collection and no cross-conversation retrieval.

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

Requirements: Python 3.12+, Node 20+, npm 10+, and an OpenAI API key.

```bash
cp .env.example .env
# Add OPENAI_API_KEY to .env

python3.12 -m venv .venv
source .venv/bin/activate
pip install -e './backend[dev]'

npm --prefix frontend install
```

Run the backend and frontend in separate terminals:

```bash
./scripts/create-dev-certs.sh
set -a && source .env && set +a
uvicorn multimedia_intelligence.main:app --reload --port 8000 \
  --ssl-keyfile certs/localhost-key.pem \
  --ssl-certfile certs/localhost.pem
```

```bash
npm --prefix frontend run dev
```

Open <http://localhost:5173>. The Vite development server proxies `/api` and `/chatkit` to the
HTTPS FastAPI server. The certificate is self-signed, so open <https://localhost:8000/api/health>
once and accept the local certificate warning if the Vite proxy cannot connect. Sign in through the
application with the built-in `ADMIN_USERNAME` and `ADMIN_PASSWORD`. Swagger remains available at
<https://localhost:8000/docs>.

Set `JWT_SECRET_KEY` to at least 32 random characters outside local development. A build-time
`VITE_API_BEARER_TOKEN` remains available for temporary testing but should not be used in a
production image.

To exercise the production-shaped build:

```bash
docker compose up --build
```

Then open <http://localhost:8000>.

## Current decisions

- **Self-hosted ChatKit:** preserves control over authentication, conversation-scoped context, tools, storage, and file lifecycle while retaining ChatKit's UI and streaming protocol.
- **Agents SDK:** owns the model/tool loop and streams through ChatKit's `stream_agent_response` adapter.
- **Root plus specialists:** the root discovers files and hands content work to modality specialists.
  Specialists return control after gathering evidence; the ingestion strategist remains a structured
  agent tool. ChatKit's model picker applies to the entire graph.
- **Client tools:** the root can list staged files. Document and structured-data specialists own the
  relevant text, PDF, CSV, and JSON browser tools. Results are typed evidence, not proof of durable
  storage.
- **Separate dictation and ingestion:** ChatKit attachments stay disabled. Composer dictation sends
  an ephemeral browser recording through the authenticated backend to `gpt-4o-mini-transcribe`;
  uploaded audio uses the custom asset and ingestion pipeline instead.
- **SQLAlchemy store:** ChatKit thread/item payloads are stored as version-tolerant JSON with indexed relational identity and timestamps. Each thread owns one OpenAI conversation ID, reused for turns and client-tool continuations and deleted with the thread. Removing local history for a retry rotates the conversation and replays only the surviving items. SQLite keeps local setup small; the same boundary can move to Postgres.
- **Conversation isolation:** attachment records carry a `thread_id`. Agent input assembly must reject any attachment not belonging to the active thread.
- **File routing before inference:** text-like files, PDFs, images, and transcribable media are classified explicitly. Unknown formats are rejected instead of being silently sent to a model.
- **Bucket first:** every accepted original is durably written to our object store before inspection or provider upload. OpenAI file IDs and vector-store IDs are disposable references, not storage.
- **Include is not upload:** a thread include is a reversible relationship to an asset. The same asset can be included in multiple threads without duplicating the original.
- **Derived artifacts are replaceable:** previews, transcripts, page renders, sampled frames, chunks, profiles, and provider/index IDs can be deleted and regenerated without losing the original.
- **Interactive ingestion:** the ingestion agent describes a provisional approach. The root performs
  bounded work through ChatKit tool calls and revises the approach as results reveal more. Tool
  boundaries validate ownership, limits, and side effects independently.
- **Text retrieval, bounded vision:** OpenAI vector stores are the default text-retrieval path. Visual embeddings are deferred; PDFs, images, and sampled video frames use bounded vision calls with retained provenance.

## Asset and inclusion pipeline

The policy currently recognizes:

- text/data: `.txt`, `.md`, `.json`, `.csv`
- documents: `.pdf`
- images: `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`
- transcribable media: `.flac`, `.mp3`, `.mp4`, `.mpeg`, `.mpga`, `.m4a`, `.ogg`, `.wav`, `.webm`

The state machine is deliberately split:

1. **Upload asset:** validate envelope, stream to a quarantine key, scan/sniff, hash, and promote to an immutable bucket key.
2. **Include asset:** create a `ThreadAssetInclude` with user intent; this does not copy or re-upload the original.
3. **Describe an approach:** inspect metadata and bounded samples, then state a provisional strategy.
4. **Work through ChatKit:** perform one validated tool call at a time and adapt to each result.
5. **Use in chat:** materialize only ready artifacts belonging to active includes for the current thread.
6. **Preview:** issue short-lived signed URLs and ranged/derived previews. Large originals are never loaded wholesale into browser memory.
7. **Exclude/delete:** excluding removes the relationship. Asset deletion is a separate lifecycle operation that checks references and cleans up provider copies.

See [the ingestion architecture](docs/ingestion-architecture.md) for strategy examples, agent roles, retrieval choices, and invariants.
See [the agent hierarchy](docs/agent-hierarchy.md) for root-agent delegation and the shared,
owner-scoped application context.

All bucket objects and disposable OpenAI file references currently have a fixed 24-hour lifetime.
The backend records the expiration, tags bucket uploads, caps preview URLs to the remaining lifetime,
and runs an hourly deletion sweep. During this prototype phase schema changes use a hard reset of the
development database; migrations are intentionally deferred.

## Tests

The fast suite includes agent graph/tool-contract tests, storage adapter tests, and file analyzers. The PDF behavioral test uses the local textbook under `tmp/files` when present and verifies that a table-of-contents region is found without an API call.

```bash
.venv/bin/ruff check --no-cache backend/src backend/tests
.venv/bin/mypy --cache-dir=/tmp/multimedia-intelligence-mypy --config-file backend/pyproject.toml backend/src
.venv/bin/pytest -q -s -p no:cacheprovider backend/tests
npm --prefix frontend run build
npm --prefix frontend run lint
npm --prefix frontend run test:e2e
```

Tests that spend OpenAI tokens or drive a complete browser/backend workflow should use the `live` or future `e2e` marker and remain opt-in. This keeps deterministic file behavior separate from provider and UI behavior.

The current live agent suite makes real `gpt-5.6` API calls for initial ingestion of text, JSON,
CSV, PDF, image, audio, and video files. It stages a real representative file for each route instead
of inserting a prewritten inspection result into the prompt. Each scenario verifies the browser
tool requests, checks that the root calls the route-appropriate overview specialist before the
ingestion strategist, and checks the resulting strategy. Load the test key without printing it and
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

- Replace development user identity with authenticated tenant/user context and authorization checks on every thread and file operation.
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
