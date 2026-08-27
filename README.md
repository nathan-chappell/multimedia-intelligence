# Multimedia Intelligence

## Setup / Quick Start

Initialize virtual environment, install dependencies, set environment variables (OpenAI API key and Clerk information required!), build frontend, start backend.
See the [AI generated instructions](#quick-start)

## Introduction 

This project is a multimedia document intelligence agent, using the OpenAI platform and related libraries.  It is designed to handle pdf, text, csv, json, as well as audio and images.  Each file type gets handled differently, and the agent has the ability to upload related artifacts to an [OpenAI vector store](https://developers.openai.com/api/docs/guides/retrieval).  An important part of the design is that file processing is not handled by the server - functionality is provided by the client, or we use OpenAI native pdf / text ingestion for the vector store.  We let the user partition files by *collection*, which is implemented with vector store metadata and used to filter files when using the OpenAI API for file search.

### Summary of file handling:

| Type | Ingestion |
|-|-|
| pdf | Agent inspects pdf and determines an ingestion strategy.  It determines if there are important images that should be extracted for viewing with image, and uses [OpenAI file inputs](https://developers.openai.com/api/docs/guides/file-inputs) to ensure the file gets processed by OCR and Vision\*.  It may decide to add to a collection, where it may be used directly or as a reverse index. |
| text | Agent can inspect file, and may decide to add it to a collection. |
| csv | Agent can inspect file with a JMESPath (the csv is converted to a JSON and queries with a JMESPath library).  It may decide to add a summary of the csv to a collection, which is used as a reverse index. |
| json | Agent can inspect file with a JMESPath.  It may decide to add a summary of the json to a collection, which is used as a reverse index, or the entire json, in which case standard text-similarity is used. |
| images | Agent can upload a description of the file to a collection, which is used as a reverse index.  Agents should always use vision with images. |
| audio | A transcript is created using OpenAI transcription capabilities, which the agent can inspect and include in a collection. |
| video | *For now, this is treated the same as audio.* |

\* *Note: it is important that pdf files are processed with vision models unless OCR or text extraction can be guaranteed to work...*

## Architecture Overview

This is a fairly standard web app - FastAPI containerized web server, requirement on a DB and Bucket (S3 compatible storage), and of course the OpenAI API.  The web server is intended to be lean - it should not be large in size or do much computation - and suitable for horizontal scaling.  A major implementation choice was the use of Chatkit by OpenAI.  This provides a good Chat UI as well as a conceptual model for storing conversation UI information.  It also provides an intersting capability, client side tool calls.  These are tool calls that are handled by the frontend (i.e. the user's browser).  We take advantage of this to handle file processing on the user's system and avoid things like pdf parsing or csv file querying on the server.

## Road to Production

The following issues require further consideration before being truly ready for a 

- Conceptual finalization (e.g. determining our restrictions / contstraints / guarantees )
- Eval plan (e.g. data collection, analysis, and principled / versioned iteration of prompts and tools)
- QA and more testing
- Database allocation and tenancy considerations
- Pricing and rate-limits (e.g. I am currently bounded by tier 3, which for [Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna) allows 5000 RPM)
- Optimizations (e.g. stress testing web server, maybe setting up a CDS for the frontend code...)
- Guardrails and safeties (e.g. detecting abuse or misuse by individual users)

## Key Technical Decisions

I'm using pretty standard backend setup for Python (fastapi / sqlalchemy).  FastAPI is pretty battle-tested at this point, and pydantic is pretty widespread as well.  I'm using chatkit due to it's ease of use with the OpenAI platform and relevant sdks (e.g. `openai-agents`), and the capabilities provided by client tools.  There are good cases to be made against becoming so vertically integrated to some extent, however it enables more rapid development, easiest access to the most advanced and specialized capabilities, and can act as a simplifying constraint.

## Engineering Standards

- We're using automated tests where appropriate, including e2e and playwright tests.
- We use popular type-checking, linting, and formatting tools.
- Using git / github for version control (once v0 is "released" move to major/minor/patch to keep backend/frontend/image aligned)
- All or nearly all code in the repository is AI generated, and not 100% of the code has been reviewed in high detail.

## AI Tools Used

All or almost all of the code in the repository has been generated with codex (`gpt-5.6-Sol`).  It had examples and similar projects in neighboring directories, so it used much of this to start work and establish the structure and architecture.  Codex has been used for everything from planning, configuration, implementation, version-control, building, running tests, and deployment to railway.

# Future Development

With more time, the following are some ideas I would pursue (other than those noted in production considerations)

- Compatible with more models: chatkit works fine with v1/completions compatible APIs, it just requires a separated memory implementation and some capabilities are somewhat restricted.
- Compatible with existing cloud file systems: some capability to work with a user's google-drive (for example) could be interesting.
- Open ended browser harness: Instead of subagents and tools, we could basically just download a bunch of libraries (maybe on demand), and give the agent the ability to write javascript using these libraries to accomplish a much larger range of tasks for the user.

---

# AI GENERATED README

## Quick start

Requirements: Python 3.12+, Node 20+, npm 10+, OpenAI credentials, a Clerk
application, a database, and S3-compatible object storage.

Create the local configuration and install dependencies:

```bash
cp .env.example .env
# Fill in the required OpenAI, Clerk, database, and object-storage values.

python3.12 -m venv .venv
source .venv/bin/activate
pip install -e './backend[dev]'

npm --prefix frontend install
```

If Node is managed by NVM inside WSL, load it before running frontend commands:

```bash
source "$NVM_DIR/nvm.sh"
nvm use
```

Create the development certificate, then start the backend:

```bash
./scripts/create-dev-certs.sh
source .venv/bin/activate
uvicorn multimedia_intelligence.main:app --reload --host 0.0.0.0 --port 8000 \
  --ssl-keyfile certs/localhost-key.pem \
  --ssl-certfile certs/localhost.pem
```

In a second terminal, start the frontend:

```bash
npm --prefix frontend run dev
```

Open <http://localhost:5173>. Vite proxies `/api` and `/chatkit` to the HTTPS
backend. If the proxy cannot connect, open <https://localhost:8000/api/health>
once and accept the local development certificate.

The backend also serves the compiled SPA at <https://localhost:8000>. Keep every
local UI origin you use in `CLERK_AUTHORIZED_PARTIES`; the supplied
`.env.example` includes both localhost entry points. Vite reads its variables
from the repository-root `.env` and fails at startup when
`VITE_CLERK_PUBLISHABLE_KEY` is absent.

For the containerized build:

```bash
docker compose up --build
```
