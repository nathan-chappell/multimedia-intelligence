# Agent hierarchy and application context

The application has one root agent and five specialist agents. Modality specialists receive the
conversation through SDK handoffs; the ingestion strategist is a structured-output agent tool.
Every agent inherits the same typed `AgentContext[RequestContext]`. No identity, database session,
or file credential is copied into model input.

```text
Root conversation agent
├── Ingestion strategist
├── Document specialist
├── Structured-data specialist
├── Media specialist
└── Image specialist
```

## Root conversation agent

Discovers current-conversation files with `list_files` and searches the owner's durable library with
`file_search`, then selects a modality specialist and synthesizes the final answer. Search is
discovery-only and restricted to the user's globally selected collection; hydration remains a
specialist capability. The root can start `prepare_ingestion`
and exposes the ingestion strategist as a tool.

For initial ingestion, the root discovers the file and hands off to the route-specific specialist.
The specialist gathers bounded evidence, then returns control. The root asks the ingestion
strategist for a provisional approach. The strategist commits a standalone retrieval description
and any confirmed PDF selection through `commit_ingestion`.

## Behavioral test execution

Agent tests keep the same root/specialist hierarchy and typed application context as production.
The test driver wraps the agent graph in a bounded continuation loop. When the active agent stops at
a browser tool, a fixture client executes it, the production Pydantic model validates the result,
and the driver resumes the same active agent with the output attached to the original call ID. The
production server similarly maps each client-tool continuation to its owning specialist.

This exercises routing, specialist tool choice, validation of untrusted browser results, handoff
back to the root, and ingestion-strategy delegation. The loop has a hard round limit.

## Specialists

- **Ingestion strategist:** owns `commit_ingestion`, commits an evidence-backed description and
  optional PDF selections to the user index, then
  returns a provisional plan containing only a summary, approach, and things to watch for.
- **Document specialist:** owns staged text reads and PDF inspect/render/extract tools, plus durable
  `get_file` hydration and bounded text reads.
- **Structured-data specialist:** owns browser JMESPath tools for included CSV/JSON and the
  owner-scoped `query_file` JMESPath tool for assets discovered through the user index.
- **Media specialist:** owns paginated `get_transcript` access for audio and video evidence.
- **Image specialist:** owns `get_file` vision hydration for canonical images.

Modality specialists cannot invoke one another; they only return control to the root. Client tools
pause the active specialist and resume that specialist after validation. Index mutation belongs
only to ingestion; retrieval and structured analysis are owner-scoped.

## Shared application context

`RequestContext` contains:

- `ClientInfo`: authenticated user ID, username, and admin status;
- `AgentDataAccess`: a narrow owner-scoped interface for database-backed entity information;
- request metadata, selected model, and reasoning settings.

`ScopedAgentDataAccess` exposes selected-collection conversation references, owner-library search,
bounded reads, structured queries, and signed inputs after ownership and collection checks. Agents never receive a raw SQLAlchemy
session, unrestricted repository, bearer token, S3 credentials, or provider keys.

## File lifetime

Bucket objects do not expire automatically. They remain durable until an explicit asset-deletion
operation removes them; the expiry on a signed preview URL only limits access through that URL.
Disposable OpenAI resources use provider-managed expiration controls.

For the current prototype there are no migrations. Schema changes are applied through a hard reset
of the development database.
