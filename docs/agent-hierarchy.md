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

Discovers staged and durable files, selects a modality specialist, and synthesizes the final answer.
It can list files but cannot read text, parse CSV or JSON, or inspect PDFs. Those capabilities belong
to specialists. The root hands off content work and exposes the ingestion strategist as a tool.

For initial ingestion, the root discovers the file and hands off to the route-specific specialist.
The specialist gathers bounded evidence, then returns control. The root asks the ingestion
strategist for a provisional approach. The approach is descriptive and may change with new results.

## Behavioral test execution

Agent tests keep the same root/specialist hierarchy and typed application context as production.
The test driver wraps the agent graph in a bounded continuation loop. When the active agent stops at
a browser tool, a fixture client executes it, the production Pydantic model validates the result,
and the driver resumes the same active agent with the output attached to the original call ID. The
production server similarly maps each client-tool continuation to its owning specialist.

This exercises routing, specialist tool choice, validation of untrusted browser results, handoff
back to the root, and ingestion-strategy delegation. The loop has a hard round limit.

## Specialists

- **Ingestion strategist:** describes a provisional, adaptable approach; can list durable file
  metadata. Its Pydantic output contains only a summary, an approach, and things to watch for.
- **Document specialist:** owns staged text reads and PDF inspect/render/extract tools, plus durable
  file lookup and bounded text reads.
- **Structured-data specialist:** owns staged CSV head/statistics and JSON character/JSONPath tools,
  plus durable file lookup and bounded text reads.
- **Media specialist:** interprets audio and video evidence; can list durable file metadata.
- **Image specialist:** interprets visual evidence; can list durable file metadata.

Modality specialists cannot invoke one another; they only return control to the root. Client tools
pause the active specialist and resume that specialist after validation. Durable tools remain
read-only and owner/thread scoped.

## Shared application context

`RequestContext` contains:

- `ClientInfo`: authenticated user ID, username, and admin status;
- `AgentDataAccess`: a narrow owner-scoped interface for database-backed entity information;
- request metadata, selected model, and reasoning settings.

`ScopedAgentDataAccess` exposes unexpired ready references and bounded text ranges for the active
owner and thread. Agents never receive a raw SQLAlchemy session, unrestricted repository, bearer
token, S3 credentials, or provider keys.

## File lifetime

All bucket objects and provider-file references expire after exactly 24 hours. Bucket uploads carry
an expiration header and `expires-at` lifecycle tag; database rows retain the same timestamp so a
cleanup worker can reconcile bucket and provider deletion. OpenAI file gateways must return an
expiring `ProviderFileReference`, and provider deletion remains an application-owned responsibility.

For the current prototype there are no migrations. Schema changes are applied through a hard reset
of the development database.
