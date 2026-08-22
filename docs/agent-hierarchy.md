# Agent hierarchy and application context

The application has one root agent and five non-recursive specialist subagents. Specialists are
invoked as Agents SDK tools, so they inherit the same typed `AgentContext[RequestContext]` as the
root. No identity, database session, or file credential is copied into model input.

```text
Root conversation agent
├── Ingestion strategist
├── Document specialist
├── Structured-data specialist
├── Media specialist
└── Image specialist
```

## Root conversation agent

Owns the ChatKit turn, resolves file references, chooses browser or durable evidence, delegates
interpretation, and synthesizes the final answer. It is the only agent that controls the user-facing
conversation. Browser tools pause the root turn and resume it after backend Pydantic validation.

For a file's initial ingestion, the root follows a fixed orchestration contract: obtain bounded
overview evidence, consult the route-specific specialist, pass that overview with metadata and user
intent to the ingestion strategist, then synthesize an overview and proposed ingestion strategy.
The specialist pair advises; deterministic backend policy remains the authority for executable plan
validation and approval.

## Behavioral test execution

Agent tests keep the same root/specialist hierarchy and typed application context as production.
The test driver wraps the root in a bounded outer continuation loop. When the SDK run stops at a
browser tool, a fixture client executes the operation against the staged test file, the production
Pydantic model validates the result, and the driver resumes from the complete SDK input history with
the validated output attached to the original function-call ID. Specialist tools continue inside
the SDK run normally; only the browser boundary is simulated.

This makes the tests exercise three distinct contracts: the root chooses an appropriate browser
operation, untrusted client results satisfy the backend schema, and the root delegates overview then
strategy after seeing the returned evidence. The continuation loop has a hard round limit so a
model repeatedly requesting client work fails clearly instead of hanging the suite.

## Specialists

- **Ingestion strategist:** recommends representation and processing plans. It does not execute or
  approve plans.
- **Document specialist:** interprets text and PDF evidence while preserving page/layout provenance.
- **Structured-data specialist:** interprets bounded CSV/JSON evidence and proposes narrow follow-up
  queries.
- **Media specialist:** handles timestamp-aligned transcript and video-frame strategy.
- **Image specialist:** interprets supplied visual evidence and recommends bounded batching.

Specialists cannot invoke one another. They may read the same conversation-scoped durable reference
list, but mutating storage and plan transitions remain deterministic application operations.

## Shared application context

`RequestContext` contains:

- `ClientInfo`: authenticated user ID, username, and admin status;
- `AgentDataAccess`: a narrow owner-scoped interface for database-backed entity information;
- request metadata, selected model, and reasoning settings.

`ScopedAgentDataAccess` currently exposes only unexpired, ready file references for the active
owner and thread. It returns stable `@asset_id` references and preview paths. Agents never receive a
raw SQLAlchemy session, unrestricted repository, bearer token, S3 credentials, or provider keys.

## File lifetime

All bucket objects and provider-file references expire after exactly 24 hours. Bucket uploads carry
an expiration header and `expires-at` lifecycle tag; database rows retain the same timestamp so a
cleanup worker can reconcile bucket and provider deletion. OpenAI file gateways must return an
expiring `ProviderFileReference`, and provider deletion remains an application-owned responsibility.

For the current prototype there are no migrations. Schema changes are applied through a hard reset
of the development database.
