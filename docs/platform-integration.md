# Quant Crucible researcher integration

The platform entry point is `core.platform`, a private HTTP service with a small
async Python client. It is separate from the legacy inspector (`web.app`), CLI,
shared memory stores, and deterministic `core.plugins.trading.ui_agent` demo.
Those legacy interfaces are **not** tenant-safe platform endpoints.

Only the QC API server calls this service. The production UI must call QC's public
`/v1/researcher/...` gateway, never this service directly. QC keeps a backend-only
schema at `src/api/internal/researcher-openapi.json`; it is not a frontend contract.
Restrict service ingress to the QC API host/network identity and keep its tenant
credentials server-side. The development playground is not a production access path.

Successful turns include usage; failed authoring turns and inspected request receipts
also include the durable request usage report. Unknown provider counts remain null in
the call events. Errors before a request is admitted, or unavailable storage, can have
usage=null; this is not a claim of zero billed tokens. QC persists call events against
its authenticated user, deduplicates call IDs, and checks user allowances independently.

## Why a service, and when to create a job

Ordinary conversation is a bounded `POST /v1/turns` request. There is no subprocess,
queue submission, autonomous research loop, or backtest on each message. Each
accepted request gets a durable receipt for idempotency, cancellation, recovery,
and metering; a receipt is not a scheduled job.

| Action | Work | Default deadline | LLM calls |
| --- | --- | --- | --- |
| `chat` | Answer using exact draft and supplied conversation | 45 seconds | 1 |
| `research` | Synthesize supplied evidence; optionally retrieve public arXiv abstracts | 90 seconds | 1 |
| `ideate` | Generate testable hypotheses and risks | 60 seconds | 1 |
| `code` | Propose full source for an algorithm or portfolio | 120 seconds | 1 |
| `build` | Code from the confirmed specification, then static validation | 130 seconds | 1 |
| `validate` | Read-only syntax, base class, method signature, metadata checks | 10 seconds | 0 |
| `workflow` | Research → ideation → coding → static validation | 240 seconds | 3 |

QC can call any action directly. For work that must survive a browser/API restart,
QC should submit a job through its existing executor architecture and have that
handler call the same endpoint. In particular, `workflow` is a natural executor
operation. This service does not introduce a competing durable job scheduler.

The service does not automatically classify a chat message into a code/workflow
request. The caller chooses an explicit action. Stop, read, replay, and validation
operations do not call an LLM. There is no automatic provider retry in the platform
profile, so hidden SDK retries do not silently increase usage.

### Live progress

While `POST /v1/turns` is running, open
`GET /v1/requests/{requestId}/events` with the same tenant bearer credential.
This SSE endpoint only observes work; it never submits or retries a turn. An initial
404 means the POST has not claimed that request yet (or it is not visible to this
tenant); retry the GET while the POST is pending. Use fetch streaming when sending
Authorization headers from a browser.

`progress` events contain `tenantId`, `requestId`, `sequence`, `stage`, `message`,
and `at`. Their SSE `id` is the sequence. Reconnect with `?after=N` or
`Last-Event-ID: N` to replay only newer events. `complete` contains `tenantId`,
`requestId`, `status`, and `stage`, and closes the stream for every terminal outcome.
The final result/error remains on the POST response and request receipt. Receipts
also include the full `progress` array. A disconnected observer does not cancel
execution; use the stop endpoint. Proxies must allow streaming without buffering;
the service sets `X-Accel-Buffering: no` and sends idle heartbeats every 15 seconds.
Apply migration `002_request_progress.sql` before updating the service.

## Ownership and tenant boundary

QC owns authenticated users, tenant membership, component/version ownership,
session/message persistence, working drafts, proposal application, runtime
validation, version publication, and execution jobs. The researcher receives an
authorized snapshot and returns an answer plus an optional pending proposal.
It has no route to publish, delete, register, or upgrade a component.

Every credential maps to exactly one tenant using `RESEARCHER_TENANT_KEYS`.
The request's `tenantId` must match that mapping. A header/body tenant ID alone
never grants authority. All receipt, cancellation, event, and summary queries are
scoped to the authenticated tenant, including identical request IDs in two tenants.
Keys stay in QC/server secret configuration, never in the browser or request body.
Provision random credentials of at least 32 characters. Multiple credentials may
map to the same tenant during rotation; restart service processes after rotation.

QC must check ownership of **all** component/version/draft/session/evidence
references before building the request. The researcher cannot independently
verify a QC record ID against a registry it does not own. Nested context inherits
the authorized request tenant. No shared vector memory, MongoDB, Neo4j, MLflow,
filesystem discovery, arbitrary import path, or caller-selected command is enabled
on this platform path. Source/evidence text cannot grant additional tool authority.

## Run locally

From the researcher repository, set environment variables through your secret
manager or shell. Example shape (replace the placeholder with a random secret):

```text
RESEARCHER_TENANT_KEYS={"replace-with-a-random-32-character-or-longer-secret":"tenant_acme_quant_01"}
RESEARCHER_DATABASE_URL=postgresql://qc_researcher:<service password>@z440.lan:5432/qc-researcher
RESEARCHER_LLM_PROVIDER=anthropic
RESEARCHER_LLM_MODEL=<your configured model ID>
ANTHROPIC_API_KEY=<provider secret>
```

For OpenAI, set provider `openai` and `OPENAI_API_KEY` instead. Provider/model
defaults otherwise come from the existing configuration/factory. Explicit
per-step model choices in the trusted profile override its default model.

```powershell
uv sync --frozen --only-group platform --no-install-project
uv run --env-file configs/platform.env --no-sync python -m core.platform
```

The default bind is `127.0.0.1:8091`; configure `RESEARCHER_HOST` and
`RESEARCHER_PORT` when needed. `/health` checks liveness, `/ready` checks the
receipt database, `/docs` serves interactive documentation, and `/openapi.json`
serves the contract. Startup fails when credentials are absent or invalid, the
database is unavailable/unmigrated, RLS is disabled, or the database role can bypass
tenant isolation. `configs/platform.env` is an ignored local secret file; the
launcher above loads it. Ordinary environment variables work in deployed services.
Model/provider connectivity is not tested by readiness.

The smaller `platform` dependency group excludes legacy Ray, MLflow, trading,
and research-storage dependencies. Use `uv sync --frozen` to restore the full
development environment before running the complete legacy test suite.

`Dockerfile.platform` runs the same service as a non-root user. Build with
`docker build -f Dockerfile.platform -t qc-researcher .`, supply secrets at runtime,
and point all replicas at the same PostgreSQL database. No local state volume is
needed. The image excludes both `configs/.env` and `configs/platform.env`.
Put network deployments behind TLS and
private service access. Pin `RESEARCHER_RELEASE` to the deployed commit/image
release; otherwise provenance explicitly reports `development`.

## Call from QC

The service is independently deployed; QC can use plain HTTP and the exported
OpenAPI contract without importing this repository. A reference async client is
provided in `core/platform/client.py` for an environment containing this code.
The service repository still uses its existing source layout rather than an
installable wheel; the container and launch command use the repository root.

```python
from uuid import uuid4
from core.platform.client import ResearcherClient
from core.platform.models import TurnRequest, content_hash

# These values come from QC's authorized tenant/session/draft services.
turn = TurnRequest(
    tenantId=authorized_tenant,
    requestId=str(uuid4()),
    sessionId=session_id,
    action="code",
    message="Add a configurable volatility filter; preserve the existing entries.",
    messages=accepted_conversation,
    component={
        "componentId": component_id,
        "kind": "algorithm",
        "baseVersionId": base_version_id,
        "draftId": draft_id,
        "revision": revision,
        "sourceCode": current_source,
        "contentHash": content_hash(current_source),
        "interfaceContext": exact_engine_interface,
        "parameterSchema": parameter_schema,
        "validationDiagnostics": diagnostics,
    },
)
async with ResearcherClient(service_url, tenant_credential, authorized_tenant) as client:
    result = await client.turn(turn)
    # Persist the answer/proposal in QC. Do not automatically apply or publish.
```

Conversation consists of accepted `user`/`assistant` messages, not user-supplied
system instructions. Include the current draft, including any edits persisted by
QC immediately before submission. For a new component, allocate a provisional
component/draft ID, send empty source with its SHA-256, and use null `baseVersionId`.
For portfolio generation, supply the actual order/position API in `interfaceContext`;
the profile instructs the model to explain missing interfaces instead of inventing them.

A proposal carries `tenantId`, session/component/draft IDs, base version,
`baseDraftRevision`, `baseContentHash`, proposed `contentHash`, explanation, and
`changes: [{file, before, after}]`. Filenames are service-assigned (`algorithm.py`
or `portfolio.py`), not model-selected filesystem paths. Proposals remain pending.

**QC's apply transaction must compare both draft revision and content hash, update
the draft once, increment its revision, and invalidate previous validation.** A
proposal is stale if the user edited the draft while generation was running. The
researcher cannot apply that transaction; it deliberately has no draft mutation API.

## Research and profile configuration

`configs/platform_profiles/quant_crucible.yaml` defines the prompts, engine
interfaces, model budgets, and bounded operation sequence. This is a platform
profile, not a legacy `main.py --profile` pipeline. Set `RESEARCHER_PROFILE_PATH`
to an operator-controlled alternative. Customization cannot change operation
boundaries or add arbitrary tools/automatic execution. Responses record the
profile name, version, content hash, executed steps, and service release.

Use `evidence: [{id, title, text, url}]` for authorized dataset summaries, prior
findings, papers, and attachments already resolved by QC. The service never
fetches caller-supplied evidence URLs. Model citation IDs are checked against the
supplied/retrieved evidence set.

An explicit `researchQuery` on `research` or `workflow` sends **only that query**
to the fixed public arXiv endpoint, with five abstracts maximum, a 15-second HTTP
timeout, and a 1 MB response cap. It does not send source/history to arXiv, scrape
arbitrary URLs, download papers, or use a cross-tenant cache. Without a query it
synthesizes supplied evidence and must acknowledge missing evidence. Provider or
research retrieval failures fail the request instead of substituting fabricated results.

## Validation boundary

Validation parses and compiles Python without executing/importing it. It checks
one direct Algorithm/Portfolio subclass imported from the expected engine module,
the required synchronous method signature, and literal metadata with the correct
component role. It neither fixes source nor runs model-generated tests.

Results identify the exact content hash, revision, checks, diagnostics, and whether
the checked source is the draft or a proposal. `scope="static"` and
`publishable=false` are always explicit. `status="passed"` means those static checks
passed; runtime checks are listed as skipped. This is not a sandbox, complete engine
compatibility check, profitability assessment, or authorization to save a version.
QC must run its isolated runtime validator before publication. Indirect inheritance
and qualified-base imports are outside this initial static contract.

## Retries, progress, stop, and failure recovery

- A request ID is an idempotency key scoped to the tenant. Same ID and exact input
  after success returns the persisted result without a new LLM call.
- Different input with an existing ID returns `409 idempotency_conflict`.
- An in-flight or failed/stopped/interrupted request returns 409 on resubmission.
  A deliberate new attempt needs a new ID; QC should link it to the original turn.
- `GET /v1/requests/{requestId}` returns the durable status, stage, timestamps,
  final result, and sanitized error. Poll only while needed; ordinary chat can
  simply await the initial POST. Progress reports real stages, not fabricated percentages.
- `POST /v1/requests/{requestId}/stop` fences publication immediately. The running
  request notices the persisted stop and cancels its async work. Already-completed
  results remain completed; stop cannot turn success into cancellation.
- Completion is conditional on an active, unexpired receipt. Late output after
  stop/expiry is discarded. Partial code is never published as a proposal.
- After a process crash, expired receipts become `interrupted` when inspected.
  They do not automatically restart or silently repeat paid calls. Inspect usage
  before creating a new attempt. Work is not durable across process death; QC's
  executor owns deliberate recovery of long workflows.
- The client does not retry transport timeouts automatically. Inspect the same
  request ID because the service may have finished despite a lost response.
- Default active limits are two requests per tenant and 32 per database. Capacity
  returns 429 before starting a call. Request bodies are capped at 512 KB and the
  validated context at 300,000 characters. Sessions/history must be summarized by QC.

## Token meter

All platform LLM calls pass through `LoggedChatModel`. Metering supports sync,
async, and streaming calls using request-local context. Each event has tenant,
request, session, call ID, operation, provider, requested model, provider-reported
model when available, input/output/total tokens, call status, and timestamps.
Events contain no prompts, source, answers, or credentials.

The ledger records `started` **before** contacting the provider and updates that
same event after completion, failure, or interruption. A storage failure before
that first write prevents the provider call. Usage is retained even if later JSON
parsing, validation, or result publication fails. Replaying a result adds no usage.

`GET /v1/usage/events?requestId=...&page=1&pageSize=100` gives paginated event details.
`GET /v1/usage/summary?since=<ISO timestamp>&until=<ISO timestamp>` groups totals by
provider/model; the time interval is start-inclusive/end-exclusive and requires
timezones. Both endpoints derive the tenant from the credential.

Absent provider usage is **unknown**, represented by null tokens and
`usageAvailable=false`. Summary fields are `knownInputTokens`, `knownOutputTokens`,
and `unknownUsageCalls`. Turn-level token totals also sum only known counts; inspect
the event flags before treating them as complete. Requested and reported models
are kept separately, so an alias is not falsely presented as a resolved model ID.
This is an observed-usage meter, not an invoice: a crash, cancellation, or provider
failure can leave billed tokens unknown. No prices or inferred token counts are fabricated.

## PostgreSQL storage and deployment

The dedicated database is **`qc-researcher`** on `z440.lan:5432`. QC's `qc` database
continues to own platform entities. Researcher tables and its migration ledger live
in the `researcher` schema. Results/events use JSONB and timestamps use `timestamptz`.
There is no SQLite fallback or automatic table creation at service startup.

`qc_researcher_owner` is a non-login migration owner. `qc_researcher` is the restricted
service login: no superuser, BYPASSRLS, database/role creation, or owner membership.
Runtime access grants SELECT/INSERT/UPDATE only on the two tenant tables; it cannot
alter schemas, truncate/delete tables, or edit the migration ledger. Both tables use
composite tenant keys and forced row-level security. Every transaction sets the
authenticated tenant with transaction-local `app.tenant_id`. Missing identity sees
no tenant data. This is defense against missing query filters; the trusted service
credential must remain private because it supplies the authenticated tenant context.

Claims serialize through a short database advisory lock. A narrowly granted
security-definer function returns only the global active-request count, allowing a
shared capacity limit without exposing another tenant's IDs or results. Its owner
has a SELECT-only owner policy; the runtime role cannot assume that owner role.
All replicas must use the same capacity configuration. Stops/completions use
conditional updates, deadlines use the database clock, and expired receipts do not
consume capacity. No transaction is held over LLM calls or research HTTP requests.
Usage events commit separately so failures in output parsing do not lose accounting.
Terminal usage records cannot be overwritten by delayed initial events.

For a new server, supply administrator credentials through environment variables
and a generated service password of at least 32 characters:

```powershell
# RESEARCHER_ADMIN_DATABASE_URL targets postgres on the desired server.
# RESEARCHER_DATABASE_PASSWORD is the generated runtime password.
uv run --no-sync python -m database.provision
# RESEARCHER_MIGRATION_DATABASE_URL targets qc-researcher as an account
# authorized to SET ROLE qc_researcher_owner (the provisioner does not save it).
uv run --no-sync python -m database up
uv run --no-sync python -m database check
```

The standalone [database migration system](../database/README.md) has its own
SQL versions, dependency group, CLI and deployment image. Use `python -m database
status` to inspect history and `python -m database new <name>` to create a version.
Provisioning preserves existing role passwords. Migrations are transactional,
serialized, versioned and checksummed; edit drift is rejected. Add a new SQL file
instead of modifying an applied migration. Never give the running service the
administrator/migration URL. Back up this database alongside QC; receipt results
can contain proprietary source. Set appropriate TLS parameters in the connection
string for your deployment. Existing SQLite files are not imported automatically;
retain any historical files until their accounting has been explicitly reconciled.

Persist accepted conversations/drafts in QC rather than making this receipt database
a second component registry. Old legacy memory stores have not been retrofitted
with tenancy; they are excluded from the platform call path. There is currently no
automated receipt retention/purge or usage billing export.

## Verification

```powershell
# RESEARCHER_TEST_ADMIN_URL targets postgres on a test-capable server.
.venv\Scripts\python.exe scripts/test-platform.py tests -q
.venv\Scripts\python.exe -m core.platform.export_openapi
```

`docs/platform-openapi.json` is generated from the actual app. Tests cover concurrent
tenants, identical IDs across tenants, denied cross-tenant reads/stops, replay,
capacity, cancellation/late publication, expiry/restart, malformed output, usage
failures, exact source/revision binding, research retrieval limits, and client calls.
Database tests create and remove uniquely named `qc-researcher-test-*` databases and
temporary restricted logins. They never write test data into `qc` or `qc-researcher`.
Without the explicit test administrator URL, direct pytest skips database tests;
the `scripts/test-platform.py` runner instead fails so integration verification
cannot silently omit them. Optional local Docker setup is in `scripts/test-postgres.sh`
and `compose.test-postgres.yaml`; use `scripts/test-platform.py --local` with it.
LLM and public research calls are mocked; tests do not assert model quality,
live-provider availability, or runtime trading safety.
# User ownership

QC is the source of truth for users, memberships and authentication. The researcher
stores only `(tenant_id, user_id)` references in `researcher.users`; it does not store
passwords or offer login endpoints. Its API is private: the UI must call QC, never
the researcher directly.

Every private `/v1` request requires both the tenant service bearer credential and
`X-User-ID`. QC sets this header from its authenticated user (or configured development
identity while authentication is disabled). Never forward a browser-supplied identity.
The request body cannot select a user. QC checks returned user IDs before accepting
results or recording usage.

Requests and LLM events have required user ownership, foreign keys to the local user
references, and forced row-level security for both tenant and user. Stored proposals,
source, evidence and validation belong to their containing request. Responses and
cached proposals/validation expose `userId`; usage events retain provider, requested
model, actual model, input/output tokens and status. Usage pages and summaries report
only the calling user's usage within the tenant. Tenant-wide concurrency limits still
count all users. Request IDs remain unique within a tenant.

Migration `003_user_ownership.sql` backfills existing records and cached responses.
Before applying it, set `RESEARCHER_LEGACY_USER_ID` to the QC seeded owner's ID. Without
that setting it uses the reserved `legacy-researcher-owner` attribution identity.
This is an attribution for historical records, not evidence of their original creator.
The migration is transactional and leaves existing token counts and models intact.
Deploy the migration and updated QC/researcher services together; an old researcher
process cannot access rows without the new user scope. Restart running processes.

Trusted Python HTTP clients should pass `ResearcherClient(..., user_id=qc_user_id)`.
Trusted in-process callers should wrap work in `core.platform.identity.user_scope(user_id)`;
the default is the legacy attribution identity, not an authenticated user.
For the standalone playground, set `RESEARCHER_PLAYGROUND_USER_ID` to its development
user ID. Its configuration and browser draft storage include this identity.
