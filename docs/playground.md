# Researcher playground

From the repository root:

```powershell
python scripts/start-playground.py
```

Open **http://127.0.0.1:8091**. The launcher starts a development server with reload,
serving both the UI and researcher API. It uses the existing `.venv`, or creates an
isolated `.tmp/playground-venv` using `uv` and the platform dependencies.

The backend uses **Claude Haiku 4.5** by default. Configure `ANTHROPIC_API_KEY` in the
shell or `configs/.env`, and `RESEARCHER_DATABASE_URL` in the shell or ignored
`configs/platform.env`. The existing local database configuration targets
`qc-researcher` on `z440.lan`. No migrations run automatically.

```powershell
python scripts/start-playground.py --port 8092
python scripts/start-playground.py --no-reload
python scripts/start-playground.py --model claude-sonnet-4-6
```

Use `--no-reload` if the reload subprocess hangs on Windows; restart manually after edits.

Choose a tenant and component type. **Send** always discusses the current idea without
changing source. **Research this** investigates the message, optionally using the
arXiv query, and carries retrieved evidence into subsequent turns.

Use **Draft specification from discussion** to summarize decisions, assumptions and
open questions into an editable working specification. Review or edit it, then click
**Confirm and generate**. That sends the exact specification to the `build` action:
one coding call followed by static validation, without repeating research or ideation.
Unresolved requirements can produce a clarification instead of code. Pending unsent
messages must be sent or incorporated into the specification before confirmation.

**Apply to local draft** accepts the proposed source in this browser only. Use
**Revise code** with an explicit message for further code changes, and **Validate draft**
for static checks without an LLM call. Normal discussion retains pending proposals.
Supply portfolio order interfaces under Extra context when needed.

The specification is saved in localStorage separately for each tenant and component
type and survives reloads/new conversations. Conversation and source drafts remain
browser-session-only. Confirmation must be repeated after reload; generated code is
never automatically applied or published. These are playground storage conventions,
not the production API's authoritative session/specification persistence.

Every model stage is instructed to stay focused on creating trading algorithms or
portfolios. Unrelated requests get a short redirection; mixed requests receive help
with only the component-related part. Supporting research, mathematics, data, risk,
and debugging remain in scope. These are model instructions, not a deterministic
topic filter or a guarantee of model compliance.

Stop cancels an active request. Inspect last request can recover a result after an
uncertain HTTP outcome. Sending again uses a new request ID. The usage panel shows
durable tenant totals and the current request's model/token events.

Activity updates appear inside the chat beneath each request: research, article retrieval,
synthesis, ideation, coding, response checks, and static validation. Updates include
timestamps and are retained in the tenant's PostgreSQL receipt. The stream reconnects
from the last received sequence without starting another model call. Inspect last
request also restores activity. This shows operational progress, not model reasoning
or individual generated tokens. Apply database migration `002_request_progress.sql`
through the separate migration runner before starting the updated server.

Invalid authoring responses show a collapsible **Raw model output** text box.
The playground retains that output in the tenant's failed request receipt, so
Inspect last request can retrieve it without repeating a paid call. This capture
is enabled only by the playground; older failures from before it was enabled
do not contain the discarded output. The normal service still returns sanitized
errors unless explicitly configured to capture model output.

Two test tenants (`playground-a` and `playground-b`) have separate browser sessions
and database records. Browser state resets on reload; metering and request receipts
remain in PostgreSQL. Service credentials are generated on server startup and exposed
to this development UI; restarting/reloading the backend rotates them, so refresh the
browser afterward. The Anthropic key is used by the backend.

This is a development-only entry point. The normal `python -m core.platform` service
does not expose the playground or its configuration endpoint.
