# Brainstorm Mode Architecture

This document describes a session-oriented brainstorming mode for the existing
research framework. The goal is to support interactive, interruptible, role-based
deliberation that can later hand off into the current proposal, planning,
implementation, and experiment pipeline.

## Summary

The recommended design is:

- keep the current graph as the execution engine
- add a separate brainstorm orchestration layer
- persist brainstorm sessions through the existing memory layer
- hand off approved brainstorm output into the existing seeded pipeline entry
  points

Brainstorm mode should not be implemented as a single new graph node. The
current graph is linear and run-oriented. Brainstorming is interactive,
checkpointed, user-steerable, and interruptible between turns. It needs a
session state machine.

## Product Behavior

The user flow should look like this:

1. The user starts a brainstorm session with a direction and a profile.
2. The orchestrator runs one or more bounded rounds of internal deliberation.
3. Configured personas discuss, criticize, research, and refine ideas.
4. The orchestrator prints a rolling summary of current thinking.
5. The session pauses when:
   - a configured time limit is reached
   - a configured round limit is reached
   - a configured token or message budget is reached
   - the user interrupts with `Ctrl-C`
   - the orchestrator detects a user-decision checkpoint
6. On pause, the system prints:
   - agreed points
   - current leading options
   - open questions
   - risks and objections
   - recommended next direction
7. The user can then:
   - ask questions
   - redirect the discussion
   - approve or reject ideas
   - ask to ignore a topic
   - ask to deepen research
   - ask to draft a concrete plan
   - ask to implement
   - ask to exit the brainstorm process
8. Once the user approves a plan, the system creates a structured handoff into
   the existing implementation pipeline.

## Core Design

Implement brainstorm mode as a separate subsystem:

```text
core/
  brainstorm/
    __init__.py
    state.py
    engine.py
    roles.py
    prompts.py
    summaries.py
    interrupt.py
    commands.py
    handoff.py
```

Recommended entrypoints:

- `main.py --mode brainstorm`
- optional dedicated CLI wrapper: `brainstorm.py`
- web endpoints under the existing inspector app for session control and
  session inspection

Current CLI usage:

```text
uv run python main.py --mode brainstorm --profile <profile> --direction "..."
uv run python main.py --mode brainstorm --profile <profile> --config configs/brainstorm/default.<profile>.brainstorm.yaml
uv run python main.py --mode brainstorm --profile <profile> --resume-brainstorm "<session-id>"
uv run python main.py --mode brainstorm --profile <profile> --source-experiment "exp-123"
uv run python main.py --mode brainstorm --profile <profile> --source-experiment "exp-123" --proposal-seed "proposal_seed:123"
uv run python main.py --mode brainstorm --profile <profile> --source-experiment "exp-123" --handoff "run_handoff:123"
uv run python main.py --mode brainstorm --profile <profile> --source-experiment "exp-123" --next-step "next_step:123"
```

Seed behavior:

- `--resume-brainstorm` loads an existing brainstorm session state
- `--source-experiment` imports a prior experiment run and proposal context into
  a new brainstorm session
- `--proposal-seed` imports a saved proposal seed into a new brainstorm session
- `--handoff` imports a saved run handoff into a new brainstorm session
- `--next-step` imports a persisted next-step recommendation into a new
  brainstorm session
- `--config` is an alias for `--brainstorm-config`

When seeded from prior work, brainstorm mode carries the imported proposal,
evidence, lineage, and campaign metadata in session state so the user can edit
or redirect the idea before using `execute` to continue into the main pipeline.

Brainstorm mode configuration should be loaded from a dedicated
`brainstorm.yaml` file, with the path supplied at runtime. Different
brainstorm setups should therefore be swappable without changing code or
profile YAML.

Suggested default location:

- `configs/brainstorm/default.<profile>.brainstorm.yaml`, for example
  `configs/brainstorm/default.trading.brainstorm.yaml`

## Architecture Layers

The design should be split into three layers.

### Conversation layer

Responsible for:

- receiving user instructions
- starting and resuming sessions
- pausing on checkpoints
- handling `Ctrl-C` interrupts
- printing progress and summaries
- translating user feedback into structured session commands

This layer owns the human-in-the-loop mechanics.

### Deliberation layer

Responsible for:

- running configured personas
- coordinating rounds
- synthesizing conflicting views
- tracking consensus, objections, and open questions
- invoking research tools for the researcher persona
- generating a plan draft when the user asks for it

This layer owns the brainstorm process itself.

### Execution layer

Responsible for:

- converting approved brainstorm output into pipeline state
- starting the existing graph from:
  - `propose_experiments`
  - `plan_implementation`
  - `implement`
- preserving lineage between brainstorm session output and downstream runs

This layer should reuse the existing graph and adapters.

## Session State

Add a canonical brainstorm state model separate from `ResearchState`.

Suggested file:

- [`core/brainstorm/state.py`](/E:/Programming/trading_researcher/core/brainstorm/state.py)

Suggested state shape:

```python
from typing import Literal, TypedDict


class BrainstormRoleConfig(TypedDict, total=False):
    name: str
    persona_type: str
    goal: str
    style: str
    enabled: bool
    llm_key: str
    model: str
    tools: list[str]
    research_budget: dict


class BrainstormTurn(TypedDict, total=False):
    turn_id: str
    round_index: int
    role_name: str
    role_type: str
    message_type: str
    content: str
    structured_points: list[dict]
    citations: list[dict]
    created_at: str


class ConsensusSnapshot(TypedDict, total=False):
    agreed_points: list[str]
    active_options: list[dict]
    rejected_options: list[dict]
    objections: list[dict]
    assumptions: list[str]
    open_questions: list[str]
    next_recommendation: str
    confidence: str


class PlanDraft(TypedDict, total=False):
    research_direction: str
    refined_ideas: list[dict]
    proposals: list[dict]
    implementation_plans: list[dict]
    constraints: list[str]
    exclusions: list[str]
    success_criteria: list[str]
    unresolved_questions: list[str]


class BrainstormState(TypedDict, total=False):
    session_id: str
    profile_name: str
    status: Literal[
        "running",
        "paused",
        "awaiting_user",
        "planning",
        "approved_for_execution",
        "executing",
        "completed",
        "cancelled",
    ]
    current_goal: str
    user_intent_notes: list[str]
    role_configs: list[BrainstormRoleConfig]
    turn_log: list[BrainstormTurn]
    consensus: ConsensusSnapshot
    plan_draft: PlanDraft
    pending_questions: list[str]
    pending_decisions: list[str]
    stop_policy: dict
    progress: dict
    execution_handoff: dict
    errors: list[str]
```

## Brainstorm Config

Brainstorm configuration should not live in `configs/config.yaml` or inside the
profile YAML as the primary source of truth. It should live in a dedicated
`brainstorm.yaml` document that is passed in as a parameter when the session
starts.

Recommended CLI/API contract:

- `--brainstorm-config <path>`
- if omitted, load a default brainstorm config path
- profile remains responsible for domain prompts, datasets, base classes, and
  adapters
- brainstorm config remains responsible for session mechanics, persona setup,
  orchestration rules, summary cadence, and tool permissions

Recommended config-loading precedence:

1. explicit `--brainstorm-config`
2. per-profile default mapping if one exists
3. fallback default such as `configs/brainstorm/default.<profile>.brainstorm.yaml`

Suggested file:

- [`core/brainstorm/config.py`](/E:/Programming/trading_researcher/core/brainstorm/config.py)

Suggested responsibilities for the brainstorm config loader:

- load YAML from a caller-provided path
- validate required brainstorm sections
- merge with safe defaults
- resolve relative paths
- validate persona tool references
- expose a normalized config object to the engine

Suggested config shape:

```yaml
name: default_brainstorm
description: Default interactive brainstorm setup

llm_defaults:
  provider: anthropic
  model: claude-sonnet-4

stop_policy:
  max_rounds_per_run: 3
  max_messages_per_run: 12
  max_seconds_per_run: 90
  summary_interval_messages: 4
  summary_interval_seconds: 20
  pause_after_research_round: true

summary:
  print_current_thinking: true
  include_evidence: true
  include_open_questions: true

roles:
  - name: facilitator
    persona_type: facilitator
    enabled: true
    llm_key: brainstorm
    model: claude-opus-4
    goal: "Drive the discussion, track consensus, and decide when to pause."
    style: "Concise, structured, neutral."

  - name: skeptic
    persona_type: skeptic
    enabled: true
    llm_key: brainstorm
    goal: "Attack weak assumptions and point out failure modes."
    style: "Direct, critical, evidence-seeking."

  - name: researcher
    persona_type: researcher
    enabled: true
    llm_key: brainstorm
    model: claude-sonnet-4
    goal: "Ground the discussion in evidence from configured research tools."
    tools:
      - tools.research_tools.collect_arxiv
      - core.tools.research_tools.collect_memory
    research_budget:
      max_tools_per_round: 2
      max_artifacts_per_tool: 4

execution_handoff:
  default_start_node: propose_experiments
  allow_direct_to_implement: true
```

The brainstorm config should own:

- persona roster
- persona prompts and styles
- default model settings
- per-persona model overrides
- stop policy
- summary cadence and rendering behavior
- interrupt behavior
- command parsing options
- researcher tool permissions and budgets
- execution handoff defaults

The profile should continue to own:

- domain prompts and datasets
- evaluation thresholds
- base classes
- adapters and execution implementation
- existing pipeline steps

## Persona System

All personas should be configurable. Do not hardcode only one panel.

Suggested default persona types:

- `facilitator`
- `optimist`
- `skeptic`
- `operator`
- `researcher`
- `planner`

Each persona should be configurable through `brainstorm.yaml`, for example:

```yaml
llm_defaults:
  model: claude-sonnet-4

roles:
  - name: facilitator
    persona_type: facilitator
    enabled: true
    llm_key: brainstorm
    model: claude-opus-4
    goal: "Drive the discussion, track consensus, and decide when to pause."
    style: "Concise, structured, neutral."
  - name: skeptic
    persona_type: skeptic
    enabled: true
    llm_key: brainstorm
    goal: "Attack weak assumptions and point out failure modes."
    style: "Direct, critical, evidence-seeking."
  - name: researcher
    persona_type: researcher
    enabled: true
    llm_key: brainstorm
    goal: "Ground the discussion in evidence from configured research tools."
    tools:
      - tools.research_tools.collect_arxiv
      - core.tools.research_tools.collect_memory
```

The brainstorm config should define:

- which roles exist
- which are enabled by default
- which LLM key each role uses
- which model each role uses, if it overrides the brainstorm default
- which research tools each role may call
- what budget limits each role gets
- what prompt template and style each role uses

Model resolution should follow this order:

1. `roles[i].model`
2. `llm_defaults.model` from `brainstorm.yaml`
3. existing framework fallback behavior, if the brainstorm config leaves the
   model unset

The loader should normalize this so the engine receives an explicit model for
each enabled persona, even when it came from the default.

## Researcher Role

The researcher role should reuse the framework's existing research-tool model.

Do not build a parallel research mechanism just for brainstorm mode. Reuse the
current tool contracts wherever possible:

- `collect_arxiv`
- `collect_prior_experiments`
- `collect_adapter_context`
- `collect_profile_context`
- `collect_strategy_library`
- `collect_memory`

Recommended behavior:

- the researcher persona can be invoked explicitly in a round
- the facilitator can request evidence from the researcher when discussion
  becomes speculative
- researcher output should become structured artifacts in the brainstorm state
- research artifacts should be citable in later summaries and plan drafts

Suggested additions in `brainstorm.yaml`:

- role-scoped allowed tools
- per-role budgets
- per-role model overrides
- brainstorm-level default model
- `require_evidence_for_plan`: boolean gate before plan approval

## Orchestrator

Suggested file:

- [`core/brainstorm/engine.py`](/E:/Programming/trading_researcher/core/brainstorm/engine.py)

The orchestrator should not simulate an endless free-form chat. It should run
explicit phases:

1. intake
2. exploration
3. critique
4. research
5. synthesis
6. checkpoint
7. plan drafting
8. approval
9. execution handoff

Recommended engine loop:

```text
while session.status in {"running", "planning"}:
  run one bounded round
  append role outputs to turn log
  update working consensus
  print current-thinking summary if due
  if stop policy triggered:
    pause and print checkpoint summary
  if interrupt flag set:
    pause immediately and print checkpoint summary
  if plan requested and enough consensus exists:
    draft or update structured plan
  if execution approved:
    build handoff and launch pipeline
```

## Periodic Summaries

The orchestrator should print periodic summaries during long runs, not only when
the round ends.

Recommended summary triggers:

- every `N` persona messages
- every `M` seconds
- after each completed round
- immediately before a forced pause
- immediately after `Ctrl-C`

Recommended summary sections:

- `Current goal`
- `Leading ideas`
- `Main objections`
- `Evidence gathered`
- `Open questions`
- `Likely next step`

These summaries should be derived from state, not generated from scratch each
time. The facilitator should update a canonical `consensus` object, and the
printer should render from that object.

## Interrupt Model

The user should be able to interrupt brainstorm execution with `Ctrl-C`.

This should not be treated as an error. It should be a first-class pause signal.

Suggested file:

- [`core/brainstorm/interrupt.py`](/E:/Programming/trading_researcher/core/brainstorm/interrupt.py)

Recommended behavior:

1. Set a session interrupt flag on `KeyboardInterrupt`.
2. Stop scheduling additional persona turns immediately.
3. Preserve all completed turn outputs.
4. Build a checkpoint summary from the latest consensus state.
5. Set session status to `awaiting_user`.
6. Print a clear prompt for the next user action, including `help` and `exit`.

Important detail:

- do not attempt to kill arbitrary subprocesses or background work used by the
  main experiment runner
- brainstorm mode itself should remain synchronous and easy to interrupt
- only downstream execution mode should launch long-running jobs

Post-interrupt behavior should be explicit:

- the first `Ctrl-C` pauses brainstorming and returns control to the user
- once paused, the user can issue `exit` to terminate the brainstorm process
- optionally, a future config flag may allow double-`Ctrl-C` to exit immediately,
  but the minimum design should support an explicit `exit` command after pause

## Stop Policies

Stop policies should be configurable both globally and per session.

Suggested `brainstorm.yaml` config:

```yaml
stop_policy:
  max_rounds_per_run: 3
  max_messages_per_run: 12
  max_seconds_per_run: 90
  summary_interval_messages: 4
  summary_interval_seconds: 20
  pause_after_research_round: true
```

Supported pause reasons should include:

- `round_limit`
- `message_limit`
- `time_limit`
- `user_interrupt`
- `needs_decision`
- `plan_ready`

## Command Model

User feedback should be parsed into explicit commands, not just appended to chat
history.

Suggested command types:

- `set_goal`
- `add_constraint`
- `add_exclusion`
- `approve_option`
- `reject_option`
- `request_research`
- `request_summary`
- `draft_plan`
- `edit_plan`
- `approve_plan`
- `start_execution`
- `help`
- `pause`
- `exit`
- `cancel`

Suggested file:

- [`core/brainstorm/commands.py`](/E:/Programming/trading_researcher/core/brainstorm/commands.py)

The CLI and web layer should translate user input into these commands before
resuming the engine.

The minimum CLI command set shown to the user after each pause should include:

- `help`: show available commands and short descriptions
- `continue`: resume brainstorming with the current state
- `summary`: print the latest current-thinking summary again
- `research`: request another evidence-gathering pass
- `plan`: draft or refresh the structured plan
- `feedback <text>`: add free-form direction or corrections
- `approve_plan`: mark the current plan draft as approved
- `execute`: hand off the approved plan into the execution pipeline
- `exit`: stop the brainstorm process without executing

Suggested CLI help text:

```text
Available commands:
  help           Show this command list
  continue       Resume the brainstorm session
  summary        Show the latest checkpoint summary
  research       Ask the researcher role to gather more evidence
  plan           Draft or refresh the current plan
  feedback ...   Add feedback, constraints, or direction
  approve_plan   Approve the current plan draft
  execute        Start downstream execution from the approved plan
  exit           Exit brainstorm mode
```

## Memory and Persistence

Brainstorm sessions should reuse the canonical memory system, but with new object
types and records.

Suggested new memory object types:

- `brainstorm_session`
- `brainstorm_turn`
- `brainstorm_summary`
- `brainstorm_plan`
- `brainstorm_decision`

Suggested persistence points:

- after each completed round
- after each checkpoint summary
- after each plan draft update
- before execution handoff

This gives:

- resumability
- auditability
- plan lineage
- retrieval of prior brainstorms during later runs

Suggested metadata for summaries and plans:

- `session_id`
- `profile_name`
- `status`
- `current_goal`
- `approved`
- `root_run_family_id`
- `source_brainstorm_session_id`

## Handoff Into Existing Pipeline

Suggested file:

- [`core/brainstorm/handoff.py`](/E:/Programming/trading_researcher/core/brainstorm/handoff.py)

The handoff boundary should be explicit and typed.

Recommended handoff payload:

```python
{
    "profile_name": "trading_researcher",
    "research_direction": "...",
    "refined_ideas": [...],
    "proposals": [...],
    "implementation_plans": [...],
    "proposal_seed_planning_notes": "...",
    "constraints": [...],
    "exclusions": [...],
    "source_brainstorm_session_id": "...",
    "source_brainstorm_plan_record_id": "...",
}
```

Start-node rules:

- if the user only approved direction and ideas:
  - start at `propose_experiments`
- if the user approved structured proposals:
  - start at `plan_implementation`
- if the user approved implementation plans:
  - start at `implement`

This aligns with the current seeded-entry behavior already used by
`proposal_seed` and `resume-from` flows.

## CLI Design

Recommended commands:

```text
uv run python main.py --mode brainstorm --profile trading_researcher --direction "..." --brainstorm-config configs/brainstorm/default.trading_researcher.brainstorm.yaml
uv run python main.py --mode brainstorm --profile trading --resume-brainstorm "<session-id>" --brainstorm-config configs/brainstorm/trading_panel.brainstorm.yaml
uv run python main.py --mode brainstorm --profile trading_researcher --brainstorm-config configs/brainstorm/research_heavy.brainstorm.yaml
```

Recommended CLI output model:

- stream persona outputs round by round
- print periodic "current thinking" summaries
- on pause, print a compact checkpoint summary
- on `Ctrl-C`, stop immediately and print the latest checkpoint
- prompt for next action when in interactive mode
- support `help` to print the available command list
- support `exit` to terminate the brainstorm process cleanly

Minimal first version:

- synchronous loop
- `Ctrl-C` pause handling
- periodic summaries
- plan drafting
- handoff into existing pipeline

## Web API Design

Extend the existing web app with session APIs:

- `POST /api/brainstorm/sessions`
- `GET /api/brainstorm/sessions/{session_id}`
- `POST /api/brainstorm/sessions/{session_id}/resume`
- `POST /api/brainstorm/sessions/{session_id}/commands`
- `POST /api/brainstorm/sessions/{session_id}/plan`
- `POST /api/brainstorm/sessions/{session_id}/execute`

The web UI can then provide:

- live session transcript
- current consensus panel
- periodic summary panel
- editable plan draft
- approve-and-execute action

## Implementation Phases

### Phase 1

Build the minimum useful path:

- `BrainstormState`
- persona config loading from profile
- synchronous orchestrator with bounded rounds
- facilitator, skeptic, researcher personas
- periodic summary rendering
- `Ctrl-C` pause handling
- `help` and `exit` interactive commands
- persistent session and summary records
- handoff into `propose_experiments` or `plan_implementation`

### Phase 2

Improve plan quality and control:

- structured command parser
- editable plan drafts
- planner persona
- richer evidence attachment and citations
- better consensus tracking and conflict markers

### Phase 3

Operationalize:

- web UI
- session resume from memory
- prior brainstorm retrieval in research tools
- campaign generation from brainstorm outputs
- optional multi-branch exploration from one brainstorm root

## Test Plan

Add a dedicated test module set:

```text
tests/
  test_brainstorm_state.py
  test_brainstorm_engine.py
  test_brainstorm_interrupt.py
  test_brainstorm_handoff.py
  test_brainstorm_commands.py
```

Key tests:

- role config loading from profile
- researcher role only uses allowed tools
- periodic summary generation from consensus state
- `Ctrl-C` converts running session into `awaiting_user`
- plan draft handoff chooses the correct pipeline start node
- brainstorm memory records persist with correct lineage
- approved brainstorm plans produce valid `proposals` or
  `implementation_plans`

## Recommendation

The recommended implementation choice is:

- keep brainstorm mode inside this repo
- implement it as a separate execution mode
- keep a strict separation between interactive deliberation and downstream
  execution

This gives the best tradeoff:

- reuse existing profiles, tools, memory, and pipeline code
- avoid forcing interactive behavior into the linear graph builder
- preserve a clean handoff into implementation and experiments
- keep the design extensible for multiple domains and configurable personas

## First Build Target

The first real milestone should be:

1. start a brainstorm session from a profile and direction
2. run `facilitator`, `skeptic`, and `researcher` for up to 3 rounds
3. print summaries every round and every 20 seconds
4. pause cleanly on `Ctrl-C`
5. show `help` with available commands when requested
6. let the user revise direction, ask for a plan, or exit
7. emit a structured approved plan
8. launch the existing pipeline from the correct downstream node

That is enough to prove the architecture before adding a richer UI or more
complex persona behaviors.
