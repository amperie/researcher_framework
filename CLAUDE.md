# NeuralSignalResearcher — Claude Code Guide

## Project Vision

A **configuration-driven, plug-and-play agentic research workflow** built on LangGraph.
The system automates the full loop of:

  literature review → ideate → refine → propose experiments → plan → implement → validate →
  create dataset → run experiment → create model → evaluate → store results → propose next steps

It is **not coupled to any specific domain**. The same pipeline code runs neuralsignal
probing experiments today and automated trading strategy research tomorrow — the domain
is fully described by a **research profile** YAML file. Adding a new research domain
requires only a new profile and (if needed) a thin plugin adapter.

---

## Design Principles

1. **Profile over code** — `configs/profiles/<name>.yaml` is the single source of truth for
   a domain. It declares: which steps run, in what order, with which LLM models, prompts,
   research sources, base classes, datasets, evaluation metrics, and storage targets.
   No domain knowledge lives anywhere else.

2. **Per-profile step list** — `pipeline.steps` is an ordered list of step names. Omit a
   step to skip it entirely. `graph/builder.py` compiles the LangGraph at runtime from
   this list.

3. **Prompts belong in the profile** — a neuralsignal ideation prompt bears no resemblance
   to a trading strategy ideation prompt. All prompts live under `prompts.<step_name>` in
   the profile YAML. Node code reads via `utils/profile_loader.py::get_prompt(profile, step, key)`.
   No prompt strings in node files.

4. **Implementation always subclasses** — the `implement` step generates Python code that
   subclasses a base class declared in `profile['base_classes']`. The base class API is
   injected verbatim into the code-generation prompt. Generated files are cached at
   `dev/experiments/<profile_name>/implementations/<ClassName>.py`.

5. **Validate with auto-run + retry** — the `validate` step generates pytest tests, runs
   them via subprocess, and on failure asks the LLM to fix the implementation. Loops up to
   `profile['validate']['max_fix_retries']` times. Failure after max retries appends to
   `state['errors']` but does not abort the pipeline.

6. **Plugin adapters for execution** — `profile['experiment_adapter']` names a Python module.
   `plugins/loader.py::load_adapter(profile)` imports it and calls `get_adapter()` if present,
�   +-- task_runner.py         # generic subprocess runner for dotted Python callables
   returning a `ResearchAdapter` instance; otherwise falls back to legacy module-level functions.
   Node code only calls `load_adapter()` — it never imports plugin internals directly.

7. **LLM-agnostic** — `get_llm(step_name, profile)` in `llm/factory.py` resolves the model:
   step override → profile default → `Config.llm_model` → provider built-in.

8. **Observability first** — `store_results` logs every run to MLflow (metrics), ChromaDB
   (semantic search), and MongoDB (full records).

---

## Repository Layout

```
NeuralSignalResearcher/
├── CLAUDE.md
├── README.md
├── pyproject.toml
│
├── configs/
│   ├── config.yaml            # global: LLM provider, backends, logging levels
│   ├── config.py              # Pydantic BaseSettings (reads .env + env vars)
│   └── profiles/
│       ├── neuralsignal.yaml  # full profile: neuralsignal probing experiments
│       └── trading.yaml       # (stub) profile: automated trading research
│
│   # NO shared prompts.py — every prompt lives inside its profile YAML
│
├── graph/
│   ├── state.py               # ResearchState TypedDict — all generic keys
│   ├── builder.py             # build_graph(profile) → compiled LangGraph
│   └── nodes/                 # one file per step; all export <name>_node(state, profile)
│       ├── research.py            # arxiv search, scoring, summarisation, digests
│       ├── ideate.py              # brainstorm ideas from research context
│       ├── refine.py              # filter ideas for feasibility
│       ├── propose_experiments.py # specify datasets, hyperparameters, success criteria
│       ├── plan_implementation.py # structured JSON plan per proposal (no code)
│       ├── implement.py           # generate + cache FeatureSetBase subclass
│       ├── validate.py            # generate pytest tests + auto-run + fix-retry loop
│       ├── prepare_experiment.py  # delegate to adapter.prepare_experiment()
│       ├── execute_experiment.py  # delegate to adapter.execute_experiment()
│       ├── evaluate.py            # LLM analysis of metrics against profile thresholds
│       ├── store_results.py       # MLflow + ChromaDB + MongoDB persistence
│       └── propose_next_steps.py  # suggest follow-up directions
│
+-- plugins/                   # domain adapters, thin, no orchestration logic
�   +-- base.py                # ResearchAdapter Protocol (adapter contract)
�   +-- loader.py              # load_adapter(profile), resolves get_adapter() or legacy module
�   +-- task_runner.py         # generic subprocess runner for dotted Python callables
�   +-- neuralsignal/
�   �   +-- adapter.py          # ResearchAdapter skeleton for neuralsignal
�   �   +-- tasks.py           # NeuralSignal task callables for subprocess execution
�   �   +-- bridge.py          # compatibility wrapper around task_runner.py
�   +-- trading/               # trading adapter scaffold
│
├── tools/                     # reusable external-service wrappers
│   ├── arxiv_tool.py          # search_arxiv, download_paper_text, digest cache
│   └── chroma_tool.py         # ChromaStore: upsert, query_similar, list_recent
│
├── utils/
│   ├── logger.py              # setup_logging(), get_logger(__name__)
│   ├── profile_loader.py      # load_profile(), get_prompt(), list_profiles()
│   └── utils.py               # extract_json_array/object, load_yaml_section, fmt_value
│
├── llm/
│   └── factory.py             # get_llm(step_name, profile) — multi-provider factory
│
├── dev/                       # local artifacts — gitignored
│   ├── state/                 # JSON state snapshots (written by run_node.py)
│   ├── experiments/
│   │   └── <profile>/
│   │       ├── implementations/  # LLM-generated subclass scripts (cached)
│   │       ├── datasets/         # created feature datasets (CSV)
│   │       └── tests/            # generated pytest files
│   └── papers/                # arxiv digest cache (<arxiv_id>.digest)
│
├── tests/
├── main.py                    # --profile <name> --direction "..."  [--loop] [--list-profiles]
└── run_node.py                # dev: run one step against a saved state snapshot
```

---

## Research Profile Schema

Full reference: `configs/profiles/neuralsignal.yaml`. Abbreviated schema:

```yaml
name: <str>                    # unique identifier, used in file paths and logging
description: <str>

pipeline:
  steps:                       # ordered; omit any step to skip it
    - research
    - ideate
    - refine
    - propose_experiments
    - plan_implementation
    - implement
    - validate
    - prepare_experiment
    - execute_experiment
    - evaluate
    - store_results
    - propose_next_steps

llm:
  default_model: claude-opus-4-6
  step_overrides:              # optional per-step model overrides
    research: claude-sonnet-4-6

research:
  sources:
    - type: arxiv
      categories: [cs.LG, cs.AI]
      max_results: 20
      relevance_score_threshold: 6
      max_papers_to_digest: 3
  domain_context: <str>        # injected into research query and scoring prompts

base_classes:                  # declared for the implement step
  - name: <ClassName>
    module: <dotted.import.path>
    description: <str>
    key_interface: |           # pasted verbatim into the implement LLM prompt
      class FeatureSetBase: ...

validate:
  auto_run: true
  max_fix_retries: 3
  test_runner: "uv run pytest"
  test_output_dir: dev/experiments/<profile>/tests

datasets:
  - name: <str>                # referenced by proposals as proposal['dataset']
    description: <str>
    storage:
      type: mongodb
      application_name: <db>
      sub_application_name: <collection>
    available_detectors: [<str>]
    available_scan_fields:
      guaranteed: [...]
      optional: [...]
      not_available: [...]
    layer_name_patterns:
      ffn: [...]
      attn: [...]

experiment_adapter: plugins.<name>.adapter  # dotted module path; must expose get_adapter()

evaluation:
  primary_metric: test_auc
  metrics: [test_auc, train_auc, test_f1, test_accuracy]
  thresholds:
    test_auc: 0.65

storage:
  mlflow_experiment: <str>
  mongodb_results_db: <str>
  chroma_collection: <str>

prompts:
  <step_name>:
    system: |                  # main system prompt for the step
      ...
    # optional extra keys (research has score_system, summary_system, digest_system;
    #                       validate has fix_system)
```

---

## Plugin Adapter Interface

Each plugin exposes `get_adapter()` in `plugins/<name>/adapter.py`, returning an object that
implements the `ResearchAdapter` Protocol from `plugins/base.py`:

```python
class ResearchAdapter(Protocol):
    def validate_environment(self, profile: dict) -> dict:
        """Return adapter readiness info without running an experiment."""

    def build_context(self, profile: dict, state: dict) -> dict:
        """Return domain-specific context to inject into planning/prompts."""

    def prepare_experiment(
        self, profile: dict, proposal: dict, implementation: dict | None, state: dict
    ) -> dict | None:
        """Prepare reusable artifacts needed before execution (dataset, parameter grid, etc.).
        Return dict must include at minimum: artifact_id, proposal_name."""

    def execute_experiment(
        self, profile: dict, proposal: dict, implementation: dict | None,
        artifact: dict | None, experiment_id: str, state: dict
    ) -> dict | None:
        """Run the experiment. Return dict must include 'metrics' key."""

    def summarize_result(self, profile: dict, result: dict) -> dict:
        """Return a compact domain-specific summary for storage/evaluation."""
```

**Legacy fallback**: if the module does not expose `get_adapter()`, `load_adapter()` returns
the module itself and nodes fall back to calling `create_dataset()`, `run_experiment()`, and
`train_model()` as module-level functions. New plugins should use the Protocol.

---

## State Flow

`graph/state.py` defines `ResearchState` — a flat `TypedDict` with no domain-specific fields.
Each step reads from state and returns a partial dict of keys it writes.

| Step | Reads | Writes |
|---|---|---|
| `research` | `research_direction` | `research_papers`, `research_summary`, `paper_digests` |
| `ideate` | `research_summary`, `paper_digests` | `ideas` |
| `refine` | `ideas`, `research_direction` | `refined_ideas` |
| `propose_experiments` | `refined_ideas` | `proposals` |
| `plan_implementation` | `proposals` | `implementation_plans` |
| `implement` | `implementation_plans` | `implementations` |
| `validate` | `implementations` | `validation_results`; updates `implementations` on fix |
| `prepare_experiment` | `proposals`, `implementations` | `experiment_artifacts` (+ `datasets` alias if artifact_type=dataset) |
| `execute_experiment` | `proposals`, `implementations`, `experiment_artifacts` | `experiment_results`, optionally `models` |
| `evaluate` | `experiment_results`, `models` | `evaluation_summary` |
| `store_results` | everything | `stored_result_ids` |
| `propose_next_steps` | `evaluation_summary`, `research_direction` | `next_steps` |

---

## Validate Step — Generate + Auto-Run Loop

```
for each implementation:
  1. LLM generates pytest tests  →  dev/experiments/<profile>/tests/test_<ClassName>.py
  2. subprocess: profile['validate']['test_runner'] test_file
  3. PASS  →  validation_results[i].passed = True
  4. FAIL + attempts < max_fix_retries:
       LLM receives code + test file + failure output  →  fixed code overwrites script_path
       attempts += 1, goto 2
  5. FAIL + attempts == max_fix_retries:
       validation_results[i].passed = False
       append to state['errors']  — pipeline continues
```

---

## Key Conventions

- **Prompts** — read via `get_prompt(profile, step_name, key="system")` from `utils/profile_loader.py`. Never hardcode prompt strings in node files.
- **Assets** — all data sources via `profile['datasets']`. Never hardcode MongoDB names, collection names, or file paths in node or tool code.
- **Logging** — `get_logger(__name__)` from `utils/logger.py`. Never use `print()` or `logging.getLogger()` directly. Console gets INFO, file gets DEBUG (both configurable in `configs/config.yaml`).
- **LLM calls** — `get_llm(step_name, profile)` from `llm/factory.py`. Never instantiate `ChatAnthropic` / `ChatOpenAI` directly in nodes.
- **Implementations** — new code always subclasses a class declared in `profile['base_classes']`. Cached at `dev/experiments/<profile>/implementations/<ClassName>.py`.
- **State** — nodes never mutate state in place. Always return a partial dict of keys written.
- **Errors** — non-fatal errors go into `state['errors']`. Fatal errors raise and abort.

---

## Running the System

```bash
# Full pipeline
uv run python main.py --profile neuralsignal --direction "attention head specialization"

# List available profiles
uv run python main.py --list-profiles

# Run a single step against a saved state snapshot
uv run python run_node.py implement --profile neuralsignal \
    --state-file dev/state/after_plan_implementation.json

# List available steps
uv run python run_node.py --list

# Tests
uv run pytest
```

---

## Environment / Configuration

All runtime settings live in `configs/config.yaml`. Secrets and environment-specific
values use `${ENV_VAR}` / `${ENV_VAR:default}` syntax and are resolved at startup from
environment variables or an optional `configs/.env` file.

To override defaults, either set environment variables or create `configs/.env`:

```ini
ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...

MONGO_URL=mongodb://hp.lan:27017
CHROMA_HOST=hp.lan
CHROMA_PORT=8000
MLFLOW_URI=http://hp.lan:8899/

# neuralsignal plugin only:
NEURALSIGNAL_PYTHON=uv run python
NEURALSIGNAL_SRC_PATH=../neuralsignal/neuralsignal
```

Everything else (timeouts, paths, logging levels) is edited directly in `configs/config.yaml`.

---

## Adding a New Research Domain

1. Create `configs/profiles/<domain>.yaml` — copy `neuralsignal.yaml` as a template
2. Fill in every section: `pipeline.steps`, `llm`, `research`, `base_classes`, `datasets`,
   `experiment_adapter`, `evaluation`, `storage`, and all `prompts`
3. Add `plugins/<domain>/adapter.py` exposing `get_adapter()` that returns a `ResearchAdapter`
   instance implementing `validate_environment`, `build_context`, `prepare_experiment`,
   `execute_experiment`, and `summarize_result`
4. Set `experiment_adapter: plugins.<domain>.adapter` in the profile
5. No changes to `graph/`, `utils/`, `tools/`, or `llm/` are needed

---

## neuralsignal Plugin Notes

- `plugins/neuralsignal/adapter.py` implements the `ResearchAdapter` skeleton; exposes `get_adapter()`
- `prepare_experiment` currently builds dataset manifests and marks them `pending_bridge`; task-runner-backed dataset creation still needs wiring
- `execute_experiment` currently reports pending task-runner work; future implementation should call `plugins.neuralsignal.tasks.create_s1_model` through `plugins/task_runner.py`
- Heavy NeuralSignal functions live in `plugins/neuralsignal/tasks.py` and run under `plugins/task_runner.py`
  using the NeuralSignal Python interpreter. `bridge.py` is only a compatibility wrapper.
  is handled by the adapter.
- Layer name patterns from `profile['datasets'][*]['layer_name_patterns']` are injected into
  `fs_cfg` at instantiation time so generated classes never use empty defaults.
- Scan field constraints from `available_scan_fields` are injected into the `implement` and
  `validate` prompts so the LLM never generates code that accesses unavailable fields.
