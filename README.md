# researcher_framework

A configuration-driven, plug-and-play agentic research pipeline built on LangGraph. The system automates the full research loop — from literature review through implementation, validation, experimentation, and result storage — for any research domain described by a **profile YAML file**.

Current profiles: `neuralsignal` (LLM internals probing & hallucination detection).

---

## What It Does

Given a research direction like `"attention head specialization"`, the pipeline:

1. Searches arxiv, scores papers for relevance, and synthesises a research summary
2. Proposes novel feature extraction ideas grounded in the summary
3. Refines ideas for feasibility against available data and base class APIs
4. Specifies concrete experiments with datasets, hyperparameters, and success criteria
5. Writes implementation plans, then generates Python classes (always subclassing a declared base class)
6. Validates the generated code: runs pytest, feeds failures back to the LLM for fixes (up to N retries)
7. Creates feature datasets, runs experiments, trains models
8. Evaluates results against configured thresholds and produces an analysis
9. Persists everything to MLflow, ChromaDB, and MongoDB
10. Proposes 3 high-value follow-up research directions

---

## Quick Start

### Prerequisites

- Python 3.11+ with [uv](https://github.com/astral-sh/uv)
- Running MongoDB, ChromaDB, and MLflow instances (for storage steps)
- Anthropic API key

### Install

```bash
uv sync
```

### Configure

All settings live in `configs/config.yaml`. Secrets use `${ENV_VAR}` syntax and are
resolved from environment variables or an optional `configs/.env` file.

Minimum: set your API key and backend URLs, either as env vars or in `configs/.env`:

```ini
ANTHROPIC_API_KEY=sk-ant-...
MONGO_URL=mongodb://localhost:27017
CHROMA_HOST=localhost
CHROMA_PORT=8000
```

Everything else (timeouts, paths, logging) is edited directly in `configs/config.yaml`.

### Run

```bash
# Full pipeline
uv run python main.py --profile neuralsignal --direction "sparse autoencoders in MLP layers"

# Auto-loop: use the top proposed next step as the next direction
uv run python main.py --profile neuralsignal --direction "attention head specialization" --loop

# List available profiles
uv run python main.py --list-profiles
```

---

## Running Individual Steps

`run_node.py` lets you run a single pipeline step against a saved state snapshot — useful for development and debugging:

```bash
# Run the research step
uv run python run_node.py research --profile neuralsignal \
    --direction "sparse autoencoders"

# Resume from a saved state (e.g., after research has already run)
uv run python run_node.py implement --profile neuralsignal \
    --state-file dev/state/after_plan_implementation.json

# List available steps
uv run python run_node.py --list

# Interactive mode (prompts for step and state file)
uv run python run_node.py
```

State snapshots are saved automatically to `dev/state/after_<step>.json` after each run.

---

## Research Profiles

A **profile** is a YAML file in `configs/profiles/` that fully describes a research domain. It is the single source of truth — no domain knowledge lives anywhere else in the codebase.

### What a profile controls

| Section | What it specifies |
|---|---|
| `pipeline.steps` | Which steps run and in what order (omit any step to skip it) |
| `llm` | Default model + optional per-step overrides |
| `research.sources` | arxiv categories, result count, relevance threshold |
| `research.domain_context` | Injected into all research/scoring prompts |
| `base_classes` | Base class name, import path, and interface — injected into the code-gen prompt |
| `validate` | Whether to auto-run tests, retry limit, test runner command |
| `datasets` / domain data config | Dataset or data-source definitions for the domain |
| `experiment_adapter` | Dotted module path to the plugin that prepares and executes domain experiments |
| `evaluation` | Primary metric, all tracked metrics, minimum acceptable thresholds |
| `storage` | MLflow experiment name, MongoDB DB, ChromaDB collection |
| `prompts` | **Every LLM system prompt** for every step in this domain |

Full reference: [`configs/profiles/neuralsignal.yaml`](configs/profiles/neuralsignal.yaml)

### Pipeline steps

| Step | What it does |
|---|---|
| `research` | arxiv search → score → digest papers → synthesise summary |
| `ideate` | Propose novel ideas from research context |
| `refine` | Filter ideas for feasibility, improve with implementation guidance |
| `propose_experiments` | Specify dataset, detector, hyperparameters, success criteria per idea |
| `plan_implementation` | Write structured JSON plan per proposal (no code yet) |
| `implement` | Generate Python class subclassing the declared base class |
| `validate` | Generate pytest tests, auto-run, LLM-fix loop up to N retries |
| `prepare_experiment` | Prepare domain artifacts via the plugin adapter |
| `execute_experiment` | Execute the experiment via the plugin adapter |
| `evaluate` | LLM analysis of metrics vs thresholds; per-proposal assessment |
| `store_results` | Log to MLflow, upsert to ChromaDB, insert to MongoDB |
| `propose_next_steps` | Suggest 3 follow-up research directions based on results |

---

## Adding a New Research Domain

1. **Create a profile**: Copy `configs/profiles/neuralsignal.yaml` to `configs/profiles/<domain>.yaml` and fill in every section — especially `prompts` (domain-specific LLM instructions for each step) and `base_classes`.

2. **Add a plugin adapter**: Create `plugins/<domain>/adapter.py` with `get_adapter()` returning an object that implements:
   ```python
   def validate_environment(profile) -> dict: ...
   def build_context(profile, state) -> dict: ...
   def prepare_experiment(profile, proposal, implementation, state) -> dict | None: ...
   def execute_experiment(profile, proposal, implementation, artifact, experiment_id, state) -> dict | None: ...
   def summarize_result(profile, result) -> dict: ...
   ```

3. **Done** — no changes to `graph/`, `utils/`, `tools/`, or `llm/` are needed.

---

## Architecture Overview

```
configs/profiles/<name>.yaml   ← domain knowledge (prompts, datasets, base classes, etc.)
        │
        ▼
graph/builder.py               ← compiles a LangGraph from profile.pipeline.steps
        │
        ▼
graph/nodes/<step>.py          ← one node per step; reads profile, calls LLM, returns state delta
        │
        ├── llm/factory.py     ← get_llm(step_name, profile) — resolves model per step
        ├── tools/             ← arxiv_tool, chroma_tool (reusable, domain-agnostic)
        └── plugins/<domain>/  ← adapter for heavy domain ops (dataset creation, training)
```

Key rules:
- Nodes never hardcode prompt strings — read via `get_prompt(profile, step, key)` from `utils/profile_loader.py`
- Nodes never hardcode data source names — all via `profile['datasets']`
- Nodes never instantiate LLM clients directly — always use `get_llm(step_name, profile)`
- Generated code always subclasses a class in `profile['base_classes']`
- Nodes return partial state dicts; never mutate state in place
- Non-fatal errors go into `state['errors']`; pipeline continues

---

## neuralsignal Plugin

The neuralsignal plugin generates `FeatureSetBase` subclasses that extract activation-based features from LLM scan snapshots and trains XGBoost hallucination detectors.

**Data flow:**
1. `implement` generates a Python class → cached at `dev/experiments/neuralsignal/implementations/<ClassName>.py`
2. `validate` generates and runs pytest tests; auto-fixes failures
3. `prepare_experiment` loads the class, wraps it in a `FeatureProcessor`, calls `neuralsignal.automation.dataset_automation_core.create_dataset()` to produce CSV output
4. `execute_experiment` records the prepared dataset and trains an XGBoost model via `create_s1_model()`, returning `test_auc` and feature importances

**Subprocess bridge**: When the neuralsignal SDK (torch, transformers) cannot be imported into the main venv, `plugins/neuralsignal/bridge.py` runs as a subprocess under the neuralsignal Python interpreter. Config is piped via stdin; results come back as JSON on stdout.

Configure the neuralsignal paths in `.env`:
```ini
NEURALSIGNAL_PYTHON=uv run python          # or: /path/to/neuralsignal/venv/python
NEURALSIGNAL_SRC_PATH=../neuralsignal/neuralsignal
```

---

## Configuration

| File | Purpose |
|---|---|
| `configs/config.yaml` | All runtime settings: backends, timeouts, paths, logging. Secrets via `${ENV_VAR}` |
| `configs/.env` | Optional local overrides (API keys, URLs). Not committed. |
| `configs/profiles/*.yaml` | Per-domain research profiles (steps, prompts, datasets, etc.) |

---

## Tests

```bash
uv run pytest
```

Generated validation tests are written to `dev/experiments/<profile>/tests/` and are run automatically during the pipeline.

---

## Dev Artifacts

All generated artifacts are local and gitignored under `dev/`:

```
dev/
├── state/                  # JSON state snapshots from run_node.py
├── experiments/
│   └── <profile>/
│       ├── implementations/  # cached LLM-generated subclass scripts
│       ├── datasets/         # created feature CSVs
│       └── tests/            # generated pytest files
└── papers/                 # arxiv digest cache
```
