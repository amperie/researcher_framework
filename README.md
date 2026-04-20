# researcher_framework

A configuration-driven, plug-and-play agentic research pipeline built on LangGraph. The system automates the full research loop — from literature review through implementation, validation, experimentation, and result storage — for any research domain described by a **profile YAML file**.

Current profiles:
- `neuralsignal` - LLM internals probing and hallucination detection
- `trading` - algorithmic trading strategy research and backtest automation scaffold

---

## What It Does

Given a research direction like `"attention head specialization"`, the pipeline:

1. Runs profile-configured research tools, scores every artifact, and synthesises a research summary
2. Proposes novel experiment ideas grounded in the highest-scored artifacts
3. Refines ideas for feasibility against available data and base class APIs
4. Specifies concrete experiments with datasets, hyperparameters, and success criteria
5. Writes implementation plans, then generates Python classes (always subclassing a declared base class)
6. Validates the generated code: runs pytest, feeds failures back to the LLM for fixes (up to N retries)
7. Submits or executes domain experiments through the profile plugin
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
| `research.tools` | Dotted research tool functions, limits, and scoring thresholds |
| `research.domain_context` | Injected into research collection/scoring prompts |
| `base_classes` | Base class name, import path, and interface — injected into the code-gen prompt |
| `validate` | Whether to auto-run tests, retry limit, test runner command |
| `datasets` / domain data config | Dataset or data-source definitions for the domain |
| `experiment_adapter` | Dotted module path to the plugin that prepares and executes domain experiments |
| `execution` | Optional async job execution settings: runner, parallelism, polling, timeouts |
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
| `submit_experiment_jobs` | Submit long-running experiment jobs and return immediately |
| `check_experiment_jobs` | Poll durable jobs, collect completed outputs, and submit next-stage jobs |
| `evaluate` | LLM analysis of metrics vs thresholds; per-proposal assessment |
| `store_results` | Log to MLflow, upsert to ChromaDB, insert to MongoDB |
| `propose_next_steps` | Suggest 3 follow-up research directions based on results |

---

## Adding a New Research Domain

1. **Create a profile**: Copy `configs/profiles/neuralsignal.yaml` to `configs/profiles/<domain>.yaml` and fill in every section — especially `prompts` (domain-specific LLM instructions for each step) and `base_classes`.

2. **Choose research tools**: Add `research.tools` entries. Each tool is a dotted function path and can set its own limits and `relevance_score_threshold`.
   ```yaml
   research:
     tools:
       - name: arxiv
         tool: tools.research_tools.collect_arxiv
         max_results: 20
         relevance_score_threshold: 6
       - name: prior_experiments
         tool: tools.research_tools.collect_prior_experiments
         n_results: 8
         relevance_score_threshold: 7
   ```

3. **Add a plugin adapter**: Create `plugins/<domain>/adapter.py` with `get_adapter()` returning an object that implements:
   ```python
   def validate_environment(profile, state) -> dict: ...
   def build_context(profile, state) -> dict: ...
   def prepare_experiment(profile, state) -> dict: ...
   def execute_experiment(profile, state) -> dict: ...
   def summarize_result(profile, state) -> dict: ...
   ```

   For long-running work, adapters can also implement the async job API:
   ```python
   def submit_experiment_jobs(profile, state) -> dict: ...
   def check_experiment_jobs(profile, state) -> dict: ...
   ```

   The synchronous `execute_experiment` method can remain as a fallback while the
   profile uses `submit_experiment_jobs` / `check_experiment_jobs` for async runs.

4. **Done** - no changes to `graph/`, `utils/`, `tools/`, or `llm/` are needed unless the domain needs a new reusable research tool.

---

## Research Tools

The `research` node is modular. Profiles declare which tools to run, and each tool returns structured artifacts. The node then asks the LLM to score every artifact with `prompts.research.artifact_score_system`, filters by the tool's threshold, and writes the selected artifacts to `state['research_artifacts']`.

Built-in tools:

| Tool | Purpose |
|---|---|
| `tools.research_tools.collect_arxiv` | Search arXiv and return paper artifacts |
| `tools.research_tools.collect_prior_experiments` | Retrieve similar past experiments from ChromaDB |
| `tools.research_tools.collect_adapter_context` | Ask the active domain adapter for environment/platform context |
| `tools.research_tools.collect_profile_context` | Expose selected profile sections as scoreable artifacts |
| `tools.research_tools.collect_strategy_library` | Inspect a local trading platform tree for strategy/backtest/risk files |

Custom tools should implement:

```python
def collect_x(direction: str, profile: dict, tool_cfg: dict, state: dict) -> list[dict]:
    ...
```

Each returned artifact should include `artifact_id`, `source`, `source_type`, `title`, `summary`, `metadata`, and optional `raw`.

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
- Long-running jobs go into `state['experiment_jobs']` and write durable files under `dev/experiments/<profile>/jobs/`

---

## Async Experiment Jobs

Long-running experiments can run asynchronously so the graph does not block on a
single synchronous process. Profiles opt in by using these pipeline steps:

```yaml
pipeline:
  steps:
    - ...
    - submit_experiment_jobs
    - check_experiment_jobs
    - evaluate
```

Execution is configured in the profile:

```yaml
execution:
  mode: async
  runner: local_process
  max_parallel_jobs: 2
  auto_submit_next_stage: true
  poll_interval_seconds: 30
  job_timeout_seconds: 7200
```

The only runner currently implemented is `local_process`. It launches detached
Python workers using the same dotted task callables used by the synchronous path.
The runner abstraction is intentionally small so future runners can plug in with
the same `submit` / `check` behavior, for example Ray, Kubernetes, Slurm, or a
remote worker service.

Each job is durable on disk:

```text
dev/experiments/<profile>/jobs/<job_id>/
  job.json
  payload.json
  status.json
  result.json
  stdout.log
  stderr.log
```

The graph state stores lightweight job metadata in `state['experiment_jobs']`.
`submit_experiment_jobs` submits work up to `execution.max_parallel_jobs`.
`check_experiment_jobs` polls existing jobs, collects completed `result.json`
files into `experiment_artifacts`, `experiment_results`, and `models`, and can
submit the next stage automatically when `auto_submit_next_stage` is true.

For development, `run_node.py` is useful for polling without rerunning earlier
steps:

```bash
uv run python run_node.py check_experiment_jobs --profile neuralsignal \
    --state-file dev/state/after_submit_experiment_jobs.json
```

If jobs are still running, run the same check step again later with the latest
saved state snapshot.

---

## neuralsignal Plugin

The neuralsignal plugin generates `FeatureSetBase` subclasses and runs NeuralSignal automation through isolated subprocess tasks. It supports both the synchronous adapter methods and the async job-node flow used by the `neuralsignal` profile.

**Data flow:**
1. `implement` generates a Python class → cached at `dev/experiments/neuralsignal/implementations/<ClassName>.py`
2. `validate` runs deterministic contract tests and can auto-fix failures
3. `submit_experiment_jobs` submits NeuralSignal dataset-creation jobs
4. `check_experiment_jobs` collects completed dataset CSVs into `experiment_artifacts`
5. `check_experiment_jobs` can then submit S1 model-training jobs automatically
6. later checks collect model metrics, params, and feature importance into `experiment_results` and `models`

**Subprocess tasks**: Heavy NeuralSignal calls live in `plugins/neuralsignal/tasks.py`. Synchronous calls use `plugins/task_runner.py`; async jobs use `plugins/job_runner.py`, which invokes the same task callables and writes durable job files. `plugins/neuralsignal/bridge.py` remains as a compatibility wrapper for old `create_dataset` / `create_s1_model` commands.

The async NeuralSignal task chain is:

```text
submit_experiment_jobs
  -> plugins.job_runner.LocalProcessRunner
    -> plugins.neuralsignal.tasks.create_dataset
      -> neuralsignal.automation.create_dataset

check_experiment_jobs
  -> collect dataset result.json
  -> submit model job
    -> plugins.neuralsignal.tasks.create_s1_model
      -> neuralsignal.automation.create_s1_model
```

The task wrapper merges the agent payload over NeuralSignal's packaged automation
defaults with `neuralsignal.automation.get_config()`, then injects the generated
feature set via a real NeuralSignal `FeatureProcessor`.

Configure the neuralsignal paths in `.env`:
```ini
NEURALSIGNAL_PYTHON=uv run python          # or: /path/to/neuralsignal/venv/python
NEURALSIGNAL_SRC_PATH=../neuralsignal/neuralsignal
```

---

## Trading Plugin

The `trading` profile is wired into the same graph and research-tool infrastructure. It currently provides prompts, risk constraints, research tools, and a plugin scaffold in `plugins/trading/adapter.py`.

To run trading experiments end to end, implement `TradingAdapter.execute_experiment()` against your trading platform's backtest engine. That method should load the generated strategy class, run a leakage-safe backtest with configured costs/slippage, and return normalized metrics such as `sharpe_ratio`, `max_drawdown`, `annual_return`, `turnover`, and `win_rate`.

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

All generated artifacts are local and gitignored under `dev/`. Async experiment
jobs are stored under `dev/experiments/<profile>/jobs/<job_id>/` with payloads,
status, results, stdout, and stderr logs:

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
