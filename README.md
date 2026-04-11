# NeuralSignalResearcher

A multi-agent research automation system built on [LangGraph](https://github.com/langchain-ai/langgraph) that autonomously searches for relevant papers, proposes neural signal feature extraction experiments, generates runnable experiment code, executes it, and iterates based on results.

The system is designed around the [neuralsignal](https://github.com/amperie/neuralsignal) SDK, which instruments transformer LLMs to capture internal activation tensors at inference time. NeuralSignalResearcher automates the research cycle: *literature review → experiment ideation → implementation → execution → analysis → follow-up*.

---

## Pipeline Overview

```
START
  └─ research_node
       └─ feature_proposal_node
            └─ code_generation_node
                 └─ experiment_runner_node
                      ├─(success)─ result_analysis_node
                      │                └─ mlflow_logger_node
                      │                     └─ generalization_eval_node
                      │                          └─ db_logger_node
                      │                               └─ followup_proposal_node
                      │                                    ├─(continue_loop)─ feature_proposal_node ↩
                      │                                    └─(done)─ END
                      └─(failure)─ END
```

### Node Status

| Node | Status | Description |
|---|---|---|
| `research_node` | **Implemented** | Searches arxiv, scores papers by relevance, synthesises a summary, extracts full-text digests |
| `feature_proposal_node` | **Implemented** | Two-stage ideation: proposes feature extraction ideas, then designs `FeatureSetBase` subclasses |
| `code_generation_node` | **Implemented** | Calls the Claude Code CLI to generate a runnable experiment script |
| `experiment_runner_node` | **Implemented** | Executes the generated script as a subprocess, captures results |
| `result_analysis_node` | Stub | Will compare results against past experiments in ChromaDB |
| `mlflow_logger_node` | Stub | Will log metrics and artefacts to MLflow |
| `generalization_eval_node` | Stub | Will run the probe on held-out data |
| `db_logger_node` | Stub | Will persist the experiment record to ChromaDB |
| `followup_proposal_node` | Stub | Will propose follow-up research directions |

---

## How It Works

### 1. Research (`research_node`)

Given a research direction (e.g. `"sparse autoencoders in LLMs"`):

1. Searches arxiv for up to `max_arxiv_papers` papers sorted by relevance.
2. Asks the LLM to score each paper 0–10 against the research direction in a single batch call.
3. Filters papers below `relevance_score_threshold`.
4. Synthesises a 3–5 paragraph research summary from the top papers.
5. Downloads the full HTML of the top `max_papers_to_digest` papers from `arxiv.org/html/<id>`, strips tags, and runs a structured LLM extraction per paper producing a ~500-word digest covering **methods**, **findings**, **applicable techniques**, and **open problems**.
6. Digests are cached to `dev/papers/<arxiv_id>.digest` (JSON) so each paper is only processed once.

State output: `arxiv_papers`, `research_summary`, `paper_digests`.

### 2. Feature Proposal (`feature_proposal_node`)

Two-stage LLM pipeline:

**Stage 1 — Ideation:** Given the research summary and paper digests, proposes 3–5 feature extraction experiments grounded in transformer internals (residual stream, attention patterns, MLP activations, logit-lens projections). Each idea includes `name`, `description`, `target_behavior`, `hypothesis`, and `rationale`.

**Stage 2 — Implementation design:** For each idea, designs a concrete `FeatureSetBase` subclass. The LLM is given the full neuralsignal scan field reference (activation tensors, layer topology, metadata) and produces a detailed spec including `class_name`, `process_feature_set_logic`, `init_logic`, `imports_needed`, and `output_columns`. This spec is what `code_generation_node` implements.

State output: `feature_proposals` — a list of fully-specified experiment proposals.

### 3. Code Generation (`code_generation_node`)

Uses the **Claude Code CLI** (`claude --print --output-format json`) invoked as a subprocess:

1. Takes `feature_proposals[0]` (highest priority proposal).
2. Assembles a prompt containing the `CODE_GENERATION` system context (FeatureSetBase API, synthetic scan template, script skeleton) plus the full proposal JSON.
3. Calls `claude --print --output-format json` with the prompt on stdin.
4. Parses the JSON envelope from the CLI output, strips any markdown fences.
5. Writes the script to `experiments/<experiment_id>.py`.
6. Assigns a UUID as `experiment_id`.

The generated script:
- Implements the `FeatureSetBase` subclass exactly as specified in the proposal
- Tests it against a synthetic scan with random torch tensors (no real LLM inference required)
- Prints a JSON results dict as the last line of stdout

State output: `generated_code`, `experiment_id`, `experiment_config`.

### 4. Experiment Runner (`experiment_runner_node`)

Runs the generated script as an isolated subprocess:

1. Verifies the script exists at `experiments/<experiment_id>.py`.
2. Sets `PYTHONPATH` to include `neuralsignal_src_path` so the script can import neuralsignal.
3. Launches `{neuralsignal_python} {script_path}` (default: `uv run python`).
4. Enforces a `experiment_timeout_seconds` wall-clock limit.
5. Parses the last non-empty stdout line as JSON → `raw_results`.
6. Routes to `result_analysis` on success, `END` on failure.

State output: `execution_stdout`, `execution_stderr`, `execution_success`, `raw_results`.

---

## Setup

### Prerequisites

- Python 3.12
- [uv](https://github.com/astral-sh/uv) package manager
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) (`npm install -g @anthropic-ai/claude-code`) — required for code generation
- An Anthropic or OpenAI API key

### Install

```bash
git clone <repo>
cd NeuralSignalResearcher
uv sync
```

### Configure

Copy the example env and fill in your API key:

```bash
cp configs/.env.example configs/.env   # if provided
# or create configs/.env manually
```

`configs/.env`:
```env
ANTHROPIC_API_KEY=sk-ant-...
# LLM_MODEL=# leave blank to use provider default (claude-opus-4-6 / gpt-4o)
LLM_PROVIDER=anthropic

NEURALSIGNAL_SRC_PATH=../../neuralsignal/neuralsignal
NEURALSIGNAL_PYTHON=uv run python
EXPERIMENTS_DIR=./experiments
EXPERIMENT_TIMEOUT_SECONDS=1800
```

`configs/config.yaml` (controls per-node behaviour):
```yaml
research_node:
  relevance_score_threshold: 6   # 0–10; papers below this are dropped
  max_papers_to_digest: 3        # full-text digests for top N papers only
```

---

## Running

### Full pipeline

```bash
uv run python main.py --direction "sparse autoencoders in transformer FFN layers"

# Interactive prompt for direction
uv run python main.py

# Automatically loop using the top follow-up proposal as the next direction
uv run python main.py --direction "..." --loop
```

### Individual nodes (development / debugging)

`run_node.py` is a dev harness that runs a single node against saved state:

```bash
# Run with CLI args
uv run python run_node.py research --direction "attention head superposition"

# Run interactively (prompts for inputs — works in IDE debugger)
uv run python run_node.py

# Re-run feature_proposal using state saved from a previous research run
uv run python run_node.py feature_proposal --state-file dev/state_after_research.json

# Save output to a custom path
uv run python run_node.py code_generation --state-file dev/state_after_feature_proposal.json --out dev/my_state.json
```

State is saved to `dev/state_after_<node>.json` after each run, making it easy to chain nodes manually.

---

## Project Structure

```
NeuralSignalResearcher/
├── main.py                        # CLI entry point
├── run_node.py                    # Dev harness for running individual nodes
├── pyproject.toml                 # uv-managed dependencies
│
├── configs/
│   ├── config.py                  # Pydantic-settings Config class
│   ├── config.yaml                # Per-node YAML configuration
│   ├── prompts.py                 # All LLM prompt strings (centralised)
│   └── .env                       # API keys and secrets (gitignored)
│
├── graph/
│   ├── graph.py                   # LangGraph pipeline assembly
│   ├── state.py                   # ResearchState TypedDict
│   └── nodes/
│       ├── research.py            # arxiv search + digest extraction
│       ├── feature_proposal.py    # two-stage experiment ideation
│       ├── code_generation.py     # Claude Code CLI script generation
│       ├── experiment_runner.py   # subprocess execution + result capture
│       ├── result_analysis.py     # stub
│       ├── mlflow_logger.py       # stub
│       ├── generalization_eval.py # stub
│       ├── db_logger.py           # stub
│       └── followup_proposal.py   # stub
│
├── llm/
│   └── factory.py                 # LangChain model factory (Anthropic / OpenAI)
│
├── tools/
│   ├── arxiv_tool.py              # arxiv search, HTML download, digest cache
│   └── chroma_tool.py             # ChromaDB client (upsert / query — stub)
│
├── utils/
│   ├── utils.py                   # extract_json_array, load_yaml_section, fmt_value
│   ├── logger.py                  # structured logging setup
│   └── __init__.py                # re-exports
│
├── tests/
│   ├── test_arxiv_tool.py         # 14 tests, all mocked
│   └── test_chroma_tool.py        # 15 tests, all mocked
│
└── dev/                           # gitignored runtime artefacts
    ├── papers/                    # cached paper digests (<arxiv_id>.digest)
    ├── state_after_*.json         # node output snapshots (from run_node.py)
    └── ...

experiments/                       # generated experiment scripts (<uuid>.py)
logs/                              # rotating log files
```

---

## Architecture Notes

### State

All nodes communicate through `ResearchState`, a `TypedDict` defined in `graph/state.py`. Every node receives the full state and returns only the fields it modifies.

### LLM abstraction

`llm/factory.py` returns a LangChain `BaseChatModel`. Provider and model are configured via `LLM_PROVIDER` / `LLM_MODEL` in `.env`. Defaults to `claude-opus-4-6` on Anthropic, `gpt-4o` on OpenAI.

### Prompts

All prompt strings live in `configs/prompts.py`. Node files import them by name — no prompt strings are defined inline in node code.

### JSON parsing

LLM responses that should be JSON arrays are parsed with a bracket-counting algorithm (`utils.extract_json_array`) that handles nested arrays and escaped strings correctly, regardless of markdown fences wrapping the output.

### neuralsignal integration

neuralsignal is not a pip-installable dependency (flat multi-package layout). Instead, `experiment_runner_node` injects `NEURALSIGNAL_SRC_PATH` into `PYTHONPATH` when launching generated scripts, so they can `import neuralsignal` without modifying the research project's environment.

### Paper digest caching

Full-text digests are cached as JSON at `dev/papers/<arxiv_id>.digest`. On subsequent runs the digest is loaded from cache, skipping the HTML download and LLM extraction call. This makes re-running experiments with the same papers fast and cheap.

---

## Running Tests

```bash
uv run pytest tests/ -v
```

All 29 tests are unit tests with all external calls (arxiv API, HTTP, ChromaDB) mocked.

---

## Dependencies

| Package | Purpose |
|---|---|
| `langgraph` | Pipeline graph orchestration |
| `langchain`, `langchain-anthropic`, `langchain-openai` | LLM abstraction |
| `arxiv` | arxiv search API client |
| `chromadb` | Vector store for experiment memory |
| `mlflow` | Experiment tracking |
| `pydantic-settings` | Typed config from `.env` |
| `python-dotenv` | `.env` file loading |
