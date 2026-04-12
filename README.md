# NeuralSignalResearcher

A multi-agent research automation system built on [LangGraph](https://github.com/langchain-ai/langgraph) that autonomously searches for relevant papers, proposes neural signal feature extraction experiments, runs them against real neuralsignal scan data, and iterates based on results.

The system is designed around the [neuralsignal](https://github.com/amperie/neuralsignal) SDK, which instruments transformer LLMs to capture internal activation tensors at inference time. NeuralSignalResearcher automates the research cycle: *literature review → experiment ideation → dataset creation → S1 model training → logging → follow-up*.

---

## Pipeline Overview

```
START
  └─ research_node
       └─ feature_proposal_node
            └─ ns_experiment_runner_node
                 ├─(success)─ followup_proposal_node
                 │                 ├─(continue_loop)─ feature_proposal_node ↩
                 │                 └─(done)─ END
                 └─(failure)─ END
```

### Nodes

| Node | Description |
|---|---|
| `research_node` | Searches arxiv, scores papers by relevance, synthesises a summary, extracts full-text digests |
| `feature_proposal_node` | Two-stage ideation: proposes feature extraction ideas, then designs `FeatureSetBase` subclass specs |
| `ns_experiment_runner_node` | Resolves datasets (cache or automation), trains XGBoost S1 model, logs to MLflow + ChromaDB + MongoDB |
| `followup_proposal_node` | Proposes follow-up research directions based on results |

---

## How It Works

### 1. Research (`research_node`)

Given a research direction (e.g. `"sparse autoencoders in LLMs"`):

1. Searches arxiv for up to `max_arxiv_papers` papers sorted by relevance.
2. Asks the LLM to score each paper 0–10 against the research direction in a single batch call.
3. Filters papers below `relevance_score_threshold`.
4. Synthesises a 3–5 paragraph research summary from the top papers.
5. Downloads the full HTML of the top `max_papers_to_digest` papers, strips tags, and runs a structured LLM extraction per paper producing a ~500-word digest covering **methods**, **findings**, **applicable techniques**, and **open problems**.
6. Digests are cached to `dev/papers/<arxiv_id>.digest` (JSON) so each paper is only processed once.

State output: `arxiv_papers`, `research_summary`, `paper_digests`.

### 2. Feature Proposal (`feature_proposal_node`)

Two-stage LLM pipeline:

**Stage 1 — Ideation:** Given the research summary and paper digests, proposes 3–5 feature extraction experiments grounded in transformer internals (residual stream, attention patterns, MLP activations, logit-lens projections). Each idea includes `name`, `description`, `target_behavior`, `hypothesis`, and `rationale`.

**Stage 2 — Implementation design:** For each idea, designs a concrete `FeatureSetBase` subclass. The LLM is given the full neuralsignal scan field reference and produces a detailed spec including `class_name`, `feature_set_name`, `process_feature_set_logic`, `init_logic`, `zone_config`, and `scan_fields_used`.

State output: `feature_proposals` — a list of fully-specified experiment proposals.

### 3. NS Experiment Runner (`ns_experiment_runner_node`)

For each feature proposal, runs the full experiment pipeline inline:

**Dataset resolution:**
1. Maps the proposal's `feature_set_name` and `zone_config` to a neuralsignal `feature_set_config` dict.
2. Checks `DatasetManager` (MongoDB + GridFS) for a cached parquet dataset matching the stable config hash.
3. On a cache hit: loads the DataFrame directly — no scan processing needed.
4. On a miss: calls `neuralsignal.automation.create_dataset()` to query MongoDB scans, run feature extraction, and write a dataset CSV. Saves the result to `DatasetManager` for future reuse.

**Model training:**
5. Feeds the DataFrame to `neuralsignal.automation.create_s1_model()` via the `cfg["dataframe"]` shortcut (bypasses file I/O).
6. XGBoost is tuned with Hyperopt, optimising for AUC (configurable). Metrics returned: `test_auc`, `train_auc`, `test_f1`, `test_accuracy`, `test_precision`, `test_recall`, cross-validation scores, and row/feature counts.

**Logging (three sinks, all non-fatal):**
7. **MLflow** — params, all numeric S1 metrics, tags (research direction, experiment ID), and artifacts (feature importance JSON, model params, proposal JSON). Experiment name is `experiment_runner.mlflow_experiment` in `config.yaml`.
8. **ChromaDB** — rich text embedding (research direction + proposal description + hypothesis + metrics summary) with flat scalar metadata for filtering. Collection is `experiment_runner.chroma_collection` in `config.yaml`.
9. **MongoDB** — full result record (proposal, metrics, S1 params, feature importance, dataset IDs, timestamps, run IDs) inserted into `{agent_experiments_db}.experiments`.

State output: `ns_experiment_results`, `ns_best_auc`, `ns_mlflow_run_ids`, `ns_chroma_record_ids`, `execution_success`, `dataset_ids`, `dataset_cache_hits`.

### 4. Follow-up Proposal (`followup_proposal_node`)

Proposes follow-up research directions based on the experiment results. When `continue_loop` is True, the top proposal's `suggested_direction` is promoted to `research_direction` and the graph loops back to `feature_proposal_node`.

---

## Dataset Registry

`DatasetManager` (`utils/dataset_manager.py`) provides persistent dataset caching backed by MongoDB + GridFS:

- **Identity:** `(feature_set_name, SHA256[:16] of stable config)` — volatile keys (`experiment_id`, `script_path`) are excluded from the hash.
- **Storage:** Feature data serialised as parquet bytes in GridFS; registry documents in `neuralsignal_datasets.datasets`.
- **Meld:** `meld(entries)` concatenates DataFrames from multiple feature sets with prefixed column names for downstream joint modelling.

---

## Setup

### Prerequisites

- Python 3.12
- [uv](https://github.com/astral-sh/uv) package manager
- An Anthropic or OpenAI API key
- MongoDB instance (default: `mongodb://hp.lan:27017`)
- ChromaDB HTTP server (default: `hp.lan:8000`)
- MLflow tracking server (default: `http://localhost:5000`)
- neuralsignal SDK cloned alongside this repo (see below)

### Install

```bash
git clone <repo>
cd NeuralSignalResearcher
uv sync
```

### Configure

Create `configs/.env`:

```env
ANTHROPIC_API_KEY=sk-ant-...
LLM_PROVIDER=anthropic
# LLM_MODEL=   # leave blank to use claude-opus-4-6

NEURALSIGNAL_SRC_PATH=../../neuralsignal/neuralsignal
MONGO_URL=mongodb://hp.lan:27017
```

`configs/config.yaml` controls per-node behaviour:

```yaml
research_node:
  relevance_score_threshold: 6   # 0–10; papers below this are dropped
  max_papers_to_digest: 3        # full-text digests for top N papers only

chromadb:
  host: hp.lan
  port: 8000
  collection: experiments        # collection for generic ChromaDB use

experiment_runner:
  mlflow_experiment: "neuralsignal_agent_experiments"
  agent_experiments_db: "agent_experiments"       # MongoDB database
  chroma_collection: "agent_experiments"          # ChromaDB collection for experiment embeddings
  modeling_row_limit: 10000
  max_evals: 20
  optimization_metric: "auc"
  default_detector: "hallucination"
  default_dataset: "HaluBench"
  dataset_row_limit: 0
```

### neuralsignal SDK

neuralsignal is not a pip-installable package (flat multi-package layout). Clone it alongside this repo:

```bash
git clone <neuralsignal-repo> ../neuralsignal
```

The default `NEURALSIGNAL_SRC_PATH=../../neuralsignal/neuralsignal` resolves to the correct package root.

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
# Run research with a direction
uv run python run_node.py research --direction "attention head superposition"

# Re-run a node using state saved from a previous run
uv run python run_node.py feature_proposal --state-file dev/state_after_research.json

# Run the experiment runner against a saved proposal state
uv run python run_node.py ns_experiment_runner --state-file dev/state_after_feature_proposal.json

# Save output to a custom path
uv run python run_node.py ns_experiment_runner --state-file dev/state_after_feature_proposal.json --out dev/my_state.json
```

State is saved to `dev/state_after_<node>.json` after each run, making it easy to chain nodes manually.

---

## Project Structure

```
NeuralSignalResearcher/
├── main.py                          # CLI entry point
├── run_node.py                      # Dev harness for running individual nodes
├── pyproject.toml                   # uv-managed dependencies
│
├── configs/
│   ├── config.py                    # Pydantic-settings Config class
│   ├── config.yaml                  # Per-node YAML configuration
│   ├── prompts.py                   # All LLM prompt strings (centralised)
│   └── .env                         # API keys and secrets (gitignored)
│
├── graph/
│   ├── graph.py                     # LangGraph pipeline assembly
│   ├── state.py                     # ResearchState TypedDict
│   └── nodes/
│       ├── research.py              # arxiv search + digest extraction
│       ├── feature_proposal.py      # two-stage experiment ideation
│       ├── ns_experiment_runner.py  # dataset caching + S1 training + logging
│       └── followup_proposal.py     # follow-up direction proposal
│
├── llm/
│   └── factory.py                   # LangChain model factory (Anthropic / OpenAI)
│
├── tools/
│   ├── arxiv_tool.py                # arxiv search, HTML download, digest cache
│   ├── chroma_tool.py               # ChromaDB client (upsert / query_similar / list_recent)
│   └── mongo_tool.py                # MongoSnapshotStore — queries neuralsignal scan data
│
├── utils/
│   ├── utils.py                     # extract_json_array, load_yaml_section, fmt_value
│   ├── logger.py                    # structured logging setup
│   ├── dataset_manager.py           # MongoDB + GridFS parquet dataset registry
│   └── __init__.py
│
├── tests/
│   ├── test_dataset_manager.py      # 11 tests (mongomock, no live DB required)
│   └── ...
│
└── dev/                             # gitignored runtime artefacts
    ├── experiments/                 # dataset CSV files from automation
    ├── papers/                      # cached paper digests (<arxiv_id>.digest)
    └── state_after_*.json           # node output snapshots (from run_node.py)

logs/                                # rotating log files (gitignored)
```

---

## Architecture Notes

### State

All nodes communicate through `ResearchState`, a `TypedDict` defined in `graph/state.py`. Every node receives the full state and returns only the fields it modifies.

### LLM abstraction

`llm/factory.py` returns a LangChain `BaseChatModel`. Provider and model are configured via `LLM_PROVIDER` / `LLM_MODEL` in `.env`. Defaults to `claude-opus-4-6` on Anthropic, `gpt-4o` on OpenAI.

### Prompts

All prompt strings live in `configs/prompts.py`. Node files import them by name — no prompt strings are defined inline in node code.

### Dataset caching

`DatasetManager` uses a two-level store: registry documents in MongoDB (`neuralsignal_datasets.datasets`) hold metadata and a GridFS reference; parquet bytes live in GridFS. The cache key is `(feature_set_name, SHA256[:16])` where the hash covers all stable config fields, excluding `experiment_id` and `script_path`. This means the same feature set + zone config always hits the same cached dataset regardless of which pipeline run created it.

### Experiment logging

Each experiment is written to three sinks: MLflow (metrics + params + artifacts), ChromaDB (semantic retrieval by research direction or hypothesis), and MongoDB (`agent_experiments.experiments` for full record storage). All three sinks are non-fatal — a failure in any one logs a warning and continues.

### Semantic experiment comparison

ChromaDB stores a rich text document per experiment combining the research direction, proposal description, hypothesis, and metric summary. `ChromaStore.query_similar(text, n)` returns the nearest neighbours by cosine distance, enabling retrieval of past experiments most relevant to a new research direction.

### Paper digest caching

Full-text digests are cached as JSON at `dev/papers/<arxiv_id>.digest`. On subsequent runs the digest is loaded from cache, skipping the HTML download and LLM extraction call.

---

## Running Tests

```bash
uv run pytest tests/ -v
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `langgraph` | Pipeline graph orchestration |
| `langchain`, `langchain-anthropic`, `langchain-openai` | LLM abstraction |
| `arxiv` | arxiv search API client |
| `chromadb` | Vector store for experiment semantic memory |
| `mlflow` | Experiment tracking and artifact storage |
| `pymongo` | MongoDB client (scan data + dataset registry + experiment logs) |
| `pandas`, `pyarrow` | DataFrame I/O and parquet serialisation |
| `pydantic-settings` | Typed config from `.env` |
| `python-dotenv` | `.env` file loading |
