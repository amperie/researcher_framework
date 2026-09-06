# Memory System

This project has a canonical memory layer under `core/memory/`. It is designed
to support both:

- semantic retrieval for planning and research
- exact object reuse for deterministic artifacts such as datasets and models

The memory system is intentionally split into a generic core plus
profile-specific builders in adapters.

## Goals

The memory layer is used for four different jobs:

1. Remember prior runs in a form the `research` step can retrieve.
2. Persist profile-specific objects such as datasets, feature sets, models,
   algorithms, portfolios, and backtests without hardcoding them into the graph.
3. Support exact-match reuse of deterministic objects before expensive work is
   recomputed.
4. Project memory into multiple backends:
   - document store for full structured records
   - vector store for semantic retrieval
   - graph store for entity/relation structure

Distillation of knowledge from raw memories is intentionally deferred. The
current implementation records enough lineage, object typing, and evidence
structure for a later distillation layer to consume.

## Core Components

### Canonical record

The central type is `MemoryRecord` in
[`core/memory/models.py`](/E:/Programming/trading_researcher/core/memory/models.py:38).

It contains:

- `record_id`: unique persisted record id
- `domain`: profile/domain name such as `trading_researcher` or `trading`
- `kind`: retrieval/use-case class such as `trading_researcher_dataset` or
  `backtest_result`
- `object_type`: canonical object class such as `dataset`, `featureset`,
  `model`, `experiment_result`, `algorithm`, `portfolio`, `backtest`
- `object_key`: stable identity for the represented object
- `object_role`: how the object is being remembered, for example `artifact`,
  `implementation`, `result`, `summary`
- `schema_version`: payload schema version for adapter evolution
- `title`, `summary`: compact retrieval-facing text
- `content`: full profile-specific structured payload
- `metadata`: compact indexed/filterable fields
- `tags`: lightweight retrieval hints
- `blob_refs`: references to large external artifacts
- `entities`, `relations`: graph projection inputs
- `lineage`: node/run/source/dependency information
- `validity`: reusable/fresh/stale/superseded status information

Records emitted by older builders are normalized by `MemoryService` before
persistence, so new canonical fields can be added without forcing every adapter
to update at once.

### Typed object specs

Profiles can declare typed memory objects under `memory.objects`. These specs
tell core how to interpret reusable domain objects without hardcoding domain
schemas into the graph:

```yaml
memory:
  objects:
    - object_type: dataset
      kind: trading_researcher_dataset
      reusable: true
      fingerprint_metadata_key: dataset_config_fingerprint
      status_metadata_key: dataset_status
      ready_statuses: [ready]
      required_blob_names: [dataset_artifact]
```

The core owns the mechanics: fingerprint fields, reusable statuses, required
blob refs, kind/schema defaults, typed lookup, and projection repair. The domain
still owns meaning: what a dataset, model, strategy, or backtest contains.

### Persistence service

`MemoryService` in
[`core/memory/service.py`](/E:/Programming/trading_researcher/core/memory/service.py:18)
coordinates:

- `document_store.upsert(record)`
- `vector_store.upsert(record_id, embedding_text, vector_metadata)`
- `graph_store.upsert(record, nodes, edges)`

It also exposes:

- `search(query, n_results=...)` for semantic retrieval
- `find_records(filters, limit=...)` for structured lookups
- `find_one_record(filters)` for exact-match reuse flows
- `query(...)` for one semantic/structured typed retrieval API
- `find_reusable(...)` for exact reuse with status and blob checks
- `emit(...)` for node-level typed memory emission
- `resolve_blob_refs(...)` / `hydrate_blobs(...)` for backend-blind blob access
- `repair_projections(...)` for rebuilding vector/graph indexes from documents

#### Service API examples

Emit one typed object from a graph node:

```python
from core.graph.nodes.memory import emit_memory_record

emit_memory_record(
    profile,
    node="prepare_experiment",
    object_type="dataset",
    payload={"dataset_config": dataset_cfg, "dataset_artifact": artifact},
    metadata={
        "dataset_config_fingerprint": fingerprint,
        "dataset_status": "ready",
    },
    blob_refs=[
        {
            "name": "dataset_artifact",
            "uri": artifact["stored_artifact_uri"],
            "artifact_id": artifact["stored_artifact_id"],
            "content_type": "text/csv",
        }
    ],
)
```

Run a backend-agnostic typed query:

```python
service = MemoryService.for_profile(profile)
hits = service.query(
    query="activation sparsity hallucination",
    domain=profile["name"],
    object_type="experiment_result",
    n_results=5,
    include_blobs=True,
)
```

Reuse a deterministic object using the profile declaration:

```python
reuse = service.find_reusable_for_profile(
    profile,
    object_type="dataset",
    fingerprint=dataset_config_fingerprint,
)
if reuse["reusable"]:
    record = reuse["record"]
```

Repair projections after changing projection code, embedding settings, or graph
backend configuration:

```python
count = service.repair_projections({"domain": profile["name"]})
```

### Backends

Current backends are defined in
[`core/memory/backends.py`](/E:/Programming/trading_researcher/core/memory/backends.py):

- `MongoMemoryDocumentStore`
  - source of truth for full records
- `ChromaMemoryVectorStore`
  - semantic search over memory projections
- `NoopMemoryGraphStore`
  - placeholder until a real graph backend is configured
- `Neo4jMemoryGraphStore`
  - optional graph projection backend when configured through profile/global
    graph settings

## How Memory Flows Through The Graph

### Writing memory

The main write point is `store_results`:

- [`core/graph/nodes/store_results.py`](/E:/Programming/trading_researcher/core/graph/nodes/store_results.py:21)

That node:

1. persists experiment outputs to MLflow/Mongo
2. builds canonical memory records
3. persists those records through `MemoryService`

Memory record construction is adapter-owned when available:

- `adapter.build_memory_records(profile, state)`

Fallback generic builders live in:

- [`core/memory/defaults.py`](/E:/Programming/trading_researcher/core/memory/defaults.py:10)

Nodes can also emit individual typed memory objects through:

- `core.graph.nodes.memory.emit_memory_record(...)`

This is useful when a node creates a reusable object before the full run is
complete, for example a generated implementation, validation result, prepared
dataset, or submitted job handle.

### Reading memory

The main read point is the `research` step through:

- [`core/tools/research_tools.py`](/E:/Programming/trading_researcher/core/tools/research_tools.py)

`collect_memory(...)` does:

1. semantic retrieval via `MemoryService.search(...)`
2. adapter-specific rendering through `memory_record_to_artifact(...)` when
   available
3. fallback rendering through `default_memory_record_to_artifact(...)`

This keeps downstream prompts working with normal `research_artifacts` instead
of raw backend records.

## Adapter Responsibilities

Adapters own profile-specific memory semantics. The shared contract lives in:

- [`core/plugins/base.py`](/E:/Programming/trading_researcher/core/plugins/base.py:205)

Three hooks matter:

### `build_memory_records(profile, state)`

Use this when a profile needs custom memory objects or custom summaries.

The adapter should:

- decide what objects are worth remembering
- construct `MemoryRecord` dictionaries
- keep the `content` payload profile-specific
- keep `metadata` compact and filterable
- build `entities` / `relations` when object graph structure matters

### `memory_record_to_artifact(profile, record, state)`

Use this when retrieved memory should appear differently in prompts than the
generic fallback artifact shape.

For example, the `trading_researcher` adapter adds dataset, detector, feature set
class, and stored artifact URI into the retrieved summary.

### `memory_object_specs(profile)`

Use this optional hook only when object specs cannot be expressed statically in
profile YAML. The core reads specs from `profile["memory"]["objects"]` by
default, so most domains should prefer YAML for clarity.

Specs can declare:

- `object_type`: canonical type such as `dataset`, `model`, or `backtest`
- `kind`: profile-specific memory kind
- `schema_version`: payload schema version
- `reusable`: whether exact reuse is allowed
- `fingerprint_fields`: dotted payload fields for core-generated fingerprints
- `fingerprint_metadata_key`: metadata key used for exact lookup
- `status_metadata_key`: metadata key used for readiness/freshness
- `ready_statuses`: statuses that count as reusable
- `required_blob_names`: blob refs that must still resolve before reuse

## Typed Objects

The memory system does not flatten all domains into one universal payload.
Instead:

- the memory envelope is generic
- the object payload is adapter-specific

Examples:

### TradingResearcher

The TradingResearcher adapter now emits multiple object types from a run:

- `object_type="dataset"`
- `object_type="featureset"`
- `object_type="model"`
- `object_type="experiment_result"`

Typical `kind` values:

- `trading_researcher_dataset`
- `trading_researcher_featureset`
- `trading_researcher_model`
- `trading_researcher_experiment`

### Trading

A trading adapter should follow the same pattern:

- `object_type="dataset"`
- `object_type="algorithm"`
- `object_type="portfolio"`
- `object_type="backtest"`

Typical `kind` values might be:

- `trading_dataset`
- `trading_algorithm`
- `trading_portfolio`
- `trading_backtest`

## Exact-Match Reuse With Fingerprints

Semantic retrieval is not enough for exact object reuse. For deterministic
objects such as datasets or trained models, memory also needs a stable
configuration fingerprint.

Fingerprint helpers live in:

- [`core/memory/fingerprints.py`](/E:/Programming/trading_researcher/core/memory/fingerprints.py)

The canonical helper is:

```python
from core.memory import fingerprint_json
```

It:

- normalizes dict key order
- normalizes lists/sets recursively
- JSON-encodes deterministically
- returns a SHA-256 hex digest

### Dataset fingerprints

For `trading_researcher`, dataset reuse is driven by a canonical dataset memory spec.
It includes only fields that actually define dataset identity, for example:

- dataset/application/sub-application
- detector names
- query
- row limits
- balancing config
- zone size
- feature set class name
- feature set source hash
- layer patterns
- backend type

It explicitly avoids path-only or run-specific noise.

The resulting hash is stored under:

- `metadata.dataset_config_fingerprint`

### Model fingerprints

The same pattern applies to models:

- hash the effective model config
- usually include the dataset fingerprint too

This is stored under:

- `metadata.model_config_fingerprint`

### Why source hashes matter

If a dataset depends on generated code, then config equality alone is not
enough. The same dataset config with different feature set source should not be
treated as the same dataset.

For that reason, the TradingResearcher adapter includes:

- `feature_set_source_hash`

inside the dataset memory spec.

## Current TradingResearcher Reuse Flow

The current `trading_researcher` adapter performs exact dataset reuse in
`prepare_experiment(...)`.

High-level flow:

1. Build the effective dataset config.
2. Compute `dataset_config_fingerprint`.
3. Call `MemoryService.find_reusable(...)` with `object_type="dataset"`,
   `fingerprint_metadata_key="dataset_config_fingerprint"`,
   `status_metadata_key="dataset_status"`, and `ready_statuses=["ready"]`.
4. If a matching dataset record exists and the referenced dataset file still
   exists, reuse it.
5. Skip the expensive dataset-generation task.

When reuse succeeds, the adapter returns a dataset artifact with:

- `dataset_source = "memory_reuse"`
- `memory_record_id = ...`
- `task_result["reused_from_memory"] = True`

This means the graph can keep using the normal `experiment_artifacts` shape
without knowing whether the dataset was newly created, reused from disk, or
reused from memory.

## Designing New Exact-Match Reuse Paths

When adding memory-backed reuse for a new object type:

1. Define the canonical object identity.
2. Build a deterministic spec containing only identity-defining fields.
3. Hash it with `fingerprint_json(...)`.
4. Persist the fingerprint under `metadata`.
5. Use `MemoryService.find_reusable(...)` before starting expensive work.
6. Let core validate reusable status and required blob refs.
7. Do any final domain-specific sanity checks that cannot be expressed as a
   generic status/blob check.

This same pattern works for:

- precomputed datasets
- generated feature sets
- trained models
- backtest outputs
- portfolio construction snapshots

## What Goes In `content` vs `metadata`

### Put in `content`

Use `content` for:

- full configs
- full results
- implementation summaries
- feature importance maps
- nested domain payloads
- data not needed for indexing or filtering

### Put in `metadata`

Use `metadata` for:

- exact-match fingerprints
- primary metrics
- assessment flags
- compact IDs and names
- profile name / dataset / detector / symbol / timeframe / benchmark
- any small scalar/string/list field you want to filter or rank on

Rule of thumb:

- `content` is for full reconstruction
- `metadata` is for search and filtering

## Entities And Relations

`entities` and `relations` are the right place to encode reusable object graph
structure.

Example TradingResearcher entities:

- `proposal:activation_sparsity`
- `dataset:HaluBench`
- `detector:hallucination`
- `feature_set:ActivationSparsity`
- `model:activation_sparsity_ab12cd34`

Example relations:

- `proposal tested_on dataset`
- `proposal used_detector detector`
- `proposal implemented_by feature_set`
- `proposal produced_model model`

Future trading entities could include:

- `algorithm:mean_reversion_v3`
- `portfolio:top20_equal_weight`
- `dataset:daily_ohlcv_us`
- `backtest:bt_2026_04_28`

## Recommended Pattern For New Profiles

When adding a new profile:

1. Keep using canonical `MemoryRecord`.
2. Introduce profile object types through:
   - `object_type`
   - `kind`
   - adapter-owned `content`
3. Add fingerprints for deterministic reusable objects.
4. Persist large files via artifact storage and reference them with `blob_refs`.
5. Use `memory_record_to_artifact(...)` to shape prompt-facing retrieval text.
6. Add `entities` and `relations` for object graph structure when useful.
7. Use `find_reusable_for_profile(...)` before expensive deterministic work.

## Logging

The memory layer logs at multiple levels:

- `DEBUG`: backend construction, record normalization, projection details,
  filter decisions, fingerprint fields, graph/no-op behavior, and blob
  resolution counts.
- `INFO`: batches persisted, records emitted, query result counts, reuse hits
  and misses, projection repair start/end.
- `WARNING`: malformed profile memory specs, adapter memory-builder failures,
  missing local blob refs, records dropped because they have no `record_id`.
- `ERROR`: misconfigured graph backends or missing optional graph dependencies.

Useful logger namespaces:

- `core.memory.service`
- `core.memory.backends`
- `core.memory.defaults`
- `core.graph.nodes.memory`

These logs are intentionally backend-agnostic at the call site. Domain code can
emit or retrieve memory without logging Mongo, Chroma, S3, or graph operations
itself; those details are recorded by the core layer.

## Example Record Shapes

### Dataset object

```python
{
  "record_id": "dataset:<fingerprint>",
  "domain": "trading_researcher",
  "kind": "trading_researcher_dataset",
  "object_type": "dataset",
  "object_key": "<fingerprint>",
  "object_role": "artifact",
  "schema_version": "1",
  "title": "activation_sparsity dataset",
  "summary": "...",
  "content": {
    "dataset_artifact": {...},
    "dataset_config": {...},
    "implementation": {...},
  },
  "metadata": {
    "dataset": "HaluBench",
    "detector": "hallucination",
    "dataset_status": "ready",
    "dataset_config_fingerprint": "...",
    "stored_artifact_uri": "...",
  },
}
```

### Experiment result object

```python
{
  "record_id": "exp-001",
  "domain": "trading_researcher",
  "kind": "trading_researcher_experiment",
  "object_type": "experiment_result",
  "object_key": "exp-001",
  "object_role": "result",
  "schema_version": "1",
  "title": "activation_sparsity",
  "summary": "...",
  "content": {
    "proposal": {...},
    "dataset_config": {...},
    "model_config": {...},
    "metrics": {...},
  },
  "metadata": {
    "assessment": "strong",
    "test_auc": 0.72,
    "dataset_config_fingerprint": "...",
    "model_config_fingerprint": "...",
  },
}
```

## Operational Notes

- Document store is the source of truth.
- Vector store is a projection.
- Graph store is a projection.
- Retrieval artifacts should be concise and readable.
- Exact-match reuse should always re-check file/blob existence before trusting a
  stale record.
- Fingerprints should be based on effective configs, not raw profile YAML dumps.
- `repair_projections(...)` can rebuild vector/graph projections from document
  records after schema, embedding, or backend changes.
- Memory logs are available through the normal project logging configuration in
  `configs/config.yaml`.

## Current Gaps / Future Work

The current implementation now supports:

- canonical typed memory records
- profile-declared memory object specs
- node-level typed memory emission API
- semantic retrieval
- typed structured retrieval through `MemoryService.query(...)`
- exact dataset reuse for TradingResearcher
- exact reuse helper with validity and blob checks
- lineage and validity fields on canonical records
- projection repair from document-store source of truth

Likely next steps:

- domain use of generic exact reuse for models/backtests/portfolios
- profile-specific retention / deduplication policies
- knowledge distillation records and jobs
