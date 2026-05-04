# Research Knowledge Graph Direction

This note captures the intended direction for the Neo4j memory graph: keep it
small, research-oriented, and insight-focused. Mongo remains the source of truth
for complete audit/history records. Neo4j should not mirror every graph node,
pipeline step, artifact, validation, job, or run detail.

## Goal

The graph should remember what was learned, not everything that happened.

Use Neo4j as a distilled research knowledge graph that supports questions like:

- Which hypotheses are supported, contradicted, or still untested?
- Which methods repeatedly produce useful evidence?
- Which limitations recur across experiments?
- What findings suggest the next high-value hypotheses?

## Minimal Node Set

Primary node types:

- `Question`: durable research question or problem.
- `Hypothesis`: testable claim.
- `Method`: reusable research approach, feature family, model pattern, or
  analysis technique.
- `Evidence`: compact summary of one or more experiments, papers, or results.
- `Finding`: distilled conclusion supported by evidence.

Optional controlled dimensions:

- `Dataset`: benchmark or data source when it affects interpretation.
- `Metric`: metric only when useful for cross-evidence comparison.
- `Limitation`: recurring caveat, failure mode, confounder, or validity risk.

Avoid graph nodes for routine operational objects:

- implementation plans
- implementations
- validation results
- artifacts
- model runs
- jobs
- evaluation summaries
- raw research artifacts
- source types
- every generated idea or next step

Those belong in Mongo and artifact storage, with record IDs attached to graph
nodes as provenance.

## Relation Vocabulary

Keep relation types sparse and canonical:

- `HAS_HYPOTHESIS`: `Question -> Hypothesis`
- `TESTED_BY`: `Hypothesis -> Evidence`
- `SUPPORTED_BY`: `Hypothesis -> Evidence`
- `CONTRADICTED_BY`: `Hypothesis -> Evidence`
- `REFINED_BY`: `Hypothesis -> Hypothesis`
- `USES_METHOD`: `Evidence -> Method`
- `ON_DATASET`: `Evidence -> Dataset`
- `PRODUCED_FINDING`: `Evidence -> Finding`
- `SUPPORTS`: `Finding -> Hypothesis`
- `HAS_LIMITATION`: `Finding|Method|Evidence -> Limitation`
- `SUGGESTS`: `Finding -> Hypothesis`

Avoid near-duplicates such as `tested_on`, `targets_dataset`,
`materializes_dataset`, `used_detector`, and `for_detector`. Pick one canonical
relation when the meaning is the same.

## Compression Rule

Raw memory records should not automatically become graph nodes. They should
first pass through a distillation step that decides:

1. Does this create or update a durable `Hypothesis`, `Method`, `Finding`, or
   `Limitation`?
2. Does it add support or contradiction to an existing hypothesis?
3. Is it merely provenance? If yes, keep it in Mongo only.

Evidence nodes should aggregate multiple low-level records when possible.

Example compact evidence payload:

```json
{
  "summary": "Sparse FFN transition features reached test_auc=0.71 on HaluBench.",
  "metric_name": "test_auc",
  "metric_value": 0.71,
  "direction": "positive",
  "confidence": "low",
  "n_experiments": 3,
  "best_record_id": "...",
  "provenance_record_ids": ["..."]
}
```

Instead of projecting:

```text
Proposal -> Dataset
Proposal -> Detector
Proposal -> Implementation
Implementation -> Validation
Proposal -> ExperimentResult
ExperimentResult -> Model
EvaluationSummary -> Proposal
NextStep -> Proposal
```

Prefer:

```text
Hypothesis -> TESTED_BY -> Evidence
Evidence -> USES_METHOD -> Method
Evidence -> ON_DATASET -> Dataset
Evidence -> PRODUCED_FINDING -> Finding
```

## Size Budget

For each research cycle, target at most:

- 0-2 new `Hypothesis` nodes
- 0-2 new `Method` nodes
- 1 new `Evidence` node
- 0-2 new `Finding` nodes
- 0-2 new `Limitation` nodes

Ten experiment runs should produce tens of graph nodes, not hundreds.

## Implementation Direction

Keep the existing canonical `MemoryRecord` mechanism, but change what is
projected into Neo4j:

- Mongo stores full state, audit details, artifacts, validation, jobs, and raw
  memory records.
- Chroma stores semantic retrieval projections.
- Neo4j stores distilled research knowledge only.

The best place to create or update KG entities is after interpretation, likely
near `evaluate` or `propose_next_steps`, not after every graph node.

Adapters should emit research KG entities and relations only when the concepts
are expected to survive across runs. Everything else should remain provenance
metadata attached to `Evidence`, `Finding`, or the source `MemoryRecord`.
