"""Centralised prompt strings for all LangGraph nodes.

Naming convention: {NODE}_{PURPOSE}
  NODE    — the node that owns the prompt (e.g. RESEARCH, FEATURE_PROPOSAL)
  PURPOSE — what the prompt does (e.g. SCORE, SUMMARY, PROPOSALS, CODE)

Only system-message content lives here. Human-message content is assembled
dynamically from state values inside each node.
"""

# ---------------------------------------------------------------------------
# research_node
# ---------------------------------------------------------------------------

RESEARCH_SCORE = """\
You are a research relevance analyst. Given a research direction and a list of
arxiv paper titles and abstracts, score each paper's relevance to the direction
on a scale of 0–10 (10 = highly relevant, 0 = unrelated).

Respond with ONLY a valid JSON array of integers, one per paper, in the same
order. Example for 3 papers: [8, 3, 9]"""

RESEARCH_SUMMARY = """\
You are a research synthesis assistant specialised in machine learning and
neuroscience. Given a research direction and a set of arxiv abstracts, write a
concise (3–5 paragraph) synthesis that:
- Identifies the key themes, methods, and findings across the papers
- Highlights techniques directly applicable to the research direction
- Notes open problems or gaps the current direction could address

Write in clear, precise prose suitable for framing an experiment proposal."""

RESEARCH_DIGEST = """\
You are a technical research analyst. Given the full text of an arxiv paper and
a research direction, extract a structured digest that will be used by a
downstream agent to propose concrete experiments.

Write four clearly labelled sections:

METHODS
Describe the key methods, algorithms, and architectures introduced or used.
Be specific: include model names, layer types, training procedures, and any
novel techniques. Skip background the reader would already know.

FINDINGS
Summarise the main empirical results and conclusions. Include numbers where
meaningful (accuracy deltas, scaling trends, ablation outcomes).

APPLICABLE TECHNIQUES
Identify methods from this paper that could be directly adapted for the research
direction. Be concrete about *how* each technique could be applied.

OPEN PROBLEMS
Note limitations, failure modes, or explicitly stated future work that is
relevant to the research direction.

Write in dense, precise prose. Do not include paper metadata, citations, or
acknowledgements. Aim for 400–700 words total."""

# ---------------------------------------------------------------------------
# feature_proposal_node  —  Stage 1: ideation
# ---------------------------------------------------------------------------

FEATURE_PROPOSAL_IDEATION = """\
You are an expert in LLM internals, mechanistic interpretability, and
activation-based behavioural probing. Given a research direction and a
synthesis of recent arxiv papers, propose 3–5 concrete feature-extraction
experiments that could be run against a transformer LLM to surface the
signal described by the research direction.

Each proposal should be grounded in what is actually measurable from a
transformer's internal activations: residual stream states, attention patterns,
MLP intermediate activations, logit-lens projections, or activation deltas
between contrastive inputs.

Respond with ONLY a valid JSON array. Each element must have these keys:
  name              — short, descriptive experiment name
  description       — 2–3 sentences on what the experiment measures and why
  target_behavior   — the specific LLM behaviour being probed (e.g. hallucination, refusal)
  hypothesis        — what signal you expect to find in the activations and why
  rationale         — why this experiment is a good fit for the research direction

Do not include any text outside the JSON array."""

# ---------------------------------------------------------------------------
# feature_proposal_node  —  Stage 2: FeatureSetBase subclass design
# ---------------------------------------------------------------------------

FEATURE_PROPOSAL_IMPLEMENTATION = """\
You are a neuralsignal SDK expert. Given a list of feature-extraction experiment
ideas, design a concrete FeatureSetBase subclass for each one.

═══════════════════════════════════════════════════════════════
NEURALSIGNAL FEATURESET API
═══════════════════════════════════════════════════════════════

── FeatureSetBase (base class to subclass) ─────────────────────
class FeatureSetBase:
    default_config = {"output_format": "name_and_value_columns"}

    def __init__(self, config: dict):
        self.config = {**self.default_config, **config}
        self.output_format = self.config["output_format"]

    def get_feature_set_name(self) -> str:
        raise NotImplementedError   # return a short descriptive string

    def process_feature_set(self, scan: dict):
        raise NotImplementedError   # see contract below

    def make_column_name(self, column_name: str) -> str:
        return f"{self.get_feature_set_name()}__{column_name}"  # use this for all column names

── process_feature_set contract ────────────────────────────────
Input  — scan: dict. Complete field reference:

  METADATA / TEXT
  scan['input']                  str        Original text input to the judge model
  scan['output']                 str        Original text output from the judge model
  scan['decoded_output']         str        Human-readable decoded output tokens
  scan['ground_truth']           str|None   Dataset label (for supervised experiments)
  scan['model_name']             str        Name of the judge model used
  scan['data_run_name']          str        Dataset/run identifier
  scan['context']                any        Optional additional context passed with input
  scan['metadata']               dict       Optional metadata dict
  scan['generation_correlation_id'] str     UUID linking related generations
  scan['detections']             dict       Keyed by behavior name → detection result

  LAYER TOPOLOGY
  scan['layer_names']            list[str]  Names of all instrumented layers
  scan['layer_order']            list       Layer IDs in execution order
  scan['layer_id_to_name']       dict       {layer_id: layer_name_str}
  scan['layer_passes']           dict       {layer_id: pass_count}
  scan['layer_passes_by_name']   dict       {layer_name: pass_count}
  scan['layer_indexes_to_include'] list     Subset of layer indexes selected by collector
  scan['layer_names_to_include'] list       Subset of layer names selected by collector
  scan['topology']               dict       Model topology info captured by collector

  ZONE / COMPRESSION CONFIG
  scan['zone_size']              int        Default token zone size (default 512)
  scan['zone_sizes_by_layer']    dict       {layer_id: int} per-layer zone sizes
  scan['zone_size_by_layer']     dict       Alternate per-layer zone size map

  ACTIVATION TENSORS  ← primary data for feature extraction
  scan['outputs']   dict  {batch_idx (int): {layer_module_id (str): Tensor}}
  scan['inputs']    dict  {batch_idx (int): {layer_module_id (str): Tensor}}

  Tensor shape: (hidden_dim,) when zone-processed (averaged over token zones),
                or raw activation shape otherwise.
  Access pattern: scan['outputs'][0][layer_id] → Tensor for batch item 0, given layer.

Output — when output_format == "name_and_value_columns" (default):
  return (col_names: list[str], col_vals: list[float])
  All column names MUST be created with self.make_column_name(...)

── FeatureSetZones example (zones per layer) ───────────────────
def process_feature_set(self, scan):
    for layer_id in scan['outputs']:
        t = scan['outputs'][layer_id]            # torch.Tensor
        t = torch.mean(t, dim=0)                 # pool if needed
        lyr_name = scan['layer_id_to_name'][layer_id]
        for idx, val in enumerate(t.tolist()):
            col_names.append(self.make_column_name(f"{lyr_name}_{idx}"))
            col_vals.append(val)
    return (col_names, col_vals)

── FeatureSetLogitLens example (project through unembedding) ───
def process_feature_set(self, scan):
    for layer_id in scan['outputs']:
        lyr_name = scan['layer_id_to_name'][layer_id]
        t = scan['outputs'][layer_id].to(self.dev_map)
        logits = self.unembed.forward(t)
        col_names.append(self.make_column_name(f"std_{lyr_name}"))
        col_vals.append(torch.std(logits).item())
        col_names.append(self.make_column_name(f"mean_{lyr_name}"))
        col_vals.append(torch.mean(logits).item())
    return (col_names, col_vals)

── Utility ─────────────────────────────────────────────────────
from neuralsignal.core.modules.feature_sets.feature_utils import is_layer_string_match_in_list
  is_layer_string_match_in_list(layer_name: str, patterns: list[str]) -> bool

═══════════════════════════════════════════════════════════════

For EACH idea in the input list, design a FeatureSetBase subclass that
implements the feature extraction described by the idea.

Return a JSON array. Each element must preserve all fields from the input idea
and add these keys:

  class_name          str   — PascalCase class name, e.g. "FeatureSetAttentionEntropy"
  feature_set_name    str   — short snake_case string returned by get_feature_set_name()
  config_schema       dict  — keys the __init__ config dict must contain, with a
                              one-line description of each as the value
  scan_fields_used    list  — which top-level scan keys process_feature_set reads
  process_feature_set_logic  str  — precise step-by-step description of what
                              process_feature_set does: which layers it iterates,
                              what tensor operations it applies, what each output
                              column represents. Be specific enough that a code
                              generator can write the method without ambiguity.
  output_columns      list  — list of dicts, each: {name_pattern, description}
                              describing the columns this feature set emits
  imports_needed      list  — Python imports required (e.g. ["torch", "numpy"])
  init_logic          str   — what __init__ does beyond calling super().__init__
                              (e.g. loading a model, precomputing a matrix)

Output ONLY the JSON array. Preserve all original idea fields."""

# ---------------------------------------------------------------------------
# code_generation_node  —  FeatureSetBase subclass + test harness
# ---------------------------------------------------------------------------

CODE_GENERATION = """\
You are an expert Python developer specialised in the neuralsignal SDK.

Given a FeatureSetBase subclass specification, write a complete, self-contained
Python script that:
  1. Implements the subclass EXACTLY as described in the proposal fields:
       class_name, feature_set_name, init_logic, process_feature_set_logic,
       imports_needed, config_schema
  2. Tests it with a SYNTHETIC scan (no real LLM inference required)
  3. Prints a JSON results dict as the VERY LAST line of stdout — nothing after it
  4. On any exception: prints {"error": "<message>", "status": "error"} and exits 1

Output ONLY raw Python code. No markdown fences, no prose, no explanation.

═══════════════════════════════════════════════════════════════
FEATURESETBASE API
═══════════════════════════════════════════════════════════════

from neuralsignal.core.modules.feature_sets.feature_set_base import FeatureSetBase
from neuralsignal.core.modules.feature_sets.feature_utils import is_layer_string_match_in_list

class FeatureSetBase:
    default_config = {"output_format": "name_and_value_columns"}

    def __init__(self, config: dict):
        self.config = {**self.default_config, **config}
        self.output_format = self.config["output_format"]

    def get_feature_set_name(self) -> str:
        raise NotImplementedError

    def make_column_name(self, column_name: str) -> str:
        return f"{self.get_feature_set_name()}__{column_name}"

    def process_feature_set(self, scan: dict):
        # Returns (col_names: list[str], col_vals: list[float])
        raise NotImplementedError

═══════════════════════════════════════════════════════════════
SYNTHETIC SCAN — use this to test the feature set
═══════════════════════════════════════════════════════════════

Build the scan dict like this (adapt NUM_LAYERS / HIDDEN_DIM as needed):

    import torch

    NUM_LAYERS = 12
    HIDDEN_DIM = 768
    BATCH_IDX  = 0

    layer_ids       = [f"layer_{i}" for i in range(NUM_LAYERS)]
    layer_id_to_name = {lid: f"transformer.h.{i}" for i, lid in enumerate(layer_ids)}

    scan = {
        # Metadata / text
        "input":    "The capital of France is",
        "output":   " Paris",
        "decoded_output": " Paris",
        "ground_truth": None,
        "model_name": "gpt2",
        "data_run_name": "synthetic_test",
        "context":  None,
        "metadata": {},
        "generation_correlation_id": "test-uuid",
        "detections": {},
        # Layer topology
        "layer_names":             list(layer_id_to_name.values()),
        "layer_order":             layer_ids,
        "layer_id_to_name":        layer_id_to_name,
        "layer_passes":            {lid: 1 for lid in layer_ids},
        "layer_passes_by_name":    {name: 1 for name in layer_id_to_name.values()},
        "layer_indexes_to_include": list(range(NUM_LAYERS)),
        "layer_names_to_include":  list(layer_id_to_name.values()),
        "topology": {},
        # Zone config
        "zone_size":             512,
        "zone_sizes_by_layer":   {lid: 512 for lid in layer_ids},
        "zone_size_by_layer":    {lid: 512 for lid in layer_ids},
        # Activation tensors — shape (HIDDEN_DIM,) per layer per batch item
        "outputs": {BATCH_IDX: {lid: torch.randn(HIDDEN_DIM) for lid in layer_ids}},
        "inputs":  {BATCH_IDX: {lid: torch.randn(HIDDEN_DIM) for lid in layer_ids}},
    }

═══════════════════════════════════════════════════════════════
REQUIRED SCRIPT SKELETON
═══════════════════════════════════════════════════════════════

    import json
    import sys
    import torch
    # ... all other imports from proposal['imports_needed'] ...

    from neuralsignal.core.modules.feature_sets.feature_set_base import FeatureSetBase
    # import feature_utils if needed

    class FeatureSet<Name>(FeatureSetBase):
        # implement __init__, get_feature_set_name, process_feature_set

    if __name__ == "__main__":
        try:
            # 1. Build synthetic scan
            # 2. Instantiate feature set with config from proposal['config_schema']
            # 3. Call process_feature_set(scan)
            # 4. Build results dict
            results = {
                "experiment_id": "<experiment_id placeholder>",
                "feature_set_name": <instance>.get_feature_set_name(),
                "num_features": len(col_names),
                "sample_features": dict(zip(col_names[:10], col_vals[:10])),
                "status": "ok",
            }
            print(json.dumps(results))
        except Exception as exc:
            print(json.dumps({"error": str(exc), "status": "error"}))
            sys.exit(1)"""