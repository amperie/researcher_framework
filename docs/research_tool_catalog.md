# Research Tool Catalog

This project supports a shared YAML catalog for reusable research-tool definitions. The goal is to keep common tool presets in one place and let profiles or brainstorm configs reference them without copying the full config block each time.

## Catalog Files

Catalog files live under [`configs/research_tools/`](/E:/Programming/NeuralSignalResearcher/configs/research_tools/).

Current built-in catalog:

- [builtins.tools.yaml](/E:/Programming/NeuralSignalResearcher/configs/research_tools/builtins.tools.yaml:1)

## Referencing Tools

Profiles use `tool` as the callable key, while brainstorm configs use `path`. The catalog supports both so the same preset can be reused in either place.

Profile example:

```yaml
research:
  tools:
    - ref: trading_arxiv
      relevance_score_threshold: 6
      max_papers_to_digest: 2
```

Brainstorm example:

```yaml
roles:
  - name: researcher
    persona_type: researcher
    tools:
      - ref: trading_arxiv
      - ref: trading_profile_context
```

Refs can be overridden inline. The referenced preset is used as the base config and the local keys win.

Example:

```yaml
tools:
  - ref: arxiv-q-bio
    max_results: 4
```

arXiv top-level archive presets are named `arxiv-<category-prefix>`, for example `arxiv-math`, `arxiv-cs`, `arxiv-q-bio`, and `arxiv-q-fin`. Each preset expands to the concrete category IDs listed in the arXiv taxonomy, such as `q-bio.BM`, `q-bio.NC`, and `q-bio.QM` for `arxiv-q-bio`.

## Current Catalog Reuse

The trading pipeline profile uses refs in [trading.yaml](/E:/Programming/NeuralSignalResearcher/configs/profiles/trading.yaml:64).

The trading brainstorm config uses refs in [default.trading.brainstorm.yaml](/E:/Programming/NeuralSignalResearcher/configs/brainstorm/default.trading.brainstorm.yaml:79).

## Built-in Tool Settings

The actual accepted keys are still defined by the Python collectors in [research_tools.py](/E:/Programming/NeuralSignalResearcher/core/tools/research_tools.py:1). The most useful built-ins are:

- `collect_arxiv`
  Keys: `max_results`, `query`, `categories`, `match_any`
- `collect_memory`
  Keys: `n_results`
- `collect_prior_experiments`
  Keys: same as `collect_memory`
- `collect_profile_context`
  Keys: `include`
- `collect_adapter_context`
  Keys: no required custom keys
- `collect_strategy_library`
  Keys: `path`, `patterns`, `max_files`
- `collect_ssrn_finance`
  Keys: `max_results`
- `collect_nber_asset_pricing`
  Keys: `max_results`
- `collect_repec_finance`
  Keys: `max_results`
- `collect_quantpedia_strategies`
  Keys: `max_results`
- `collect_quantconnect_strategies`
  Keys: `max_results`
- `collect_reddit_trading`
  Keys: `max_results`, `default_terms`
- `collect_aqr_research`
  Keys: `max_results`
- `collect_fred_series_context`
  Keys: none
- `collect_cftc_cot`
  Keys: none
- `collect_sec_filings_signals`
  Keys: none

## Notes

- Catalog refs are resolved at load time for both profile loading and brainstorm config loading.
- Legacy inline tool definitions still work.
- Legacy brainstorm string entries like `core.tools.research_tools.collect_memory` still work.
- If a `ref` is unknown, loading fails with a config error instead of silently skipping it.
