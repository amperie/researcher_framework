# `main.py` CLI Usage

This document describes the supported ways to run [`main.py`](/E:/Programming/NeuralSignalResearcher/main.py:1), including pipeline mode, brainstorm mode, seeded runs, resume flows, and discovery commands.

## Help

Show the built-in CLI help:

```powershell
uv run python main.py help
```

## Discovery

List available profiles:

```powershell
uv run python main.py --list-profiles
```

List pipeline nodes for a profile:

```powershell
uv run python main.py --profile trading --list-nodes
```

## Pipeline Mode

Run the default pipeline from a fresh direction:

```powershell
uv run python main.py --profile neuralsignal --direction "attention head specialization"
```

Prompt for the direction interactively:

```powershell
uv run python main.py --profile neuralsignal
```

Run once, then stop:

```powershell
uv run python main.py --profile trading --direction "regime-aware mean reversion"
```

Run once, then execute each proposed next step once:

```powershell
uv run python main.py --profile trading --direction "SPY intraday momentum" --run-next-steps-once
```

Loop continuously using the top promoted next step:

```powershell
uv run python main.py --profile trading --direction "cross-asset macro regime filters" --loop
```

## Pipeline Seeding

Start from a saved next step:

```powershell
uv run python main.py --profile neuralsignal --next-step "next_step:123"
```

Start from a saved proposal seed:

```powershell
uv run python main.py --profile trading --source-experiment "exp-123" --proposal-seed "proposal_seed:123"
```

Start from a saved run handoff:

```powershell
uv run python main.py --profile trading --source-experiment "exp-123" --handoff "run_handoff:123"
```

## Resume Mode

Resume from a persisted record and continue from the pipeline entry node:

```powershell
uv run python main.py --profile trading --resume-from "experiment_result:123"
```

Resume from a persisted record and restart at a later node:

```powershell
uv run python main.py --profile trading --resume-from "experiment_result:123" --start-node implement
```

## Brainstorm Mode

Start a brainstorm session from a fresh direction:

```powershell
uv run python main.py --mode brainstorm --profile trading --direction "using regime detection and technical indicators to trade SPY"
```

Start brainstorm mode with an explicit brainstorm config:

```powershell
uv run python main.py --mode brainstorm --profile trading --config configs/brainstorm/default.trading.brainstorm.yaml
```

The long form also works:

```powershell
uv run python main.py --mode brainstorm --profile trading --brainstorm-config configs/brainstorm/default.trading.brainstorm.yaml
```

If no brainstorm config is passed:

- the CLI uses the only config under `configs/brainstorm` automatically
- if multiple configs exist, it prompts you to choose one

Resume a saved brainstorm session:

```powershell
uv run python main.py --mode brainstorm --profile trading --resume-brainstorm "brainstorm-session-id"
```

## Brainstorm Seeding

Import prior run context from a source experiment:

```powershell
uv run python main.py --mode brainstorm --profile trading --source-experiment "exp-123"
```

Import a saved proposal seed into brainstorm mode:

```powershell
uv run python main.py --mode brainstorm --profile trading --source-experiment "exp-123" --proposal-seed "proposal_seed:123"
```

Import a saved run handoff into brainstorm mode:

```powershell
uv run python main.py --mode brainstorm --profile trading --source-experiment "exp-123" --handoff "run_handoff:123"
```

Import a promoted next step into brainstorm mode:

```powershell
uv run python main.py --mode brainstorm --profile trading --source-experiment "exp-123" --next-step "next_step:123"
```

## Campaign Metadata

Attach campaign metadata to a run:

```powershell
uv run python main.py --profile trading --direction "factor rotation under macro regimes" --campaign-id "camp-1" --campaign-title "Macro rotation study" --campaign-variant-id "v1" --campaign-variant-title "Baseline" --campaign-variant-index 1 --campaign-size 3
```

## Interactive Brainstorm Commands

Once brainstorm mode pauses, the CLI accepts:

```text
help
continue
summary
research
plan
feedback <text>
approve_plan
execute
exit
```

## Common Notes

- `--mode` defaults to `pipeline`.
- `--config` is an alias for `--brainstorm-config`.
- `--direction` cannot be combined with `--source-experiment`, `--handoff`, `--proposal-seed`, or `--next-step`.
- `--resume-from` cannot be combined with fresh-direction or seed-selection arguments.
- `--start-node` requires `--resume-from` unless you start at the pipeline entry node.
