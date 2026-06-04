# `main.py` CLI Usage

[`main.py`](/E:/Programming/NeuralSignalResearcher/main.py:1) is the primary entrypoint for profile discovery, pipeline runs, resume flows, seeded runs, and interactive brainstorm sessions.

## Help

```powershell
uv run python main.py help
uv run python main.py --help
```

## Discovery

```powershell
uv run python main.py --list-profiles
uv run python main.py --profile neuralsignal --list-nodes
```

If `--profile` is omitted and more than one profile exists, the CLI prompts for one.

## Pipeline Mode

`pipeline` is the default mode.

```powershell
uv run python main.py --profile neuralsignal --direction "attention head specialization"
uv run python main.py --mode pipeline --profile trading --direction "SPY regime filters"
```

If `--direction` is omitted in pipeline mode, the CLI prompts for a research direction and run mode.

Run modes:

- default: run the pipeline once
- `--run-next-steps-once`: run the initial direction, then run each proposed `next_step` once
- `--loop`: keep running the top proposed `next_step` until no new step remains

```powershell
uv run python main.py --profile trading --direction "SPY intraday momentum" --run-next-steps-once
uv run python main.py --profile neuralsignal --direction "residual entropy features" --loop
```

## Pipeline Seeding

Start from persisted work instead of a fresh direction:

```powershell
uv run python main.py --profile neuralsignal --next-step "next_step:123"
uv run python main.py --profile trading --source-experiment "exp-123" --proposal-seed "proposal_seed:123"
uv run python main.py --profile trading --source-experiment "exp-123" --handoff "run_handoff:123"
```

Seed behavior:

- `--next-step`: runs a promoted follow-up recommendation
- `--proposal-seed`: starts from saved proposals
- `--handoff`: starts from a saved run handoff
- `--source-experiment`: supplies lineage and prior run context for proposal/handoff seeds

## Resume Mode

Resume from memory:

```powershell
uv run python main.py --profile trading --resume-from "experiment_result:123"
uv run python main.py --profile trading --resume-from "experiment_result:123" --start-node implement
```

Resume from a state snapshot:

```powershell
uv run python main.py --profile neuralsignal --start-node check_experiment_jobs --resume-snapshot
uv run python main.py --profile neuralsignal --start-node implement --resume-snapshot dev/state/neuralsignal/after_plan_implementation.json
```

With no value, `--resume-snapshot` loads `dev/state/<profile>/after_<previous-node>.json` based on `--start-node`.

## Brainstorm Mode

Start a new brainstorm session:

```powershell
uv run python main.py --mode brainstorm --profile trading --direction "using regime detection and technical indicators to trade SPY"
```

Use an explicit brainstorm config:

```powershell
uv run python main.py --mode brainstorm --profile trading --config configs/brainstorm/default.trading.brainstorm.yaml
uv run python main.py --mode brainstorm --profile trading --brainstorm-config configs/brainstorm/default.trading.brainstorm.yaml
```

If no config is passed, the CLI prefers a profile-named config, uses the only config when exactly one exists, or prompts when multiple configs exist.

Resume a saved brainstorm session:

```powershell
uv run python main.py --mode brainstorm --profile trading --resume-brainstorm "brainstorm-session-id"
```

## Brainstorm Seeding

Import prior work into a brainstorm session:

```powershell
uv run python main.py --mode brainstorm --profile trading --source-experiment "exp-123"
uv run python main.py --mode brainstorm --profile trading --source-experiment "exp-123" --proposal-seed "proposal_seed:123"
uv run python main.py --mode brainstorm --profile trading --source-experiment "exp-123" --handoff "run_handoff:123"
uv run python main.py --mode brainstorm --profile trading --source-experiment "exp-123" --next-step "next_step:123"
```

Imported context is editable in the brainstorm session before `execute` hands it to the pipeline.

## Interactive Brainstorm Commands

Once brainstorm mode pauses:

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

## Campaign Metadata

Attach campaign lineage to stored results:

```powershell
uv run python main.py --profile trading --direction "factor rotation under macro regimes" --campaign-id "camp-1" --campaign-title "Macro rotation study" --campaign-variant-id "v1" --campaign-variant-title "Baseline" --campaign-variant-index 1 --campaign-size 3
```

For batched campaign configs, use [`run_campaign.py`](/E:/Programming/NeuralSignalResearcher/run_campaign.py:1):

```powershell
uv run python run_campaign.py configs/campaigns/neuralsignal_campaign.yaml
uv run python run_campaign.py configs/campaigns/neuralsignal_campaign.yaml --dry-run
```

## NeuralSignal Dataset Refresh

Force NeuralSignal dataset regeneration instead of memory/local CSV reuse:

```powershell
uv run python main.py --profile neuralsignal --direction "new detector" --force-dataset-refresh
uv run python main.py --profile neuralsignal --start-node submit_experiment_jobs --resume-snapshot --force-dataset-refresh
```

## Option Compatibility

- `--mode` defaults to `pipeline`.
- `--config` is an alias for `--brainstorm-config`.
- `--direction` cannot be combined with `--source-experiment`, `--handoff`, `--proposal-seed`, or `--next-step`.
- `--resume-from` and `--resume-snapshot` cannot be used together.
- Resume options cannot be combined with fresh-direction or seed-selection arguments.
- `--start-node` requires `--resume-from` or `--resume-snapshot` unless it is the profile's first pipeline node.
- `--loop` and `--run-next-steps-once` are mutually exclusive.
