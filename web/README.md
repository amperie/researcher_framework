# Web Inspector

This app provides an aggregated run-centric view across the project backends:

- Mongo memory records
- Chroma vector projection
- Neo4j graph projection
- MLflow run metadata
- artifact references stored in Mongo and linked to MinIO/S3

## Run

From the project root:

```powershell
uv run python -m web
```

Then open:

```text
http://127.0.0.1:8090
```

## What it shows

- run list across profiles
- single run summary
- related memory records from Mongo
- Chroma projection payload
- graph view from canonical entities/relations plus Neo4j node hydration when enabled
- artifact links
- MLflow run metadata and UI link

## Current scope

The app treats the canonical `experiment_result` memory record as the anchor for a run.
It does not yet provide:

- live polling
- diffed state snapshots
- inline artifact previews
- global graph exploration beyond the selected run
