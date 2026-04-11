"""Database logger node — persist the full experiment record to ChromaDB."""
from __future__ import annotations

from graph.state import ResearchState
from logger import get_logger

log = get_logger(__name__)


def db_logger_node(state: ResearchState) -> dict:
    """Upsert the full experiment record into ChromaDB.

    Reads:
        state['experiment_id']
        state['research_direction']
        state['research_summary']
        state['feature_proposals']
        state['generated_code']
        state['raw_results']
        state['analysis_summary']
        state['generalization_results']
        state['mlflow_run_id']
        state['mlflow_generalization_run_id']

    Writes:
        state['chroma_record_id']   — same as experiment_id

    Document format (concatenated text for embedding):
        Research direction: {research_direction}
        Summary: {research_summary}
        Top proposal: {name} — {description}
        Analysis: {analysis_summary}

    Metadata: flat scalar dict with timestamps, run IDs, key metrics.

    TODO: instantiate ChromaStore from config.
    TODO: compose document text from state fields.
    TODO: call chroma.upsert(experiment_id, document, metadata).
    """
    exp_id = state.get("experiment_id", "")
    raw = state.get("raw_results") or {}
    numeric_results = {k: v for k, v in raw.items() if isinstance(v, (int, float))}
    log.info("db_logger_node | Storing experiment record — experiment_id=%r", exp_id)
    log.debug("db_logger_node | Numeric metrics to embed in metadata: %s", numeric_results)

    # TODO: compose document
    # log.debug("db_logger_node | Document length: %d chars", len(document))

    # TODO: upsert
    # log.info("db_logger_node | Record upserted successfully — chroma_record_id=%r", exp_id)

    raise NotImplementedError("db_logger_node is not yet implemented")
