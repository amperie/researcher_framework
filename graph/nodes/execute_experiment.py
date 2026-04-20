"""Execute prepared experiments using the active domain adapter."""
from __future__ import annotations

from uuid import uuid4

from graph.state import ResearchState
from plugins.loader import adapter_has, load_adapter
from utils.logger import get_logger

log = get_logger(__name__)


def execute_experiment_node(state: ResearchState, profile: dict) -> dict:
    proposals = state.get("proposals") or []
    implementations = state.get("implementations") or []
    artifacts = state.get("experiment_artifacts") or state.get("datasets") or []

    if not proposals:
        log.warning("execute_experiment_node | No proposals - skipping")
        return {"experiment_results": []}

    try:
        adapter = load_adapter(profile)
    except Exception as exc:
        return {
            "experiment_results": [],
            "errors": (state.get("errors") or []) + [f"execute_experiment: adapter load failed: {exc}"],
        }

    proposal_by_name = {p.get("name", ""): p for p in proposals}
    impl_by_name = {i.get("proposal_name", ""): i for i in implementations}
    artifact_by_name = {a.get("proposal_name", ""): a for a in artifacts}

    results: list[dict] = []
    models: list[dict] = []
    errors = list(state.get("errors") or [])

    runnable_names = list(proposal_by_name)
    for proposal_name in runnable_names:
        proposal = proposal_by_name.get(proposal_name, {})
        impl = impl_by_name.get(proposal_name)
        artifact = artifact_by_name.get(proposal_name)
        experiment_id = str(uuid4())

        log.info("execute_experiment_node | Running proposal=%r", proposal_name)
        try:
            if adapter_has(adapter, "execute_experiment"):
                result = adapter.execute_experiment(
                    profile=profile,
                    proposal=proposal,
                    implementation=impl,
                    artifact=artifact,
                    experiment_id=experiment_id,
                    state=state,
                )
            elif adapter_has(adapter, "run_experiment"):
                result = adapter.run_experiment(profile, proposal, impl, artifact, experiment_id)
                if result and adapter_has(adapter, "train_model"):
                    model = adapter.train_model(profile, result, artifact)
                    if model:
                        model.setdefault("experiment_id", experiment_id)
                        model.setdefault("proposal_name", proposal_name)
                        models.append(model)
                        result.setdefault("metrics", {}).update(model.get("metrics") or {})
                        result.setdefault("model", model)
            else:
                errors.append(f"execute_experiment: adapter cannot execute {proposal_name}")
                continue

            if result:
                result.setdefault("experiment_id", experiment_id)
                result.setdefault("proposal_name", proposal_name)
                result.setdefault("proposal", proposal)
                results.append(result)
            else:
                errors.append(f"execute_experiment: no result returned for {proposal_name}")
        except Exception as exc:
            log.error("execute_experiment_node | Failed for %r: %s", proposal_name, exc, exc_info=True)
            errors.append(f"execute_experiment: {proposal_name} failed: {exc}")

    delta: dict = {"experiment_results": results, "errors": errors}
    if models:
        delta["models"] = models
    return delta

