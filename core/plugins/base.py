"""Shared experiment adapter contract.

Adapters are the boundary between the generic research graph and domain-specific
execution. The graph owns orchestration, LLM prompting, state snapshots, retries,
and storage. A domain adapter owns the parts that cannot be made generic:
dataset preparation, platform inspection, experiment execution, model training,
backtesting, metric calculation, and domain-specific result summaries.

New plugins should expose a module-level ``get_adapter()`` function that returns
an object satisfying ``ResearchAdapter``:

    def get_adapter() -> ResearchAdapter:
        return MyDomainAdapter()

Adapter methods receive the loaded profile and the full graph state. Methods that
perform graph work return a *partial state update*, just like LangGraph nodes do.
They should not mutate ``state`` in place.
"""
from __future__ import annotations

from typing import Any, Protocol


class ResearchAdapter(Protocol):
    """Protocol implemented by domain experiment adapters.

    The adapter is intentionally structural: a class does not have to inherit
    from this protocol at runtime. It only needs methods with compatible names
    and behavior.

    Common state keys available during the lifecycle:
    - ``research_direction``: user-supplied research question.
    - ``research_artifacts``: scored artifacts from profile-configured research tools.
    - ``ideas`` / ``refined_ideas`` / ``proposals``: LLM-generated research plan.
    - ``implementation_plans`` / ``implementations``: generated code metadata.
    - ``validation_results``: generated test outcomes.
    - ``experiment_artifacts``: outputs from ``prepare_experiment``.
    - ``experiment_results`` / ``models``: outputs from ``execute_experiment``.
    - ``errors``: accumulated non-fatal errors.

    Return conventions:
    - Return dictionaries containing only keys the method wants to update.
    - Preserve prior non-fatal errors by starting from ``list(state.get("errors") or [])``.
    - Use serializable values where possible; large in-memory objects should be
      private helper keys such as ``_dataframe`` and should not be relied on by
      storage nodes.
    """

    def validate_environment(
        self,
        profile: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Report whether the domain runtime appears usable.

        Called by research/context tools, not by the main experiment execution
        path. This method should be cheap and side-effect-light. Use it to report
        configuration and dependency readiness, for example:
        - whether a required package can be imported
        - whether a configured source path exists
        - which interpreter or subprocess bridge will be used
        - whether required environment variables are present

        Do not run experiments, download large datasets, train models, or create
        durable artifacts here.

        Return:
            A serializable diagnostic dictionary. The research node may turn this
            into a scoreable ``research_artifact`` so downstream prompts know the
            platform constraints.
        """
        ...

    def build_context(self, profile: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        """Return domain context that helps the LLM plan realistic experiments.

        Called by research/context tools. This is where an adapter can expose
        platform facts that should influence ideation and implementation:
        available datasets, market data fields, symbol universes, base classes,
        allowed APIs, risk constraints, metric definitions, benchmark names, or
        feature extraction constraints.

        Keep this concise. It is often included in LLM prompts after scoring, so
        prefer structured summaries over large raw data dumps.

        Return:
            A serializable dictionary. Example keys:
            ``{"data_sources": [...], "base_classes": [...], "evaluation": {...}}``.
        """
        ...

    def prepare_experiment(
        self,
        profile: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Prepare domain artifacts needed before execution.

        Called by the generic ``prepare_experiment`` graph node after proposal,
        implementation, and validation steps. The adapter receives the full state
        and decides how to map proposals to artifacts.

        Typical responsibilities:
        - NeuralSignal: load generated feature classes, create feature datasets,
          cache CSV paths, and attach lightweight dataset metadata.
        - Trading: build backtest configs, resolve universes/date ranges, prepare
          parameter grids, or cache market data bundles.
        - Other domains: prepare simulation configs, benchmark inputs, generated
          assets, evaluation fixtures, or execution manifests.

        Expected return:
            A partial state update. Most adapters should return:

            {
                "experiment_artifacts": [
                    {
                        "artifact_id": "...",
                        "artifact_type": "dataset|backtest_config|...",
                        "proposal_name": "...",
                        ...
                    }
                ],
                "errors": [...]
            }

        Compatibility:
            If an artifact is a dataset, also returning ``"datasets"`` as an
            alias is useful for older debug tooling.

        Do not:
            Run the final experiment or calculate final performance metrics here.
            That belongs in ``execute_experiment``.
        """
        ...

    def execute_experiment(
        self,
        profile: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Run domain experiments and return normalized results.

        Called by the generic ``execute_experiment`` graph node after
        ``prepare_experiment``. The adapter owns execution strategy. It may run
        one experiment per proposal, parameter sweeps, walk-forward folds,
        benchmarks, cost-sensitivity checks, or any other domain-appropriate
        execution plan.

        Typical responsibilities:
        - load generated implementations from ``state["implementations"]``
        - consume ``state["experiment_artifacts"]``
        - call the domain platform or subprocess bridge
        - enforce deterministic guardrails such as no-lookahead checks, costs,
          validation splits, risk limits, or API constraints
        - calculate normalized metrics
        - attach lightweight artifact references such as report paths, trade
          logs, feature importance summaries, or model IDs

        Expected return:
            A partial state update. Common keys:

            {
                "experiment_results": [
                    {
                        "experiment_id": "...",
                        "proposal_name": "...",
                        "metrics": {"primary_metric": 0.0, ...},
                        "artifacts": {...},
                        ...
                    }
                ],
                "models": [...],        # optional, if the domain trains models
                "errors": [...]
            }

        Metrics should use the names declared in ``profile["evaluation"]`` so
        the generic ``evaluate`` and ``store_results`` nodes can interpret them.
        """
        ...

    def summarize_result(
        self,
        profile: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Return a compact domain-specific summary of completed experiments.

        This method is for adapters that need a domain-shaped summary in
        addition to the generic ``experiment_results`` list. It should read from
        full state and produce a concise serializable object suitable for
        storage, retrieval, or prompt context.

        Examples:
        - NeuralSignal: best proposal by AUC, feature importance highlights, and
          dataset sizes.
        - Trading: best strategy by Sharpe, drawdown/turnover summary, benchmark
          comparison, and risk warnings.

        Return:
            A serializable dictionary. This method should not run new
            experiments, mutate state, or write external records.
        """
        ...
