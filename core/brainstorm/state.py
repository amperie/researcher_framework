from __future__ import annotations

from typing import Literal, TypedDict


class BrainstormRoleConfig(TypedDict, total=False):
    name: str
    persona_type: str
    goal: str
    style: str
    enabled: bool
    llm_key: str
    model: str
    provider: str
    tools: list[dict]
    research_budget: dict
    prompt_overrides: dict


class BrainstormTurn(TypedDict, total=False):
    turn_id: str
    round_index: int
    role_name: str
    role_type: str
    message_type: str
    content: str
    structured_points: list[dict]
    citations: list[dict]
    created_at: str


class ConsensusSnapshot(TypedDict, total=False):
    agreed_points: list[str]
    active_options: list[dict]
    rejected_options: list[dict]
    objections: list[dict]
    assumptions: list[str]
    open_questions: list[str]
    next_recommendation: str
    confidence: str
    evidence: list[dict]


class PlanDraft(TypedDict, total=False):
    research_direction: str
    refined_ideas: list[dict]
    proposals: list[dict]
    implementation_plans: list[dict]
    constraints: list[str]
    exclusions: list[str]
    success_criteria: list[str]
    unresolved_questions: list[str]


class BrainstormState(TypedDict, total=False):
    session_id: str
    profile_name: str
    brainstorm_config_path: str
    brainstorm_config_name: str
    status: Literal[
        "running",
        "paused",
        "awaiting_user",
        "planning",
        "approved_for_execution",
        "executing",
        "completed",
        "cancelled",
    ]
    current_goal: str
    user_intent_notes: list[str]
    role_configs: list[BrainstormRoleConfig]
    turn_log: list[BrainstormTurn]
    consensus: ConsensusSnapshot
    plan_draft: PlanDraft
    pending_questions: list[str]
    pending_decisions: list[str]
    stop_policy: dict
    summary_config: dict
    prompt_config: dict
    progress: dict
    execution_handoff: dict
    approved_plan: bool
    last_summary: str
    errors: list[str]
    source_experiment_record_id: str
    source_next_step_record_id: str
    source_next_step_title: str
    source_proposal_seed_record_id: str
    source_proposal_seed_title: str
    proposal_seed_planning_notes: str
    seed_context_artifacts: list[dict]
    root_run_family_id: str
    root_research_direction: str
    campaign_id: str
    campaign_title: str
    campaign_variant_id: str
    campaign_variant_title: str
    campaign_variant_index: int
    campaign_size: int
