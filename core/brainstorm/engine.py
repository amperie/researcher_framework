from __future__ import annotations

from datetime import datetime, timezone
import importlib
import json
from typing import Any, Callable
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from core.brainstorm.commands import parse_brainstorm_command
from core.brainstorm.handoff import build_execution_handoff, choose_start_node
from core.graph.nodes.plan_implementation import _coerce_plan_string, _normalize_implementation_plans
from core.graph.nodes.research import _normalize_artifact, _score_artifacts
from core.brainstorm.seeding import apply_brainstorm_seed
from core.brainstorm.state import BrainstormState
from core.brainstorm.summaries import render_consensus_summary
from core.llm.factory import get_llm
from core.memory import MemoryService, build_memory_record
from core.utils import extract_json_object
from core.utils.logger import get_logger

log = get_logger(__name__)


def create_brainstorm_state(
    *,
    profile_name: str,
    direction: str,
    brainstorm_cfg: dict[str, Any],
    session_id: str | None = None,
    seed: dict[str, Any] | None = None,
) -> BrainstormState:
    state: BrainstormState = {
        "session_id": session_id or str(uuid4()),
        "profile_name": profile_name,
        "brainstorm_config_path": str(brainstorm_cfg.get("path") or ""),
        "brainstorm_config_name": str(brainstorm_cfg.get("name") or ""),
        "status": "running",
        "current_goal": str(direction or "").strip(),
        "user_intent_notes": [],
        "role_configs": [dict(item) for item in (brainstorm_cfg.get("roles") or []) if item.get("enabled", True)],
        "turn_log": [],
        "consensus": _empty_consensus(),
        "plan_draft": _empty_plan_draft(str(direction or "")),
        "pending_questions": [],
        "pending_decisions": [],
        "stop_policy": dict(brainstorm_cfg.get("stop_policy") or {}),
        "summary_config": dict(brainstorm_cfg.get("summary") or {}),
        "prompt_config": dict(brainstorm_cfg.get("prompts") or {}),
        "progress": {
            "round_index": 0,
            "message_count": 0,
        },
        "execution_handoff": {},
        "approved_plan": False,
        "last_summary": "",
        "errors": [],
        "source_experiment_record_id": "",
        "source_next_step_record_id": "",
        "source_next_step_title": "",
        "source_proposal_seed_record_id": "",
        "source_proposal_seed_title": "",
        "proposal_seed_planning_notes": "",
        "seed_context_artifacts": [],
        "root_run_family_id": "",
        "root_research_direction": "",
        "campaign_id": "",
        "campaign_title": "",
        "campaign_variant_id": "",
        "campaign_variant_title": "",
        "campaign_variant_index": 0,
        "campaign_size": 0,
    }
    apply_brainstorm_seed(state, dict(seed or {}))
    state["execution_handoff"] = build_execution_handoff(state, brainstorm_cfg)
    state["last_summary"] = render_consensus_summary(state)
    return state


class BrainstormEngine:
    def __init__(self, profile: dict[str, Any], brainstorm_cfg: dict[str, Any]) -> None:
        self.profile = profile
        self.brainstorm_cfg = brainstorm_cfg

    def run_until_pause(
        self,
        state: BrainstormState,
        *,
        emit: Callable[[str], None] | None = None,
        on_role_start: Callable[[dict[str, Any], int], None] | None = None,
        on_role_end: Callable[[dict[str, Any], int], None] | None = None,
    ) -> BrainstormState:
        emitter = emit or (lambda _text: None)
        role_started = on_role_start or (lambda _role, _round_index: None)
        role_ended = on_role_end or (lambda _role, _round_index: None)
        stop_policy = dict(state.get("stop_policy") or {})
        max_rounds = int(stop_policy.get("max_rounds_per_run", 3) or 3)
        summary_interval_messages = int(stop_policy.get("summary_interval_messages", 4) or 4)
        pause_after_research = bool(stop_policy.get("pause_after_research_round", True))
        state["status"] = "running"

        try:
            while int((state.get("progress") or {}).get("round_index", 0) or 0) < max_rounds:
                state["progress"]["round_index"] = int(state["progress"].get("round_index", 0) or 0) + 1
                round_index = int(state["progress"]["round_index"])
                had_research_turn = False
                for role in list(state.get("role_configs") or []):
                    if not role.get("enabled", True):
                        continue
                    role_started(role, round_index)
                    try:
                        turn = self._run_role_turn(state, role, round_index)
                    finally:
                        role_ended(role, round_index)
                    state["turn_log"].append(turn)
                    state["progress"]["message_count"] = int(state["progress"].get("message_count", 0) or 0) + 1
                    if turn.get("role_type") == "researcher":
                        had_research_turn = True
                    emitter(self._render_turn(turn))
                    self._refresh_consensus(state)
                    if summary_interval_messages > 0 and int(state["progress"]["message_count"]) % summary_interval_messages == 0:
                        summary = render_consensus_summary(state)
                        state["last_summary"] = summary
                        emitter("\n[Current thinking]\n" + summary + "\n")
                if pause_after_research and had_research_turn:
                    break
            state["status"] = "awaiting_user"
            state["last_summary"] = render_consensus_summary(state)
            emitter("\n[Checkpoint]\n" + state["last_summary"] + "\n")
            emitter("\nType `help` to see available commands.\n")
            return state
        except KeyboardInterrupt:
            state["status"] = "awaiting_user"
            state["last_summary"] = render_consensus_summary(state)
            emitter("\n[Interrupted]\n" + state["last_summary"] + "\n")
            emitter("\nType `help` for commands or `exit` to leave brainstorm mode.\n")
            return state

    def apply_command(
        self,
        state: BrainstormState,
        command: dict[str, Any] | str,
        *,
        emit: Callable[[str], None] | None = None,
        on_role_start: Callable[[dict[str, Any], int], None] | None = None,
        on_role_end: Callable[[dict[str, Any], int], None] | None = None,
    ) -> BrainstormState:
        emitter = emit or (lambda _text: None)
        cmd = parse_brainstorm_command(command) if isinstance(command, str) else dict(command)
        cmd_type = str(cmd.get("type") or "").strip()
        if cmd_type == "help":
            from core.brainstorm.commands import HELP_TEXT

            emitter(HELP_TEXT)
            return state
        if cmd_type == "summary":
            summary = state.get("last_summary") or render_consensus_summary(state)
            state["last_summary"] = summary
            emitter(summary + "\n")
            return state
        if cmd_type == "feedback":
            text = str(cmd.get("text") or "").strip()
            if text:
                state["user_intent_notes"] = list(state.get("user_intent_notes") or []) + [text]
                state["current_goal"] = text
                self._reset_deliberation_state(state, preserve_seed_evidence=True)
                emitter("Direction updated. Previous brainstorm consensus was cleared.\n")
            return state
        if cmd_type == "request_research":
            state["progress"]["round_index"] = 0
            return self.run_until_pause(state, emit=emit, on_role_start=on_role_start, on_role_end=on_role_end)
        if cmd_type == "draft_plan":
            self._draft_plan(state)
            emitter(self._render_plan(state) + "\n")
            return state
        if cmd_type == "approve_plan":
            if state.get("plan_draft"):
                state["approved_plan"] = True
                state["status"] = "awaiting_user"
                state["execution_handoff"] = build_execution_handoff(state, self.brainstorm_cfg)
                emitter("Plan approved. Use `execute` to start downstream execution.\n")
            return state
        if cmd_type == "start_execution":
            state["approved_plan"] = True
            state["status"] = "approved_for_execution"
            state["execution_handoff"] = build_execution_handoff(state, self.brainstorm_cfg)
            return state
        if cmd_type == "exit":
            state["status"] = "cancelled"
            return state
        return self.run_until_pause(state, emit=emit, on_role_start=on_role_start, on_role_end=on_role_end)

    def _run_role_turn(self, state: BrainstormState, role: dict[str, Any], round_index: int) -> dict[str, Any]:
        role_type = str(role.get("persona_type") or role.get("name") or "persona")
        if role_type == "researcher":
            return self._run_researcher_turn(state, role, round_index)

        prompt_cfg = self._role_prompt_config(state, role)
        prompt_context = {
            "role_name": str(role.get("name") or role_type),
            "role_type": role_type,
            "role_goal": str(role.get("goal") or ""),
            "role_style": str(role.get("style") or ""),
            "current_goal": str(state.get("current_goal") or ""),
            "consensus_summary": render_consensus_summary(state),
            "user_notes_json": json.dumps(state.get("user_intent_notes") or [], indent=2),
        }
        system_prompt = self._render_template(str(prompt_cfg.get("system_template") or ""), prompt_context)
        human_prompt = self._render_template(
            str(prompt_cfg.get("human_template") or ""),
            prompt_context,
        )
        llm = get_llm(
            step_name=str(role.get("llm_key") or "brainstorm"),
            profile=self.profile,
            provider=(str(role.get("provider") or "").strip() or None),
            model=(str(role.get("model") or "").strip() or None),
        )
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])
        return {
            "turn_id": str(uuid4()),
            "round_index": round_index,
            "role_name": str(role.get("name") or role_type),
            "role_type": role_type,
            "message_type": "discussion",
            "content": str(getattr(response, "content", "") or ""),
            "structured_points": [],
            "citations": [],
            "created_at": _now_iso(),
        }

    def _run_researcher_turn(self, state: BrainstormState, role: dict[str, Any], round_index: int) -> dict[str, Any]:
        artifacts: list[dict[str, Any]] = []
        researcher_prompt_cfg = dict((state.get("prompt_config") or {}).get("researcher") or {})
        tool_defs = [dict(item) for item in list(role.get("tools") or []) if isinstance(item, dict) and item.get("enabled", True)]
        max_tools_per_round = int(((role.get("research_budget") or {}).get("max_tools_per_round") or 0))
        if max_tools_per_round > 0:
            tool_defs = tool_defs[:max_tools_per_round]
        for tool_def in tool_defs:
            tool_path = str(tool_def.get("path") or "").strip()
            if not tool_path:
                continue
            tool = _load_callable(tool_path)
            if tool is None:
                continue
            tool_cfg = dict(tool_def)
            tool_cfg.setdefault("name", str(tool_def.get("name") or tool_path.rsplit(".", 1)[-1]))
            try:
                result = tool(str(state.get("current_goal") or ""), self.profile, tool_cfg, state)
            except Exception as exc:
                result = [{
                    "artifact_id": f"research_error:{tool_path}",
                    "source": str(tool_cfg.get("name") or role.get("name") or "researcher"),
                    "source_type": "error",
                    "title": f"Research tool failed: {tool_path}",
                    "summary": str(exc),
                    "metadata": {"tool": tool_path, "tool_name": tool_cfg.get("name", "")},
                    "raw": {},
                }]
            artifacts.extend(list(result or []))
        capped = artifacts[: int(((role.get("research_budget") or {}).get("max_artifacts_per_tool") or 4) * max(len(tool_defs), 1))]
        normalized = [
            _normalize_artifact(item, self._matching_tool_config(tool_defs, item))
            for item in capped
            if isinstance(item, dict)
        ]
        scored = self._score_brainstorm_research(state, role, normalized, tool_defs)
        selected = self._select_brainstorm_research(role, scored)
        state["consensus"]["evidence"] = list(state.get("consensus", {}).get("evidence") or []) + selected[:5]
        line_template = str(researcher_prompt_cfg.get("result_line_template") or "- {title}: {summary_excerpt}")
        summary_excerpt_chars = int(researcher_prompt_cfg.get("summary_excerpt_chars", 100) or 100)
        summary_lines = []
        for item in selected[:3]:
            summary_lines.append(
                self._render_template(
                    line_template,
                    {
                        "artifact_id": str(item.get("artifact_id") or ""),
                        "title": str(item.get("title") or ""),
                        "summary": str(item.get("summary") or ""),
                        "summary_excerpt": str(item.get("summary") or "")[:summary_excerpt_chars],
                        "source": str(item.get("source") or ""),
                        "source_type": str(item.get("source_type") or ""),
                    },
                )
            )
        content = "\n".join(summary_lines) if summary_lines else str(researcher_prompt_cfg.get("empty_message") or "No evidence gathered.")
        return {
            "turn_id": str(uuid4()),
            "round_index": round_index,
            "role_name": str(role.get("name") or "researcher"),
            "role_type": "researcher",
            "message_type": "research",
            "content": content,
            "structured_points": selected[:5],
            "citations": [{"title": item.get("title", ""), "url": item.get("url", "")} for item in selected[:5]],
            "created_at": _now_iso(),
        }

    def _matching_tool_config(self, tool_defs: list[dict[str, Any]], artifact: dict[str, Any]) -> dict[str, Any]:
        artifact_source = str(artifact.get("source") or "").strip()
        for tool_def in tool_defs:
            if str(tool_def.get("name") or "").strip() == artifact_source:
                return tool_def
        return {}

    def _score_brainstorm_research(
        self,
        state: BrainstormState,
        role: dict[str, Any],
        artifacts: list[dict[str, Any]],
        tool_defs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not artifacts:
            return []
        llm = get_llm(
            step_name=str(role.get("llm_key") or "research"),
            profile=self.profile,
            provider=(str(role.get("provider") or "").strip() or None),
            model=(str(role.get("model") or "").strip() or None),
        )
        direction = str(state.get("current_goal") or "")
        return _score_artifacts(direction, self.profile, artifacts, tool_defs, llm)

    def _select_brainstorm_research(self, role: dict[str, Any], scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not scored:
            return []
        selected = [
            item for item in scored
            if int(item.get("relevance_score", 0)) >= int(item.get("score_threshold", 6))
        ]
        selected.sort(key=lambda item: int(item.get("relevance_score", 0)), reverse=True)
        max_evidence = int(((role.get("research_budget") or {}).get("max_evidence_items") or 5))
        return selected[:max_evidence]

    def _refresh_consensus(self, state: BrainstormState) -> None:
        recent_turns = list(state.get("turn_log") or [])[-6:]
        if not recent_turns:
            return
        prompt_cfg = dict((state.get("prompt_config") or {}).get("consensus") or {})
        consensus_input_json = json.dumps(
            {
                "goal": state.get("current_goal", ""),
                "recent_turns": [
                    {
                        "role_name": item.get("role_name"),
                        "role_type": item.get("role_type"),
                        "content": item.get("content"),
                    }
                    for item in recent_turns
                ],
            },
            indent=2,
        )
        system_prompt = self._render_template(
            str(prompt_cfg.get("system_template") or ""),
            {"consensus_input_json": consensus_input_json},
        )
        human_prompt = self._render_template(
            str(prompt_cfg.get("human_template") or "{consensus_input_json}"),
            {"consensus_input_json": consensus_input_json},
        )
        facilitator = next(
            (item for item in (state.get("role_configs") or []) if str(item.get("persona_type")) == "facilitator"),
            None,
        )
        llm = get_llm(
            step_name="brainstorm_consensus",
            profile=self.profile,
            provider=(str((facilitator or {}).get("provider") or "").strip() or None),
            model=(str((facilitator or {}).get("model") or "").strip() or None),
        )
        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ])
            consensus = extract_json_object(getattr(response, "content", "") or "")
            if isinstance(consensus, dict):
                merged = dict(state.get("consensus") or {})
                for key, value in consensus.items():
                    merged[key] = value
                state["consensus"] = merged
        except Exception as exc:
            log.warning("brainstorm.engine | Consensus refresh failed: %s", exc)

    def _draft_plan(self, state: BrainstormState) -> None:
        prompt_cfg = dict((state.get("prompt_config") or {}).get("plan") or {})
        plan_input_json = json.dumps(
            {
                "goal": state.get("current_goal", ""),
                "consensus": state.get("consensus") or {},
                "user_notes": state.get("user_intent_notes") or [],
            },
            indent=2,
        )
        system_prompt = self._render_template(
            str(prompt_cfg.get("system_template") or ""),
            {"plan_input_json": plan_input_json},
        )
        human_prompt = self._render_template(
            str(prompt_cfg.get("human_template") or "{plan_input_json}"),
            {"plan_input_json": plan_input_json},
        )
        planner = next(
            (item for item in (state.get("role_configs") or []) if str(item.get("persona_type")) in {"planner", "facilitator"}),
            None,
        )
        llm = get_llm(
            step_name="brainstorm_plan",
            profile=self.profile,
            provider=(str((planner or {}).get("provider") or "").strip() or None),
            model=(str((planner or {}).get("model") or "").strip() or None),
        )
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])
        plan = extract_json_object(getattr(response, "content", "") or "")
        if isinstance(plan, dict):
            normalized_proposals, proposal_errors = _normalize_brainstorm_proposals(plan.get("proposals") or [])
            normalized_implementation_plans, implementation_errors = _normalize_implementation_plans(
                list(plan.get("implementation_plans") or []),
                normalized_proposals,
            )
            state["plan_draft"] = {
                "research_direction": str(plan.get("research_direction") or state.get("current_goal") or ""),
                "refined_ideas": list(plan.get("refined_ideas") or []),
                "proposals": normalized_proposals,
                "implementation_plans": normalized_implementation_plans,
                "constraints": list(plan.get("constraints") or []),
                "exclusions": list(plan.get("exclusions") or []),
                "success_criteria": list(plan.get("success_criteria") or []),
                "unresolved_questions": list(plan.get("unresolved_questions") or []),
            }
            normalization_errors = proposal_errors + implementation_errors
            if normalization_errors:
                state["errors"] = list(state.get("errors") or []) + normalization_errors
            state["execution_handoff"] = build_execution_handoff(state, self.brainstorm_cfg)
            state["last_summary"] = render_consensus_summary(state)

    @staticmethod
    def _render_turn(turn: dict[str, Any]) -> str:
        return f"[{turn.get('role_name')}] {turn.get('content', '').strip()}\n"

    @staticmethod
    def _render_plan(state: BrainstormState) -> str:
        plan = dict(state.get("plan_draft") or {})
        sections: list[str] = ["[Plan]"]

        direction = str(plan.get("research_direction") or state.get("current_goal") or "").strip()
        if direction:
            sections.append(f"Direction: {direction}")

        sections.append(_render_plan_list("Refined Ideas", plan.get("refined_ideas") or []))
        sections.append(_render_plan_list("Proposals", plan.get("proposals") or []))
        sections.append(_render_plan_list("Implementation Plans", plan.get("implementation_plans") or []))
        sections.append(_render_plan_list("Constraints", plan.get("constraints") or []))
        sections.append(_render_plan_list("Exclusions", plan.get("exclusions") or []))
        sections.append(_render_plan_list("Success Criteria", plan.get("success_criteria") or []))
        sections.append(_render_plan_list("Unresolved Questions", plan.get("unresolved_questions") or []))

        return "\n".join(item for item in sections if item.strip())

    @staticmethod
    def _reset_deliberation_state(state: BrainstormState, *, preserve_seed_evidence: bool) -> None:
        seed_evidence = [dict(item) for item in (state.get("seed_context_artifacts") or []) if isinstance(item, dict)]
        state["consensus"] = _empty_consensus(evidence=seed_evidence if preserve_seed_evidence else [])
        state["plan_draft"] = _empty_plan_draft(str(state.get("current_goal") or ""))
        state["pending_questions"] = []
        state["pending_decisions"] = []
        state["execution_handoff"] = {}
        state["approved_plan"] = False
        state["last_summary"] = render_consensus_summary(state)
        state["progress"]["round_index"] = 0
        state["turn_log"] = [
            dict(item)
            for item in (state.get("turn_log") or [])
            if str(item.get("message_type") or "") == "seed_import"
        ]

    def _role_prompt_config(self, state: BrainstormState, role: dict[str, Any]) -> dict[str, Any]:
        base_cfg = dict((state.get("prompt_config") or {}).get("role_turn") or {})
        overrides = dict(role.get("prompt_overrides") or {})
        if overrides:
            base_cfg.update(overrides)
        return base_cfg

    @staticmethod
    def _render_template(template: str, context: dict[str, Any]) -> str:
        class _SafeDict(dict):
            def __missing__(self, key: str) -> str:
                return "{" + key + "}"

        return str(template or "").format_map(_SafeDict({key: "" if value is None else value for key, value in context.items()}))


def persist_brainstorm_session(profile: dict[str, Any], brainstorm_cfg: dict[str, Any], state: BrainstormState) -> dict[str, Any]:
    session_id = str(state.get("session_id") or "")
    summary = state.get("last_summary") or render_consensus_summary(state)
    session_record = build_memory_record(
        profile=profile,
        object_type="brainstorm_session",
        payload={
            "record_id": f"brainstorm_session:{session_id}",
            "session_id": session_id,
            "brainstorm_config_name": str(brainstorm_cfg.get("name") or ""),
            "brainstorm_config_path": str(brainstorm_cfg.get("path") or ""),
            "state": dict(state),
        },
        node="brainstorm",
        kind="brainstorm_session",
        object_key=session_id,
        object_role="summary",
        title=f"Brainstorm session {session_id}",
        summary=summary[:1200],
        metadata={
            "profile": str(profile.get("name") or ""),
            "session_id": session_id,
            "status": str(state.get("status") or ""),
            "brainstorm_config_name": str(brainstorm_cfg.get("name") or ""),
            "source_experiment_record_id": str(state.get("source_experiment_record_id") or ""),
            "root_run_family_id": str(state.get("root_run_family_id") or ""),
        },
        tags=[str(profile.get("name") or ""), "brainstorm_session"],
    )
    service = MemoryService.for_profile(profile)
    service.persist_record(session_record)
    if state.get("plan_draft"):
        plan_record = build_memory_record(
            profile=profile,
            object_type="brainstorm_plan",
            payload={
                "record_id": f"brainstorm_plan:{session_id}",
                "session_id": session_id,
                "plan_draft": dict(state.get("plan_draft") or {}),
            },
            node="brainstorm",
            kind="brainstorm_plan",
            object_key=session_id,
            object_role="recommendation",
            title=f"Brainstorm plan {session_id}",
            summary=json.dumps(state.get("plan_draft") or {}, default=str)[:1200],
            metadata={
                "profile": str(profile.get("name") or ""),
                "session_id": session_id,
                "approved": bool(state.get("approved_plan")),
            },
            tags=[str(profile.get("name") or ""), "brainstorm_plan"],
            source_record_ids=[str(session_record.get("record_id") or "")],
        )
        service.persist_record(plan_record)
    return session_record


def load_brainstorm_session(profile: dict[str, Any], session_id: str) -> BrainstormState:
    service = MemoryService.for_profile(profile)
    record = service.find_one_record({"object_type": "brainstorm_session", "metadata.session_id": session_id})
    if not record:
        raise KeyError(f"Brainstorm session not found: {session_id}")
    content = dict(record.get("content") or {})
    state = dict(content.get("state") or {})
    return state


def execute_brainstorm_handoff(
    state: BrainstormState,
    brainstorm_cfg: dict[str, Any],
    *,
    build_initial_state_fn: Callable[..., dict[str, Any]],
    run_pipeline_graph_fn: Callable[..., dict[str, Any]],
    profile_name: str,
    profile: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    handoff = build_execution_handoff(state, brainstorm_cfg)
    start_node = choose_start_node(handoff, brainstorm_cfg)
    seed = {
        "research_direction": str(handoff.get("research_direction") or state.get("current_goal") or ""),
        "proposal_seed_planning_notes": str(handoff.get("proposal_seed_planning_notes") or ""),
        "source_experiment_record_id": str(handoff.get("source_experiment_record_id") or ""),
        "source_next_step_record_id": str(handoff.get("source_next_step_record_id") or ""),
        "source_next_step_title": str(handoff.get("source_next_step_title") or ""),
        "source_proposal_seed_record_id": str(handoff.get("source_proposal_seed_record_id") or ""),
        "source_proposal_seed_title": str(handoff.get("source_proposal_seed_title") or ""),
        "root_run_family_id": str(handoff.get("root_run_family_id") or ""),
        "root_research_direction": str(handoff.get("root_research_direction") or ""),
        "campaign_id": str(handoff.get("campaign_id") or ""),
        "campaign_title": str(handoff.get("campaign_title") or ""),
        "campaign_variant_id": str(handoff.get("campaign_variant_id") or ""),
        "campaign_variant_title": str(handoff.get("campaign_variant_title") or ""),
        "campaign_variant_index": int(handoff.get("campaign_variant_index") or 0),
        "campaign_size": int(handoff.get("campaign_size") or 0),
    }
    extra_state = {
        "refined_ideas": list(handoff.get("refined_ideas") or []),
        "proposals": list(handoff.get("proposals") or []),
        "implementation_plans": list(handoff.get("implementation_plans") or []),
    }
    initial_state = build_initial_state_fn(
        profile_name,
        str(seed["research_direction"]),
        seed,
        continue_loop=False,
        extra_state=extra_state,
    )
    result = run_pipeline_graph_fn(
        profile_name,
        profile,
        initial_state=initial_state,
        start_node=start_node,
        print_results=True,
    )
    return start_node, result


def _load_callable(dotted_path: str) -> Callable[..., Any] | None:
    module_name, _, attr_name = str(dotted_path or "").rpartition(".")
    if not module_name or not attr_name:
        return None
    module = importlib.import_module(module_name)
    candidate = getattr(module, attr_name, None)
    return candidate if callable(candidate) else None


def _empty_consensus(*, evidence: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "agreed_points": [],
        "active_options": [],
        "rejected_options": [],
        "objections": [],
        "assumptions": [],
        "open_questions": [],
        "next_recommendation": "",
        "confidence": "low",
        "evidence": list(evidence or []),
    }


def _empty_plan_draft(direction: str) -> dict[str, Any]:
    return {
        "research_direction": str(direction or "").strip(),
        "refined_ideas": [],
        "proposals": [],
        "implementation_plans": [],
        "constraints": [],
        "exclusions": [],
        "success_criteria": [],
        "unresolved_questions": [],
    }


def _normalize_brainstorm_proposals(raw_proposals: list[Any]) -> tuple[list[dict[str, Any]], list[str]]:
    normalized: list[dict[str, Any]] = []
    errors: list[str] = []

    for idx, item in enumerate(raw_proposals):
        if isinstance(item, dict):
            proposal = dict(item)
            if not proposal.get("name"):
                proposal["name"] = f"proposal_{idx + 1}"
            normalized.append(proposal)
            continue

        if isinstance(item, str) and item.strip():
            parsed = _coerce_plan_string(item)
            if isinstance(parsed, dict):
                proposal = dict(parsed)
                if not proposal.get("name"):
                    proposal["name"] = f"proposal_{idx + 1}"
                normalized.append(proposal)
                continue
            normalized.append({
                "name": f"proposal_{idx + 1}",
                "description": item.strip(),
            })
            errors.append(
                f"brainstorm_plan: coerced string proposal entry at index {idx} into a description-only proposal"
            )
            continue

        errors.append(
            f"brainstorm_plan: dropped malformed proposal entry at index {idx}: expected object, got {type(item).__name__}"
        )

    return normalized, errors


def _render_plan_list(title: str, items: list[Any]) -> str:
    if not items:
        return ""
    lines = [f"{title}:"]
    for item in items[:5]:
        lines.append(f"  - {_plan_item_text(item)}")
    return "\n".join(lines)


def _plan_item_text(item: Any) -> str:
    if isinstance(item, dict):
        for key in ("title", "name", "proposal_name", "class_name", "description", "summary", "rationale"):
            value = str(item.get(key) or "").strip()
            if value:
                extra = ""
                if key in {"title", "name", "proposal_name", "class_name"}:
                    detail = str(item.get("description") or item.get("rationale") or item.get("summary") or "").strip()
                    if detail:
                        extra = f": {detail}"
                return value + extra
        return json.dumps(item, default=str)[:160]
    return str(item).strip()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
