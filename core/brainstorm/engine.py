from __future__ import annotations

from datetime import datetime, timezone
import importlib
import json
from typing import Any, Callable
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from core.brainstorm.commands import parse_brainstorm_command
from core.brainstorm.handoff import build_execution_handoff, choose_start_node
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
) -> BrainstormState:
    return {
        "session_id": session_id or str(uuid4()),
        "profile_name": profile_name,
        "brainstorm_config_path": str(brainstorm_cfg.get("path") or ""),
        "brainstorm_config_name": str(brainstorm_cfg.get("name") or ""),
        "status": "running",
        "current_goal": str(direction or "").strip(),
        "user_intent_notes": [],
        "role_configs": [dict(item) for item in (brainstorm_cfg.get("roles") or []) if item.get("enabled", True)],
        "turn_log": [],
        "consensus": {
            "agreed_points": [],
            "active_options": [],
            "rejected_options": [],
            "objections": [],
            "assumptions": [],
            "open_questions": [],
            "next_recommendation": "",
            "confidence": "low",
            "evidence": [],
        },
        "plan_draft": {
            "research_direction": str(direction or "").strip(),
            "refined_ideas": [],
            "proposals": [],
            "implementation_plans": [],
            "constraints": [],
            "exclusions": [],
            "success_criteria": [],
            "unresolved_questions": [],
        },
        "pending_questions": [],
        "pending_decisions": [],
        "stop_policy": dict(brainstorm_cfg.get("stop_policy") or {}),
        "summary_config": dict(brainstorm_cfg.get("summary") or {}),
        "progress": {
            "round_index": 0,
            "message_count": 0,
        },
        "execution_handoff": {},
        "approved_plan": False,
        "last_summary": "",
        "errors": [],
    }


class BrainstormEngine:
    def __init__(self, profile: dict[str, Any], brainstorm_cfg: dict[str, Any]) -> None:
        self.profile = profile
        self.brainstorm_cfg = brainstorm_cfg

    def run_until_pause(
        self,
        state: BrainstormState,
        *,
        emit: Callable[[str], None] | None = None,
    ) -> BrainstormState:
        emitter = emit or (lambda _text: None)
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
                    turn = self._run_role_turn(state, role, round_index)
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
            return state
        if cmd_type == "request_research":
            state["progress"]["round_index"] = 0
            return self.run_until_pause(state, emit=emit)
        if cmd_type == "draft_plan":
            self._draft_plan(state)
            summary = state.get("last_summary") or render_consensus_summary(state)
            emitter(summary + "\n")
            return state
        if cmd_type == "approve_plan":
            if state.get("plan_draft"):
                state["approved_plan"] = True
                state["status"] = "approved_for_execution"
            return state
        if cmd_type == "start_execution":
            state["approved_plan"] = True
            state["status"] = "approved_for_execution"
            state["execution_handoff"] = build_execution_handoff(state, self.brainstorm_cfg)
            return state
        if cmd_type == "exit":
            state["status"] = "cancelled"
            return state
        return self.run_until_pause(state, emit=emit)

    def _run_role_turn(self, state: BrainstormState, role: dict[str, Any], round_index: int) -> dict[str, Any]:
        role_type = str(role.get("persona_type") or role.get("name") or "persona")
        if role_type == "researcher":
            return self._run_researcher_turn(state, role, round_index)

        system_prompt = (
            f"You are the {role.get('name')} persona in a brainstorming session.\n"
            f"Role type: {role_type}\n"
            f"Goal: {role.get('goal', '')}\n"
            f"Style: {role.get('style', '')}\n"
            "Produce concise but concrete reasoning. Mention weak assumptions."
        )
        human_prompt = (
            f"Current goal: {state.get('current_goal', '')}\n\n"
            f"Current consensus summary:\n{render_consensus_summary(state)}\n\n"
            f"Recent user notes:\n{json.dumps(state.get('user_intent_notes') or [], indent=2)}"
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
        for tool_path in list(role.get("tools") or []):
            tool = _load_callable(tool_path)
            if tool is None:
                continue
            tool_cfg = {"name": role.get("name", "researcher")}
            try:
                result = tool(str(state.get("current_goal") or ""), self.profile, tool_cfg, state)
            except Exception as exc:
                result = [{
                    "artifact_id": f"research_error:{tool_path}",
                    "source": str(role.get("name") or "researcher"),
                    "source_type": "error",
                    "title": f"Research tool failed: {tool_path}",
                    "summary": str(exc),
                    "metadata": {"tool": tool_path},
                    "raw": {},
                }]
            artifacts.extend(list(result or []))
        capped = artifacts[: int(((role.get("research_budget") or {}).get("max_artifacts_per_tool") or 4) * max(len(role.get("tools") or []), 1))]
        state["consensus"]["evidence"] = list(state.get("consensus", {}).get("evidence") or []) + capped[:5]
        summary_lines = []
        for item in capped[:5]:
            summary_lines.append(f"- {item.get('title', '')}: {str(item.get('summary', ''))[:180]}")
        content = "\n".join(summary_lines) if summary_lines else "No evidence gathered."
        return {
            "turn_id": str(uuid4()),
            "round_index": round_index,
            "role_name": str(role.get("name") or "researcher"),
            "role_type": "researcher",
            "message_type": "research",
            "content": content,
            "structured_points": capped[:5],
            "citations": [{"title": item.get("title", ""), "url": item.get("url", "")} for item in capped[:5]],
            "created_at": _now_iso(),
        }

    def _refresh_consensus(self, state: BrainstormState) -> None:
        recent_turns = list(state.get("turn_log") or [])[-6:]
        if not recent_turns:
            return
        system_prompt = (
            "Summarize the brainstorming session into JSON with keys: "
            "agreed_points, active_options, rejected_options, objections, assumptions, "
            "open_questions, next_recommendation, confidence. "
            "Use arrays for lists and keep values concise."
        )
        human_prompt = json.dumps(
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
        system_prompt = (
            "Generate a structured brainstorming plan as JSON with keys: "
            "research_direction, refined_ideas, proposals, implementation_plans, "
            "constraints, exclusions, success_criteria, unresolved_questions."
        )
        human_prompt = json.dumps(
            {
                "goal": state.get("current_goal", ""),
                "consensus": state.get("consensus") or {},
                "user_notes": state.get("user_intent_notes") or [],
            },
            indent=2,
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
            state["plan_draft"] = {
                "research_direction": str(plan.get("research_direction") or state.get("current_goal") or ""),
                "refined_ideas": list(plan.get("refined_ideas") or []),
                "proposals": list(plan.get("proposals") or []),
                "implementation_plans": list(plan.get("implementation_plans") or []),
                "constraints": list(plan.get("constraints") or []),
                "exclusions": list(plan.get("exclusions") or []),
                "success_criteria": list(plan.get("success_criteria") or []),
                "unresolved_questions": list(plan.get("unresolved_questions") or []),
            }
            state["execution_handoff"] = build_execution_handoff(state, self.brainstorm_cfg)
            state["last_summary"] = render_consensus_summary(state)

    @staticmethod
    def _render_turn(turn: dict[str, Any]) -> str:
        return f"[{turn.get('role_name')}] {turn.get('content', '').strip()}\n"


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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
