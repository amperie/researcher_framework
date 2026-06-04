from __future__ import annotations

from typing import Any

HELP_TEXT = """Available commands:
  help           Show this command list
  continue       Resume the brainstorm session
  summary        Show the latest checkpoint summary
  research       Ask the researcher role to gather more evidence
  research <q>   Ask the researcher to search for a specific topic or question
  plan           Draft or refresh the current plan
  edit_plan      Open the current plan in $EDITOR for direct editing
  feedback ...   Add feedback, constraints, or direction
  approve_plan   Approve the current plan draft without executing it
  execute        Start downstream execution from the approved plan
  exit           Exit brainstorm mode
"""


def parse_brainstorm_command(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {"type": "continue"}
    head, _, tail = raw.partition(" ")
    command = head.strip().lower()
    rest = tail.strip()

    if command == "help":
        return {"type": "help"}
    if command == "continue":
        return {"type": "continue"}
    if command == "summary":
        return {"type": "summary"}
    if command == "research":
        return {"type": "request_research", "query": rest}
    if command == "plan":
        return {"type": "draft_plan"}
    if command == "edit_plan":
        return {"type": "edit_plan"}
    if command == "approve_plan":
        return {"type": "approve_plan"}
    if command == "execute":
        return {"type": "start_execution"}
    if command == "exit":
        return {"type": "exit"}
    if command == "feedback":
        return {"type": "feedback", "text": rest}
    return {"type": "feedback", "text": raw}
