"""Bounded authoring steps, callable directly or from QC's job executor."""
from copy import deepcopy
import json
import os
from pathlib import Path
from uuid import uuid4
import yaml
from pydantic import ValidationError
from langchain_core.messages import HumanMessage, SystemMessage

from core.llm.factory import get_llm
from core.llm.usage import get_usage_report
from core.platform.identity import current_user
from core.platform.models import ModelOutput, Proposal, TurnRequest, TurnResult, content_hash
from core.platform.research import collect_public_research
from core.platform.validation import validate_component
from core.platform.prompt_contract import instructions, VERSION as CONTRACT_VERSION

PROFILE_PATH = Path(__file__).resolve().parents[2] / "configs" / "platform_profiles" / "quant_crucible.yaml"


class InvalidModelOutput(ValueError):
    def __init__(self, message, raw_output=None):
        super().__init__(message)
        self.raw_output = raw_output if isinstance(raw_output, str) else json.dumps(raw_output, ensure_ascii=False)


class PlatformWorkflow:
    def __init__(self, *, profile=None, model_factory=get_llm, researcher=collect_public_research):
        path = Path(os.environ.get("RESEARCHER_PROFILE_PATH", str(PROFILE_PATH)))
        self.profile = deepcopy(profile) if profile is not None else yaml.safe_load(path.read_text(encoding="utf-8"))
        expected = {"chat": ["chat"], "research": ["research"], "ideate": ["ideate"],
                    "code": ["code"], "build": ["code", "validate"], "validate": ["validate"], "workflow": ["research", "ideate", "code", "validate"]}
        if self.profile.get("operations") != expected:
            raise ValueError("Platform profiles must preserve the bounded operation boundaries")
        for env, key in (("RESEARCHER_LLM_PROVIDER", "provider"), ("RESEARCHER_LLM_MODEL", "default_model")):
            if os.environ.get(env):
                self.profile["llm"][key] = os.environ[env]
        self.model_factory = model_factory
        self.researcher = researcher

    async def run(self, request: TurnRequest, progress=None) -> TurnResult:
        evidence = list(request.evidence)
        outputs, ideas, proposal, validation = {}, [], None, None
        def report(stage, message):
            if progress:
                progress(stage, message)
        for step in self.profile["operations"][request.action]:
            if step == "validate" and request.action == "build" and outputs["code"].sourceCode is None:
                report(step, "No source generated; awaiting clarification before validation")
                continue
            report(step, {"chat": "Preparing a reply", "research": "Starting research",
                "ideate": "Developing testable ideas", "code": "Authoring component source",
                "validate": "Running static Python and component contract checks"}[step])
            if step == "validate":
                validation = validate_component(request.tenantId, request.component,
                    source=proposal.changes[0].after if proposal else None)
                report(step, f"Static checks {validation.status}; runtime checks were not run")
                continue
            if step == "research" and request.researchQuery:
                report(step, "Searching arXiv and retrieving article abstracts")
                collected = await self.researcher(request.researchQuery)
                report(step, f"Retrieved {len(collected)} article abstracts")
                for article in collected:
                    report(step, f"Retrieved article: {article.title}")
                evidence = list({item.id: item for item in [*evidence, *collected]}.values())
            if step == "research":
                report(step, f"Synthesizing {len(evidence)} evidence items" if evidence else "No evidence supplied; identifying research gaps")
            context = {
                "operation": step, "message": request.message,
                "conversation": [item.model_dump() for item in request.messages],
                "component": request.component.model_dump() if request.component else None,
                "constraints": request.constraints, "evidence": [item.model_dump() for item in evidence],
                "previousSteps": {key: value.model_dump() for key, value in outputs.items()},
            }
            kind = request.component.kind if request.component else None
            interface = self.profile["interfaces"].get(kind, "") if step == "code" else ""
            messages = [SystemMessage(content="\n".join([self.profile["system"], self.profile["prompts"][step], interface,
                        instructions(step, kind, [item.id for item in evidence])])),
                        HumanMessage(content=json.dumps(context))]
            response = await self.model_factory(step, self.profile).ainvoke(messages)
            report(step, "Model response received; checking response format and citations")
            output = self._parse(response.content)
            cited = set(output.evidenceIds) | {key for idea in output.ideas for key in idea.evidenceIds}
            if cited - {item.id for item in evidence}:
                raise InvalidModelOutput("Model cited evidence outside the supplied context", response.content)
            if step != "code" and output.sourceCode is not None:
                raise InvalidModelOutput("Only the code operation may propose source changes", response.content)
            outputs[step] = output
            if output.ideas:
                ideas = output.ideas
            if step == "code" and output.sourceCode is not None:
                if not output.sourceCode.strip():
                    raise InvalidModelOutput("Generated source is empty", response.content)
                component = request.component
                if output.sourceCode != component.sourceCode:
                    proposal = Proposal(tenantId=request.tenantId, userId=current_user(), proposalId=str(uuid4()), sessionId=request.sessionId,
                        componentId=component.componentId, draftId=component.draftId, baseVersionId=component.baseVersionId,
                        baseDraftRevision=component.revision, baseContentHash=component.contentHash,
                        contentHash=content_hash(output.sourceCode), explanation=output.content,
                        changes=[{"file": component.kind + ".py", "before": component.sourceCode, "after": output.sourceCode}])
            report(step, f"{step.capitalize()} complete")
        content = "\n\n".join(item.content for item in outputs.values())
        if not content and validation:
            content = f"Static checks {validation.status}; runtime validation is still required."
        return TurnResult(tenantId=request.tenantId, userId=current_user(), requestId=request.requestId, sessionId=request.sessionId,
            content=content, ideas=ideas, evidence=evidence,
            evidenceIds=sorted({key for output in outputs.values() for key in output.evidenceIds}),
            proposal=proposal, validation=validation, usage=get_usage_report(),
            provenance={"profile": self.profile["name"], "profileVersion": self.profile["version"], "contractVersion": CONTRACT_VERSION,
                "profileHash": content_hash(json.dumps(self.profile, sort_keys=True)),
                "serviceRelease": os.environ.get("RESEARCHER_RELEASE", "development"),
                "steps": list(self.profile["operations"][request.action]),
                "externalResearch": bool(request.researchQuery)})

    @staticmethod
    def _parse(content) -> ModelOutput:
        raw_output = content
        if isinstance(content, list):
            content = "".join(block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text")
        if not isinstance(content, str):
            raise InvalidModelOutput("Model did not return text", raw_output)
        if content.startswith("```json\n") and content.rstrip().endswith("```"):
            content = content.removeprefix("```json\n").rstrip()[:-3]
        try:
            return ModelOutput.model_validate_json(content)
        except ValidationError as exc:
            details = []
            for error in exc.errors(include_input=False, include_url=False)[:8]:
                path = ".".join(map(str, error["loc"])) or "response"
                message = "Return one complete JSON object with no surrounding prose" if error["type"] == "json_invalid" else error["msg"]
                details.append(f"{path}: {message}")
            raise InvalidModelOutput("Model returned an invalid authoring response: " + "; ".join(details), raw_output) from exc
