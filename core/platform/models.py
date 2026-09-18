"""Versioned contracts. QC owns the authoritative session and draft."""
from hashlib import sha256
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

Identity = Annotated[str, Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9_.:-]+$")]
Text = Annotated[str, Field(min_length=1, max_length=24000)]
Source = Annotated[str, Field(max_length=150000)]


def content_hash(source: str) -> str:
    return sha256(source.encode("utf-8")).hexdigest()


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Message(Contract):
    role: Literal["user", "assistant"]
    content: Text


class Evidence(Contract):
    id: Identity
    title: Annotated[str, Field(min_length=1, max_length=1000)]
    text: Text
    url: Annotated[str, Field(max_length=2000)] | None = None


class ComponentContext(Contract):
    componentId: Identity
    kind: Literal["algorithm", "portfolio"]
    baseVersionId: Identity | None = None
    draftId: Identity
    revision: Annotated[int, Field(ge=0, strict=True)]
    sourceCode: Source
    contentHash: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    description: Annotated[str, Field(max_length=8000)] = ""
    interfaceContext: Annotated[str, Field(max_length=24000)] = ""
    parameterSchema: dict = Field(default_factory=dict)
    validationDiagnostics: list[dict] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def verify_hash(self):
        if content_hash(self.sourceCode) != self.contentHash:
            raise ValueError("contentHash does not match the exact sourceCode")
        return self


class TurnRequest(Contract):
    schemaVersion: Literal["1"] = "1"
    tenantId: Identity
    requestId: Identity
    sessionId: Identity
    action: Literal["chat", "research", "ideate", "code", "build", "validate", "workflow"] = "chat"
    message: Text
    messages: list[Message] = Field(default_factory=list, max_length=40)
    component: ComponentContext | None = None
    constraints: list[Text] = Field(default_factory=list, max_length=20)
    evidence: list[Evidence] = Field(default_factory=list, max_length=20)
    researchQuery: Annotated[str, Field(min_length=1, max_length=300)] | None = None

    @model_validator(mode="after")
    def check_context(self):
        if self.action in {"code", "build", "validate", "workflow"} and self.component is None:
            raise ValueError("This action requires an exact component draft")
        if self.researchQuery and self.action not in {"research", "workflow"}:
            raise ValueError("External research must be explicitly requested")
        if len({item.id for item in self.evidence}) != len(self.evidence):
            raise ValueError("Evidence IDs must be unique")
        if len(self.model_dump_json()) > 300000:
            raise ValueError("Context exceeds 300000 characters; summarize older messages")
        return self


class Idea(Contract):
    name: Annotated[str, Field(min_length=1, max_length=200)]
    hypothesis: Text
    rationale: Text
    risks: list[Text] = Field(default_factory=list, max_length=10)
    evidenceIds: list[Identity] = Field(default_factory=list, max_length=20)


class ModelOutput(Contract):
    content: Text
    sourceCode: Source | None = None
    ideas: list[Idea] = Field(default_factory=list, max_length=8)
    evidenceIds: list[Identity] = Field(default_factory=list, max_length=20)


class Change(Contract):
    file: str
    before: str
    after: str


class Proposal(Contract):
    tenantId: Identity
    proposalId: Identity
    sessionId: Identity
    componentId: Identity
    draftId: Identity
    baseVersionId: Identity | None
    baseDraftRevision: int
    baseContentHash: str
    contentHash: str
    explanation: str
    changes: list[Change]
    state: Literal["pending"] = "pending"


class Check(Contract):
    id: str
    label: str
    status: Literal["passed", "failed", "skipped"]


class Diagnostic(Contract):
    code: str
    severity: Literal["info", "warning", "error"]
    message: str
    path: str | None = None
    actionable: str | None = None


class Validation(Contract):
    tenantId: Identity
    validationId: Identity
    componentId: Identity
    draftId: Identity
    revision: int
    contentHash: str
    status: Literal["passed", "failed"]
    scope: Literal["static"] = "static"
    target: Literal["draft", "proposal"] = "draft"
    publishable: Literal[False] = False
    checks: list[Check]
    diagnostics: list[Diagnostic]
    performedAt: str


class UsageEvent(Contract):
    callId: str
    tenantId: Identity | None
    requestId: Identity | None
    sessionId: Identity | None
    step: str
    provider: str
    requestedModel: str
    reportedModel: str | None
    model: str
    status: Literal["started", "succeeded", "failed", "interrupted"]
    usageAvailable: bool
    promptTokens: int | None
    completionTokens: int | None
    totalTokens: int | None
    startedAt: str
    finishedAt: str | None


class UsageReport(Contract):
    provider: str | None
    model: str | None
    promptTokens: int
    completionTokens: int
    totalTokens: int
    calls: int
    steps: list[UsageEvent]


class TurnResult(Contract):
    schemaVersion: Literal["1"] = "1"
    tenantId: Identity
    requestId: Identity
    sessionId: Identity
    content: str
    ideas: list[Idea] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    evidenceIds: list[Identity] = Field(default_factory=list)
    proposal: Proposal | None = None
    validation: Validation | None = None
    usage: UsageReport
    provenance: dict


class OperationError(Contract):
    rawModelOutput: str | None = None
    code: str
    message: str
    retryable: bool


class ErrorEnvelope(Contract):
    tenantId: Identity | None = None
    error: OperationError
    usage: UsageReport | None = None
    diagnostics: list[dict] = Field(default_factory=list)


class ProgressEvent(Contract):
    sequence: int
    stage: str
    message: str
    at: str


class RequestReceipt(Contract):
    tenantId: Identity
    requestId: Identity
    sessionId: Identity
    status: Literal["running", "succeeded", "failed", "stopped", "interrupted"]
    stage: str
    progress: list[ProgressEvent] = Field(default_factory=list)
    usage: UsageReport | None = None
    startedAt: str
    expiresAt: str
    finishedAt: str | None
    result: TurnResult | None
    error: OperationError | None


class UsageTotals(Contract):
    calls: int
    unknownUsageCalls: int
    knownInputTokens: int
    knownOutputTokens: int


class ModelUsage(UsageTotals):
    provider: str
    model: str


class UsageSummary(Contract):
    tenantId: Identity
    models: list[ModelUsage]
    totals: UsageTotals


class UsagePage(Contract):
    tenantId: Identity
    items: list[UsageEvent]
    pagination: dict[str, int | bool]
