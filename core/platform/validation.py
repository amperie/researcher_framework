"""Read-only static checks. Never import, execute, or repair submitted source."""
import ast
from datetime import datetime, timezone
from uuid import uuid4
from core.platform.identity import current_user
from core.platform.models import ComponentContext, Validation, content_hash

COMPONENT_INTERFACES = {
    "algorithm": ("Algorithm", "on_data_logic", ("self", "data"), "trading.core.algorithm"),
    "portfolio": ("Portfolio", "process_tick_market_signals_logic", ("self", "signals", "tick"), "trading.core.portfolio"),
}


def validate_component(tenant_id: str, component: ComponentContext, *, source: str | None = None) -> Validation:
    code = component.sourceCode if source is None else source
    checks, diagnostics = [], []
    def check(name, passed, message):
        checks.append({"id": name, "label": name.replace("_", " ").capitalize(), "status": "passed" if passed else "failed"})
        if not passed:
            diagnostics.append({"code": name, "severity": "error", "message": message})
    try:
        tree = ast.parse(code)
        compile(tree, "component.py", "exec")
        check("syntax", bool(code.strip()), "Source must not be empty")
    except (SyntaxError, ValueError, RecursionError) as exc:
        tree = ast.Module(body=[], type_ignores=[])
        check("syntax", False, f"Invalid Python source: {exc}")
    base, method, args, module = COMPONENT_INTERFACES[component.kind]
    arg_count = len(args)
    imports = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == module:
            imports.update({alias.asname or alias.name: alias.name for alias in node.names})
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and any(
        isinstance(parent, ast.Name) and imports.get(parent.id) == base for parent in node.bases)]
    check("base_class", len(classes) == 1, f"Define exactly one subclass imported from {module}.{base}")
    if len(classes) == 1:
        cls = classes[0]
        methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method]
        valid = len(methods) == 1 and len(methods[0].args.posonlyargs + methods[0].args.args) == arg_count
        if valid:
            args = methods[0].args
            valid = not methods[0].decorator_list and not args.vararg and not args.kwarg and all(v is not None for v in args.kw_defaults)
        check("method_signature", valid, f"Implement instance method {method} with {arg_count} positional arguments including self")
        metadata = None
        for node in cls.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "crucible_metadata" for t in node.targets):
                try:
                    metadata = ast.literal_eval(node.value)
                except (ValueError, TypeError, SyntaxError, RecursionError):
                    pass
        check("metadata", isinstance(metadata, dict) and metadata.get("role") == component.kind,
              f"crucible_metadata must be a literal dictionary with role={component.kind!r}")
    checks.append({"id": "runtime", "label": "Isolated runtime contract checks", "status": "skipped"})
    diagnostics.append({"code": "runtime_not_checked", "severity": "warning",
        "message": "Static checks do not execute source or establish runtime compatibility, safety, or profitability.",
        "actionable": "Run QC's isolated runtime validator before publishing a component version."})
    return Validation(tenantId=tenant_id, userId=current_user(), validationId=str(uuid4()), componentId=component.componentId,
        draftId=component.draftId, revision=component.revision, contentHash=content_hash(code),
        target="draft" if source is None else "proposal", checks=checks, diagnostics=diagnostics,
        status="failed" if any(c["status"] == "failed" for c in checks) else "passed",
        performedAt=datetime.now(timezone.utc).isoformat())
