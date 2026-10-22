"""Controlled in-memory builder operations. These are not agent runtime tools."""
from __future__ import annotations

import ast
import json
import re

from app.modules.agentic.generation_schemas import GeneratedApplicationBundle, GeneratedSourceFile
from app.modules.agentic.generator import (
    GenerationOutputError, _validate_stage_files, _validate_backend_routes,
    _validate_api_source, deterministic_support_files, validate_generation_compatibility,
)
from app.modules.agentic.schemas import ArchitecturePlan

FILE_STAGES = {"backend/main.py": "backend", "agents/main.py": "agents", "frontend/app/page.tsx": "frontend"}
INTERFACE = "from agents.main import run_agent; synchronous run_agent(payload: dict) -> dict"


def sanitize_build_logs(logs: str) -> str:
    from app.modules.agentic.build.repository import _redact
    from app.core.config.settings import settings
    value = re.sub(r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----", "[REDACTED]", str(logs), flags=re.S)
    value = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", value)
    key = getattr(settings, "openrouter_api_key", None)
    if key:
        value = value.replace(str(key), "[REDACTED]")
    return _redact(value[-4000:])[-2000:]


def validate_file(path: str, content: str, endpoints) -> None:
    file = GeneratedSourceFile(path=path, content=content, purpose="Application source")
    _validate_stage_files("application", [file], set(), 0)
    if path.endswith(".py"):
        try:
            tree = ast.parse(content, filename=path)
        except SyntaxError as exc:
            raise GenerationOutputError(f"Invalid Python source: {path}") from exc
        if path == "backend/main.py":
            _validate_backend_routes([file], endpoints)
            imported = any(isinstance(node, ast.ImportFrom) and node.module == "agents.main"
                           and any(alias.name == "run_agent" and alias.asname in (None, "run_agent") for alias in node.names)
                           for node in ast.walk(tree))
            called = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                         and node.func.id == "run_agent" and len(node.args) == 1 for node in ast.walk(tree))
            if not imported or not called:
                raise GenerationOutputError("Backend must import and call run_agent(payload) from agents.main.")
        else:
            functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_agent"]
            if not functions or len(functions[0].args.args) != 1 or functions[0].args.vararg or functions[0].args.kwarg or functions[0].args.kwonlyargs:
                raise GenerationOutputError("Agent source must define synchronous run_agent(payload).")
    if path != "agents/main.py":
        _validate_api_source(path, [file], endpoints)
        for endpoint in endpoints:
            for field in {*endpoint.request_schema, *endpoint.response_schema}:
                if not re.search(r"(?<![A-Za-z0-9_])" + re.escape(field) + r"(?![A-Za-z0-9_])", content):
                    raise GenerationOutputError(f"Source omits contracted API field: {field}")
    if path.endswith(".tsx") and (not content.strip() or "export default" not in content or not re.search(r"<[A-Za-z]", content)):
        raise GenerationOutputError("Frontend must contain a non-empty default TSX page.")


class CodingWorkspace:
    """Only three fixed source paths are editable; no OS or shell capability is exposed."""
    def __init__(self, plan: ArchitecturePlan, bundle: GeneratedApplicationBundle):
        self.plan = plan.model_copy(deep=True)
        self.scaffold = validate_generation_compatibility(self.plan)
        self._bundle = bundle.model_copy(deep=True)
        self._files = {file.path: file.model_copy(deep=True) for file in bundle.files}
        if len(self._files) != len(bundle.files):
            raise GenerationOutputError("Workspace contains duplicate paths.")
        self._logs = ""

    @classmethod
    def create(cls, plan):
        from app.modules.agentic.version_service import _template_files
        scaffold = validate_generation_compatibility(plan)
        files = [*scaffold.files, *deterministic_support_files(scaffold).files]
        files.append(GeneratedSourceFile(path="agentic-api-contract.json", language="json", purpose="Approved API contract",
            content=json.dumps({"endpoints": [e.model_dump(mode="json") for e in scaffold.api_endpoints]}, indent=2) + "\n"))
        bundle = GeneratedApplicationBundle(application=scaffold.application, manifest=scaffold.manifest, files=files)
        bundle.files.extend(file for file in _template_files(bundle) if file.path not in {item.path for item in files})
        return cls(plan, bundle)

    def list_files(self):
        return sorted(self._files)

    def read_file(self, path):
        if path not in self._files:
            raise GenerationOutputError("Workspace file not found.")
        return self._files[path].content

    def _store(self, path, content, *, replace):
        if path not in FILE_STAGES or (path in self._files) != replace:
            raise GenerationOutputError("Builder write is outside its assigned scope or duplicates a file.")
        validate_file(path, content, self.scaffold.api_endpoints)
        file = GeneratedSourceFile(path=path, content=content, language="typescript" if path.endswith(".tsx") else "python", purpose="Approved application source")
        candidates = [item for key, item in self._files.items() if key != path] + [file]
        # Final path, count, byte, credential and manifest checks also apply after every write.
        from app.modules.agentic.version_service import validate_generated_files
        # During initial construction only entrypoints are intentionally absent.
        from app.modules.agentic.constants import MAX_GENERATED_TOTAL_BYTES, MAX_GENERATED_FILES
        if len(candidates) > MAX_GENERATED_FILES or sum(len(item.content.encode("utf-8")) for item in candidates) > MAX_GENERATED_TOTAL_BYTES:
            raise GenerationOutputError("Workspace exceeds source limits.")
        if set(FILE_STAGES).issubset({item.path for item in candidates}):
            validate_generated_files(self._bundle.model_copy(update={"files": candidates}))
        self._files[path] = file

    def write_file(self, path, content):
        self._store(path, content, replace=False)

    def replace_file(self, path, content):
        self._store(path, content, replace=True)

    def read_spec(self, path):
        if path not in FILE_STAGES:
            raise GenerationOutputError("Unknown builder source path.")
        contract = [endpoint.model_dump(mode="json") for endpoint in self.scaffold.api_endpoints]
        if path == "backend/main.py":
            return {"purpose": self.plan.application.summary, "api_contract": contract, "backend": self.plan.backend.model_dump(mode="json"), "agent_interface": INTERFACE}
        if path == "agents/main.py":
            return {"agents": [a.model_dump(mode="json") for a in self.plan.agents],
                    "tools": [t.model_dump(mode="json") for t in self.plan.tools],
                    "workflow": [step.model_dump(mode="json") for step in self.plan.workflow],
                    "data": self.plan.data.model_dump(mode="json"),
                    "integrations": [item.model_dump(mode="json") for item in self.plan.integrations],
                    "api_contract": contract, "agent_interface": INTERFACE}
        return {"api_contract": contract, "frontend": self.plan.frontend.model_dump(mode="json")}

    def set_build_logs(self, logs):
        self._logs = sanitize_build_logs(logs)

    def read_build_logs(self):
        return self._logs

    def bundle(self):
        if not set(FILE_STAGES).issubset(self._files):
            raise GenerationOutputError("Workspace application is incomplete.")
        return self._bundle.model_copy(update={"files": list(self._files.values())}, deep=True)


def affected_files(result: dict, logs: str) -> list[str]:
    """Repair source failures only; infrastructure/configuration failures fail truthfully."""
    if not any(result.get(key) == "failed" for key in ("frontend_build", "backend_validation", "backend_tests")):
        return []
    paths = [path for path in FILE_STAGES if path in logs]
    if paths:
        return paths
    if result.get("frontend_build") == "failed":
        return ["frontend/app/page.tsx"]
    if result.get("backend_validation") == "failed" or result.get("backend_tests") == "failed":
        return ["backend/main.py", "agents/main.py"]
    return []
