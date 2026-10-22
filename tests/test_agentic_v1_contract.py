from copy import deepcopy
from typing import get_args
import ast

import pytest
from pydantic import ValidationError

from app.modules.agentic.constants import ToolType, MAX_CONTRACT_NAME, MAX_ROLE
from app.modules.agentic.schemas import ArchitecturePlan, ApiEndpoint
from app.modules.agentic.generation_schemas import GeneratedApiEndpoint, GeneratedToolContract
from app.modules.agentic.generator import deterministic_scaffold, deterministic_support_files, validate_generation_compatibility
from app.modules.agentic.service import AgenticError
from test_agentic_p1 import architecture, make_service
from test_agentic_staged_generation import generator, staged_outputs


@pytest.mark.parametrize("kind", get_args(ToolType))
def test_every_shared_tool_type_is_consumable(kind):
    data = architecture().model_dump(mode="json")
    data["tools"][0]["type"] = kind
    plan = ArchitecturePlan.model_validate(data)
    scaffold = validate_generation_compatibility(plan)
    assert scaffold.manifest.tools[0].type == kind
    assert GeneratedToolContract.model_fields["type"].annotation == ToolType


def test_legacy_alias_and_framework_normalize_before_generation():
    data = architecture().model_dump(mode="json")
    data["tools"][0]["type"] = "retrieval"
    data["frontend"]["type"] = "Web app"
    del data["backend"]["api_endpoints"][0]["request_schema"]
    del data["backend"]["api_endpoints"][0]["response_schema"]
    plan = ArchitecturePlan.model_validate(data)
    assert plan.tools[0].type == "knowledge_retrieval"
    assert plan.frontend.type == "Next.js"
    assert deterministic_scaffold(plan).api_endpoints == plan.backend.api_endpoints


@pytest.mark.parametrize("path", ["https://host/api", "api/run", "/../run", "/run?x=1", "/a//b", "/{x}/{x}", "/bad path"])
def test_invalid_api_paths_rejected_at_p1(path):
    with pytest.raises(ValidationError):
        ApiEndpoint(method="POST", path=path, purpose="Run")


@pytest.mark.parametrize("method", ["HEAD", "OPTIONS", "post"])
def test_unsupported_methods_rejected(method):
    with pytest.raises(ValidationError):
        ApiEndpoint(method=method, path="/run", purpose="Run")


def test_p1_p2_endpoint_contract_is_identical():
    assert GeneratedApiEndpoint is ApiEndpoint
    data = architecture().model_dump(mode="json")
    data["backend"]["api_endpoints"][0].update(
        request_schema={"question": "string", "count": "integer"},
        response_schema={"result": "object"})
    plan = ArchitecturePlan.model_validate(data)
    assert deterministic_scaffold(plan).api_endpoints == plan.backend.api_endpoints


def test_duplicate_api_pairs_rejected_at_p1():
    data = architecture().model_dump(mode="json")
    data["backend"]["api_endpoints"].append(deepcopy(data["backend"]["api_endpoints"][0]))
    with pytest.raises(ValidationError, match="Duplicate API"):
        ArchitecturePlan.model_validate(data)


@pytest.mark.parametrize("field,value", [
    ("tool", "browser"), ("frontend", "Vue"), ("backend", "Flask"),
    ("name", "x" * (MAX_CONTRACT_NAME + 1)), ("role", "x" * (MAX_ROLE + 1))])
def test_unsupported_values_and_shared_limits_rejected(field, value):
    data = architecture().model_dump(mode="json")
    if field == "tool": data["tools"][0]["type"] = value
    elif field == "frontend": data["frontend"]["type"] = value
    elif field == "backend": data["backend"]["framework"] = value
    else: data["agents"][0][field] = value
    with pytest.raises(ValidationError):
        ArchitecturePlan.model_validate(data)


@pytest.mark.asyncio
async def test_incompatible_saved_plan_cannot_be_approved():
    service, repository = make_service()
    project = await service.create_project("user-a", "Support", "Improve support outcomes.", [])
    plan = await service.generate_plan(project["id"], "user-a")
    repository.plans[plan["id"]]["plan"]["tools"][0]["type"] = "unsupported"
    with pytest.raises(AgenticError) as error:
        await service.approve(project["id"], "user-a")
    assert error.value.code == "AGENTIC_PLAN_INCOMPATIBLE"
    assert error.value.status_code == 422
    assert repository.plans[plan["id"]]["status"] == "generated"
    assert repository.projects[project["id"]]["status"] == "plan_ready"


def test_required_support_files_are_deterministic_and_smoke_tests_are_real():
    scaffold = validate_generation_compatibility(architecture())
    files = deterministic_support_files(scaffold).files
    assert {file.path for file in files} == {"frontend/tsconfig.json", "backend/tests/test_smoke.py", "README.md"}
    code = next(file.content for file in files if file.path.endswith("test_smoke.py"))
    ast.parse(code)
    assert "client.request" in code and "assert response.status_code" in code
    assert "assert True" not in code


@pytest.mark.asyncio
async def test_small_v1_support_files_do_not_use_optional_llm():
    outputs = staged_outputs()
    outputs["tests_docs"]["files"] = [{"path": "README.md", "content": "unsafe", "purpose": "Not used"}]
    app_generator, provider = generator(outputs)
    result = await app_generator.generate(project_name="Support", problem_statement="Support.",
                                          plan=architecture(), enhance_docs=True)
    assert len(provider.calls) == 3
    assert sum(file.path == "backend/tests/test_smoke.py" for file in result.files) == 1
    assert "client.request" in next(file.content for file in result.files if file.path.endswith("test_smoke.py"))
