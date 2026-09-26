import json

import pytest

from app.modules.agentic.generator import ApplicationGenerator, GenerationOutputError
from app.modules.agentic.generation_schemas import GenerationScaffold
from app.modules.agentic.model_router import AgenticModelRouter
from app.modules.agentic.service import AgenticError
from app.modules.agentic.version_service import validate_generated_files
from test_agentic_openrouter import configuration
from test_agentic_p1 import architecture
from test_agentic_p2 import bundle, service


CONTRACT = [{
    "method": "POST", "path": "/requests", "purpose": "Submit request",
    "request_schema": {"request": "string"},
    "response_schema": {"answer": "string"},
}]


def source_file(path, content):
    return {"path": path, "language": None, "purpose": "Generated application", "content": content}


def staged_outputs():
    original = bundle()
    return {
        "scaffold": {
            "application": original.application.model_dump(mode="json"),
            "manifest": original.manifest.model_dump(mode="json"),
            "api_endpoints": CONTRACT,
            "files": [
                source_file("frontend/package.json", json.dumps({
                    "scripts": {"build": "next build"},
                    "dependencies": {"next": "16.3.1", "react": "19.2.4", "react-dom": "19.2.4"},
                })),
                source_file("backend/requirements.txt", "fastapi\nuvicorn\npytest\n"),
            ],
        },
        "backend": {
            "api_endpoints": CONTRACT,
            "files": [source_file("backend/main.py", "from fastapi import FastAPI\napp = FastAPI()\n@app.post('/requests')\ndef submit(request: dict):\n    return {'answer': 'ok'}\n")],
        },
        "agents": {"files": [source_file("agents/support.py", "def answer(request):\n    return 'ok'\n")]},
        "frontend": {
            "api_endpoints": CONTRACT,
            "files": [source_file("frontend/app/page.tsx", "export default function Page() { fetch('/requests'); return <main>Support</main>; }\n")],
        },
        "tests_docs": {"files": [
            source_file("backend/tests/test_smoke.py", "def test_smoke():\n    assert True\n"),
            source_file("frontend/tsconfig.json", '{"compilerOptions":{"jsx":"preserve"}}\n'),
            source_file("README.md", "# Support App\n"),
        ]},
    }


class StageProvider:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    async def complete(self, config, messages, reasoning, **kwargs):
        stage = json.loads(messages[1]["content"])["stage"]
        self.calls.append((stage, config, messages, kwargs))
        return json.dumps(self.outputs[stage])


def generator(outputs):
    provider = StageProvider(outputs)
    return ApplicationGenerator(provider, AgenticModelRouter(configuration())), provider


class ResponseProvider:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    async def complete(self, config, messages, reasoning, **kwargs):
        self.calls.append((config, messages, kwargs))
        return self.responses.pop(0)


def scaffold_generator(*responses):
    provider = ResponseProvider(*responses)
    return ApplicationGenerator(provider, AgenticModelRouter(configuration())), provider


@pytest.mark.asyncio
async def test_valid_scaffold_first_response_needs_one_call():
    value = json.dumps(staged_outputs()["scaffold"])
    app_generator, provider = scaffold_generator(value)
    result = await app_generator._stage("scaffold", {}, GenerationScaffold)
    assert result.application.name == "Support App"
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_fenced_scaffold_json_is_accepted_without_repair():
    value = json.dumps(staged_outputs()["scaffold"])
    app_generator, provider = scaffold_generator(f"```json\n{value}\n```")
    result = await app_generator._stage("scaffold", {}, GenerationScaffold)
    assert result.api_endpoints[0].path == "/requests"
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_invalid_scaffold_schema_gets_one_successful_repair():
    invalid = '{"files": []}'
    valid = json.dumps(staged_outputs()["scaffold"])
    app_generator, provider = scaffold_generator(invalid, valid)
    result = await app_generator._stage("scaffold", {}, GenerationScaffold)
    assert result.application.name == "Support App"
    assert len(provider.calls) == 2
    repair = json.loads(provider.calls[1][1][1]["content"])
    assert repair["stage"] == "scaffold"
    assert repair["previous_response"] == invalid
    assert repair["required_json_schema"] == GenerationScaffold.model_json_schema()
    assert repair["validation_errors"]
    assert "ONLY one corrected JSON object" in repair["instruction"]
    assert all(call[0].model == "example/coder:free" for call in provider.calls)


@pytest.mark.asyncio
async def test_invalid_scaffold_repair_fails_after_exactly_two_calls():
    app_generator, provider = scaffold_generator('{"files": []}', "not JSON")
    with pytest.raises(GenerationOutputError, match="scaffold returned malformed structured output"):
        await app_generator._stage("scaffold", {}, GenerationScaffold)
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_five_focused_calls_combine_into_one_valid_immutable_bundle():
    app_generator, provider = generator(staged_outputs())
    result = await app_generator.generate(
        project_name="Support", problem_statement="Improve support outcomes.", plan=architecture()
    )
    assert [call[0] for call in provider.calls] == [
        "scaffold", "backend", "agents", "frontend", "tests_docs"
    ]
    assert all(call[1].model == "example/coder:free" for call in provider.calls)
    assert all(call[1].max_output_tokens == 8192 for call in provider.calls)
    assert all(call[3]["response_format"] == {"type": "json_object"} for call in provider.calls)
    assert len(result.files) == 9
    validated = validate_generated_files(result)
    assert len(validated) == 11  # Contract, manifest, and .env.example are stored once.
    contract = next(item for item in validated if item["path"] == "agentic-api-contract.json")
    assert json.loads(contract["content"])["endpoints"] == CONTRACT
    assert all(len(item["sha256"]) == 64 for item in validated)
    assert "api_contract" in json.loads(provider.calls[3][2][1]["content"])


@pytest.mark.asyncio
async def test_staged_generation_commits_one_ready_version():
    app_generator, provider = generator(staged_outputs())
    version_service, repository = service(app_generator)
    version = await version_service.generate("approved", "user-a")
    assert version["status"] == "ready"
    assert version["version_number"] == 1
    assert len(repository.versions) == 1
    assert len(repository.files[version["id"]]) == 11
    assert repository.projects["approved"]["current_version_id"] == version["id"]
    assert len(provider.calls) == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("stage,change,expected_calls", [
    ("scaffold", lambda value: value["files"][0].update(content='{"scripts":{"build":"echo unsafe"}}'), 1),
    ("backend", lambda value: value["files"].append(source_file("backend/Main.py", "duplicate")), 2),
    ("backend", lambda value: value["files"].append(source_file("backend/../escape.py", "escape")), 2),
    ("backend", lambda value: value.update(api_endpoints=[]), 2),
    ("backend", lambda value: value["files"][0].update(content="from fastapi import FastAPI\napp = FastAPI()\n"), 2),
    ("backend", lambda value: value["files"][0].update(content="from fastapi import FastAPI\napp = FastAPI()\n# /requests\n"), 2),
    ("agents", lambda value: value["files"].append(source_file("agents/SUPPORT.py", "duplicate")), 3),
    ("frontend", lambda value: value.update(api_endpoints=[]), 4),
    ("frontend", lambda value: value["files"][0].update(content="export default function Page() { return <main>Support</main>; }"), 4),
    ("tests_docs", lambda value: value["files"].append(source_file("README.md", "duplicate")), 5),
])
async def test_invalid_stage_stops_before_later_calls(stage, change, expected_calls):
    outputs = staged_outputs()
    change(outputs[stage])
    app_generator, provider = generator(outputs)
    with pytest.raises(GenerationOutputError):
        await app_generator.generate(
            project_name="Support", problem_statement="Improve support outcomes.", plan=architecture()
        )
    assert len(provider.calls) == expected_calls


@pytest.mark.asyncio
async def test_failed_stage_never_creates_a_ready_version():
    outputs = staged_outputs()
    outputs["frontend"]["api_endpoints"] = []
    app_generator, provider = generator(outputs)
    version_service, repository = service(app_generator)
    with pytest.raises(AgenticError, match="frontend API contract conflicts"):
        await version_service.generate("approved", "user-a")
    failed = next(iter(repository.versions.values()))
    assert failed["status"] == "failed"
    assert repository.projects["approved"]["current_version_id"] is None
    assert repository.files == {}
    assert len(provider.calls) == 4
