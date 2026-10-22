import json
import pytest
from app.modules.agentic.generator import ApplicationGenerator, GenerationOutputError
from app.modules.agentic.generation_schemas import GeneratedFileContentResponse
from app.modules.agentic.model_router import AgenticModelRouter
from app.modules.agentic.version_service import validate_generated_files
from test_agentic_openrouter import configuration
from test_agentic_p1 import architecture
from test_agentic_p2 import service

CONTRACT = [{"method": "POST", "path": "/requests", "purpose": "Submit request",
             "request_schema": {"request": "string"}, "response_schema": {"answer": "string"}}]

def staged_outputs():
    return {
        "backend": {"content": "from fastapi import FastAPI\nfrom agents.main import run_agent\napp = FastAPI()\n@app.post('/requests')\ndef submit(request: dict):\n    result = run_agent(request)\n    return {'answer': result['answer']}\n"},
        "agents": {"content": "def run_agent(payload: dict) -> dict:\n    return {'answer': str(payload.get('request', ''))}\n"},
        "frontend": {"content": "'use client';\nexport default function Page() { async function submit() { const response = await fetch('/requests', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({request:'hello'}) }); const result = await response.json(); return result.answer; } return <main><button onClick={submit}>Run</button></main>; }\n"},
        "tests_docs": {"files": []},
    }

class StageProvider:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []
    async def complete(self, config, messages, reasoning, **kwargs):
        stage = json.loads(messages[1]['content'])['stage']
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
async def test_three_controlled_file_calls_and_small_inputs():
    app, provider = generator(staged_outputs())
    bundle = await app.generate(project_name='Support', problem_statement='Support', plan=architecture())
    assert [call[0] for call in provider.calls] == ['backend', 'agents', 'frontend']
    files = validate_generated_files(bundle)
    assert len(files) == 12 and all(len(f['sha256']) == 64 for f in files)
    for call in provider.calls:
        payload = json.loads(call[2][1]['content'])
        assert set(payload['required_json_schema']['properties']) == {'content'}
        assert 'approved_architecture' not in payload and 'failed_file' not in payload
        assert 'def submit' not in str(payload)
        assert call[3]['response_format'] == {'type':'json_object'}
    assert 'agents' not in json.loads(provider.calls[0][2][1]['content'])
    assert 'frontend' not in json.loads(provider.calls[1][2][1]['content'])

@pytest.mark.asyncio
async def test_fenced_file_response():
    fence = chr(96)*3
    app, provider = scaffold_generator(fence+'json\n'+json.dumps(staged_outputs()['backend'])+'\n'+fence)
    result = await app._stage('backend', {}, GeneratedFileContentResponse, [1])
    assert result.content.startswith('from fastapi') and len(provider.calls) == 1

@pytest.mark.asyncio
async def test_shared_one_structured_repair_budget():
    outputs = staged_outputs()
    app, provider = scaffold_generator('{}', json.dumps(outputs['backend']), '{}')
    with pytest.raises(GenerationOutputError, match='repair budget exhausted'):
        await app.generate(project_name='Support', problem_statement='Support', plan=architecture())
    assert len(provider.calls) == 3

@pytest.mark.asyncio
async def test_successful_generation_publishes_one_verified_version():
    app, provider = generator(staged_outputs())
    svc, repo = service(app)
    version = await svc.generate('approved','user-a')
    assert version['status'] == 'ready' and len(repo.files[version['id']]) == 12
    assert len(provider.calls) == 3

@pytest.mark.asyncio
@pytest.mark.parametrize('stage,content', [
    ('backend', 'from fastapi import FastAPI\napp = FastAPI()\n'),
    ('agents', 'def run_agent(:'),
    ('agents', 'def other(payload): return {}'),
    ('agents', '-----BEGIN PRIVATE KEY-----'),
    ('agents', 'binary\x00'),
    ('frontend', 'export default function Page() { return <main>Hi</main>; }'),
])
async def test_invalid_file_stops_before_later_calls(stage, content):
    outputs = staged_outputs(); outputs[stage]['content'] = content
    app, provider = generator(outputs)
    svc, repo = service(app)
    from app.modules.agentic.service import AgenticError
    with pytest.raises(AgenticError):
        await svc.generate('approved','user-a')
    assert next(iter(repo.versions.values()))['status'] == 'failed' and repo.files == {}
    assert len(provider.calls) == ['backend','agents','frontend'].index(stage)+1

@pytest.mark.asyncio
async def test_repair_failure_never_restarts_routes():
    from app.modules.genai.exceptions import ProviderConnectionError
    class Provider(ResponseProvider):
        async def complete(self, config, messages, reasoning, **kwargs):
            if self.calls:
                self.calls.append((config, messages, kwargs))
                raise ProviderConnectionError("Unavailable")
            return await super().complete(config, messages, reasoning, **kwargs)
    provider = Provider("{}")
    app_generator = ApplicationGenerator(provider, AgenticModelRouter(configuration()))
    with pytest.raises(GenerationOutputError, match="repair is unavailable"):
        await app_generator._stage("backend", {}, GeneratedFileContentResponse)
    assert len(provider.calls) == 2
    assert all(call[0].model == "example/coder:free" for call in provider.calls)

@pytest.mark.asyncio
async def test_repair_uses_responding_fallback_only():
    from app.modules.genai.exceptions import ProviderConnectionError
    class Provider(ResponseProvider):
        async def complete(self, config, messages, reasoning, **kwargs):
            if not self.calls:
                self.calls.append((config, messages, kwargs))
                raise ProviderConnectionError("Empty primary")
            return await super().complete(config, messages, reasoning, **kwargs)
    provider = Provider("{}", json.dumps(staged_outputs()["backend"]))
    app_generator = ApplicationGenerator(provider, AgenticModelRouter(configuration()))
    await app_generator._stage("backend", {}, GeneratedFileContentResponse)
    assert [call[0].model for call in provider.calls] == ["example/coder:free", "example/fallback:free", "example/fallback:free"]

@pytest.mark.asyncio
async def test_hung_stage_is_cancelled(monkeypatch):
    import asyncio
    from app.modules.agentic.generator import STAGE_TIMEOUTS
    monkeypatch.setitem(STAGE_TIMEOUTS, "backend", 0.01)
    class Provider:
        calls = 0
        cancelled = False
        async def complete(self, *args, **kwargs):
            self.calls += 1
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled = True
    provider = Provider()
    app_generator = ApplicationGenerator(provider, AgenticModelRouter(configuration()))
    with pytest.raises(GenerationOutputError, match="backend generation timed out"):
        await app_generator._stage("backend", {}, GeneratedFileContentResponse)
    assert provider.calls == 1 and provider.cancelled

@pytest.mark.asyncio
async def test_empty_completion_advances_once():
    from app.modules.agentic.model_router import complete_agentic
    from app.modules.genai.constants import ReasoningLevel
    provider = ResponseProvider("", '{"ok": true}')
    result = await complete_agentic(provider, AgenticModelRouter(configuration()), "coder", [], ReasoningLevel.DEEP)
    assert result == '{"ok": true}'
    assert [call[0].model for call in provider.calls] == ["example/coder:free", "example/fallback:free"]

def test_json_extraction_respects_braces_and_rejects_multiple_objects():
    from app.modules.agentic.generator import _extract_json_object
    value = {"code": 'return {"answer": "}"}'}
    fence = chr(96) * 3
    assert json.loads(_extract_json_object("Result:\n" + fence + "json\n" + json.dumps(value) + "\n" + fence)) == value
    with pytest.raises(ValueError):
        _extract_json_object('{"first": true} {"second": true}')


def test_stage_and_attempt_budgets_are_separate():
    from app.modules.agentic.generator import STAGE_TIMEOUTS, PROVIDER_ATTEMPT_TIMEOUT_SECONDS
    from app.modules.agentic.constants import GENERATION_TIMEOUT_SECONDS
    assert STAGE_TIMEOUTS["backend"] == 120.0 and STAGE_TIMEOUTS["agents"] == 120.0 and STAGE_TIMEOUTS["frontend"] == 120.0
    assert PROVIDER_ATTEMPT_TIMEOUT_SECONDS == 35.0
    assert PROVIDER_ATTEMPT_TIMEOUT_SECONDS < STAGE_TIMEOUTS["backend"]
    assert GENERATION_TIMEOUT_SECONDS <= 600


@pytest.mark.asyncio
async def test_generation_routes_advance_and_keep_failure_details_server_side(monkeypatch, caplog):
    import asyncio
    from app.modules.genai.exceptions import ProviderConnectionError
    from test_agentic_openrouter import provider_error
    monkeypatch.setattr("app.modules.agentic.generator.PROVIDER_ATTEMPT_TIMEOUT_SECONDS", 0.01)
    class Provider(ResponseProvider):
        async def complete(self, config, messages, reasoning, **kwargs):
            self.calls.append((config, messages, kwargs))
            if len(self.calls) == 1:
                await asyncio.Event().wait()
            if len(self.calls) == 2:
                raise provider_error(429, "limited test-key")
            raise ProviderConnectionError("The inference service returned an empty response.")
    provider = Provider()
    app_generator = ApplicationGenerator(provider, AgenticModelRouter(configuration()))
    with pytest.raises(GenerationOutputError) as error:
        await app_generator._stage("backend", {}, GeneratedFileContentResponse)
    assert len(provider.calls) == 3
    assert "provider routes are unavailable" in str(error.value)
    assert "example/" not in str(error.value) and "test-key" not in str(error.value)
    assert "stage=backend" in caplog.text
    assert all(outcome in caplog.text for outcome in ("TIMEOUT", "HTTP_429", "EMPTY_RESPONSE"))
    assert "test-key" not in caplog.text


@pytest.mark.asyncio
async def test_stage_deadline_logs_the_active_model(monkeypatch, caplog):
    import asyncio
    from app.modules.agentic.generator import STAGE_TIMEOUTS
    monkeypatch.setitem(STAGE_TIMEOUTS, "backend", 0.01)
    class Provider:
        async def complete(self, *args, **kwargs):
            await asyncio.Event().wait()
    app_generator = ApplicationGenerator(Provider(), AgenticModelRouter(configuration()))
    with pytest.raises(GenerationOutputError, match="backend generation timed out"):
        await app_generator._stage("backend", {}, GeneratedFileContentResponse)
    assert "stage=backend" in caplog.text and "example/coder:free" in caplog.text
    assert "TIMEOUT" in caplog.text


@pytest.mark.asyncio
async def test_repair_has_its_own_short_timeout(monkeypatch):
    import asyncio
    monkeypatch.setattr("app.modules.agentic.generator.PROVIDER_ATTEMPT_TIMEOUT_SECONDS", 0.01)
    class Provider(ResponseProvider):
        async def complete(self, config, messages, reasoning, **kwargs):
            if self.calls:
                self.calls.append((config, messages, kwargs))
                await asyncio.Event().wait()
            return await super().complete(config, messages, reasoning, **kwargs)
    provider = Provider("{}")
    app_generator = ApplicationGenerator(provider, AgenticModelRouter(configuration()))
    with pytest.raises(GenerationOutputError, match="repair is unavailable"):
        await app_generator._stage("backend", {}, GeneratedFileContentResponse)
    assert len(provider.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('stage,path,language', [
    ('backend','backend/main.py','python'),
    ('agents','agents/main.py','python'),
    ('frontend','frontend/app/page.tsx','tsx'),
])
@pytest.mark.parametrize('wrapper', ['content','code','source','path','filename','fenced'])
async def test_safe_wrappers_normalize_without_repair(stage, path, language, wrapper, caplog):
    content = staged_outputs()[stage]['content']
    if wrapper == 'fenced':
        fence = chr(96)*3
        raw = fence + language + '\n' + content + fence
    elif wrapper in ('path','filename'):
        raw = json.dumps({wrapper:path, 'content':content})
    else:
        raw = json.dumps({wrapper:content})
    app, provider = scaffold_generator(raw)
    budget = [1]
    result = await app._stage(stage, {}, GeneratedFileContentResponse, budget)
    assert result.content.rstrip('\n') == content.rstrip('\n')
    assert len(provider.calls) == 1 and budget == [1]
    assert 'structured output repair' not in caplog.text


@pytest.mark.parametrize('wrapper', [
    {'path':'../main.py','content':'x'},
    {'path':'agents/main.py','content':'x'},
    {'filename':'page.tsx','content':'x'},
    {'filename':'../main.py','content':'x'},
    {'path':['backend/main.py'],'content':'x'},
    {'code':'x','source':'y'},
    {'content':'x','code':'x'},
    {'content':'x','unexpected':'metadata'},
    {'path':'backend/main.py','filename':'main.py','content':'x'},
    {'code':42}, {'source':''}, {'content':['x']},
])
def test_ambiguous_unsafe_or_non_string_wrappers_are_rejected(wrapper):
    from app.modules.agentic.generator import _parse_stage_output
    with pytest.raises(ValueError):
        _parse_stage_output(json.dumps(wrapper), 'backend', GeneratedFileContentResponse)


def test_multiple_source_fences_are_rejected():
    from app.modules.agentic.generator import _parse_stage_output
    fence = chr(96)*3
    with pytest.raises(ValueError):
        _parse_stage_output(fence+'python\npass\n'+fence+'\n'+fence+'python\npass\n'+fence,
                            'backend', GeneratedFileContentResponse)


@pytest.mark.asyncio
async def test_normalization_also_applies_to_existing_one_repair_response():
    app, provider = scaffold_generator('{}', json.dumps({'code':staged_outputs()['backend']['content']}))
    result = await app._stage('backend', {}, GeneratedFileContentResponse, [1])
    assert result.content.startswith('from fastapi') and len(provider.calls) == 2


@pytest.mark.asyncio
async def test_normalized_sources_still_pass_through_workspace_validation():
    outputs = staged_outputs()
    outputs['backend'] = {'path':'backend/main.py', 'code':outputs['backend']['content']}
    outputs['agents'] = {'source':outputs['agents']['content']}
    outputs['frontend'] = {'filename':'page.tsx', 'content':outputs['frontend']['content']}
    app, provider = generator(outputs)
    bundle = await app.generate(project_name='Support', problem_statement='Support', plan=architecture())
    assert len(provider.calls) == 3
    assert len(validate_generated_files(bundle)) == 12


@pytest.mark.asyncio
@pytest.mark.parametrize('content', ['invalid Python', 'binary\x00', '-----BEGIN PRIVATE KEY-----'])
async def test_source_aliases_do_not_bypass_file_validation(content):
    outputs = staged_outputs()
    outputs['agents'] = {'code':content}
    app, provider = generator(outputs)
    with pytest.raises(GenerationOutputError):
        await app.generate(project_name='Support', problem_statement='Support', plan=architecture())
    assert len(provider.calls) == 2
