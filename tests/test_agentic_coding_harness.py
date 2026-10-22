import json
from copy import deepcopy
import pytest

from app.modules.agentic.coding_harness import CodingWorkspace, affected_files, sanitize_build_logs
from app.modules.agentic.build.candidate import CandidateBuildVerifier
from app.modules.agentic.build.docker_runner import DockerBuildError
from app.modules.agentic.build.schemas import BuildResult
from app.modules.agentic.generator import GenerationOutputError
from app.modules.agentic.service import AgenticError
from test_agentic_p1 import architecture
from test_agentic_p2 import service
from test_agentic_p3 import MemoryBuildRepository
from test_agentic_staged_generation import generator, staged_outputs

class CandidateRepository(MemoryBuildRepository):
    async def create_build(self, owner, project, version, *, private_candidate=False, candidate_worker=None):
        result = await super().create_build(owner, project, version)
        if result and private_candidate:
            self.builds[result["id"]]["private_candidate"] = True
        if result and candidate_worker:
            self.builds[result['id']].update(status='running', lease_owner=candidate_worker, attempt=1)
        return result

class Runner:
    def __init__(self, outcomes, repo):
        self.outcomes = outcomes
        self.repo = repo
        self.sources = []
    async def build(self, workspace, event, cancelled):
        assert all(v['status'] == 'generating' for v in self.repo.versions.values() if v['id'] != 'old')
        assert self.repo.files == {} or set(self.repo.files) == {'old'}
        self.sources.append((workspace/'frontend/app/page.tsx').read_text())
        success = self.outcomes.pop(0)
        result = BuildResult(package_validation='passed', backend_validation='passed', backend_tests='passed', frontend_build='passed' if success else 'failed')
        if not success:
            await event('log','building_frontend','frontend/app/page.tsx: Type error; api_key=secret-value')
            raise DockerBuildError('Next.js build failed.', 'building_frontend', result)
        return result

async def setup(outcomes):
    app, provider = generator(staged_outputs())
    svc, repo = service(app)
    builds = CandidateRepository()
    runner = Runner(list(outcomes), repo)
    svc.build_verifier = CandidateBuildVerifier(builds, repo, lambda: runner)
    return svc, repo, builds, runner, provider


def test_deterministic_workspace_seals_config_and_bounds_tools():
    workspace = CodingWorkspace.create(architecture())
    expected = {'frontend/package.json','frontend/tsconfig.json','backend/requirements.txt',
                'backend/tests/test_smoke.py','README.md','agentic-api-contract.json','agentic-manifest.json','.env.example'}
    assert expected.issubset(workspace.list_files())
    original = workspace.read_file('frontend/package.json')
    for path in ('../escape','README.md','frontend/package.json','other.py'):
        with pytest.raises(GenerationOutputError): workspace.write_file(path,'unsafe')
    assert workspace.read_file('frontend/package.json') == original
    workspace.write_file('backend/main.py', staged_outputs()['backend']['content'])
    with pytest.raises(GenerationOutputError): workspace.write_file('backend/main.py', staged_outputs()['backend']['content'])
    with pytest.raises(GenerationOutputError): workspace.replace_file('agents/main.py','bad')
    with pytest.raises(GenerationOutputError): workspace.read_file('../escape')


@pytest.mark.asyncio
async def test_p3_build_success_precedes_ready_publication():
    svc, repo, builds, runner, provider = await setup([True])
    ready = await svc.generate('approved','user-a')
    assert ready['status'] == 'ready' and len(provider.calls) == 3
    assert builds.builds['build-1']['status'] == 'succeeded'
    assert len(runner.sources) == 1

@pytest.mark.asyncio
async def test_failed_build_repairs_only_affected_file_then_reuses_p3():
    svc, repo, builds, runner, provider = await setup([False, True])
    ready = await svc.generate('approved','user-a')
    assert ready['status'] == 'ready' and len(runner.sources) == 2
    assert [call[0] for call in provider.calls] == ['backend','agents','frontend','frontend']
    payload = json.loads(provider.calls[-1][2][1]['content'])
    assert payload['failed_file'] == staged_outputs()['frontend']['content']
    assert 'secret-value' not in payload['build_error']
    assert 'backend_files' not in payload and 'agents' not in payload
    assert [item['status'] for item in builds.builds.values()] == ['failed','succeeded']

@pytest.mark.asyncio
async def test_max_two_repairs_failed_candidate_preserves_old_ready_version():
    svc, repo, builds, runner, provider = await setup([False, False, False])
    repo.projects['approved']['current_version_id'] = 'old'
    repo.versions['old'] = {'id':'old','status':'ready','project_id':'approved','owner_id':'user-a','version_number':1}
    repo.files['old'] = [{'content':'immutable old'}]
    previous = deepcopy(repo.files['old'])
    with pytest.raises(AgenticError, match='build verification failed'):
        await svc.generate('approved','user-a')
    assert len(runner.sources) == 3 and len(provider.calls) == 5
    assert repo.projects['approved']['current_version_id'] == 'old'
    assert repo.versions['old']['status'] == 'ready' and repo.files['old'] == previous
    candidate = [v for v in repo.versions.values() if v['id'] != 'old'][0]
    assert candidate['status'] == 'failed' and candidate['parent_version_id'] == 'old'
    assert set(repo.files) == {'old'}

@pytest.mark.asyncio
async def test_provider_failure_mid_file_does_not_publish_or_build():
    svc, repo, builds, runner, provider = await setup([True])
    from app.modules.genai.exceptions import ProviderConnectionError
    original = provider.complete
    async def failing(config, messages, reasoning, **kwargs):
        if json.loads(messages[1]['content'])['stage'] == 'agents':
            raise ProviderConnectionError('Unavailable')
        return await original(config, messages, reasoning, **kwargs)
    provider.complete = failing
    with pytest.raises(AgenticError, match='provider routes are unavailable'):
        await svc.generate('approved','user-a')
    assert repo.files == {} and builds.builds == {} and runner.sources == []
    assert next(iter(repo.versions.values()))['status'] == 'failed'

@pytest.mark.asyncio
async def test_missing_build_verifier_fails_closed():
    svc, repo = service()
    svc.build_verifier = None
    with pytest.raises(AgenticError, match='build verification is unavailable'):
        await svc.generate('approved','user-a')
    assert repo.files == {}

@pytest.mark.asyncio
async def test_absent_smoke_check_cannot_publish_ready():
    svc, repo, builds, runner, provider = await setup([True])
    async def incomplete(workspace, event, cancelled):
        return BuildResult(package_validation='passed', backend_validation='passed', backend_tests='absent', frontend_build='passed')
    runner.build = incomplete
    with pytest.raises(AgenticError, match='build verification failed'):
        await svc.generate('approved','user-a')
    assert next(iter(repo.versions.values()))['status'] == 'failed' and repo.files == {}


def test_repair_targets_and_log_sanitization():
    assert affected_files({'frontend_build':'failed'}, 'frontend/app/page.tsx: Type error') == ['frontend/app/page.tsx']
    assert affected_files({'backend_tests':'failed'}, 'agents/main.py: error') == ['agents/main.py']
    assert affected_files({'package_validation':'passed'}, 'Docker unavailable') == []
    value = sanitize_build_logs('api_key=secret sk-or-v1-secrettoken '+'x'*5000)
    assert 'secret' not in value and len(value) <= 2000


def test_each_file_has_strict_content_only_schema():
    from pydantic import ValidationError
    from app.modules.agentic.generation_schemas import GeneratedFileContentResponse
    assert set(GeneratedFileContentResponse.model_json_schema()['properties']) == {'content'}
    for value in ({'content':'x','path':'evil.py'}, {'content':123}, {'content':''}):
        with pytest.raises(ValidationError): GeneratedFileContentResponse.model_validate(value)


def test_utf8_byte_limit_applies_before_file_acceptance():
    from app.modules.agentic.constants import MAX_GENERATED_FILE_BYTES
    workspace = CodingWorkspace.create(architecture())
    with pytest.raises(GenerationOutputError, match='source size limits'):
        workspace.write_file('agents/main.py', '#' + '\u20ac' * (MAX_GENERATED_FILE_BYTES // 3 + 1))
    assert 'agents/main.py' not in workspace.list_files()


@pytest.mark.asyncio
async def test_failed_repair_validation_cannot_publish_ready():
    svc, repo, builds, runner, provider = await setup([False, True])
    original = provider.complete
    async def bad_repair(config, messages, reasoning, **kwargs):
        payload = json.loads(messages[1]['content'])
        if 'failed_file' in payload:
            return json.dumps({'content':'invalid frontend'})
        return await original(config, messages, reasoning, **kwargs)
    provider.complete = bad_repair
    with pytest.raises(AgenticError): await svc.generate('approved','user-a')
    assert repo.files == {} and len(builds.builds) == 1 and len(runner.sources) == 1
    assert next(iter(repo.versions.values()))['status'] == 'failed'


@pytest.mark.asyncio
async def test_candidate_adapter_enforces_owner_and_state():
    svc, repo, builds, runner, provider = await setup([True])
    from app.modules.agentic.version_service import validate_generated_files
    app, unused = generator(staged_outputs())
    bundle = await app.generate(project_name='Support', problem_statement='Support',plan=architecture())
    version = await repo.create_version('approved','user-a','plan-approved',None)
    with pytest.raises(GenerationOutputError):
        await svc.build_verifier.verify('user-b','approved',version['id'],validate_generated_files(bundle))
    repo.versions[version['id']]['status'] = 'ready'
    with pytest.raises(GenerationOutputError):
        await svc.build_verifier.verify('user-a','approved',version['id'],validate_generated_files(bundle))
    assert builds.builds == {}


@pytest.mark.asyncio
async def test_production_candidate_is_queued_and_consumed_by_existing_worker():
    from app.modules.agentic.build.worker import execute_claimed_build
    from app.modules.agentic.build.candidate import PrivateCandidateSource
    svc, versions, builds, unused, provider = await setup([True])
    svc.build_verifier = CandidateBuildVerifier(builds, versions)
    original = builds.get_build
    seen_hashes = []
    class QueuedRunner:
        async def build(self, workspace, event, cancelled):
            candidate = next(iter(versions.versions.values()))
            assert candidate['status'] == 'generating'
            with pytest.raises(AgenticError):
                await svc.read_file('approved',candidate['id'],'user-a','backend/main.py')
            seen_hashes.extend((file['path'],file['sha256']) for file in versions.files[candidate['id']])
            return BuildResult(package_validation='passed', backend_validation='passed', backend_tests='passed', frontend_build='passed')
    async def drive_queue(owner, project, build_id):
        item = builds.builds[build_id]
        if item['status'] == 'queued':
            assert item['private_candidate'] is True
            claimed = await builds.claim_next('worker-a')
            source = PrivateCandidateSource(versions, owner, project, item['version_id'])
            await execute_claimed_build(claimed, builds, source, 'worker-a', lambda: QueuedRunner())
        return await original(owner, project, build_id)
    builds.get_build = drive_queue
    ready = await svc.generate('approved','user-a')
    assert ready['status'] == 'ready' and len(provider.calls) == 3
    assert seen_hashes == [(file['path'],file['sha256']) for file in versions.files[ready['id']]]


@pytest.mark.asyncio
async def test_queued_candidate_timeout_cancels_build_and_cleans_private_source(monkeypatch):
    import asyncio
    svc, versions, builds, unused, provider = await setup([True])
    svc.build_verifier = CandidateBuildVerifier(builds, versions)
    original = builds.get_build
    async def hang(owner, project, build_id):
        await asyncio.Event().wait()
    builds.get_build = hang
    task = asyncio.create_task(svc.generate('approved','user-a'))
    # Await the queued build without sleeping or contacting any real worker.
    for unused in range(100):
        if builds.builds: break
        await asyncio.sleep(0)
    assert builds.builds
    task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    assert next(iter(builds.builds.values()))['status'] == 'cancelled'
    assert next(iter(versions.versions.values()))['status'] == 'failed'
    assert versions.files == {}
