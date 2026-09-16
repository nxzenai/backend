from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from copy import deepcopy

import pytest
from fastapi import Depends, FastAPI, Request
from httpx import ASGITransport, AsyncClient

from app.core import audit
from app.core.config.settings import Settings
from app.core.database import mongodb
from app.core.database.indexes import ensure_indexes


def test_database_names_do_not_depend_on_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017")
    monkeypatch.setenv("STUDIO_DB", "ai_studio_db")
    monkeypatch.setenv("LEADS_DB", "nxzenai_leads")
    monkeypatch.setenv("AUDIT_DB", "nxzenai_audit")
    monkeypatch.setenv("DATABASE_NAME", "wrong")
    monkeypatch.setenv("MARKETING_DB", "wrong")
    settings = Settings()
    assert (settings.database_name, settings.marketing_database_name, settings.audit_database_name) == (
        "ai_studio_db", "nxzenai_leads", "nxzenai_audit")


@pytest.mark.asyncio
async def test_shared_client_and_database_routing(monkeypatch):
    client = MagicMock()
    client.admin.command = AsyncMock()
    databases = {name: MagicMock(name=name) for name in ("ai_studio_db", "nxzenai_leads", "nxzenai_audit")}
    client.__getitem__.side_effect = databases.__getitem__
    constructor = MagicMock(return_value=client)
    indexes = AsyncMock()
    monkeypatch.setattr(mongodb, "AsyncIOMotorClient", constructor)
    monkeypatch.setattr("app.core.database.indexes.ensure_indexes", indexes)
    monkeypatch.setattr(mongodb, "settings", Settings(_env_file=None))
    try:
        await mongodb.MongoDB.connect()
        assert mongodb.get_database() is databases["ai_studio_db"]
        assert mongodb.get_leads_database() is databases["nxzenai_leads"]
        assert mongodb.get_audit_database() is databases["nxzenai_audit"]
        assert mongodb.get_sync_database() is databases["ai_studio_db"].delegate
        constructor.assert_called_once()
        indexes.assert_awaited_once_with(*databases.values())
    finally:
        await mongodb.MongoDB.disconnect()
    client.close.assert_called_once()


@pytest.mark.asyncio
async def test_essential_indexes(monkeypatch):
    class DB:
        def __init__(self): self.collections = {}
        def __getitem__(self, key):
            return self.collections.setdefault(key, SimpleNamespace(create_index=AsyncMock()))
        __getattr__ = __getitem__
    studio, leads, audit_db = DB(), DB(), DB()
    await ensure_indexes(studio, leads, audit_db)
    studio.users.create_index.assert_awaited_once_with("email", unique=True)
    leads.crm_deals.create_index.assert_awaited_once_with("lead_id", unique=True)
    assert set(audit_db.collections) == audit.COLLECTIONS
    assert "leads" not in studio.collections and "users" not in leads.collections


@pytest.mark.asyncio
async def test_audit_excludes_secrets_and_views_and_keeps_actor(monkeypatch):
    events = {}
    class DB:
        def __getitem__(self, key):
            async def insert(document): events.setdefault(key, []).append(document)
            return SimpleNamespace(insert_one=insert)
    monkeypatch.setattr(audit, "get_audit_database", lambda: DB())
    app = FastAPI()
    app.add_middleware(audit.AuditMiddleware)

    @app.post("/api/v1/automl/train")
    async def chat(request: Request):
        audit.actor.set(("user-1", "user"))
        await request.json()
        return {"token": "secret-response"}

    @app.get("/api/v1/genai/conversations")
    async def conversations(): return []

    @app.post("/api/v1/auth/login")
    async def login(): return {"access_token": "secret-token"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.post("/api/v1/automl/train?api_key=secret-query",
            json={"password": "secret-password", "prompt": "secret-prompt"},
            headers={"Authorization": "Bearer secret-header"})).status_code == 200
        await client.get("/api/v1/genai/conversations")
        await client.post("/api/v1/auth/login")
    assert len(events["activity_logs"]) == 1
    assert events["activity_logs"][0]["owner_id"] == "user-1"
    assert len(events["module_usage"]) == 1
    assert events["auth_logs"][0]["owner_id"] is None
    assert "secret" not in repr(events)


@pytest.mark.asyncio
async def test_audit_failure_does_not_break_completed_action(monkeypatch):
    monkeypatch.setattr(audit, "get_audit_database", MagicMock(side_effect=RuntimeError("secret-uri")))
    await audit.log_event("auth_logs", "login", "auth", status_code=401)


def test_sql_history_does_not_store_literals(monkeypatch):
    from app.modules.sql import history
    db = MagicMock()
    monkeypatch.setattr(history, "get_sync_database", lambda: db)
    history.record_execution("user-1", "SELECT 'secret-token'", .1, True)
    row = db.sql_lab_history.insert_one.call_args.args[0]
    assert row["owner_id"] == "user-1" and row["operation"] == "SELECT"
    assert "secret-token" not in repr(row)
    with pytest.raises(ValueError): history.record_execution(None, "SELECT 1", .1, True)


def test_registry_owner_scope_and_stage_authorization(monkeypatch):
    from app.core import ai_model_registry as registry
    db = MagicMock()
    monkeypatch.setattr(registry, "get_sync_database", lambda: db)
    db.ai_model_registry.find_one.return_value = None
    assert registry.get_model("model-1", "owner-1") is None
    db.ai_model_registry.find_one.assert_called_with({"_id": "model-1", "owner_id": "owner-1"})
    db.ai_model_registry.find_one.return_value = {
        "_id": "model-1", "owner_id": "owner-1", "module": "autonlp",
        "lifecycle_stage": "draft", "configuration": {}}
    with pytest.raises(PermissionError): registry.change_stage("model-1", "owner-1", "production")
    db.ai_model_registry.find_one_and_update.assert_not_called()


def test_autodl_audit_uses_audit_database_and_excludes_details():
    from app.modules.autodl_v2.repository import AutoDLV2Repository
    from app.core.config.settings import settings
    db = MagicMock()
    repository = AutoDLV2Repository(db)
    repository.add_audit("owner-1", "run-1", "training_failed", {"error": "secret-password"})
    audit_db = db.client[settings.audit_database_name]
    row = audit_db["activity_logs"].insert_one.call_args.args[0]
    assert row["owner_id"] == "owner-1"
    assert "secret" not in repr(row)


@pytest.mark.asyncio
async def test_genai_usage_is_audit_only_and_whitelists_fields(monkeypatch):
    from app.modules.genai.repository import GenAIRepository
    database = SimpleNamespace(module_usage=SimpleNamespace(update_one=AsyncMock()))
    monkeypatch.setattr(mongodb, "get_audit_database", lambda: database)
    repository = GenAIRepository.__new__(GenAIRepository)
    repository.generations = SimpleNamespace(update_one=AsyncMock())
    await repository.record_request("request-1", "owner-1", {
        "status": "completed", "model_latency_ms": 10,
        "password": "secret-password", "prompt": "secret-prompt", "owner_id": "other-owner"})
    repository.generations.update_one.assert_not_awaited()
    row = database.module_usage.update_one.call_args.args[1]["$set"]
    assert row["owner_id"] == "owner-1" and row["module"] == "genai"
    assert row["model_latency_ms"] == 10 and "secret" not in repr(row)


@pytest.mark.asyncio
async def test_crm_router_denies_anonymous_access():
    from app.modules.crm.router import router
    from app.core.exceptions.handlers import register_exception_handlers
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router, prefix="/api/v1")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # get_current_user also depends on the studio database.
        app.dependency_overrides[mongodb.get_database] = lambda: MagicMock()
        assert (await client.get("/api/v1/crm/leads")).status_code == 401


class Rows(list):
    def sort(self, key, direction):
        return Rows(sorted(self, key=lambda row: row[key], reverse=direction == -1))
    def limit(self, limit): return Rows(self[:limit])


class SyncCollection:
    def __init__(self, database): self.database, self.rows = database, []
    def find(self, query, projection=None):
        def matches(row):
            return all(row.get(key) != value["$ne"] if isinstance(value, dict) and "$ne" in value
                       else row.get(key) == value for key, value in query.items())
        return Rows(deepcopy([row for row in self.rows if matches(row)]))
    def find_one(self, query, **kwargs): return next(iter(self.find(query)), None)
    def insert_one(self, row): self.rows.append(deepcopy(row))
    def find_one_and_update(self, query, update, upsert=False, **kwargs):
        match = self.find_one(query)
        if match is None:
            if not upsert: return None
            match = {**query, **update.get("$setOnInsert", {})}
            self.rows.append(match)
        else:
            match = next(row for row in self.rows if row == match)
        match.update(update.get("$set", {}))
        for key, value in update.get("$inc", {}).items(): match[key] = match.get(key, 0) + value
        return deepcopy(match)
    def update_one(self, *args, **kwargs): return self.find_one_and_update(*args, **kwargs)
    def count_documents(self, query): return len(self.find(query))
    def delete_one(self, query):
        found = self.find_one(query)
        if found: self.rows.remove(found)


class SyncDatabase:
    def __init__(self): self.collections = {}
    def __getitem__(self, name): return self.collections.setdefault(name, SyncCollection(self))
    __getattr__ = __getitem__


def test_registry_lifecycle_preserves_owner_links_and_audit_separation(monkeypatch):
    from app.core import ai_model_registry as registry
    studio, audit_db = SyncDatabase(), SyncDatabase()
    monkeypatch.setattr(registry, "get_sync_database", lambda: studio)
    monkeypatch.setattr(registry, "get_audit_database", lambda: SimpleNamespace(delegate=audit_db))
    monkeypatch.setattr(registry, "get_artifact_storage", lambda: SimpleNamespace(artifact_location=lambda *args: "artifact-key"))
    arguments = dict(module="autonlp", job_id="job-1", owner_id="owner-1",
                     manifest={"task": "classification"}, configuration={"password": "secret-config"})
    first = registry.register_completed_model(**arguments)
    assert registry.register_completed_model(**arguments).id == first.id
    second = registry.register_completed_model(**{**arguments, "job_id": "job-2", "source_model_id": first.id})
    assert second.version == 2 and second.source_model_id == first.id
    assert registry.list_models("stranger") == []
    assert [model.version for model in registry.list_versions(first.id, "owner-1")] == [2, 1]
    with pytest.raises(LookupError):
        registry.register_completed_model(**{**arguments, "owner_id": "stranger", "job_id": "job-3", "source_model_id": first.id})
    assert registry.change_stage(first.id, "owner-1", "archived").lifecycle_stage == "archived"
    assert [model.id for model in registry.list_models("owner-1")] == [second.id]
    registry.record_prediction(module="autonlp", job_id="job-2", owner_id="owner-1", success=True, latency_ms=3)
    observation = registry.list_prediction_observations("owner-1")[0]
    assert observation.model_id == second.id
    with pytest.raises(LookupError): registry.record_prediction_feedback(observation.id, "stranger", "label")
    assert registry.record_prediction_feedback(observation.id, "owner-1", "label").actual_label == "label"
    assert registry.list_audit_events(first.id, "stranger") == []
    assert "secret-config" not in repr(audit_db.activity_logs.rows)
    assert all(row["owner_id"] == "owner-1" for row in audit_db.activity_logs.rows)


@pytest.mark.asyncio
async def test_automl_artifact_and_mongo_metadata_roundtrip(monkeypatch, tmp_path):
    from app.modules.automl import mongo_metadata
    from app.modules.automl.router import save_training_artifact
    from app.modules.automl.service import AutoMLService, AutoMLServiceConfig
    from test_automl_artifact_prediction import _numeric_artifact, _training_result
    db = SyncDatabase()
    monkeypatch.setattr(mongo_metadata, "get_sync_database", lambda: db)
    service = AutoMLService(AutoMLServiceConfig(model_directory=str(tmp_path)))
    filename = await save_training_artifact(service, _training_result(_numeric_artifact()), "owner-1")
    assert service.list_models_for_owner("owner-1") == [filename]
    assert service.list_models_for_owner("stranger") == []
    assert service.load_owned_artifact(filename, "owner-1").metadata["owner_id"] == "owner-1"
    mongo_metadata.delete_model("stranger", filename)
    assert service.list_models_for_owner("owner-1") == [filename]
    mongo_metadata.delete_model("owner-1", filename)
    assert service.list_models_for_owner("owner-1") == []
