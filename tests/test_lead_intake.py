import asyncio
from copy import deepcopy
from datetime import datetime
from email.message import EmailMessage
import re
from types import SimpleNamespace

from bson import ObjectId
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from app.core.database.mongodb import get_database, get_marketing_database
from app.core.exceptions.handlers import register_exception_handlers
from app.modules.auth.dependencies import get_current_user
from app.modules.crm.repository import CRMRepository
from models.lead import VerificationUpdate
from routers.lead_intake import router as intake_router
from routers.leads import router as public_router
from services.lead_intake import LeadIntakeService
from services import demo_email


def matches(row, query):
    for key, expected in query.items():
        if key == "$or":
            if not any(matches(row, term) for term in expected):
                return False
            continue
        value = row.get(key)
        if not isinstance(expected, dict):
            if value != expected:
                return False
            continue
        for op, operand in expected.items():
            if op == "$ne" and value == operand: return False
            if op == "$nin" and value in operand: return False
            if op == "$in" and value not in operand: return False
            if op == "$lt" and (value is None or value >= operand): return False
            if op == "$gte" and (value is None or value < operand): return False
            if op == "$regex" and not re.search(operand, str(value or ""), re.I): return False
    return True


class Cursor:
    def __init__(self, rows): self.rows = deepcopy(rows)
    def sort(self, key, direction):
        self.rows.sort(key=lambda row: row[key], reverse=direction == -1)
        return self
    def skip(self, count): self.rows = self.rows[count:]; return self
    def limit(self, count): self.rows = self.rows[:count]; return self
    async def to_list(self, length): return self.rows[:length]
    def __aiter__(self):
        async def iterate():
            for row in self.rows: yield row
        return iterate()


class Collection:
    def __init__(self): self.rows = []
    async def insert_one(self, row):
        row = deepcopy(row)
        row["_id"] = ObjectId()
        self.rows.append(row)
        return SimpleNamespace(inserted_id=row["_id"])
    async def find_one(self, query, projection=None):
        return next((deepcopy(row) for row in self.rows if matches(row, query)), None)
    def find(self, query=None, projection=None):
        return Cursor([row for row in self.rows if matches(row, query or {})])
    async def update_one(self, query, update):
        found = await self.find_one_and_update(query, update)
        return SimpleNamespace(matched_count=int(found is not None))
    async def find_one_and_update(self, query, update, **kwargs):
        for row in self.rows:
            if matches(row, query):
                row.update(deepcopy(update["$set"]))
                return deepcopy(row)
        return None
    async def count_documents(self, query):
        return sum(matches(row, query) for row in self.rows)
    async def distinct(self, field):
        return list({row.get(field) for row in self.rows})
    async def delete_one(self, query):
        self.rows = [row for row in self.rows if not matches(row, query)]


class Database:
    def __init__(self): self.leads = Collection()
    def __getitem__(self, key):
        assert key == "leads", "Lead intake must use the existing leads collection"
        return self.leads


@pytest.fixture
def db(): return Database()


@pytest.fixture
def app(db):
    instance = FastAPI()
    register_exception_handlers(instance)
    instance.include_router(public_router)
    instance.include_router(intake_router, prefix="/api/v1")
    instance.dependency_overrides[get_marketing_database] = lambda: db
    instance.dependency_overrides[get_database] = lambda: db
    return instance


def payload(source="website"):
    return dict(name="Ada Lovelace", email="ada@example.com", phone="+91 90000 00000",
                profession="Student", program_interest="AI Engineering", source=source,
                city="Pune", qualification="BSc", consent=True,
                preferred_demo_date="" if source == "training_registration" else "2026-09-12")


async def seed(db, status="pending", **extra):
    result = await db.leads.insert_one({**payload(), "created_at": datetime(2026, 9, 10),
                                      "status": "new", "priority": "warm",
                                      "verification_status": status, "crm_status": "not_pushed", **extra})
    return str(result.inserted_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["website", "demo", "training_registration"])
async def test_public_submission_same_db_and_real_email_rendering(app, db, monkeypatch, source):
    deliveries = []
    recipients = []
    def message(subject, to):
        recipients.append(to)
        mail = EmailMessage()
        mail["Subject"] = subject
        return mail
    monkeypatch.setattr(demo_email, "_message", message)
    monkeypatch.setattr(demo_email, "_send", deliveries.append)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/leads/", json=payload(source))
    assert response.status_code == 200
    assert len(db.leads.rows) == 1
    row = db.leads.rows[0]
    assert row["source"] == source
    assert row["verification_status"] == "pending"
    assert row["crm_status"] == "not_pushed"
    assert row["customer_confirmation_sent"] is True
    assert row["admin_notification_sent"] is True
    assert row["email_notification_status"] == "sent"
    assert len(deliveries) == 2
    assert ["ada@example.com"] in recipients
    assert demo_email.settings.smtp_admin_recipients in recipients
    if source == "training_registration":
        assert all("To be arranged" in mail.get_body(preferencelist=("plain",)).get_content() for mail in deliveries)


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("phone", "abc"), ("phone", "123"), ("city", " "), ("qualification", ""), ("consent", False), ("profession", "invalid"), ("preferred_demo_date", "bad-date")])
async def test_training_validation(app, db, field, value):
    request = payload("training_registration")
    request[field] = value
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/leads/", json=request)
    assert response.status_code == 422
    assert not db.leads.rows


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [None, "user", "instructor", "admin", "super_admin"])
async def test_admin_routes_enforce_roles_and_verification(app, db, role):
    value = await seed(db)
    if role:
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id="admin-1", role=role)
    expected = 401 if role is None else 403 if role in ("user", "instructor") else 200
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for method, path, body in [
            ("GET", "/api/v1/leads/", None), ("GET", f"/api/v1/leads/{value}", None),
            ("PATCH", f"/api/v1/leads/{value}/verification", {"verification_status": "verified", "verification_notes": "Called candidate"}),
            ("POST", f"/api/v1/leads/{value}/push-to-crm", None), ("POST", "/api/v1/leads/push-verified", None),
        ]:
            response = await client.request(method, path, json=body)
            assert response.status_code == expected, response.text
        if expected == 200:
            response = await client.patch(f"/api/v1/leads/{value}/verification", json={"verification_status": "not_verified", "verification_notes": "Could not confirm"})
            assert response.json()["verification_status"] == "not_verified"
            assert response.json()["verified_by"] == "admin-1"
            assert response.json()["verified_at"]
            assert response.json()["verification_notes"] == "Could not confirm"
        else:
            for path in ("/api/leads/", "/api/leads/export/csv"):
                assert (await client.get(path)).status_code == expected
    assert "email_notification_status" not in db.leads.rows[0]


@pytest.mark.asyncio
async def test_push_atomic_idempotency_linkage_and_legacy_crm_preservation(db):
    legacy_id = await seed(db, status="verified", crm_status="pushed")
    legacy = db.leads.rows[0]
    del legacy["crm_status"]
    del legacy["verification_status"]
    before = deepcopy(legacy)
    value = await seed(db, "verified")
    intake = LeadIntakeService(db)
    crm = CRMRepository(db)
    assert await crm.get_lead(value) is None
    assert (await crm.dashboard())["total"] == 1
    results = await asyncio.gather(*(intake.push(value, "admin-1") for _ in range(10)))
    assert sum(result["pushed"] for result in results) == 1
    lead = await intake.get(value)
    assert lead["crm_status"] == "pushed"
    assert lead["crm_lead_id"] == value
    assert lead["pushed_to_crm_by"] == "admin-1"
    assert lead["pushed_to_crm_at"]
    assert len(db.leads.rows) == 2
    assert db.leads.rows[0] == before
    assert (await crm.get_lead(value))["id"] == value
    assert (await crm.get_lead(legacy_id))["id"] == legacy_id
    assert (await crm.dashboard())["total"] == 2
    assert not (await intake.push(legacy_id, "admin-1"))["pushed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["pending", "not_verified"])
async def test_ineligible_push_and_crm_mutation_cannot_bypass_verification(db, status):
    value = await seed(db, status)
    intake = LeadIntakeService(db)
    with pytest.raises(Exception) as error:
        await intake.push(value, "admin-1")
    assert error.value.status_code == 409
    crm = CRMRepository(db)
    assert await crm.update_lead(value, {"status": "enrolled"}) is None
    await crm.delete_lead(value)
    assert len(db.leads.rows) == 1
    assert db.leads.rows[0]["status"] == "new"


@pytest.mark.asyncio
async def test_bulk_push_only_eligible_and_repeat_safe(db):
    intake = LeadIntakeService(db)
    await seed(db, "pending")
    await seed(db, "not_verified")
    await seed(db, "verified", crm_status="pushed")
    eligible = [await seed(db, "verified"), await seed(db, "verified")]
    result = await intake.bulk_push("admin-1")
    assert result == {"pushed": 2, "skipped": 0, "failed": []}
    assert (await intake.bulk_push("admin-1"))["pushed"] == 0
    assert len(db.leads.rows) == 5
    for value in eligible:
        assert (await intake.get(value))["crm_lead_id"] == value


@pytest.mark.asyncio
async def test_filter_search_dates_summary_and_legacy_defaults(db):
    value = await seed(db, "verified", name="Ada [test]", source="training_registration")
    await seed(db, "not_verified", name="Grace Hopper", program_interest="AI Foundations")
    intake = LeadIntakeService(db)
    for search in ("[test]", "ada@example.com", "90000"):
        result = await intake.list(search=search, status="verified", course="AI Engineering", source="training_registration", start=datetime(2026, 9, 10), end=datetime(2026, 9, 11))
        assert [lead["id"] for lead in result["items"]] == [value]
        assert result["summary"] == {"total": 2, "pending": 0, "verified": 1, "not_verified": 1, "ready": 1}
    assert not (await intake.list(start=datetime(2026, 9, 11)))["items"]


@pytest.mark.asyncio
async def test_missing_and_invalid_ids_and_enum(app, db):
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id="admin-1", role="admin")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for value, expected in (("invalid", 400), (str(ObjectId()), 404)):
            assert (await client.get(f"/api/v1/leads/{value}")).status_code == expected
            assert (await client.patch(f"/api/v1/leads/{value}/verification", json={"verification_status": "verified"})).status_code == expected
            assert (await client.post(f"/api/v1/leads/{value}/push-to-crm")).status_code == expected
        value = await seed(db)
        assert (await client.patch(f"/api/v1/leads/{value}/verification", json={"verification_status": True})).status_code == 422
        assert (await client.get("/api/v1/leads/?start=2026-09-12&end=2026-09-10")).status_code == 422
