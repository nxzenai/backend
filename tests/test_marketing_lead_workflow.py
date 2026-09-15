from __future__ import annotations

import pytest
from bson import ObjectId
from fastapi import BackgroundTasks
from types import SimpleNamespace

from models.lead import LeadCreate
from routers import leads as leads_router


@pytest.mark.parametrize("origin", [
    "https://www.nxzenai.com", "https://nxzenai.com",
    "http://localhost:3000", "http://localhost:3001",
    "http://127.0.0.1:3000", "http://127.0.0.1:3001",
    "https://configured.example.com",
])
def test_registration_post_and_production_cors(origin, monkeypatch):
    import ast
    from pathlib import Path
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.testclient import TestClient

    app = FastAPI()
    # Exercise the actual middleware declaration without starting unrelated AI modules.
    tree = ast.parse((Path(__file__).parents[1] / "main.py").read_text())
    middleware = next(node for node in tree.body if isinstance(node, ast.Expr)
                      and isinstance(node.value, ast.Call)
                      and isinstance(node.value.func, ast.Attribute)
                      and node.value.func.attr == "add_middleware")
    exec(compile(ast.Module(body=[middleware], type_ignores=[]), "main.py", "exec"), {
        "app": app, "CORSMiddleware": CORSMiddleware,
        "settings": SimpleNamespace(cors_origins=["https://configured.example.com/"]),
    })
    app.include_router(leads_router.router)
    database = _MarketingDatabase()
    app.dependency_overrides[leads_router.get_marketing_database] = lambda: database
    deliveries = []
    monkeypatch.setattr(leads_router, "send_customer_demo_confirmation", lambda p: deliveries.append("customer"))
    monkeypatch.setattr(leads_router, "send_admin_demo_notification", lambda p: deliveries.append("admin"))
    payload = dict(name="Ada Lovelace", email="ada@example.com", phone="+91 90000 00000",
                   profession="Student", program_interest="AI Engineering", city="Hyderabad",
                   qualification="Graduate", organization="Example College", experience="Fresher",
                   referral_source="Website", source="training_registration", consent=True,
                   preferred_demo_date="", message="Training enquiry")
    with TestClient(app) as client:
        preflight = client.options("/api/leads/", headers={
            "Origin": origin, "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        })
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == origin
        response = client.post("/api/leads/", json=payload, headers={"Origin": origin})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    assert response.json()["id"] == "507f1f77bcf86cd799439011"
    for field, value in payload.items():
        assert database.leads.inserted[field] == value
    assert sorted(deliveries) == ["admin", "customer"]


class _LeadCollection:
    def __init__(self) -> None:
        self.updates = []

    async def update_one(self, query, update):
        self.updates.append((query, update))

    async def insert_one(self, lead):
        self.inserted = lead
        return SimpleNamespace(inserted_id=ObjectId("507f1f77bcf86cd799439011"))

    async def find_one(self, query, projection):
        return None


class _MarketingDatabase:
    def __init__(self) -> None:
        self.leads = _LeadCollection()


@pytest.mark.asyncio
async def test_marketing_lead_submission_uses_existing_collection_and_email_task():
    payload = {
        "name": "Ada Lovelace",
        "email": "ada@example.com",
        "phone": "+91 90000 00000",
        "profession": "Website enquiry",
        "program_interest": "AI Engineering",
        "preferred_demo_date": "2026-09-12",
        "message": "Team training",
    }
    database = _MarketingDatabase()
    background_tasks = BackgroundTasks()

    response = await leads_router.create_lead(
        LeadCreate(**payload),
        background_tasks,
        database,
    )

    assert response == {
        "message": "Lead created successfully",
        "id": "507f1f77bcf86cd799439011",
    }
    assert database.leads.inserted["email"] == payload["email"]
    assert database.leads.inserted["program_interest"] == payload["program_interest"]
    assert len(background_tasks.tasks) == 1
    task = background_tasks.tasks[0]
    assert task.func is leads_router._deliver_demo_booking_emails
    for field, value in payload.items():
        assert task.args[1][field] == value
    assert task.args[2] is database


@pytest.mark.asyncio
async def test_existing_lead_email_workflow_sends_customer_and_admin_notifications(monkeypatch):
    deliveries = []

    def customer_confirmation(payload):
        deliveries.append(("customer", payload))

    def admin_notification(payload):
        deliveries.append(("admin", payload))

    monkeypatch.setattr(leads_router, "send_customer_demo_confirmation", customer_confirmation)
    monkeypatch.setattr(leads_router, "send_admin_demo_notification", admin_notification)

    payload = {
        "name": "Ada Lovelace",
        "email": "ada@example.com",
        "phone": "+91 90000 00000",
        "profession": "Website enquiry",
        "program_interest": "AI Engineering",
        "preferred_demo_date": "2026-09-12",
        "message": "Team training",
    }
    database = _MarketingDatabase()

    await leads_router._deliver_demo_booking_emails(
        "507f1f77bcf86cd799439011",
        payload,
        database,
    )

    assert deliveries == [("customer", payload), ("admin", payload)]
    assert len(database.leads.updates) == 1
    update = database.leads.updates[0][1]["$set"]
    assert update["email_notification_status"] == "sent"
    assert update["customer_confirmation_sent"] is True
    assert update["admin_notification_sent"] is True
