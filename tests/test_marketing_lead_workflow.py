from __future__ import annotations

import pytest
from bson import ObjectId
from fastapi import BackgroundTasks
from types import SimpleNamespace

from models.lead import LeadCreate
from routers import leads as leads_router


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
