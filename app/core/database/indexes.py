"""Indexes for active queries; no data migration or backfill."""


async def ensure_indexes(studio, leads, audit):
    # ---------------------------------------------------------
    # Users
    # ---------------------------------------------------------

    await studio.users.create_index(
        "email",
        unique=True,
    )

    await studio.users.create_index(
        [
            ("account_status", 1),
            ("organization_id", 1),
            ("role", 1),
        ]
    )

    await studio.users.create_index(
        [
            ("batch_id", 1),
            ("account_status", 1),
        ]
    )

    # ---------------------------------------------------------
    # Organizations / Courses / Batches
    # ---------------------------------------------------------

    # Must match the existing MongoDB index specification.
    # Existing index: name_1 with unique=True
    await studio.organizations.create_index(
        [("name", 1)],
        unique=True,
    )

    await studio.courses.create_index(
        [
            ("organization_id", 1),
            ("name", 1),
        ]
    )

    await studio.batches.create_index(
        [
            ("organization_id", 1),
            ("course_id", 1),
            ("end_at", -1),
        ]
    )

    # ---------------------------------------------------------
    # User-owned resources
    # ---------------------------------------------------------

    for name, timestamp in (
        ("eda_projects", "created_at"),
        ("datasets", "created_at"),
        ("notebooks", "updated_at"),
    ):
        await studio[name].create_index(
            [
                ("owner_id", 1),
                ("is_deleted", 1),
                (timestamp, -1),
            ]
        )

    # ---------------------------------------------------------
    # SQL Lab
    # ---------------------------------------------------------

    await studio.sql_lab_workspaces.create_index(
        "owner_id",
        unique=True,
    )

    await studio.sql_lab_history.create_index(
        [
            ("owner_id", 1),
            ("created_at", -1),
        ]
    )

    # ---------------------------------------------------------
    # AutoML / AI registry
    # ---------------------------------------------------------

    await studio.automl_models.create_index(
        [
            ("owner_id", 1),
            ("filename", 1),
        ],
        unique=True,
    )

    await studio.ai_model_registry.create_index(
        [
            ("module", 1),
            ("winning_job_id", 1),
        ],
        unique=True,
    )

    await studio.ai_model_registry.create_index(
        [
            ("owner_id", 1),
            ("created_at", -1),
        ]
    )

    await studio.ai_model_registry.create_index(
        [
            ("model_group_id", 1),
            ("version", -1),
        ],
        unique=True,
    )

    await studio.ai_prediction_observations.create_index(
        [
            ("owner_id", 1),
            ("created_at", -1),
        ]
    )

    # ---------------------------------------------------------
    # Leads / CRM
    # ---------------------------------------------------------

    await leads.leads.create_index(
        [
            ("created_at", -1),
        ]
    )

    await leads.leads.create_index(
        [
            ("crm_status", 1),
            ("status", 1),
            ("created_at", -1),
        ]
    )

    await leads.leads.create_index(
        [
            ("email", 1),
            ("preferred_demo_date", 1),
            ("email_notification_status", 1),
        ]
    )

    await leads.crm_deals.create_index(
        "lead_id",
        unique=True,
    )

    for name in (
        "lead_activities",
        "crm_notes",
    ):
        await leads[name].create_index(
            [
                ("lead_id", 1),
                ("created_at", -1),
            ]
        )

    # ---------------------------------------------------------
    # Audit / Usage
    # ---------------------------------------------------------

    for name in (
        "activity_logs",
        "auth_logs",
        "admin_audit_logs",
        "module_usage",
    ):
        await audit[name].create_index(
            [
                ("owner_id", 1),
                ("created_at", -1),
            ]
        )

    await audit.module_usage.create_index(
        [
            ("organization_id", 1),
            ("batch_id", 1),
            ("created_at", -1),
        ]
    )

    await audit.activity_logs.create_index(
        [
            ("model_id", 1),
            ("created_at", -1),
        ]
    )