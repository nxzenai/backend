from enum import Enum


PLANNER_SCHEMA_VERSION = "1.0"


class ProjectStatus(str, Enum):
    DRAFT = "draft"
    PLANNING = "planning"
    PLAN_READY = "plan_ready"
    APPROVED = "approved"
    PLANNING_FAILED = "planning_failed"


class PlanStatus(str, Enum):
    GENERATED = "generated"
    APPROVED = "approved"
    SUPERSEDED = "superseded"
