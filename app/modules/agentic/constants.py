import os
from enum import Enum


PLANNER_SCHEMA_VERSION = "1.0"
GENERATOR_SCHEMA_VERSION = "1.0"
MAX_GENERATED_FILES = int(os.getenv("AGENTIC_MAX_GENERATED_FILES", "80"))
MAX_GENERATED_FILE_BYTES = int(os.getenv("AGENTIC_MAX_GENERATED_FILE_BYTES", "262144"))
MAX_GENERATED_TOTAL_BYTES = int(os.getenv("AGENTIC_MAX_GENERATED_TOTAL_BYTES", "2097152"))
GENERATION_TIMEOUT_SECONDS = float(os.getenv("AGENTIC_GENERATION_TIMEOUT_SECONDS", "180"))
MAX_GENERATED_PATH_DEPTH = 12
MAX_GENERATED_PATH_LENGTH = 240


class ProjectStatus(str, Enum):
    DRAFT = "draft"
    PLANNING = "planning"
    PLAN_READY = "plan_ready"
    APPROVED = "approved"
    PLANNING_FAILED = "planning_failed"
    GENERATING = "generating"
    GENERATED = "generated"
    GENERATION_FAILED = "generation_failed"


class PlanStatus(str, Enum):
    GENERATED = "generated"
    APPROVED = "approved"
    SUPERSEDED = "superseded"


class VersionStatus(str, Enum):
    GENERATING = "generating"
    READY = "ready"
    FAILED = "failed"
