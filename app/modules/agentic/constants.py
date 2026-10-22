import os
from enum import Enum
from typing import Literal


PLANNER_SCHEMA_VERSION = "1.0"
GENERATOR_SCHEMA_VERSION = "1.0"
MAX_GENERATED_FILES = int(os.getenv("AGENTIC_MAX_GENERATED_FILES", "80"))
MAX_GENERATED_FILE_BYTES = int(os.getenv("AGENTIC_MAX_GENERATED_FILE_BYTES", "262144"))
MAX_GENERATED_TOTAL_BYTES = int(os.getenv("AGENTIC_MAX_GENERATED_TOTAL_BYTES", "2097152"))
GENERATION_TIMEOUT_SECONDS = float(os.getenv("AGENTIC_GENERATION_TIMEOUT_SECONDS", "600"))
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

# Shared Agentic V1 contract: P1 planning and P2 generation use the same values.
MAX_APPLICATION_NAME = 160
MAX_CONTRACT_NAME = 120
MAX_SUMMARY = 2000
MAX_ROLE = 500
MAX_CONTRACT_TEXT = 1000
ToolType = Literal["document_search", "csv_analysis", "calculator", "knowledge_retrieval", "report_generation", "http_api", "sql_query"]
FrontendFramework = Literal["Next.js"]
BackendFramework = Literal["FastAPI"]
ApiMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
TOOL_TYPE_ALIASES = {"retrieval": "knowledge_retrieval"}
TOOL_CONFIGURATION = {
    "document_search": ["document_source"], "csv_analysis": ["csv_path"],
    "knowledge_retrieval": ["knowledge_source"], "http_api": ["endpoint"],
    "sql_query": ["connection_string"], "calculator": [], "report_generation": [],
}

def normalize_tool_type(value):
    return TOOL_TYPE_ALIASES.get(value, value) if isinstance(value, str) else value
