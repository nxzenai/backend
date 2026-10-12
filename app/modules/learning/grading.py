"""Owner-scoped learning checks using the existing lab execution services."""
from __future__ import annotations

from collections import Counter
from decimal import Decimal
import re
from uuid import uuid4

from fastapi import HTTPException

from app.core.config.settings import settings
from app.modules.execution.dependencies import get_kernel_manager
from app.modules.execution.exceptions import HostExecutionDisabled
from app.modules.execution.models import ExecutionOutputType
from app.modules.sql.dependencies import get_sql_service

from .curriculum import PYTHON_CHALLENGES, SQL_CHALLENGES
from .service import record_score


def public_challenge(spec: dict) -> dict:
    return {key: value for key, value in spec.items() if key not in {"hidden_tests", "expected_rows", "expected_columns", "fixture_cte"}}


def python_grading_capability() -> dict:
    # The current Python Lab uses a host Jupyter kernel. No isolated production
    # executor is wired; configured runner images and subprocess helpers are not sandboxes.
    if settings.environment.lower() == "production":
        return {"grading_available": False,
                "reason": "Secure Python execution is not enabled for this environment."}
    return {"grading_available": True}


async def grade_python(db, user, topic_id: str, code: str) -> dict:
    spec = PYTHON_CHALLENGES.get(topic_id)
    if not spec:
        raise HTTPException(404, "Python challenge not found.")
    if not user.id:
        raise HTTPException(401, "Authenticated learner identity is unavailable.")
    capability = python_grading_capability()
    if not capability["grading_available"]:
        return capability
    owner_id = str(user.id)
    if await db.learning_attempts.count_documents({"owner_id": owner_id, "topic_id": topic_id, "activity_type": "python_execution"}) >= spec["max_attempts"]:
        raise HTTPException(409, "Maximum attempts reached for this challenge.")
    manager = get_kernel_manager()
    kernel_id = str(uuid4())
    passed = 0
    execution_success = False
    feedback = "Code execution failed. Review your code and try again."
    try:
        await manager.start_kernel(kernel_id)
        outputs, _ = await manager.execute(kernel_id, code, timeout=10)
        execution_success = not any(item.output_type == ExecutionOutputType.ERROR or item.metadata.get("truncated") for item in outputs)
        if execution_success:
            for check in spec["hidden_tests"]:
                outputs, _ = await manager.execute(kernel_id, check, timeout=10)
                if not any(item.output_type == ExecutionOutputType.ERROR or item.metadata.get("truncated") for item in outputs):
                    passed += 1
            feedback = f"Passed {passed} of {len(spec['hidden_tests'])} checks."
    except HostExecutionDisabled:
        return {"grading_available": False,
                "reason": "Secure Python execution is not enabled for this environment."}
    except Exception:
        # Runtime availability and execution errors are failed attempts, never verified passes.
        execution_success = False
        passed = 0
        feedback = "Python Lab execution was unavailable or failed."
    finally:
        if manager.kernel_exists(kernel_id):
            try:
                await manager.shutdown_kernel(kernel_id)
            except Exception:
                pass
    score = round(spec["score"] * passed / len(spec["hidden_tests"])) if execution_success else 0
    result = await record_score(db, owner_id, topic_id, "coding", score, "python_execution", code, feedback,
                                evidence_type="PYTHON_EXECUTION", source_reference=kernel_id)
    return {**result, "grading_available": True, "execution_success": execution_success, "passed": passed,
            "total": len(spec["hidden_tests"]), "verified_score": score}


def _scalar(value):
    if value is None:
        return ("null", "")
    if isinstance(value, bool):
        return ("bool", str(value))
    if isinstance(value, (int, float, Decimal)):
        number = Decimal(str(value))
        if number.is_nan() or number.is_infinite():
            return ("nonfinite", str(number))
        return ("number", str((Decimal(0) if number == 0 else number).normalize()))
    if isinstance(value, str):
        return ("text", value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return ("blob", bytes(value).hex())
    return (type(value).__name__, repr(value))


def _rows(rows):
    return [tuple(_scalar(value) for value in row) for row in rows]


async def grade_sql(db, user, topic_id: str, query: str) -> dict:
    spec = SQL_CHALLENGES.get(topic_id)
    if not spec:
        raise HTTPException(404, "SQL challenge not found.")
    if not user.id:
        raise HTTPException(401, "Authenticated learner identity is unavailable.")
    owner_id = str(user.id)
    if await db.learning_attempts.count_documents({"owner_id": owner_id, "topic_id": topic_id, "activity_type": "sql_execution"}) >= spec["max_attempts"]:
        raise HTTPException(409, "Maximum attempts reached for this challenge.")
    # Grading is read-only even though the ordinary SQL Lab supports mutations.
    read_only = bool(re.match(r"(?is)^\s*select\b", query)) and ";" not in query.strip().rstrip(";")
    passed = False
    feedback = "Query execution failed. Review the SQL and try again."
    try:
        if not read_only:
            feedback = "Submit a single read-only SELECT query."
        else:
            # A fixed CTE shadows the mutable lab table while retaining SQLService
            # validation, owner context, and the existing SQL execution engine.
            result = get_sql_service().execute(user, spec["fixture_cte"] + query.strip().rstrip(";"))
            columns_ok = result["columns"] == spec["expected_columns"]
            actual = _rows(result["rows"])
            expected = _rows(spec["expected_rows"])
            rows_ok = actual == expected if spec["order_sensitive"] else Counter(actual) == Counter(expected)
            passed = columns_ok and rows_ok
            feedback = "Result matched expected output." if passed else (
                "Column names or order differ from the required schema." if not columns_ok else "Returned rows do not match the expected result.")
    except Exception:
        pass
    score = spec["score"] if passed else 0
    result = await record_score(db, owner_id, topic_id, "coding", score, "sql_execution", query, feedback,
                                evidence_type="SQL_EXECUTION", source_reference=topic_id)
    return {**result, "passed": passed, "verified_score": score, "feedback": feedback}
