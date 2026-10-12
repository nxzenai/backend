"""Authenticated, owner-scoped learning API; existing lab routes are untouched."""
from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel, Field

from app.core.database import get_database
from app.modules.auth.dependencies import get_current_user
from app.modules.auth.models import UserModel

from .ai import LearningAIService
from .curriculum import ACTIVITY_BY_ID, ACTIVITIES, BY_ID, CASE_BY_ID, CASES, LABS, TOPICS, PYTHON_CHALLENGES, SQL_CHALLENGES
from .faculty import assignments_for, cohort_summary, create_assignment, faculty_scope, ASSIGNMENT_TYPES, PRACTICAL_TYPES
from .grading import grade_python, grade_sql, public_challenge
from .evidence import evidence_options, verify_evidence
from .lifecycle import state as capstone_state, submit_stage
from .portfolio import business_report, experiments, model_card, model_package
from .service import dashboard, path_detail, path_list, progress_for, public, record_score, score_profile, update_case

router = APIRouter(prefix="/learning", tags=["learning"])


def owner(user: UserModel) -> str:
    return user.id or str(user.email)


class ProgressInput(BaseModel):
    topic_id: str


class AnswerInput(BaseModel):
    topic_id: str
    answer_index: int | None = None
    answer_indices: list[int] | None = None
    text: str | None = Field(default=None, max_length=12000)
    activity_type: str = "mcq"
    lab_reference: str | None = Field(default=None, max_length=200)


class CaseProgressInput(BaseModel):
    completed_steps: list[str] = Field(default_factory=list, max_length=30)
    lab_references: list[str] = Field(default_factory=list, max_length=10)


class InterviewStartInput(BaseModel):
    module: str
    topic_id: str | None = None
    mode: str = Field(pattern="^(practice|mock)$")
    level: int = Field(default=4, ge=1, le=10)
    target_role: str = Field(default="", max_length=80)
    question_count: int = 5
    question_type: str | None = Field(default=None, pattern="^(THEORY|CODING|DEBUGGING|CASE_STUDY|RESULT_INTERPRETATION|ARCHITECTURE)$")


class InterviewAnswerInput(BaseModel):
    answer: str = Field(min_length=1, max_length=6000)


class AssessmentInput(BaseModel):
    activity_id: str
    selected_indices: list[int] = Field(min_length=1, max_length=10)


class ReviewInput(BaseModel):
    category: str = Field(pattern="^(coding|practical)$")
    score: int = Field(ge=0, le=100)
    feedback: str = Field(min_length=1, max_length=1000)


class CodeSubmission(BaseModel):
    code: str = Field(min_length=1, max_length=12000)


class QuerySubmission(BaseModel):
    query: str = Field(min_length=1, max_length=12000)


class AutoMLProof(BaseModel):
    topic_id: str
    filename: str = Field(min_length=1, max_length=255)


class EvidenceProof(BaseModel):
    topic_id: str
    reference_id: str = Field(min_length=1, max_length=255)


class CapstoneStageInput(BaseModel):
    draft: bool = False
    choice: str = Field(default="", max_length=80)
    metric: str = Field(default="", max_length=80)
    response: str = Field(default="", max_length=2000)
    reference: str = Field(default="", max_length=255)


class ActivityCheckInput(BaseModel):
    topic_id: str
    kind: str = Field(pattern="^(concept|reasoning|quiz|code_reading)$")
    answer_index: int = Field(ge=0)
    explanation: str = Field(default="", max_length=1000)


class CaseProof(BaseModel):
    attempt_id: str


class AssignmentInput(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    type: str
    reference_id: str
    batch_id: str | None = None
    due_date: datetime | None = None


def safe_session(item: dict) -> dict:
    value = public(item)
    value.pop("pending_answer", None)
    if value.get("current_question"):
        value["current_question"] = {key: field for key, field in value["current_question"].items()
                                     if key != "expected_concepts" and not (value.get("mode") == "mock" and key == "competencies")}
    hidden = {"expected_concepts"} | ({"evaluation", "score"} if value.get("mode") == "mock" and value.get("status") != "completed" else set())
    if value.get("mode") == "mock" and value.get("status") != "completed":
        value.pop("overall_score", None)
    value["answers"] = [{key: field for key, field in answer.items() if key not in hidden}
                        for answer in value.get("answers", [])]
    value.pop("retry_history", None)
    return value


@router.get("/paths")
async def paths(user: UserModel = Depends(get_current_user)):
    return path_list()


@router.get("/paths/{module}")
async def path(module: str, user: UserModel = Depends(get_current_user),
               db: AsyncIOMotorDatabase = Depends(get_database)):
    detail = path_detail(module)
    from .mastery_roadmaps import path_metadata
    from .topic_journey import roadmap_progress
    detail.update(path_metadata(module, detail["lab_url"]))
    known_groups = {group["id"] for group in detail["roadmap_groups"]}
    if any(topic.get("roadmap_group") not in known_groups for topic in detail["topics"]):
        detail["roadmap_groups"].append({"id": "additional", "title": "Additional Topics", "order": 999, "lifecycle_stages": []})
    detail["roadmap_progress"] = await roadmap_progress(db, owner(user), detail["topics"])
    return detail


@router.get("/paths/{module}/topics/{topic_id}")
async def lesson(module: str, topic_id: str, user: UserModel = Depends(get_current_user)):
    item = BY_ID.get(topic_id)
    if not item or item["module"] != module:
        raise HTTPException(404, "Lesson not found.")
    hidden = {"quick_check", "reasoning_check", "quiz_check", "code_check"}
    return {key: value for key, value in item.items() if key not in hidden} | {
        key: {field: value for field, value in item[key].items() if field != "answer_index"}
        for key in hidden if key in item}



class TopicReflectionInput(BaseModel):
    response: str = Field(min_length=10, max_length=2000)

class TopicTutorInput(BaseModel):
    question: str = Field(min_length=1, max_length=1000)

@router.get("/topics/{topic_id}/journey")
async def topic_journey(topic_id: str, db: AsyncIOMotorDatabase = Depends(get_database),
                        user: UserModel = Depends(get_current_user)):
    from .topic_journey import journey
    return await journey(db, owner(user), topic_id)

@router.post("/topics/{topic_id}/reflection")
async def topic_reflection(topic_id: str, payload: TopicReflectionInput,
                           db: AsyncIOMotorDatabase = Depends(get_database), user: UserModel = Depends(get_current_user)):
    if len(payload.response.strip()) < 10: raise HTTPException(422, "Write a short explanation first.")
    from .topic_journey import save_reflection
    return await save_reflection(db, owner(user), topic_id, payload.response)

@router.post("/topics/{topic_id}/tutor")
async def topic_tutor(topic_id: str, payload: TopicTutorInput,
                      db: AsyncIOMotorDatabase = Depends(get_database), user: UserModel = Depends(get_current_user)):
    from .topic_journey import tutor
    return await tutor(db, owner(user), topic_id, payload.question)


@router.get("/progress")
async def progress(db: AsyncIOMotorDatabase = Depends(get_database), user: UserModel = Depends(get_current_user)):
    return await progress_for(db, owner(user))


@router.get("/capstones/{topic_id}")
async def get_capstone(topic_id: str, db: AsyncIOMotorDatabase = Depends(get_database),
                       user: UserModel = Depends(get_current_user)):
    return await capstone_state(db, owner(user), topic_id)


@router.post("/capstones/{topic_id}/stages/{stage_id}")
async def complete_capstone_stage(topic_id: str, stage_id: str, payload: CapstoneStageInput,
                                  db: AsyncIOMotorDatabase = Depends(get_database),
                                  user: UserModel = Depends(get_current_user)):
    return await submit_stage(db, owner(user), topic_id, stage_id, draft=payload.draft,
                              choice=payload.choice, metric=payload.metric,
                              response=payload.response, reference=payload.reference)


@router.get("/capstones/{topic_id}/model-card")
async def capstone_model_card(topic_id: str, db: AsyncIOMotorDatabase = Depends(get_database),
                             user: UserModel = Depends(get_current_user)):
    return await model_card(db, owner(user), topic_id)


@router.get("/capstones/{topic_id}/report")
async def capstone_report(topic_id: str, db: AsyncIOMotorDatabase = Depends(get_database),
                          user: UserModel = Depends(get_current_user)):
    return await business_report(db, owner(user), topic_id)


@router.get("/capstones/{topic_id}/experiments")
async def capstone_experiments(topic_id: str, db: AsyncIOMotorDatabase = Depends(get_database),
                              user: UserModel = Depends(get_current_user)):
    return await experiments(db, owner(user), topic_id)


@router.get("/capstones/{topic_id}/model-package")
async def capstone_model_package(topic_id: str, db: AsyncIOMotorDatabase = Depends(get_database),
                                user: UserModel = Depends(get_current_user)):
    return await model_package(db, owner(user), topic_id)


@router.post("/progress")
async def start_topic(payload: ProgressInput, db: AsyncIOMotorDatabase = Depends(get_database),
                      user: UserModel = Depends(get_current_user)):
    if payload.topic_id not in BY_ID:
        raise HTTPException(404, "Lesson not found.")
    owner_id = owner(user)
    topic = BY_ID[payload.topic_id]
    if topic.get("is_capstone"):
        current = await progress_for(db, owner_id)
        missing = [ref for ref in topic["prerequisites"] if current.get(ref, {}).get("status") != "MASTERED"]
        if missing:
            raise HTTPException(409, "Master the listed prerequisites before starting this capstone.")
    elif topic.get("prerequisites"):
        current = await progress_for(db, owner_id)
        if any(((current.get(ref, {}).get("scores") or {}).get("theory") or 0) < 70
               for ref in topic["prerequisites"]):
            raise HTTPException(409, "Complete the prerequisite quick check before opening this topic.")
    now = datetime.now(UTC)
    await db.learning_progress.update_one({"owner_id": owner_id, "topic_id": payload.topic_id},
        {"$set": {"last_viewed_at": now, "updated_at": now},
         "$setOnInsert": {"_id": str(uuid4()), "owner_id": owner_id, "topic_id": payload.topic_id,
                          "module": BY_ID[payload.topic_id]["module"], "path_id": BY_ID[payload.topic_id]["module"],
                          "status": "IN_PROGRESS", "scores": {}, "best_score": 0, "mastery_score": 0, "attempts": 0,
                          "activity_started_at": now}},
        upsert=True)
    if topic.get("is_capstone"):
        await db.learning_progress.update_one({"owner_id": owner_id, "topic_id": payload.topic_id,
                                               "activity_started_at": {"$exists": False}},
                                              {"$set": {"activity_started_at": now}})
    return {"topic_id": payload.topic_id, "status": "IN_PROGRESS"}


@router.post("/activities/check")
async def check_activity(payload: ActivityCheckInput, db: AsyncIOMotorDatabase = Depends(get_database),
                         user: UserModel = Depends(get_current_user)):
    topic = BY_ID.get(payload.topic_id)
    if not topic or topic.get("is_capstone"):
        raise HTTPException(404, "Foundation activity not found.")
    key = {"concept": "quick_check", "reasoning": "reasoning_check", "quiz": "quiz_check",
           "code_reading": "code_check"}[payload.kind]
    check = topic.get(key)
    if not check or payload.answer_index >= len(check["choices"]):
        raise HTTPException(422, "Invalid activity choice.")
    if payload.kind == "reasoning" and not payload.explanation.strip():
        raise HTTPException(422, "Briefly explain your choice.")
    category = "theory" if payload.kind == "concept" else payload.kind
    correct = payload.answer_index == check["answer_index"]
    return await record_score(db, owner(user), payload.topic_id, category, 100 if correct else 0,
                              f"foundation_{payload.kind}", f"Choice {payload.answer_index}. {payload.explanation.strip()}",
                              ("Correct choice. Your explanation was recorded, not auto-graded." if payload.kind == "reasoning"
                               else "Correct choice.") if correct else "Review the lesson and try again.")


@router.post("/attempts")
async def submit_attempt(payload: AnswerInput, db: AsyncIOMotorDatabase = Depends(get_database),
                         user: UserModel = Depends(get_current_user)):
    item = BY_ID.get(payload.topic_id)
    if not item:
        raise HTTPException(404, "Lesson not found.")
    if payload.activity_type == "mcq" and payload.answer_index is not None:
        correct = payload.answer_index == item["quick_check"]["answer_index"]
        answer = item["quick_check"]["choices"][item["quick_check"]["answer_index"]]
        return await record_score(db, owner(user), payload.topic_id, "theory", 100 if correct else 0,
                                  "mcq", str(payload.answer_index),
                                  "Correct." if correct else f"Review the lesson. Correct answer: {answer}")
    if payload.activity_type in {"coding", "sql", "practical"} and payload.lab_reference:
        # A reference records practice, never a verified coding/practical score.
        now = datetime.now(UTC)
        record = {"_id": str(uuid4()), "owner_id": owner(user), "module": item["module"],
                  "topic_id": payload.topic_id, "activity_type": payload.activity_type,
                  "lab_reference": payload.lab_reference, "response": (payload.text or "")[:12000],
                  "verification_status": "unverified", "created_at": now}
        await db.learning_attempts.insert_one(record)
        return {"attempt": public(record), "feedback": "Lab work recorded. It needs verification before it contributes to mastery."}
    raise HTTPException(422, "Submit a quick-check choice or a lab reference for practice.")


@router.get("/attempts")
async def attempts(topic_id: str | None = None, db: AsyncIOMotorDatabase = Depends(get_database),
                   user: UserModel = Depends(get_current_user)):
    query = {"owner_id": owner(user)}
    if topic_id:
        query["topic_id"] = topic_id
    return [public(item) for item in await db.learning_attempts.find(query).sort("created_at", -1).limit(200).to_list(length=200)]


@router.get("/assessments")
async def assessments(module: str | None = None, user: UserModel = Depends(get_current_user)):
    if module and module not in LABS:
        raise HTTPException(404, "Learning path not found.")
    return [{key: value for key, value in item.items() if key not in {"correct_indices", "explanation"}}
            for item in ACTIVITIES if not module or item["module"] == module]


@router.post("/assessments")
async def submit_assessment(payload: AssessmentInput, db: AsyncIOMotorDatabase = Depends(get_database),
                            user: UserModel = Depends(get_current_user)):
    item = ACTIVITY_BY_ID.get(payload.activity_id)
    if not item:
        raise HTTPException(404, "Assessment activity not found.")
    if any(index < 0 or index >= len(item["choices"]) for index in payload.selected_indices):
        raise HTTPException(422, "Invalid answer choice.")
    correct = set(payload.selected_indices) == set(item["correct_indices"])
    # Assessment choices check understanding; case mastery needs verified lab evidence.
    category = "theory" if item["category"] in {"coding", "case"} else item["category"]
    result = await record_score(db, owner(user), item["topic_id"], category, 100 if correct else 0,
                                item["type"], str(sorted(set(payload.selected_indices))),
                                "Correct. " + item["explanation"] if correct else "Review this topic. " + item["explanation"],
                                source_reference=item["id"])
    return result


@router.get("/python-challenges/{topic_id}")
async def python_challenge(topic_id: str, user: UserModel = Depends(get_current_user)):
    spec = PYTHON_CHALLENGES.get(topic_id)
    if not spec:
        raise HTTPException(404, "Python challenge not found.")
    return {"topic_id": topic_id, **public_challenge(spec)}


@router.post("/python-challenges/{topic_id}/submit")
async def submit_python(topic_id: str, payload: CodeSubmission, db: AsyncIOMotorDatabase = Depends(get_database),
                        user: UserModel = Depends(get_current_user)):
    return await grade_python(db, user, topic_id, payload.code)


@router.get("/sql-challenges/{topic_id}")
async def sql_challenge(topic_id: str, user: UserModel = Depends(get_current_user)):
    spec = SQL_CHALLENGES.get(topic_id)
    if not spec:
        raise HTTPException(404, "SQL challenge not found.")
    return {"topic_id": topic_id, **public_challenge(spec)}


@router.post("/sql-challenges/{topic_id}/submit")
async def submit_sql(topic_id: str, payload: QuerySubmission, db: AsyncIOMotorDatabase = Depends(get_database),
                     user: UserModel = Depends(get_current_user)):
    return await grade_sql(db, user, topic_id, payload.query)


@router.post("/automl-evidence")
async def verify_automl(payload: AutoMLProof, db: AsyncIOMotorDatabase = Depends(get_database),
                        user: UserModel = Depends(get_current_user)):
    if BY_ID.get(payload.topic_id, {}).get("module") != "automl" or not user.id:
        raise HTTPException(422, "AutoML learning topic or owner is invalid.")
    return await verify_evidence(db, owner(user), payload.topic_id, payload.filename)


@router.get("/evidence/options")
async def lab_evidence_options(module: str, topic_id: str | None = None, db: AsyncIOMotorDatabase = Depends(get_database),
                               user: UserModel = Depends(get_current_user)):
    return await evidence_options(db, owner(user), module, topic_id)


@router.post("/evidence/verify")
async def verify_lab_evidence(payload: EvidenceProof, db: AsyncIOMotorDatabase = Depends(get_database),
                              user: UserModel = Depends(get_current_user)):
    return await verify_evidence(db, owner(user), payload.topic_id, payload.reference_id)


@router.get("/assignments")
async def my_assignments(db: AsyncIOMotorDatabase = Depends(get_database), user: UserModel = Depends(get_current_user)):
    return await assignments_for(db, user)


@router.get("/faculty/reference-options")
async def faculty_reference_options(type: str, user: UserModel = Depends(get_current_user)):
    faculty_scope(user)
    if type not in ASSIGNMENT_TYPES:
        raise HTTPException(422, "Unsupported assignment type.")
    if type == "assessment":
        return [{"id": item["id"], "label": f"{item['module']} · {item['prompt'][:90]}"} for item in ACTIVITIES]
    if type == "case_study":
        return [{"id": item["id"], "label": item["title"]} for item in CASES]
    if type == "mock_interview":
        return [{"id": module, "label": value[0]} for module, value in LABS.items()]
    module = PRACTICAL_TYPES[type][0] if type in PRACTICAL_TYPES else {"python_challenge": "python", "sql_challenge": "sql"}.get(type)
    allowed = PYTHON_CHALLENGES if type == "python_challenge" else SQL_CHALLENGES if type == "sql_challenge" else None
    return [{"id": item["id"], "label": f"{LABS[item['module']][0]} · {item['title']}"} for item in TOPICS
            if (not module or item["module"] == module) and (allowed is None or item["id"] in allowed)
            and (type not in {"capstone", "automl_practical", "autodl_practical", "autonlp_practical"} or item.get("is_capstone"))]


@router.get("/faculty/batches")
async def faculty_batches(db: AsyncIOMotorDatabase = Depends(get_database), user: UserModel = Depends(get_current_user)):
    scope = faculty_scope(user)
    values = await db.users.distinct("batch_id", {**scope, "role": {"$in": ["trainee", "user"]}})
    return [{"id": str(value), "label": str(value)} for value in values if value]


@router.post("/assignments")
async def assign(payload: AssignmentInput, db: AsyncIOMotorDatabase = Depends(get_database),
                 user: UserModel = Depends(get_current_user)):
    return await create_assignment(db, user, payload)


@router.get("/faculty/cohort-summary")
async def faculty_cohort_summary(batch_id: str | None = None, db: AsyncIOMotorDatabase = Depends(get_database),
                                 user: UserModel = Depends(get_current_user)):
    return await cohort_summary(db, user, batch_id)


@router.get("/mastery")
async def mastery(db: AsyncIOMotorDatabase = Depends(get_database), user: UserModel = Depends(get_current_user)):
    progress = await progress_for(db, owner(user))
    return {topic_id: {key: item[key] for key in ("scores", "mastery_score", "mastery_requirements", "verification_unavailable", "status")}
            for topic_id, item in progress.items()}


@router.get("/recommendations")
async def recommendations(db: AsyncIOMotorDatabase = Depends(get_database), user: UserModel = Depends(get_current_user)):
    summary = await dashboard(db, owner(user))
    return {"recommended_next": summary["recommended_next"], "weak_areas": summary["weak_areas"]}


@router.get("/dashboard")
async def learning_dashboard(db: AsyncIOMotorDatabase = Depends(get_database), user: UserModel = Depends(get_current_user)):
    return await dashboard(db, owner(user))


@router.get("/case-studies")
async def case_studies(db: AsyncIOMotorDatabase = Depends(get_database), user: UserModel = Depends(get_current_user)):
    records = await db.learning_projects.find({"owner_id": owner(user)}).limit(100).to_list(length=100)
    by_id = {item["case_id"]: {**public(item),
              "status": "completed" if item.get("verified_attempt_id") else "in_progress"} for item in records}
    return [{**case, "progress": by_id.get(case["id"])} for case in CASES]


@router.post("/case-studies/{case_id}/progress")
async def case_progress(case_id: str, payload: CaseProgressInput, db: AsyncIOMotorDatabase = Depends(get_database),
                        user: UserModel = Depends(get_current_user)):
    return await update_case(db, owner(user), case_id, payload.completed_steps, payload.lab_references)


@router.post("/case-studies/{case_id}/verify")
async def verify_case(case_id: str, payload: CaseProof, db: AsyncIOMotorDatabase = Depends(get_database),
                      user: UserModel = Depends(get_current_user)):
    case = CASE_BY_ID.get(case_id)
    if not case:
        raise HTTPException(404, "Case study not found.")
    attempt = await db.learning_attempts.find_one({"_id": payload.attempt_id, "owner_id": owner(user),
        "verified": True, "combined_score": {"$gte": 70}, "evidence_type": case["evidence_type"],
        "topic_id": case["topic_id"]})
    if not attempt:
        raise HTTPException(422, "Choose a verified result for this case.")
    if case.get("expected_task") and attempt.get("task_type") != case["expected_task"]:
        raise HTTPException(422, "The saved result has the wrong task for this case.")
    if case.get("compatible_tasks") and attempt.get("task_type") not in case["compatible_tasks"]:
        raise HTTPException(422, "The saved result has the wrong task for this case.")
    if case.get("expected_target") and str(attempt.get("target_column") or "").casefold() != case["expected_target"]:
        raise HTTPException(422, "The saved result target does not match this case.")
    if await db.learning_projects.find_one({"owner_id": owner(user), "case_id": {"$ne": case_id},
                                            "verified_attempt_id": payload.attempt_id}, {"_id": 1}):
        raise HTTPException(409, "This lab result was already submitted for another case.")
    result = await record_score(db, owner(user), case["topic_id"], "case", 100, "verified_case",
                                case["title"], "Case completed with verified lab evidence.",
                                source_reference=payload.attempt_id)
    await db.learning_projects.update_one({"owner_id": owner(user), "case_id": case_id},
        {"$set": {"module": case["module"], "status": "completed", "verified_attempt_id": payload.attempt_id,
                  "updated_at": datetime.now(UTC)}, "$setOnInsert": {"_id": str(uuid4()),
                  "owner_id": owner(user), "case_id": case_id, "created_at": datetime.now(UTC)}}, upsert=True)
    return result


@router.post("/interviews")
async def start_interview(payload: InterviewStartInput, db: AsyncIOMotorDatabase = Depends(get_database),
                          user: UserModel = Depends(get_current_user)):
    if payload.module not in LABS or payload.question_count not in {5, 10, 15}:
        raise HTTPException(422, "Choose a supported module and 5, 10, or 15 questions.")
    topic_id = payload.topic_id or next(item["id"] for item in TOPICS if item["module"] == payload.module)
    if topic_id not in BY_ID or BY_ID[topic_id]["module"] != payload.module:
        raise HTTPException(422, "Interview topic does not belong to this module.")
    now = datetime.now(UTC)
    item = {"_id": str(uuid4()), "owner_id": owner(user), "module": payload.module,
            "topic_id": topic_id, "mode": payload.mode, "target_role": payload.target_role,
            "question_type": payload.question_type,
            "difficulty": payload.level, "question_count": payload.question_count,
            "status": "in_progress", "current_question": None, "answers": [],
            "started_at": now, "updated_at": now}
    await db.interview_sessions.insert_one(item)
    return safe_session(item)


async def owned_session(db: AsyncIOMotorDatabase, owner_id: str, session_id: str) -> dict:
    item = await db.interview_sessions.find_one({"_id": session_id, "owner_id": owner_id})
    if not item:
        raise HTTPException(404, "Interview session not found.")
    return item


@router.post("/interviews/{session_id}/question")
async def next_question(session_id: str, db: AsyncIOMotorDatabase = Depends(get_database),
                        user: UserModel = Depends(get_current_user)):
    owner_id = owner(user)
    item = await owned_session(db, owner_id, session_id)
    if item["status"] == "completed":
        return safe_session(item)
    if item.get("current_question"):
        return safe_session(item)  # Cached until answered; no extra provider call.
    if len(item["answers"]) >= item["question_count"]:
        await db.interview_sessions.update_one({"_id": session_id, "owner_id": owner_id}, {"$set": {"status": "completed", "completed_at": datetime.now(UTC)}})
        return safe_session(await owned_session(db, owner_id, session_id))
    progress = await progress_for(db, owner_id)
    weak = [key for key, value in (progress.get(item["topic_id"], {}).get("scores") or {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value < 60]
    question = await LearningAIService().generate_interview_question(item["module"], item["difficulty"],
                     item.get("target_role", ""), weak, [answer["question"] for answer in item["answers"]],
                     item.get("question_type"))
    question["id"] = str(uuid4())
    question["difficulty"] = item["difficulty"]
    await db.interview_sessions.update_one({"_id": session_id, "owner_id": owner_id, "current_question": None},
        {"$set": {"current_question": question, "updated_at": datetime.now(UTC)}})
    return safe_session(await owned_session(db, owner_id, session_id))


@router.post("/interviews/{session_id}/answer")
async def submit_interview_answer(session_id: str, payload: InterviewAnswerInput,
                                  db: AsyncIOMotorDatabase = Depends(get_database),
                                  user: UserModel = Depends(get_current_user)):
    owner_id = owner(user)
    item = await owned_session(db, owner_id, session_id)
    question = item.get("current_question")
    if not question or item["status"] != "in_progress":
        raise HTTPException(409, "Request the next question first.")
    pending = item.get("pending_answer")
    if pending and pending.get("question_id") != question["id"]:
        raise HTTPException(409, "Another answer is pending.")
    answer_text = pending["text"] if pending else payload.answer
    await db.interview_sessions.update_one({"_id": session_id, "owner_id": owner_id},
        {"$set": {"pending_answer": {"question_id": question["id"], "text": answer_text}, "updated_at": datetime.now(UTC)}})
    try:
        evaluation = await LearningAIService().evaluate_open_answer(question, answer_text)
    except HTTPException:
        await db.interview_sessions.update_one({"_id": session_id, "owner_id": owner_id},
            {"$set": {"evaluation_status": "provider_failed"}})
        return {"status": "provider_failed", "message": "Your answer is saved. Retry evaluation when inference is available."}
    answer = {"id": str(uuid4()), "question_id": question["id"], "question": question["question"],
              "type": question["type"], "difficulty": question["difficulty"],
              "competencies": question.get("competencies", []),
              "expected_concepts": question.get("expected_concepts", []),
              "user_answer": answer_text, "evaluation": evaluation, "score": evaluation["score"] * 10,
              "created_at": datetime.now(UTC)}
    answers = [*item["answers"], answer]
    next_level = max(1, min(10, item["difficulty"] + (1 if evaluation["score"] >= 8 else -1 if evaluation["score"] <= 4 else 0)))
    complete = len(answers) >= item["question_count"]
    await db.interview_sessions.update_one({"_id": session_id, "owner_id": owner_id, "current_question.id": question["id"]},
        {"$set": {"answers": answers, "current_question": None, "pending_answer": None,
                  "evaluation_status": "completed", "difficulty": next_level,
                  "status": "completed" if complete else "in_progress",
                  "overall_score": round(sum(a["score"] for a in answers) / len(answers)),
                  "updated_at": datetime.now(UTC), "completed_at": datetime.now(UTC) if complete else None}})
    await record_score(db, owner_id, item["topic_id"], "interview", answer["score"],
                       "interview", answer_text, evaluation["feedback"])
    return {"answer": {key: value for key, value in answer.items() if key != "expected_concepts"} if item["mode"] == "practice" or complete else None,
            "session": safe_session(await owned_session(db, owner_id, session_id))}


@router.post("/interviews/{session_id}/retry")
async def retry_practice_question(session_id: str, db: AsyncIOMotorDatabase = Depends(get_database),
                                  user: UserModel = Depends(get_current_user)):
    item = await owned_session(db, owner(user), session_id)
    if item["mode"] != "practice" or item.get("current_question") or not item.get("answers"):
        raise HTTPException(409, "Only the most recently answered practice question can be retried.")
    previous = item["answers"][-1]
    question = {"id": str(uuid4()), "question": previous["question"], "type": previous["type"],
                "difficulty": previous["difficulty"], "competencies": previous.get("competencies", []),
                "expected_concepts": previous.get("expected_concepts", [])}
    remaining = item["answers"][:-1]
    await db.interview_sessions.update_one({"_id": session_id, "owner_id": owner(user), "current_question": None},
        {"$set": {"answers": remaining, "current_question": question, "difficulty": previous["difficulty"],
                  "status": "in_progress", "completed_at": None,
                  "overall_score": round(sum(a["score"] for a in remaining) / len(remaining)) if remaining else None,
                  "updated_at": datetime.now(UTC)}, "$push": {"retry_history": previous}})
    return safe_session(await owned_session(db, owner(user), session_id))


@router.get("/interviews/history")
async def interview_history(db: AsyncIOMotorDatabase = Depends(get_database), user: UserModel = Depends(get_current_user)):
    items = await db.interview_sessions.find({"owner_id": owner(user)}, {"answers": 0, "current_question": 0, "pending_answer": 0, "retry_history": 0}).sort("started_at", -1).limit(30).to_list(length=30)
    return [{key: value for key, value in public(item).items()
             if not (item.get("mode") == "mock" and item.get("status") != "completed" and key == "overall_score")}
            for item in items]


@router.get("/interviews/{session_id}")
async def get_interview(session_id: str, db: AsyncIOMotorDatabase = Depends(get_database),
                        user: UserModel = Depends(get_current_user)):
    return safe_session(await owned_session(db, owner(user), session_id))


@router.get("/interview-readiness")
async def interview_readiness(db: AsyncIOMotorDatabase = Depends(get_database), user: UserModel = Depends(get_current_user)):
    items = await db.interview_sessions.find({"owner_id": owner(user), "status": "completed"},
                                             {"module": 1, "overall_score": 1, "answers": 1}).sort("completed_at", -1).limit(30).to_list(length=30)
    by_module = {}
    by_competency = {}
    for item in items:
        by_module.setdefault(item["module"], []).append(item.get("overall_score") or 0)
        for answer in item.get("answers") or []:
            by_competency.setdefault(item["module"], {}).setdefault(answer.get("type", "THEORY"), []).append(answer.get("score") or 0)
    return {module: {"score": round(sum(scores) / len(scores)), "sessions": len(scores),
                     "competencies": {key: round(sum(values) / len(values)) for key, values in by_competency.get(module, {}).items()},
                     "weakest_competency": min(by_competency.get(module, {}),
                                               key=lambda key: sum(by_competency[module][key]) / len(by_competency[module][key]),
                                               default=None)} for module, scores in by_module.items()}


@router.get("/faculty/overview")
async def faculty_overview(db: AsyncIOMotorDatabase = Depends(get_database), user: UserModel = Depends(get_current_user)):
    faculty_scope(user)
    summary = await cohort_summary(db, user)
    return {"learner_count": summary["learner_count"], "module_averages": summary["module_averages"],
            "interview_readiness": summary["interview_readiness_average"],
            "assessment_average": None, "assessment_participation": summary["assessment_participation"],
            "common_weak_topics": summary["common_weak_topics"]}


@router.get("/faculty/reviews")
async def pending_reviews(db: AsyncIOMotorDatabase = Depends(get_database), user: UserModel = Depends(get_current_user)):
    scope = faculty_scope(user)
    learners = await db.users.find({**scope,
                                    "role": {"$in": ["trainee", "user"]}}, {"_id": 1}).limit(1000).to_list(length=1000)
    ids = [str(item["_id"]) for item in learners]
    items = await db.learning_attempts.find({"owner_id": {"$in": ids}, "verification_status": "unverified"}).sort("created_at", 1).limit(50).to_list(length=50)
    return [public(item) for item in items]


@router.post("/faculty/reviews/{attempt_id}")
async def review_attempt(attempt_id: str, payload: ReviewInput,
                         db: AsyncIOMotorDatabase = Depends(get_database),
                         user: UserModel = Depends(get_current_user)):
    faculty_scope(user)
    attempt = await db.learning_attempts.find_one({"_id": attempt_id, "verification_status": "unverified"})
    if not attempt:
        raise HTTPException(404, "Pending attempt not found.")
    if BY_ID.get(attempt.get("topic_id"), {}).get("is_capstone") and payload.category == "practical":
        raise HTTPException(422, "Use saved-result verification for capstone practical work.")
    try:
        learner_id = ObjectId(attempt["owner_id"])
    except Exception as exc:
        raise HTTPException(404, "Learner not found.") from exc
    scope = {"_id": learner_id, "organization_id": user.organization_id}
    if user.role == "trainer" and user.batch_id:
        scope["batch_id"] = user.batch_id
    learner = await db.users.find_one({**scope,
                                       "role": {"$in": ["trainee", "user"]}}, {"_id": 1})
    if not learner:
        raise HTTPException(404, "Pending attempt not found.")
    result = await db.learning_attempts.update_one({"_id": attempt_id, "verification_status": "unverified"},
        {"$set": {"verification_status": "reviewed", "reviewed_by": owner(user),
                  "reviewed_at": datetime.now(UTC), "review_score": payload.score,
                  "review_feedback": payload.feedback}})
    if result.modified_count != 1:
        raise HTTPException(409, "This attempt was already reviewed.")
    return await record_score(db, attempt["owner_id"], attempt["topic_id"], payload.category,
                              payload.score, "faculty_review", attempt.get("lab_reference", ""), payload.feedback)
