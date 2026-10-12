"""Deterministic assessment, progress, mastery, and recommendations."""
from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from fastapi import HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.core.config.settings import settings

from .curriculum import BY_ID, CASE_BY_ID, CASES, LABS, TOPICS

WEIGHTS = {"theory": 20, "reasoning": 20, "quiz": 20, "code_reading": 15,
           "coding": 25, "practical": 25, "case": 15, "interview": 15}
MASTERY_MINIMUMS = {"theory": 70, "reasoning": 70, "quiz": 70, "code_reading": 70,
                    "coding": 70, "practical": 70, "case": 70, "interview": 65}
CATEGORIES = tuple(WEIGHTS)


def public(item: dict) -> dict:
    return {"id": str(item["_id"]), **{key: value for key, value in item.items() if key not in {"_id", "owner_id"}}}


def score_profile(scores: dict, topic_id: str | None = None) -> dict:
    observed = {key: max(0, min(100, int(scores[key]))) for key in CATEGORIES if scores.get(key) is not None}
    if not scores.get("case_verified"):
        observed.pop("case", None)
    requirements = dict(BY_ID.get(topic_id, {}).get("mastery_requirements") or {"theory": 70})
    unavailable = []
    if topic_id in {"python-basics", "python-collections", "python-pandas"} and settings.environment.lower() == "production":
        requirements.pop("coding", None)
        unavailable.append("coding")
    total_weight = sum(WEIGHTS[key] for key in requirements)
    overall = round(sum(observed.get(key, 0) * WEIGHTS[key] for key in requirements) / total_weight)
    mastered = all(observed.get(key, -1) >= threshold for key, threshold in requirements.items())
    if BY_ID.get(topic_id, {}).get("is_capstone"):
        mastered = mastered and scores.get("lifecycle_complete") is True
    return {"scores": {key: observed.get(key) for key in CATEGORIES}, "mastery_score": overall,
            "mastery_requirements": requirements, "verification_unavailable": unavailable,
            "status": "MASTERED" if mastered else "IN_PROGRESS" if observed else "NOT_STARTED"}


def recommend(profile: dict) -> str:
    values = profile["scores"]
    if values["coding"] is not None and values["coding"] < 60:
        return "Complete a coding challenge in Python or SQL Lab."
    if values["interview"] is not None and values["interview"] < 60 and (values["theory"] or 0) >= 75:
        return "Practice explaining your reasoning in an interview."
    if values["practical"] is not None and values["practical"] >= 70 and (values["theory"] or 0) < 60:
        return "Review the underlying concepts before the next lab."
    return "Continue the next available lesson and complete its quick check."


async def progress_for(db: AsyncIOMotorDatabase, owner_id: str) -> dict[str, dict]:
    items = await db.learning_progress.find({"owner_id": owner_id}).limit(500).to_list(length=500)
    verified = await db.learning_attempts.find({"owner_id": owner_id, "category": {"$in": ["coding", "practical"]},
        "$or": [{"verified": True}, {"activity_type": "faculty_review"}]},
        {"topic_id": 1, "category": 1, "combined_score": 1, "source_reference": 1,
         "evidence_type": 1, "created_at": 1}).sort([("created_at", 1), ("_id", 1)]).limit(2000).to_list(length=2000)
    best = {}
    claimed = {}
    for attempt in verified:
        source = attempt.get("source_reference")
        if attempt.get("category") == "practical" and source and attempt.get("evidence_type"):
            claim = (attempt["evidence_type"], source)
            prior = claimed.setdefault(claim, attempt.get("topic_id"))
            if prior != attempt.get("topic_id") and not BY_ID.get(attempt.get("topic_id"), {}).get("allow_practical_reuse", False):
                continue
        key = (attempt.get("topic_id"), attempt.get("category"))
        best[key] = max(best.get(key, 0), attempt.get("combined_score") or 0)
    result = {}
    for item in items:
        value = public(item)
        scores = dict(value.get("scores") or {})
        for category in ("coding", "practical"):
            scores[category] = best.get((item["topic_id"], category))
        value["scores"] = scores
        value.update(score_profile({**scores, "lifecycle_complete": item.get("lifecycle_complete") is True}, item["topic_id"]))
        if BY_ID.get(item["topic_id"], {}).get("is_capstone") and not item.get("lifecycle_complete"):
            from .lifecycle import completion_percent
            value["mastery_score"] = completion_percent(item["topic_id"], item.get("lifecycle_stages") or {})
        result[item["topic_id"]] = value
    return result


async def record_score(db: AsyncIOMotorDatabase, owner_id: str, topic_id: str, category: str,
                       score: int, activity_type: str, response: str, feedback: str,
                       *, evidence_type: str | None = None, source_reference: str | None = None,
                       attempt_id: str | None = None, evidence_metadata: dict | None = None) -> dict:
    topic = BY_ID.get(topic_id)
    if not topic or category not in CATEGORIES:
        raise HTTPException(404, "Learning topic or score category not found.")
    if topic.get("is_capstone") and category == "practical" and activity_type != "verified_lab_result":
        raise HTTPException(422, "Capstone practical mastery requires a qualified saved lab result.")
    if category in {"coding", "practical"} and not evidence_type and activity_type != "faculty_review":
        raise HTTPException(422, "Verified lab evidence is required for coding or practical mastery.")
    if not topic.get("is_capstone") and topic.get("prerequisites"):
        prior = await progress_for(db, owner_id)
        if any(((prior.get(ref, {}).get("scores") or {}).get("theory") or 0) < 70
               for ref in topic["prerequisites"]):
            raise HTTPException(409, "Complete the prerequisite quick check before this topic.")
    now = datetime.now(UTC)
    attempt = {"_id": attempt_id or str(uuid4()), "owner_id": owner_id, "module": topic["module"],
               "topic_id": topic_id, "activity_type": activity_type, "category": category,
               "response": response[:12000], "deterministic_score": score if activity_type != "interview" else None,
               "ai_score": score if activity_type == "interview" else None,
               "combined_score": score, "feedback": feedback[:2000], "created_at": now,
               "verified": bool(evidence_type or activity_type == "faculty_review"),
               "verification_status": "verified" if evidence_type or activity_type == "faculty_review" else None,
               "evidence_type": evidence_type or ("FACULTY_REVIEW" if activity_type == "faculty_review" else None),
               "source_module": (evidence_type or "").split("_")[0].lower() if evidence_type else None,
               "source_reference": source_reference, **(evidence_metadata or {})}
    await db.learning_attempts.insert_one(attempt)
    current = await db.learning_progress.find_one({"owner_id": owner_id, "topic_id": topic_id})
    scores = dict(current.get("scores") or {}) if current else {}
    if category == "case" and activity_type == "verified_case":
        scores["case_verified"] = True
    if category not in {"coding", "practical"}:
        scores[category] = max(score, scores.get(category) or 0)
    verified = await db.learning_attempts.find({"owner_id": owner_id, "topic_id": topic_id,
        "category": {"$in": ["coding", "practical"]},
        "$or": [{"verified": True}, {"activity_type": "faculty_review"}]},
        {"category": 1, "combined_score": 1}).limit(2000).to_list(length=2000)
    for practical_category in ("coding", "practical"):
        values = [entry.get("combined_score") or 0 for entry in verified if entry.get("category") == practical_category]
        scores[practical_category] = max(values) if values else None
    profile = score_profile({**scores, "lifecycle_complete": bool(current and current.get("lifecycle_complete"))}, topic_id)
    await db.learning_progress.update_one(
        {"owner_id": owner_id, "topic_id": topic_id},
        {"$set": {"module": topic["module"], "path_id": topic["module"], "scores": scores,
                  "best_score": max(score, current.get("best_score") or 0) if current else score,
                  "mastery_score": profile["mastery_score"], "status": profile["status"],
                  "updated_at": now, "completed_at": now if profile["status"] == "MASTERED" else None},
         "$inc": {"attempts": 1}, "$setOnInsert": {"_id": str(uuid4()), "owner_id": owner_id, "topic_id": topic_id}},
        upsert=True,
    )
    return {"attempt": public(attempt), "progress": profile, "recommendation": recommend(profile)}


async def dashboard(db: AsyncIOMotorDatabase, owner_id: str) -> dict:
    progress = await progress_for(db, owner_id)
    # Prompt Lab evaluations remain in their own collection; consume only evaluated attempts.
    prompt_attempts = await db.genai_prompt_practice_attempts.find(
        {"owner_id": owner_id, "evaluation_score": {"$gte": 0}},
        {"evaluation_score": 1},
    ).sort("created_at", -1).limit(100).to_list(length=100)
    modules = []
    for module, (title, lab_url, _) in LABS.items():
        topics = [item for item in TOPICS if item["module"] == module]
        score = round(sum(progress.get(item["id"], {}).get("mastery_score", 0) for item in topics) / len(topics)) if topics else 0
        next_topic = next((item for item in topics if progress.get(item["id"], {}).get("status") != "MASTERED"), None)
        modules.append({"module": module, "title": title, "lab_url": lab_url, "mastery_score": score,
                        "current_level": next_topic["level"] if next_topic else "MASTERED",
                        "next_topic": {"id": next_topic["id"], "title": next_topic["title"]} if next_topic else None,
                        "mastered": sum(progress.get(item["id"], {}).get("status") == "MASTERED" for item in topics),
                        "topic_count": len(topics),
                        "prompt_lab_evaluated_attempts": len(prompt_attempts) if module == "prompting" else None})
    weighted = round(sum(item["mastery_score"] for item in modules) / len(modules)) if modules else 0
    weak = sorted((item for item in modules if item["mastery_score"] < 70), key=lambda item: item["mastery_score"])[:3]
    recent = await db.learning_attempts.find({"owner_id": owner_id}).sort("created_at", -1).limit(5).to_list(length=5)
    verified_practicals = [item for item in progress.values()
                           if (item.get("scores") or {}).get("practical") is not None
                           and item["scores"]["practical"] >= 70]
    return {"overall_readiness": weighted, "modules": modules, "weak_areas": weak,
            "recent_assessments": [public(item) for item in recent],
            "verified_practical_topics": len(verified_practicals),
            "recommended_next": next((item["next_topic"] for item in modules if item["next_topic"]), None)}


def path_list() -> list[dict]:
    return [{"id": module, "title": title, "lab_url": lab_url,
             "topic_count": sum(item["module"] == module for item in TOPICS)}
            for module, (title, lab_url, _) in LABS.items()]


def path_detail(module: str) -> dict:
    if module not in LABS:
        raise HTTPException(404, "Learning path not found.")
    title, lab_url, _ = LABS[module]
    return {"id": module, "title": title, "lab_url": lab_url,
            "topics": [{key: value for key, value in item.items() if key not in {"quick_check", "reasoning_check", "quiz_check", "code_check", "example", "theory_sections", "common_mistake"}}
                       for item in TOPICS if item["module"] == module]}


async def update_case(db: AsyncIOMotorDatabase, owner_id: str, case_id: str, completed_steps: list[str],
                      lab_references: list[str]) -> dict:
    case = CASE_BY_ID.get(case_id)
    if not case:
        raise HTTPException(404, "Case study not found.")
    allowed = set(case["steps"])
    if not set(completed_steps) <= allowed:
        raise HTTPException(422, "Unknown case-study step.")
    now = datetime.now(UTC)
    previous = await db.learning_projects.find_one({"owner_id": owner_id, "case_id": case_id}, {"lab_references": 1, "verified_attempt_id": 1})
    references = [value[:200] for value in lab_references[:10]] if lab_references else (previous or {}).get("lab_references", [])
    verified = bool((previous or {}).get("verified_attempt_id"))
    await db.learning_projects.update_one({"owner_id": owner_id, "case_id": case_id},
        {"$set": {"module": case["module"], "completed_steps": completed_steps,
                  "lab_references": references,
                  "status": "completed" if verified else "in_progress", "updated_at": now},
         "$setOnInsert": {"_id": str(uuid4()), "owner_id": owner_id, "case_id": case_id, "created_at": now}},
        upsert=True)
    return {"case_id": case_id, "completed_steps": completed_steps,
            "status": "completed" if verified else "in_progress"}
