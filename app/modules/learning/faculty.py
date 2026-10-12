"""Organization-scoped learning assignments and bounded cohort summaries."""
from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from fastapi import HTTPException

from .curriculum import ACTIVITY_BY_ID, BY_ID, CASE_BY_ID, LABS, TOPICS, PYTHON_CHALLENGES, SQL_CHALLENGES
from .service import public, score_profile, progress_for
from .lifecycle import completion_percent, definition

ASSIGNMENT_TYPES = {"learning_topic", "capstone", "assessment", "python_challenge", "sql_challenge",
                    "case_study", "mock_interview", "automl_practical", "autodl_practical",
                    "autonlp_practical", "eda_practical", "genai_practical", "prompting_practical"}
PRACTICAL_TYPES = {"automl_practical": ("automl", "AUTOML_RUN"),
                   "autodl_practical": ("autodl", "AUTODL_RUN"),
                   "autonlp_practical": ("autonlp", "AUTONLP_RUN"),
                   "eda_practical": ("eda", "EDA_REPORT"),
                   "genai_practical": ("genai", "GENAI_ACTIVITY"),
                   "prompting_practical": ("prompting", "PROMPT_LAB_RUN")}


def overdue(due_date):
    if not due_date:
        return False
    return (due_date.replace(tzinfo=UTC) if due_date.tzinfo is None else due_date) < datetime.now(UTC)


def faculty_scope(user):
    if user.role not in {"trainer", "admin", "super_admin"} or not user.organization_id or not user.id:
        raise HTTPException(403, "Organization-scoped faculty access is required.")
    if user.role == "trainer" and not user.batch_id:
        raise HTTPException(403, "A trainer batch is required for learner access.")
    return {"organization_id": user.organization_id, **({"batch_id": user.batch_id} if user.role == "trainer" and user.batch_id else {})}


def validate_reference(kind: str, reference: str) -> None:
    if kind not in ASSIGNMENT_TYPES:
        raise HTTPException(422, "Unsupported assignment type.")
    if kind in {"learning_topic", "capstone", "python_challenge", "sql_challenge"} or kind in PRACTICAL_TYPES:
        item = BY_ID.get(reference)
        expected = {"python_challenge": "python", "sql_challenge": "sql", **{key: value[0] for key, value in PRACTICAL_TYPES.items()}}.get(kind)
        if not item or (expected and item["module"] != expected):
            raise HTTPException(422, "Invalid learning topic for assignment type.")
        if (kind == "capstone" or kind in {"automl_practical", "autodl_practical", "autonlp_practical"}) and not item.get("is_capstone"):
            raise HTTPException(422, "Choose a capstone for this assignment.")
        if (kind == "python_challenge" and reference not in PYTHON_CHALLENGES) or (kind == "sql_challenge" and reference not in SQL_CHALLENGES):
            raise HTTPException(422, "Executable challenge not found.")
    elif kind == "assessment" and reference not in ACTIVITY_BY_ID:
        raise HTTPException(422, "Assessment not found.")
    elif kind == "case_study" and reference not in CASE_BY_ID:
        raise HTTPException(422, "Case study not found.")
    elif kind == "mock_interview" and reference not in {item["module"] for item in BY_ID.values()}:
        raise HTTPException(422, "Interview learning path not found.")


def capstone_lifecycle_summary(rows: list[dict], learner_count: int, topic_ids: list[str]) -> dict:
    selected = [row for row in rows if row.get("topic_id") in topic_ids]
    started = [row for row in selected if row.get("activity_started_at")]
    required = list(dict.fromkeys(item["id"] for topic_id in topic_ids
                                  for item in definition(topic_id)["stages"] if item["required"]))
    gaps = {stage: sum(row.get("lifecycle_stages", {}).get(stage, {}).get("status") != "COMPLETED"
                       for row in started) for stage in required}
    return {"started": len(started),
            "completed": sum(row.get("lifecycle_complete") is True for row in selected),
            "average_completion_percent": round(sum(completion_percent(row["topic_id"],
                row.get("lifecycle_stages") or {}, row.get("lifecycle_complete") is True)
                for row in selected) / (learner_count * len(topic_ids))),
            "most_incomplete_stage": max(gaps, key=gaps.get) if gaps and max(gaps.values()) else None}


async def cohort_users(db, user, batch_id=None):
    scope = faculty_scope(user)
    if batch_id:
        if user.role == "trainer" and user.batch_id and batch_id != user.batch_id:
            raise HTTPException(403, "Batch is outside your scope.")
        scope["batch_id"] = batch_id
    members = await db.users.find({**scope, "role": {"$in": ["trainee", "user"]}},
                                  {"_id": 1, "id": 1, "batch_id": 1}).limit(1000).to_list(length=1000)
    return [item.get("id") or str(item["_id"]) for item in members]


async def assignment_status(db, assignment, owner_id):
    kind, ref = assignment["type"], assignment["reference_id"]
    match = {"owner_id": owner_id}
    if kind in {"learning_topic", "capstone"}:
        record = await db.learning_progress.find_one({**match, "topic_id": ref}, {"scores": 1, "lifecycle_complete": 1})
        done = bool(record and (await progress_for(db, owner_id)).get(ref, {}).get("status") == "MASTERED")
        started = bool(record)
    elif kind == "case_study":
        record = await db.learning_projects.find_one({**match, "case_id": ref}, {"status": 1, "verified_attempt_id": 1})
        done = bool(record and record.get("verified_attempt_id"))
        started = bool(record)
    elif kind == "mock_interview":
        record = await db.interview_sessions.find_one({**match, "module": ref, "mode": "mock", "status": "completed"}, {"_id": 1})
        done = bool(record)
        started = done or bool(await db.interview_sessions.find_one({**match, "module": ref, "mode": "mock"}, {"_id": 1}))
    else:
        query = {**match, "source_reference": ref} if kind == "assessment" else {**match, "topic_id": ref}
        if kind == "python_challenge": query["activity_type"] = "python_execution"
        if kind == "sql_challenge": query["activity_type"] = "sql_execution"
        if kind in PRACTICAL_TYPES:
            query["evidence_type"] = PRACTICAL_TYPES[kind][1]
            if kind in {"automl_practical", "autodl_practical", "autonlp_practical"} and not BY_ID.get(ref, {}).get("is_capstone"):
                return "OVERDUE" if overdue(assignment.get("due_date")) else "NOT_STARTED"
        record = await db.learning_attempts.find_one(query, {"_id": 1})
        started = bool(record)
        passed_query = {**query, "combined_score": {"$gte": 70}}
        if kind != "assessment": passed_query["verified"] = True
        done = bool(await db.learning_attempts.find_one(passed_query, {"_id": 1}))
        if kind in PRACTICAL_TYPES:
            profile = (await progress_for(db, owner_id)).get(ref, {})
            done = ((profile.get("scores") or {}).get("practical") or 0) >= 70
        if kind in {"automl_practical", "autodl_practical", "autonlp_practical"}:
            capstone = await db.learning_progress.find_one({**match, "topic_id": ref, "lifecycle_complete": True}, {"_id": 1})
            done = done and bool(capstone)
    status = "COMPLETED" if done else "OVERDUE" if overdue(assignment.get("due_date")) else "IN_PROGRESS" if started else "NOT_STARTED"
    return status


async def assignments_for(db, user):
    if not user.organization_id:
        return []
    query = {"organization_id": user.organization_id, "batch_id": {"$in": [None, user.batch_id]}}
    items = await db.learning_assignments.find(query).sort("created_at", -1).limit(100).to_list(length=100)
    return [{**public(item), "status": await assignment_status(db, item, str(user.id))} for item in items]


async def create_assignment(db, user, payload):
    scope = faculty_scope(user)
    validate_reference(payload.type, payload.reference_id)
    batch_id = payload.batch_id or user.batch_id
    if user.role == "trainer" and user.batch_id and batch_id != user.batch_id:
        raise HTTPException(403, "Batch is outside your scope.")
    if batch_id and not await db.users.find_one({"organization_id": user.organization_id, "batch_id": batch_id,
                                                  "role": {"$in": ["trainee", "user"]}}, {"_id": 1}):
        raise HTTPException(422, "No learners in this organization and batch.")
    item = {"_id": str(uuid4()), "organization_id": scope["organization_id"], "batch_id": batch_id,
            "title": payload.title, "type": payload.type, "reference_id": payload.reference_id,
            "due_date": payload.due_date, "assigned_by": str(user.id), "created_at": datetime.now(UTC),
            "status": "ACTIVE"}
    await db.learning_assignments.insert_one(item)
    return public(item)


async def cohort_summary(db, user, batch_id=None):
    ids = await cohort_users(db, user, batch_id)
    if not ids:
        return {"learner_count": 0, "average_overall_mastery": 0, "module_averages": {},
                "assessment_completion": None, "assessment_participation": 0,
                "foundation_mastery": {}, "capstone_completion": {}, "capstone_lifecycle": {},
                "assignment_completion": {"assigned": 0, "completed": 0, "overdue": 0},
                "interview_readiness_average": None, "skill_distribution": {"<50": 0, "50-69": 0, "70-84": 0, "85+": 0},
                "common_weak_topics": [], "learners_needing_attention": 0, "attention_list": []}
    members = await db.users.find({"organization_id": user.organization_id,
        "role": {"$in": ["trainee", "user"]}}, {"_id": 1, "id": 1, "batch_id": 1, "full_name": 1}).limit(1000).to_list(length=1000)
    batch_by_owner = {item.get("id") or str(item["_id"]): item.get("batch_id") for item in members}
    name_by_owner = {item.get("id") or str(item["_id"]): item.get("full_name") or "Learner" for item in members}
    progress = await db.learning_progress.find({"owner_id": {"$in": ids}}, {"owner_id": 1, "module": 1, "topic_id": 1, "scores": 1, "lifecycle_complete": 1, "lifecycle_stages": 1, "activity_started_at": 1}).limit(30000).to_list(length=30000)
    verified = await db.learning_attempts.find({"owner_id": {"$in": ids}, "category": {"$in": ["coding", "practical"]},
        "$or": [{"verified": True}, {"activity_type": "faculty_review"}]},
        {"owner_id": 1, "topic_id": 1, "category": 1, "combined_score": 1,
         "source_reference": 1, "evidence_type": 1, "created_at": 1}).sort([("created_at", 1), ("_id", 1)]).limit(30000).to_list(length=30000)
    evidence_scores = {}
    claimed = {}
    for attempt in verified:
        source = attempt.get("source_reference")
        if attempt.get("category") == "practical" and source and attempt.get("evidence_type"):
            claim = (attempt["owner_id"], attempt["evidence_type"], source)
            first_topic = claimed.setdefault(claim, attempt.get("topic_id"))
            if first_topic != attempt.get("topic_id") and not BY_ID.get(attempt.get("topic_id"), {}).get("allow_practical_reuse", False):
                continue
        key = (attempt.get("owner_id"), attempt.get("topic_id"), attempt.get("category"))
        evidence_scores[key] = max(evidence_scores.get(key, 0), attempt.get("combined_score") or 0)
    topic_scores_by_user, topics = {}, {}
    for item in progress:
        scores = dict(item.get("scores") or {})
        for category in ("coding", "practical"):
            scores[category] = evidence_scores.get((item["owner_id"], item["topic_id"], category))
        score = score_profile({**scores, "lifecycle_complete": item.get("lifecycle_complete") is True}, item.get("topic_id"))["mastery_score"]
        if BY_ID.get(item.get("topic_id"), {}).get("is_capstone") and not item.get("lifecycle_complete"):
            score = completion_percent(item["topic_id"], item.get("lifecycle_stages") or {})
        topic_scores_by_user[(item["owner_id"], item["topic_id"])] = score
        topics.setdefault(item.get("topic_id", ""), []).append(score)
    module_topics = {module: [item["id"] for item in TOPICS if item["module"] == module] for module in LABS}
    foundation_topics = {module: [item["id"] for item in TOPICS if item["module"] == module and not item.get("is_capstone")] for module in LABS}
    capstone_topics = {module: [item["id"] for item in TOPICS if item["module"] == module and item.get("is_capstone")] for module in LABS}
    module_scores = {module: [] for module in LABS}
    averages = []
    for owner_id in ids:
        for module, topic_ids in module_topics.items():
            module_scores[module].append(round(sum(topic_scores_by_user.get((owner_id, topic_id), 0) for topic_id in topic_ids) / len(topic_ids)) if topic_ids else 0)
        averages.append(round(sum(module_scores[module][-1] for module in LABS) / len(LABS)))
    dist = {"<50": sum(s < 50 for s in averages), "50-69": sum(50 <= s < 70 for s in averages),
            "70-84": sum(70 <= s < 85 for s in averages), "85+": sum(s >= 85 for s in averages)}
    sessions = await db.interview_sessions.find({"owner_id": {"$in": ids}, "status": "completed"}, {"overall_score": 1}).limit(5000).to_list(length=5000)
    readiness = [item["overall_score"] for item in sessions if item.get("overall_score") is not None]
    assessed = await db.learning_attempts.distinct("owner_id", {"owner_id": {"$in": ids}, "combined_score": {"$gte": 70},
        "activity_type": {"$in": ["mcq", "multi_select", "output_prediction", "debugging", "complete_missing_code", "business_coding", "query_correction", "business_problem", "case_study", "result_interpretation"]}})
    scope = faculty_scope(user)
    if batch_id:
        scope["batch_id"] = {"$in": [None, batch_id]}
    elif user.role == "trainer":
        scope["batch_id"] = {"$in": [None, user.batch_id]}
    assignments = await db.learning_assignments.find(scope).limit(100).to_list(length=100)
    counts = {"assigned": 0, "completed": 0, "overdue": 0}
    for assignment in assignments:
        kind, ref = assignment["type"], assignment["reference_id"]
        assigned_ids = [owner_id for owner_id in ids if not assignment.get("batch_id") or
                        batch_by_owner.get(owner_id) == assignment["batch_id"]]
        counts["assigned"] += len(assigned_ids)
        query = {"owner_id": {"$in": assigned_ids}}
        if kind in {"learning_topic", "capstone"}:
            completed = [item["owner_id"] for item in progress if item["owner_id"] in assigned_ids and item.get("topic_id") == ref and
                         score_profile({**(item.get("scores") or {}), **{category: evidence_scores.get((item["owner_id"], ref, category)) for category in ("coding", "practical")}, "lifecycle_complete": item.get("lifecycle_complete") is True}, ref)["status"] == "MASTERED"]
        elif kind == "case_study":
            completed = await db.learning_projects.distinct("owner_id", {**query, "case_id": ref, "verified_attempt_id": {"$exists": True}})
        elif kind == "mock_interview":
            completed = await db.interview_sessions.distinct("owner_id", {**query, "module": ref, "mode": "mock", "status": "completed"})
        else:
            if kind == "assessment": query["source_reference"] = ref
            else: query["topic_id"] = ref; query["verified"] = True
            if kind == "python_challenge": query["activity_type"] = "python_execution"
            if kind == "sql_challenge": query["activity_type"] = "sql_execution"
            if kind in PRACTICAL_TYPES: query["evidence_type"] = PRACTICAL_TYPES[kind][1]
            completed = ([] if kind in {"automl_practical", "autodl_practical", "autonlp_practical"}
                         and not BY_ID.get(ref, {}).get("is_capstone") else
                         await db.learning_attempts.distinct("owner_id", {**query, "combined_score": {"$gte": 70}}))
            if kind in PRACTICAL_TYPES:
                completed = ([] if kind in {"automl_practical", "autodl_practical", "autonlp_practical"}
                             and not BY_ID.get(ref, {}).get("is_capstone") else
                             [candidate for candidate in assigned_ids
                              if (evidence_scores.get((candidate, ref, "practical")) or 0) >= 70])
            if kind in {"automl_practical", "autodl_practical", "autonlp_practical"}:
                completed = [candidate for candidate in completed if any(item.get("owner_id") == candidate and
                    item.get("topic_id") == ref and item.get("lifecycle_complete") is True for item in progress)]
        count = len(set(completed) & set(assigned_ids))
        counts["completed"] += count
        if overdue(assignment.get("due_date")):
            counts["overdue"] += len(assigned_ids) - count
    assigned_assessments = [item for item in assignments if item["type"] == "assessment"]
    assessment_total = sum(sum(not item.get("batch_id") or batch_by_owner.get(owner_id) == item["batch_id"] for owner_id in ids)
                           for item in assigned_assessments)
    assessment_done = 0
    for item in assigned_assessments:
        assigned_ids = [owner_id for owner_id in ids if not item.get("batch_id") or batch_by_owner.get(owner_id) == item["batch_id"]]
        assessment_done += len(await db.learning_attempts.distinct("owner_id", {"owner_id": {"$in": assigned_ids},
            "source_reference": item["reference_id"], "combined_score": {"$gte": 70}}))
    needing_attention = [{"learner_id": owner_id, "learner_name": name_by_owner.get(owner_id, "Learner"), "mastery": score,
                          "weak_topics": [topic_id for (candidate, topic_id), value in topic_scores_by_user.items() if candidate == owner_id and value < 50][:3]}
                         for owner_id, score in zip(ids, averages) if score < 50][:20]
    return {"learner_count": len(ids), "average_overall_mastery": round(sum(averages) / len(averages)),
            "module_averages": {key: round(sum(values) / len(values)) for key, values in module_scores.items()},
            "foundation_mastery": {module: round(sum(topic_scores_by_user.get((owner_id, topic_id), 0)
                for owner_id in ids for topic_id in topic_ids) / (len(ids) * len(topic_ids)))
                for module, topic_ids in foundation_topics.items() if topic_ids},
            "capstone_completion": {module: {"completed": sum(any(item.get("owner_id") == owner_id and item.get("topic_id") == topic_id and item.get("lifecycle_complete") is True for item in progress)
                for owner_id in ids for topic_id in topic_ids), "possible": len(ids) * len(topic_ids)}
                for module, topic_ids in capstone_topics.items() if topic_ids},
            "capstone_lifecycle": {module: {**capstone_lifecycle_summary(progress, len(ids), topic_ids),
                "assigned": sum(sum(not assignment.get("batch_id") or batch_by_owner.get(owner_id) == assignment["batch_id"]
                                    for owner_id in ids) for assignment in assignments
                                if assignment.get("reference_id") in topic_ids and
                                assignment.get("type") in {"capstone", "automl_practical", "autodl_practical", "autonlp_practical"})}
                for module, topic_ids in capstone_topics.items() if topic_ids},
            "assessment_completion": round(100 * assessment_done / assessment_total) if assessment_total else None,
            "assessment_participation": round(100 * len(assessed) / len(ids)), "assignment_completion": counts,
            "interview_readiness_average": round(sum(readiness) / len(readiness)) if readiness else None,
            "skill_distribution": dist, "common_weak_topics": [key for key, values in sorted(topics.items(), key=lambda entry: sum(entry[1]) / len(entry[1]))[:5]],
            "learners_needing_attention": dist["<50"], "attention_list": needing_attention}
