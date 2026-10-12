"""Task-aware teaching lifecycle; existing capstone evidence IDs remain stable."""
from __future__ import annotations

SUPERVISED = [
    ("business", "Business Problem Understanding", "Define the decision, stakeholders, constraints and costs of mistakes."),
    ("ingestion", "Data Collection / Ingestion", "Understand files, APIs, databases, sensors/logs, sampling quality and data provenance."),
    ("cleaning", "Data Cleaning & Preprocessing", "Review missing, duplicate and invalid values, encoding and scaling; fit learned preprocessing without leakage."),
    ("eda", "Exploratory Data Analysis", "Inspect distributions, relationships, outliers, missingness, imbalance and correlations; correlation is not causation."),
    ("features", "Feature Engineering & Selection", "Engineering creates or transforms variables; selection chooses predictors. Identifiers are not automatically useful predictors."),
    ("split", "Train / Validation / Test Strategy", "Training fits parameters, validation guides choices, and test data checks generalization. Consider cross-validation and leakage; no split ratio fits every task."),
    ("selection", "Model Selection", "Choose candidate algorithms using task, data, interpretability, complexity, resources and evaluation objective. A leaderboard winner is not automatically the recommended model."),
    ("training", "Model Training", "Fit model parameters on training data. Candidate selection and fitting are different activities."),
    ("tuning", "Hyperparameter Tuning", "Hyperparameters configure learning; parameters are fitted. Tune against validation evidence and avoid repeatedly optimizing against the test set."),
    ("evaluation", "Model Evaluation", "Use task-aware independent evaluation and explain what each metric means for the decision."),
    ("errors", "Error Analysis & Interpretation", "Inspect errors and available native feature, text or visual evidence. Explain limitations without causal claims."),
    ("inference", "Prediction / Inference", "Use a fitted model on new data; inference is separate from training."),
    ("recommendation", "Business Recommendation", "Translate evidence into possible actions, risks, limitations and next steps without promising certainty."),
]
CLUSTERING = [
    ("business", "Business Problem Understanding", "Define why discovering groups could support a decision; no labelled target or positive class is required."),
    ("ingestion", "Data Collection / Ingestion", SUPERVISED[1][2]),
    ("cleaning", "Data Cleaning & Preprocessing", "Review missing values, duplicates, invalid values and encodings before unsupervised grouping."),
    ("eda", "Exploratory Data Analysis", "Inspect feature distributions, relationships, outliers and potential grouping structure."),
    ("features", "Feature Selection", "Choose grouping features; review identifiers and redundant variables without selecting a supervised target."),
    ("scaling", "Scaling / Preparation", "Distance-based groups can depend strongly on units and scale. Explain the preparation actually used."),
    ("strategy", "Clustering Strategy", "Choose the desired type of grouping and how stability, separation and usefulness will be investigated."),
    ("selection", "Algorithm Selection", "Consider geometry, noise, interpretability, computational resources and assignment capabilities."),
    ("parameters", "Cluster Count / Parameters", "Explain cluster count or density settings where applicable; an elbow is evidence, not proof of a uniquely correct count."),
    ("training", "Model Training", "Fit the clustering algorithm to discover groups. Do not frame this as positive-class detection."),
    ("evaluation", "Internal Evaluation", "Silhouette measures cohesion/separation (higher); Davies-Bouldin compares scatter/separation (lower); Calinski-Harabasz compares between/within dispersion (higher). Inspect stability and usefulness too."),
    ("profiling", "Cluster Profiling", "Describe measured native differentiators, sizes and distributions before assigning business interpretations."),
    ("interpretation", "Interpretation", "Inspect poor separation, small/unstable groups and ambiguous profiles. External context cannot verify the discovered cluster or replace its statistics."),
    ("recommendation", "Business Recommendation", "Suggest possible actions with risks and limitations; do not treat discovered groups as confirmed business truth."),
]
STAGE_MAP = {
    "BUSINESS_UNDERSTANDING": ["business"], "PROBLEM_FORMULATION": ["business"],
    "DATA_UNDERSTANDING": ["ingestion"], "EDA": ["eda"],
    "DATA_PREPARATION": ["cleaning", "features", "split"], "BASELINE": ["selection"],
    "MODEL_DEVELOPMENT": ["selection", "training", "tuning"], "EVALUATION": ["evaluation"],
    "ERROR_ANALYSIS": ["errors"], "PREDICTION": ["inference"], "INTERPRETATION": ["errors"],
    "BUSINESS_RECOMMENDATION": ["recommendation"], "VERIFICATION": [],
}
CLUSTER_MAP = {**STAGE_MAP, "PROBLEM_FORMULATION": ["business", "strategy"],
    "DATA_PREPARATION": ["cleaning", "features", "scaling"], "BASELINE": ["selection", "parameters"],
    "MODEL_DEVELOPMENT": ["selection", "parameters", "training"], "ERROR_ANALYSIS": ["profiling", "interpretation"],
    "INTERPRETATION": ["profiling", "interpretation"], "PREDICTION": ["assignment"]}

def learner_lifecycle(task: str = "classification", prediction_supported: bool | None = None) -> list[dict]:
    clustering = task == "clustering"
    rows = list(CLUSTERING if clustering else SUPERVISED)
    if clustering and prediction_supported is True:
        rows.insert(-1, ("assignment", "New Observation Assignment", "This fitted algorithm supports assigning new observations. Other clustering algorithms may not; never assume a prediction API exists."))
    if task == "regression":
        rows = [(key, title, "MAE expresses typical absolute error in target units; RMSE penalizes larger errors; R² compares explained variation with a baseline." if key == "evaluation"
                 else "Inspect large residuals and systematic under/over prediction; separate model influence from causation." if key == "errors" else description) for key, title, description in rows]
    elif not clustering:
        rows = [(key, title, "Accuracy measures overall correctness; precision measures false-alarm burden; recall measures detected cases for the specified class; F1 balances precision/recall. State binary, macro or weighted averaging. ROC-AUC measures threshold discrimination; inspect confusion and per-class support." if key == "evaluation"
                 else "Inspect false positives, false negatives and class-specific errors alongside available feature/text/visual evidence." if key == "errors" else description) for key, title, description in rows]
    return [{"id": key, "title": title, "description": description, "order": index + 1} for index, (key, title, description) in enumerate(rows)]

def stage_presentation(stage_id: str, task: str, original_title: str, original_instruction: str) -> dict:
    mapping = (CLUSTER_MAP if task == "clustering" else STAGE_MAP).get(stage_id, [])
    lifecycle = learner_lifecycle(task, prediction_supported=True)
    indexed = {row["id"]: row for row in lifecycle}
    titles = [indexed[key]["title"] for key in mapping if key in indexed]
    instruction = original_instruction
    if task == "clustering":
        instruction = {
            "DATA_UNDERSTANDING": "Inspect collected inputs, provenance, size and grouping features. No target or positive class is required.",
            "DATA_PREPARATION": "Review missing/duplicate/invalid values, encodings, excluded identifiers and scaling used for grouping. Do not impose a classification train/test split.",
            "BASELINE": "Discuss a candidate algorithm and cluster-count/parameter strategy using existing evidence; no extra training is started here.",
            "EVALUATION": "Interpret the saved internal metric alongside separation, stability, size and cluster profiles. Internal scores alone do not prove business usefulness.",
            "ERROR_ANALYSIS": "Inspect poor separation, small or unstable groups, noise and ambiguous native profiles.",
            "INTERPRETATION": "Explain the measured cluster pattern and its limitations. Research can provide context, never prove the discovered cluster correct.",
            "PREDICTION": "Assign a new observation only if the fitted algorithm supports it; otherwise this stage can be skipped.",
        }.get(stage_id, instruction)
    return {"internal_stage": stage_id, "learner_stages": mapping,
            "learner_stage": " / ".join(titles) or original_title,
            "stage_group": "quality_gate" if stage_id == "VERIFICATION" else "interpretation" if stage_id in {"ERROR_ANALYSIS", "INTERPRETATION", "BUSINESS_RECOMMENDATION"} else "modeling" if stage_id in {"BASELINE", "MODEL_DEVELOPMENT", "EVALUATION", "PREDICTION"} else "understanding_and_data",
            "description": " ".join(indexed[key]["description"] for key in mapping if key in indexed) or "Verify the existing owned capstone evidence; this is a quality gate, not a new model-training stage.",
            "instruction": instruction}

def teaching_context(task: str = "classification") -> str:
    rows = learner_lifecycle(task)
    return ("NxZenAI teaching lifecycle: " + " → ".join(row["title"] for row in rows) +
            ". AutoML automates parts of this wider lifecycle; upload → train → leaderboard is not the whole process. "
            "The lifecycle is iterative, not a claim that every recorded native run executed every stage. "
            "For clustering use its unsupervised lifecycle without a target, positive class or mandatory supervised split. "
            "For NLP allow text cleaning/tokenization/vectorization; for DL allow tensor/image preparation and validation.")


# Overview groups reference the same canonical lifecycle IDs used inside topics.
ROADMAP_GROUPS = [
    ("problem", "Understand the Problem", ["business"]),
    ("data", "Understand the Data", ["ingestion", "eda"]),
    ("preparation", "Prepare the Data", ["cleaning", "features", "scaling"]),
    ("modeling", "Prepare for Modeling", ["split", "strategy"]),
    ("training", "Choose & Train Models", ["selection", "parameters", "training", "tuning"]),
    ("evaluation", "Evaluate & Interpret", ["evaluation", "errors", "profiling", "interpretation"]),
    ("apply", "Apply to Business Problems", ["inference", "recommendation"]),
]

def overview_roadmap() -> dict:
    return {"roadmap_groups": [{"id": key, "title": title, "lifecycle_stages": stages}
                                for key, title, stages in ROADMAP_GROUPS],
            "lifecycles": {task: learner_lifecycle(task) for task in ("classification", "regression", "clustering")}}
