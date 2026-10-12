"""Domain lifecycle and roadmap presentation over the existing curriculum."""
from __future__ import annotations
from .learner_lifecycle import learner_lifecycle, overview_roadmap, stage_presentation, teaching_context

# Educational stages, not new curriculum items or evidence requirements.
DOMAINS = {
    "autodl": [
        ("problem", "Problem Definition", "Define the prediction task, constraints and costs of errors."),
        ("collection", "Data Collection", "Review data sources, permissions, coverage and label quality."),
        ("preparation", "Data Preparation", "Inspect missing/corrupt inputs and supported normalization or augmentation."),
        ("representation", "Input Representation", "Understand tensors, shapes, channels and input encoding."),
        ("architecture", "Architecture Selection", "Choose candidate networks using task, data, complexity and resources."),
        ("split", "Train / Validation Strategy", "Keep validation and independent evaluation separate; avoid leakage."),
        ("training", "Model Training", "Connect forward pass, loss, gradients and optimizer updates."),
        ("tuning", "Hyperparameter Tuning", "Use validation to investigate learning rate, batch size and regularization."),
        ("evaluation", "Evaluation", "Inspect held-out task metrics and class support rather than training score alone."),
        ("errors", "Error Analysis", "Investigate confusion, failed examples and training-validation gaps."),
        ("interpretation", "Interpretation", "Use available native evidence; do not invent activation or spatial explanations."),
        ("inference", "Inference", "Run a supported fitted model on compatible new inputs."),
        ("application", "Optimization / Deployment Considerations", "Discuss latency, size, resources, monitoring and limitations without claiming deployment happened."),
    ],
    "autonlp": [
        ("problem", "Problem Definition", "Define what the text application should accomplish and how errors matter."),
        ("collection", "Text Collection", "Check text sources, permissions, labels, language and coverage."),
        ("cleaning", "Text Cleaning", "Inspect noise; retain punctuation or negation when it carries meaning."),
        ("representation", "Tokenization / Representation", "Explain tokens, vocabulary, n-grams and numeric representations."),
        ("features", "Feature / Embedding Strategy", "Compare count features, TF-IDF and embeddings for the task."),
        ("selection", "Model Selection", "Compare a classical baseline with more complex language models where justified."),
        ("training", "Training", "Fit text preparation and model components on the appropriate training data."),
        ("evaluation", "Evaluation", "Use held-out and per-class metrics, confusion patterns and task-appropriate checks."),
        ("errors", "Error Analysis", "Review ambiguous language, minority classes and systematic mistakes."),
        ("inference", "Inference", "Apply the saved text pipeline to new text without rebuilding it."),
        ("interpretation", "Interpretation", "Explain only available influential text patterns and measured results."),
        ("application", "Application", "Connect NLP evidence to routing, classification or other supported uses with limitations."),
    ],
    "python": [
        ("problem", "Problem Understanding", "Define inputs, expected outputs and constraints before writing code."),
        ("values", "Values & Data Types", "Understand values, variables and the operations their types support."),
        ("expressions", "Expressions & Operators", "Use arithmetic, comparisons and Boolean expressions clearly."),
        ("control", "Control Flow", "Choose branches and iterations that match the problem."),
        ("functions", "Functions", "Design clear parameters, scope and returned results."),
        ("collections", "Collections", "Choose lists, dictionaries and sets using ordering and lookup needs."),
        ("modules", "Modules", "Import and organize reusable code without relying on hidden state."),
        ("files", "File / Data Handling", "Read and write data carefully using supported files and context managers."),
        ("libraries", "Libraries", "Use array, table and plotting libraries while checking their inputs and shapes."),
        ("debugging", "Errors & Debugging", "Read errors, handle expected failures and investigate incorrect behavior."),
        ("solving", "Problem Solving", "Break a problem into small reusable steps and check the result."),
        ("application", "Mini Application", "Combine existing skills into a small useful program without assuming completion."),
    ],
    "sql": [
        ("model", "Understand the Data Model", "Identify entities, relationships and the question the data can answer."),
        ("keys", "Tables & Keys", "Understand row grain and join keys before combining tables."),
        ("projection", "SELECT / Projection", "Select the required columns and expressions."),
        ("filtering", "Filtering & Sorting", "Apply row conditions and explicit ordering."),
        ("aggregation", "Aggregation", "Summarize at the correct grain and distinguish WHERE from HAVING."),
        ("joins", "Joins", "Combine related records while checking unmatched rows and multiplication."),
        ("subqueries", "Subqueries / CTEs", "Structure intermediate calculations using supported query workflows."),
        ("windows", "Window Functions", "Calculate across related rows without collapsing each row."),
        ("validation", "Query Validation", "Check NULL behavior, row counts, joins and totals before trusting a query."),
        ("efficiency", "Query Optimization / Efficiency", "Consider unnecessary work and readable efficient queries; no optimizer is added here."),
        ("application", "Business Query / Reporting", "Answer a defined question with validated results and clear limitations."),
    ],
    "eda": [
        ("problem", "Business Question", "Define the question and intended decision before drawing conclusions."),
        ("loading", "Data Collection / Loading", "Inspect sources, provenance, rows and columns."),
        ("quality", "Data Quality Assessment", "Review types, units, missingness, duplicates and suspicious values."),
        ("cleaning", "Cleaning", "Document supported changes; recommendations are not silent automatic fixes."),
        ("descriptive", "Descriptive Analysis", "Summarize data with counts, proportions and numeric statistics."),
        ("univariate", "Univariate Analysis", "Examine each variable's distribution and support."),
        ("relationships", "Bivariate / Multivariate Analysis", "Explore relationships while considering outliers and confounding."),
        ("visualization", "Visualization", "Choose charts using variable types and the comparison goal."),
        ("outliers", "Outliers / Anomalies", "Investigate unusual values without automatically deleting them."),
        ("interpretation", "Pattern Interpretation", "Describe observed patterns and uncertainty; association does not prove causation."),
        ("insights", "Insight Formation", "Connect defensible observations to the original question."),
        ("recommendation", "Business Recommendation", "Explain possible actions, limitations and next checks."),
    ],
    "genai": [
        ("problem", "Problem Definition", "Define the task, constraints and acceptable output."),
        ("fundamentals", "LLM Fundamentals", "Understand tokens, context windows and uncertainty in generated text."),
        ("selection", "Model Selection", "Consider capability, cost, context and evaluation needs."),
        ("prompt", "Prompt / Instruction Design", "Specify task, constraints and output requirements."),
        ("context", "Context Design", "Supply relevant bounded context and respect evidence boundaries."),
        ("retrieval", "Retrieval / RAG", "Retrieve relevant evidence when the application benefits from it; RAG is optional."),
        ("generation", "Generation", "Generate under the configured workflow; fluency is not proof of truth."),
        ("evaluation", "Evaluation", "Check task completion, support, format and failure behavior."),
        ("safety", "Grounding / Safety", "Maintain citations, permissions and honest uncertainty."),
        ("tools", "Tool Use", "Use complete arguments and report actual outcomes or failures."),
        ("agents", "Agents / Orchestration", "Coordinate supported tools when needed; not every application needs an agent."),
        ("application", "Production Application", "Consider reliability, monitoring and limits without claiming an application was deployed."),
    ],
    "prompting": [
        ("goal", "Define the Goal", "State the task and observable success criteria."),
        ("context", "Provide Context", "Specify audience, background and relevant evidence."),
        ("instructions", "Write Clear Instructions", "Use explicit directions that match the desired task."),
        ("constraints", "Add Constraints", "Set measurable limits and required output format."),
        ("templates", "Variables / Templates", "Reuse instructions while resolving required variables safely."),
        ("examples", "Few-shot Examples", "Provide consistent relevant input-output examples."),
        ("structured", "Structured Output", "Specify fields, allowed values and missing-value behavior."),
        ("evaluation", "Evaluate Output", "Check the output against the same goal and evidence."),
        ("iteration", "Iterate / Refine", "Revise based on observed shortcomings rather than assumptions."),
        ("comparison", "Compare Models / Runs", "Keep tasks, evidence and criteria consistent across comparisons."),
        ("grounding", "Knowledge Grounding", "Use supporting documents and admit when evidence is absent."),
        ("production", "Production Prompt", "Consider reusable inputs, safe failure and versioned evaluation."),
    ],
}
GROUPS = {
    "autodl": [("foundations", "Foundations", ["problem"]), ("data", "Data & Representation", ["collection", "preparation", "representation", "split"]), ("network", "Build the Network", ["architecture"]), ("training", "Train & Improve", ["training", "tuning"]), ("evaluation", "Evaluate & Interpret", ["evaluation", "errors", "interpretation"]), ("apply", "Apply", ["inference", "application"])],
    "autonlp": [("foundations", "NLP Foundations", ["problem", "collection"]), ("preparation", "Text Preparation", ["cleaning"]), ("representation", "Representation", ["representation", "features"]), ("modeling", "Modeling", ["selection", "training"]), ("evaluation", "Evaluation", ["evaluation", "errors", "interpretation"]), ("apply", "Application", ["inference", "application"])],
    "python": [("foundations", "Foundations", ["problem", "values", "expressions"]), ("control", "Logic & Control", ["control"]), ("functions", "Functions & Reuse", ["functions", "modules"]), ("collections", "Data Structures", ["collections"]), ("files", "Files & Libraries", ["files", "libraries"]), ("debugging", "Debugging & Problem Solving", ["debugging", "solving"]), ("apply", "Apply", ["application"])],
    "sql": [("foundations", "Relational Foundations", ["model", "keys"]), ("basics", "Query Basics", ["projection", "filtering", "aggregation"]), ("combining", "Combining Data", ["joins"]), ("advanced", "Advanced Queries", ["subqueries", "windows"]), ("validation", "Validation & Performance", ["validation", "efficiency"]), ("apply", "Business Application", ["application"])],
    "eda": [("problem", "Understand the Problem", ["problem"]), ("data", "Understand the Data", ["loading", "quality"]), ("cleaning", "Clean & Prepare", ["cleaning"]), ("explore", "Explore", ["descriptive", "univariate", "relationships", "outliers"]), ("visualize", "Visualize", ["visualization"]), ("interpret", "Interpret", ["interpretation", "insights"]), ("apply", "Recommend", ["recommendation"])],
    "genai": [("foundations", "Foundations", ["problem", "fundamentals", "selection", "generation"]), ("context", "Prompt & Context", ["prompt", "context"]), ("retrieval", "Retrieval", ["retrieval"]), ("evaluation", "Evaluation & Safety", ["evaluation", "safety"]), ("tools", "Tools & Agents", ["tools", "agents"]), ("apply", "Application", ["application"])],
    "prompting": [("foundations", "Foundations", ["goal", "context"]), ("construction", "Prompt Construction", ["instructions", "constraints", "structured"]), ("reusable", "Reusable Prompts", ["templates"]), ("evaluation", "Evaluation", ["evaluation", "iteration", "comparison"]), ("advanced", "Advanced Techniques", ["examples", "grounding"]), ("apply", "Production", ["production"])],
}
LAB_LABELS = {"automl": "AutoML Lab", "autodl": "AutoDL", "autonlp": "AutoNLP", "python": "Python Lab", "sql": "SQL Lab", "eda": "EDA Hub", "genai": "GenAI", "prompting": "Prompt Lab"}
NOTES = {
    "autodl": "Neural-network workflow: representation, architecture and gradient-based training are distinct concepts.",
    "autonlp": "Text workflow: preserve task-relevant meaning and compare classical baselines with more complex models where useful.",
    "python": "Programming progression, not a mandatory machine-learning workflow.",
    "sql": "Query-learning progression. Modification is not included because current topics do not teach it; conceptual stages do not grant execution capability.",
    "eda": "Exploration and interpretation, not model training; association does not establish causation.",
    "genai": "RAG, tool use and agents are optional choices, not requirements for every application.",
    "prompting": "Prompt design and evaluation, not model training. Grounding needs actual supporting evidence.",
}
PLACEMENTS = {
    "autodl": {
        "foundations": ("foundations", ["problem"]), "neurons": ("foundations", ["representation", "training"]),
        "tensors": ("data", ["representation"]), "layers": ("network", ["architecture"]),
        "activation": ("network", ["architecture"]), "forward": ("network", ["training"]),
        "cnn": ("network", ["architecture", "split"]), "transfer": ("network", ["architecture", "tuning"]),
        "training": ("training", ["training"]), "pytorch": ("training", ["training"]),
        "loss": ("training", ["training"]), "backprop": ("training", ["training"]), "optimizers": ("training", ["training"]),
        "learning-rate": ("training", ["tuning"]), "regularization": ("evaluation", ["evaluation", "errors", "tuning"]),
    },
    "autonlp": {
        "foundations": ("foundations", ["problem", "collection"]), "cleaning": ("preparation", ["cleaning"]),
        "preprocessing": ("preparation", ["cleaning", "representation"]), "stopwords": ("preparation", ["cleaning"]),
        "word-forms": ("preparation", ["cleaning"]), "bow": ("representation", ["representation", "features"]),
        "tfidf": ("representation", ["features"]), "ngrams": ("representation", ["representation"]),
        "embeddings": ("representation", ["features"]), "context": ("representation", ["representation"]),
        "attention": ("representation", ["representation"]), "pipeline": ("modeling", ["selection", "training"]),
        "transformers": ("modeling", ["selection"]), "baseline": ("modeling", ["selection", "evaluation"]),
        "metrics": ("evaluation", ["evaluation", "errors"]),
    },
    "python": {
        "basics": ("foundations", ["values", "control"]), "operators": ("foundations", ["expressions"]),
        "conditions": ("control", ["control"]), "loops": ("control", ["control"]),
        "parameters": ("functions", ["functions"]), "reusable": ("functions", ["functions", "solving"]),
        "collections": ("collections", ["collections", "functions"]), "maps-sets": ("collections", ["collections"]),
        "files-errors": ("files", ["files", "modules", "debugging"]), "numpy": ("files", ["libraries"]),
        "pandas": ("files", ["libraries"]), "filter-group": ("files", ["libraries"]), "plotting": ("files", ["libraries"]),
        "models": ("apply", ["solving", "application"]),
    },
    "sql": {
        "relations": ("foundations", ["model", "keys"]), "select": ("basics", ["projection", "filtering"]),
        "sort-distinct": ("basics", ["filtering"]), "aggregates": ("basics", ["aggregation"]),
        "groups": ("basics", ["aggregation"]), "joins": ("combining", ["joins", "subqueries"]),
        "left-join": ("combining", ["joins"]), "subqueries": ("advanced", ["subqueries"]),
        "ctes": ("advanced", ["subqueries"]), "windows": ("advanced", ["windows"]),
        "case-null": ("validation", ["validation"]), "revenue": ("apply", ["application"]), "business": ("apply", ["validation", "application"]),
    },
    "eda": {
        "purpose": ("problem", ["problem"]), "schema": ("data", ["loading", "quality"]), "types": ("data", ["quality"]),
        "quality": ("cleaning", ["quality", "cleaning"]), "transformations": ("cleaning", ["cleaning"]),
        "categories": ("explore", ["descriptive", "univariate"]), "outliers": ("explore", ["univariate", "outliers"]),
        "relationships": ("explore", ["relationships", "interpretation"]), "charts": ("visualize", ["visualization"]),
        "insights": ("apply", ["insights", "recommendation"]),
    },
    "genai": {
        "llm": ("foundations", ["fundamentals"]), "temperature": ("foundations", ["generation"]),
        "tokens": ("context", ["fundamentals", "context"]), "embeddings": ("retrieval", ["retrieval"]),
        "retrieval": ("retrieval", ["retrieval"]), "rag": ("retrieval", ["retrieval", "generation"]),
        "hallucinations": ("evaluation", ["safety"]), "citations": ("evaluation", ["safety"]),
        "evaluation": ("evaluation", ["evaluation"]), "agents": ("tools", ["tools", "agents"]), "guardrails": ("tools", ["tools", "safety"]),
    },
    "prompting": {
        "clarity": ("foundations", ["goal", "instructions"]), "context": ("construction", ["context"]),
        "format": ("construction", ["constraints"]), "extraction": ("construction", ["structured"]),
        "variables": ("reusable", ["templates"]), "compare": ("evaluation", ["comparison"]),
        "evaluation": ("evaluation", ["evaluation", "iteration"]), "few-shot": ("advanced", ["examples"]),
        "grounding": ("advanced", ["grounding"]), "reduce-hallucination": ("advanced", ["grounding"]),
        "tool-agent": ("advanced", ["production"]),
    },
}

def module_lifecycle(module: str, task: str = "classification") -> list[dict]:
    if module == "automl":
        return learner_lifecycle(task)
    return [{"id": key, "title": title, "description": description, "order": index + 1}
            for index, (key, title, description) in enumerate(DOMAINS.get(module, []))]

def path_metadata(module: str, lab_url: str) -> dict:
    if module == "automl":
        data = overview_roadmap()
        data.update({"lifecycle_title": "Data Science lifecycle", "default_lifecycle_key": "classification",
            "lifecycle_variants": [
                {"id": "classification", "title": "Classification", "description": "Supervised class prediction: separate selection, fitting, tuning and inference."},
                {"id": "regression", "title": "Regression", "description": "Supervised numeric prediction: evaluate error magnitude and residual patterns."},
                {"id": "clustering", "title": "Clustering", "description": "Unsupervised grouping: no target, positive class or required supervised split. New-observation assignment depends on the fitted algorithm."}]})
    else:
        data = {"roadmap_groups": [{"id": key, "title": title, "lifecycle_stages": stages}
                                  for key, title, stages in GROUPS.get(module, [])],
                "lifecycles": {"default": module_lifecycle(module)}, "default_lifecycle_key": "default",
                "lifecycle_title": "Professional lifecycle / learning progression",
                "lifecycle_variants": [{"id": "default", "title": LAB_LABELS.get(module, module), "description": NOTES.get(module, "")}]}
    data["roadmap_groups"] = [{**group, "order": index + 1} for index, group in enumerate(data["roadmap_groups"])]
    return {**data, "native_lab_label": LAB_LABELS.get(module, "Related Lab"), "native_lab_url": lab_url}

def attach_topic_metadata(topic: dict, lab_url: str) -> None:
    module = topic["module"]
    if module != "automl":
        key = topic["id"].removeprefix(module + "-")
        group, stages = (("apply", ["application"]) if topic.get("is_capstone")
                         else PLACEMENTS.get(module, {}).get(key, ("additional", [])))
        topic.update({"roadmap_group": group, "lifecycle_stages": stages,
                      "learner_lifecycle": module_lifecycle(module)})
    groups = path_metadata(module, lab_url)["roadmap_groups"]
    topic["roadmap_group_order"] = next((group["order"] for group in groups if group["id"] == topic.get("roadmap_group")), 999)
    stage_ids = topic.get("lifecycle_stages") or []
    stages = topic.get("learner_lifecycle") or []
    primary = next((stage for stage in stages if stage["id"] in stage_ids), None)
    task_notes = {"classification": "Classification · class prediction", "regression": "Regression · numeric prediction", "clustering": "Clustering · unsupervised grouping"}
    topic["task_note"] = task_notes.get(topic.get("learning_task"), "")
    topic.update({"lifecycle_stage": primary["id"] if primary else None,
                  "lifecycle_stage_order": primary["order"] if primary else None,
                  "learner_stage_name": primary["title"] if primary else "Additional Topics",
                  "learner_stage_description": primary["description"] if primary else "Existing curriculum content without a lifecycle placement."})

def module_teaching_context(module: str, task: str) -> str:
    if module == "automl":
        return teaching_context(task)
    return ("Use the " + LAB_LABELS.get(module, module) + " learning progression: " +
            " → ".join(stage["title"] for stage in module_lifecycle(module)) +
            ". " + NOTES.get(module, "") + " Conceptual stages alone are not evidence of mastery or completed native work.")

def capstone_presentation(module: str, task: str, stage_id: str, title: str, instruction: str) -> dict:
    if module == "automl":
        return stage_presentation(stage_id, task, title, instruction)
    mapping = {"BUSINESS_UNDERSTANDING": ["problem"], "PROBLEM_FORMULATION": ["problem"],
               "DATA_UNDERSTANDING": ["collection"], "EDA": ["preparation"] if module == "autodl" else ["cleaning"],
               "DATA_PREPARATION": ["preparation", "representation", "split"] if module == "autodl" else ["cleaning", "representation", "features"],
               "BASELINE": ["architecture"] if module == "autodl" else ["selection"],
               "MODEL_DEVELOPMENT": ["training"], "EVALUATION": ["evaluation"], "ERROR_ANALYSIS": ["errors"],
               "PREDICTION": ["inference"], "INTERPRETATION": ["interpretation"], "BUSINESS_RECOMMENDATION": ["application"]}
    ids = mapping.get(stage_id, [])
    stages = [stage for stage in module_lifecycle(module) if stage["id"] in ids]
    return {"internal_stage": stage_id, "learner_stages": ids, "learner_stage": " / ".join(stage["title"] for stage in stages) or title,
            "stage_group": "quality_gate" if stage_id == "VERIFICATION" else "domain_workflow",
            "description": " ".join(stage["description"] for stage in stages) or "Verify existing owned capstone evidence; this is a quality gate.",
            "instruction": instruction}
