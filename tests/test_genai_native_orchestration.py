from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.modules.genai.lab_adapters import (
    GenAILabAdapters,
    LabPredictionInputRequired,
    LabResourceSelectionRequired,
    _confirmed_autodl_task,
    _autodl_status_result,
    _followup_values,
    _prediction_result,
    _schema_values,
)
from app.modules.genai.schemas import ChatRequest
from app.modules.genai.service import GenAIService
from app.modules.genai.tools import (
    GenAIToolRegistry, ToolDefinition, ToolExecutionContext, ToolResult, ToolRouter,
)


OWNER = "owner-1"
USER = SimpleNamespace(id=OWNER, role="user")


class FakeRepository:
    def __init__(self, pending_prediction=None):
        self.conversation = {
            "id": "conversation-1", "title": "New chat", "project_id": None,
            "pending_prediction": pending_prediction, "pending_confirmation": None,
        }
        self.confirmation = None
        self.cleared_prediction = False
        self.messages = []

    async def get_conversation(self, conversation_id, owner_id):
        assert owner_id == OWNER
        return dict(self.conversation)

    async def attach_files_to_conversation(self, owner_id, attachment_ids, conversation_id, project_id):
        return [{"id": value, "filename": "dataset.csv", "content_type": "text/csv", "extraction": {"format": "csv"}} for value in attachment_ids]

    async def set_pending_confirmation(self, conversation_id, owner_id, state):
        self.confirmation = dict(state)
        self.conversation["pending_confirmation"] = dict(state)

    async def consume_pending_confirmation(self, conversation_id, owner_id, confirmation_id):
        if self.confirmation and self.confirmation["id"] == confirmation_id:
            value, self.confirmation = self.confirmation, None
            self.conversation["pending_confirmation"] = None
            return value
        return None

    async def clear_pending_confirmation(self, conversation_id, owner_id):
        self.confirmation = None
        self.conversation["pending_confirmation"] = None

    async def clear_pending_prediction(self, conversation_id, owner_id):
        self.cleared_prediction = True
        self.conversation["pending_prediction"] = None

    async def set_active_autodl_run(self, conversation_id, owner_id, run_id, metadata=None):
        resources = self.conversation.setdefault("active_lab_resources", {})
        resources["autodl"] = {"run_id": run_id, **(metadata or {})}

    async def set_pending_prediction(self, conversation_id, owner_id, state):
        self.conversation["pending_prediction"] = dict(state)

    async def add_message(self, owner_id, conversation_id, role, content, **kwargs):
        message = {"id": str(len(self.messages) + 1), "role": role, "content": content, "created_at": "now", "metadata": kwargs.get("metadata", {})}
        self.messages.append(message)
        return message

    async def rename_conversation(self, *args):
        return self.conversation


class FakeAdapters:
    def __init__(self, result=None):
        self.executions = []
        self.result = result or ToolResult("automl", True, "native training completed")

    async def resolve(self, tool, user, arguments, query, selected_attachments):
        return {**arguments, "action": arguments.get("action") or "train", "target_column": arguments.get("target_column") or "label"}

    async def has_compatible_automl_structured_input(self, user, query):
        return True

    @staticmethod
    def requires_confirmation(tool, arguments):
        return arguments.get("action") in {"train", "predict"}

    async def execute(self, tool, user, arguments):
        self.executions.append((tool, arguments))
        result = self.result
        result.tool = tool
        return result


def test_prediction_and_training_routes_are_lab_isolated():
    router = ToolRouter()
    assert router.route("Predict the sentiment of: excellent service", [], []) == ["autonlp"]
    assert router.route("AutoDL predict feature_1=0.8", [], []) == ["autodl"]
    assert router.route("Train a churn classification model", [], ["file-1"]) == ["automl"]
    assert router.route("Train a sentiment model", [], ["file-1"]) == ["autonlp"]
    assert router.route("Train a tabular classification model", [], ["file-1"]) == ["autodl"]


def test_explicit_autodl_task_family_is_authoritative_after_detection():
    assert _confirmed_autodl_task("classification", "tabular_regression", "csv") == "tabular_classification"
    assert _confirmed_autodl_task("regression", "time_series_classification", "csv") == "time_series_regression"


@pytest.mark.asyncio
async def test_compound_train_and_predict_is_rejected_before_native_execution():
    repository = FakeRepository()
    adapters = FakeAdapters()
    service = GenAIService(repository, adapters)
    events = [event async for event in service.stream_chat(ChatRequest(
        conversation_id="conversation-1", message="Train and predict with AutoML",
        tools=["automl"], tool_arguments={"automl": {}},
    ), OWNER, USER)]
    assert any(event["type"] == "error" and event["code"] == "LAB_ACTION_AMBIGUOUS" for event in events)
    assert adapters.executions == []


def test_automl_structured_values_and_ordered_followup_use_persisted_schema():
    schema = {
        "expected_features": ["age", "monthly_visits", "avg_order_value", "support_tickets"],
        "required_fields": ["age", "monthly_visits", "avg_order_value", "support_tickets"],
        "columns": {name: {"dtype": "float"} for name in ["age", "monthly_visits", "avg_order_value", "support_tickets"]},
    }
    row, missing = _schema_values(
        "age: 34, monthly_visits: 11, avg_order_value: 774, support_tickets: 6", schema,
    )
    assert not missing
    assert row == {"age": 34, "monthly_visits": 11, "avg_order_value": 774, "support_tickets": 6}

    partial = {"age": 34}
    _followup_values(partial, "original prediction\n11, 774, 6", schema)
    assert partial == {"age": 34, "monthly_visits": 11, "avg_order_value": 774, "support_tickets": 6}


def test_schema_slot_normalization_accepts_abbreviations_and_natural_names():
    schema = {
        "expected_features": ["avg_order_value"], "required_fields": ["avg_order_value"],
        "columns": {"avg_order_value": {"dtype": "float"}},
    }
    for phrase in ("avg_order_value=774", "avg order value 774", "average order value: 774"):
        row, missing = _schema_values(phrase, schema)
        assert row == {"avg_order_value": 774}
        assert missing == []


@pytest.mark.asyncio
async def test_bare_structured_values_resolve_native_automl_without_llm():
    repository = FakeRepository()
    adapters = FakeAdapters()
    events = [event async for event in GenAIService(repository, adapters).stream_chat(ChatRequest(
        conversation_id="conversation-1",
        message="age=34, monthly_visits=11, avg_order_value=774, support_tickets=6",
    ), OWNER, USER)]
    confirmation = next(event for event in events if event["type"] == "confirmation_required")
    assert confirmation["tool"] == "automl"
    assert confirmation["arguments"]["action"] == "predict"
    assert not any(event["type"] == "delta" for event in events)


@pytest.mark.asyncio
async def test_explicit_sentiment_analysis_routes_to_native_autonlp_prediction():
    repository = FakeRepository()
    adapters = FakeAdapters()
    events = [event async for event in GenAIService(repository, adapters).stream_chat(ChatRequest(
        conversation_id="conversation-1",
        message="Analyze the sentiment of: The company reported record profits.",
    ), OWNER, USER)]
    confirmation = next(event for event in events if event["type"] == "confirmation_required")
    assert confirmation["tool"] == "autonlp"
    assert confirmation["arguments"]["action"] == "predict"
    assert not any(event["type"] == "delta" for event in events)


@pytest.mark.asyncio
async def test_structured_automl_resolution_prefers_best_schema_match():
    adapters = GenAILabAdapters(SimpleNamespace())

    class AutoML:
        @staticmethod
        def list_models_for_owner(owner_id):
            return ["partial.pkl", "complete.pkl"]

        @staticmethod
        def load_owned_artifact(filename, owner_id):
            features = ["age"] if filename == "partial.pkl" else ["age", "monthly_visits"]
            schema = {
                "expected_features": features, "required_fields": features,
                "columns": {feature: {"dtype": "float"} for feature in features},
            }
            return SimpleNamespace(
                model_name=filename, target_column="churn", task="classification",
                metadata={"prediction_schema": schema}, original_feature_names=features,
            )

    adapters.__dict__["automl"] = AutoML()
    resolved = await adapters.resolve(
        "automl", USER, {"action": "predict", "_schema_match_required": True},
        "age=34, monthly_visits=11", [],
    )
    assert resolved["model_filename"] == "complete.pkl"
    assert resolved["rows"] == [{"age": 34, "monthly_visits": 11}]


def test_uncalibrated_native_scores_are_never_labelled_confidence():
    automl = _prediction_result("automl", {
        "task": "classification", "predictions": [1], "prediction_confidences": [0.9],
        "model_name": "decision_tree", "score_is_calibrated": False,
    })
    autodl = _prediction_result("autodl", {
        "prediction": {"predicted_class": "cat", "model_score": 0.8, "score_is_calibrated": False,
                       "explanation": "High confidence output."},
    })
    assert "Model score" in automl.content and "Confidence" not in automl.content
    assert "Model score" in autodl.content and "confidence" not in autodl.content.casefold()


@pytest.mark.asyncio
async def test_ambiguous_automl_models_require_selection_and_keep_owner_scope():
    adapters = GenAILabAdapters(SimpleNamespace())
    owners = []

    class AutoML:
        def list_models_for_owner(self, owner_id):
            owners.append(owner_id)
            return ["a.pkl", "b.pkl"]

        def load_owned_artifact(self, filename, owner_id):
            owners.append(owner_id)
            return SimpleNamespace(model_name=filename[0].upper(), target_column="label", task="classification", metadata={})

    adapters.__dict__["automl"] = AutoML()
    with pytest.raises(LabResourceSelectionRequired) as error:
        await adapters.resolve("automl", USER, {"action": "predict"}, "predict this row", [])
    assert len(error.value.candidates) == 2
    assert set(owners) == {OWNER}


@pytest.mark.asyncio
async def test_autonlp_prediction_accepts_current_message_csv_attachment():
    adapters = GenAILabAdapters(SimpleNamespace())

    class AutoNLP:
        @staticmethod
        def list_models(owner_id):
            assert owner_id == OWNER
            return [SimpleNamespace(
                model_id="nlp-1", model_type="linear_svm", task="sentiment_analysis",
                artifact_available=True,
            )]

    adapters.__dict__["autonlp"] = AutoNLP()
    resolved = await adapters.resolve(
        "autonlp", USER, {"action": "predict"}, "Predict sentiment in this CSV",
        [{"id": "attachment-1", "filename": "texts.csv", "extraction": {"format": "csv"}}],
    )
    assert resolved["model_id"] == "nlp-1"
    assert resolved["attachment_id"] == "attachment-1"
    assert resolved["text"] == ""


@pytest.mark.asyncio
async def test_autonlp_csv_batch_never_requests_single_prediction_text():
    adapters = GenAILabAdapters(SimpleNamespace())

    class AutoNLP:
        @staticmethod
        def list_models(owner_id):
            return [SimpleNamespace(
                model_id="nlp-1", model_type="linear_svm", task="sentiment_analysis",
                artifact_available=True,
            )]

    adapters.__dict__["autonlp"] = AutoNLP()
    parsed = GenAIService._tool_arguments(
        "autonlp", "Use my sentiment model on the CSV. Use text as the input column.",
        {"action": "predict"},
    )
    assert parsed["text_column"] == "text"
    resolved = await adapters.resolve(
        "autonlp", USER, parsed,
        "Use text as the input column", [{"id": "csv-1", "filename": "batch.csv", "content_type": "text/csv"}],
    )
    assert resolved["attachment_id"] == "csv-1"
    assert resolved["text"] == ""


@pytest.mark.asyncio
async def test_autonlp_missing_single_text_uses_natural_question():
    adapters = GenAILabAdapters(SimpleNamespace())

    class AutoNLP:
        @staticmethod
        def list_models(owner_id):
            return [SimpleNamespace(
                model_id="nlp-1", model_type="linear_svm", task="sentiment_analysis",
                artifact_available=True,
            )]

    adapters.__dict__["autonlp"] = AutoNLP()
    with pytest.raises(LabPredictionInputRequired) as error:
        await adapters.resolve("autonlp", USER, {"action": "predict"}, "Predict sentiment", [])
    assert str(error.value) == "What text would you like me to analyze?"
    assert "Provide:" not in str(error.value)


@pytest.mark.asyncio
async def test_autonlp_csv_executes_native_batch_and_preserves_rows():
    adapters = GenAILabAdapters(SimpleNamespace())

    class Repository:
        @staticmethod
        async def read_attachment(attachment_id, owner_id):
            assert (attachment_id, owner_id) == ("attachment-1", OWNER)
            return {"filename": "texts.csv", "extraction": {"format": "csv"}}, b"text\na\nb\nc\n"

    class AutoNLP:
        @staticmethod
        def _registered_model(model_id, owner_id):
            return SimpleNamespace(configuration={"text_column": "text"})

        @staticmethod
        def predict_batch(**kwargs):
            assert kwargs["text_column"] == "text"
            return {
                "model_id": "nlp-1", "text_column": "text", "total_rows": 3,
                "valid_rows": 3, "failed_rows": 0,
                "rows": [{"row_index": index, "predicted_label": label, "model_score": 0.8}
                         for index, label in enumerate(("positive", "neutral", "negative"))],
            }

    adapters.__dict__["autonlp"] = AutoNLP()
    result = await adapters.execute("autonlp", USER, {
        "action": "predict", "model_id": "nlp-1", "attachment_id": "attachment-1",
        "text_column": "text", "_repository": Repository(),
    })
    assert result.ok
    assert "3 of 3 rows completed" in result.content
    assert len(result.data["result"]["rows"]) == 3
    assert "positive: 1" in result.content


@pytest.mark.asyncio
async def test_automl_training_without_target_requests_only_target():
    adapters = GenAILabAdapters(SimpleNamespace())
    with pytest.raises(Exception) as error:
        await adapters.resolve(
            "automl", USER, {"action": "train", "attachment_id": "attachment-1", "task": "classification"},
            "Train an AutoML classification model", [{"id": "attachment-1"}],
        )
    assert getattr(error.value, "missing_fields", None) == ["target column"]


def test_training_followup_binds_the_single_requested_target():
    resolved = GenAIService._tool_arguments(
        "automl", "churn", {"action": "train", "task": "classification", "_requested_fields": ["target column"]},
    )
    assert resolved["target_column"] == "churn"


@pytest.mark.asyncio
async def test_forged_confirmed_tools_do_not_start_native_training():
    repository = FakeRepository()
    adapters = FakeAdapters()
    service = GenAIService(repository, adapters)
    request = ChatRequest(
        conversation_id="conversation-1", message="AutoML train model target is label",
        tools=["automl"], confirmed_tools=["automl"],
        tool_arguments={"automl": {"action": "train", "target_column": "label"}},
    )
    events = [event async for event in service.stream_chat(request, OWNER, USER)]
    assert any(event["type"] == "confirmation_required" for event in events)
    assert adapters.executions == []


@pytest.mark.asyncio
async def test_server_confirmation_executes_exact_native_action_once():
    repository = FakeRepository()
    adapters = FakeAdapters()
    service = GenAIService(repository, adapters)
    initial = ChatRequest(
        conversation_id="conversation-1", message="AutoML train model target is label",
        tools=["automl"], tool_arguments={"automl": {"action": "train", "target_column": "label"}},
    )
    first = [event async for event in service.stream_chat(initial, OWNER, USER)]
    confirmation_id = next(event["confirmation_id"] for event in first if event["type"] == "confirmation_required")
    confirmed = ChatRequest(
        conversation_id="conversation-1", message="confirm", confirmation_id=confirmation_id,
        tools=["autodl"], tool_arguments={"autodl": {"action": "train"}},
    )
    second = [event async for event in service.stream_chat(confirmed, OWNER, USER)]
    assert adapters.executions[0][0] == "automl"
    assert adapters.executions[0][1]["target_column"] == "label"
    assert any(event["type"] == "done" for event in second)


@pytest.mark.asyncio
async def test_stale_automl_prediction_cannot_hijack_explicit_autodl_request():
    repository = FakeRepository({"tool": "automl", "action": "predict", "arguments": {}, "attachment_ids": []})
    adapters = FakeAdapters(ToolResult("autodl", False, error_code="NATIVE_FAILED", error_message="native failure"))
    service = GenAIService(repository, adapters)
    request = ChatRequest(
        conversation_id="conversation-1", message="AutoDL predict feature_1=0.8",
        tools=["autodl"], tool_arguments={"autodl": {"action": "predict", "run_id": "run-1", "input": {"feature_1": 0.8}}},
    )
    events = [event async for event in service.stream_chat(request, OWNER, USER)]
    assert repository.cleared_prediction
    assert all(tool == "autodl" for tool, _ in adapters.executions) or not adapters.executions
    assert not any(event["type"] == "delta" for event in events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "binding"),
    (("automl", {"model_filename": "fixed.pkl"}), ("autodl", {"run_id": "run-1", "model_id": "model-1"})),
)
async def test_prediction_followup_keeps_exact_native_model_binding(tool, binding):
    pending = {
        "tool": tool, "action": "predict",
        "arguments": {"action": "predict", **binding, "_requested_fields": ["feature_1"]},
        "attachment_ids": [],
    }
    repository = FakeRepository(pending)
    adapters = FakeAdapters()
    service = GenAIService(repository, adapters)
    request = ChatRequest(
        conversation_id="conversation-1", message="feature_1=31", tools=[tool],
        tool_arguments={tool: {"action": "predict", "model_filename": "other.pkl", "run_id": "other-run"}},
    )
    events = [event async for event in service.stream_chat(request, OWNER, USER)]
    confirmation = next(event for event in events if event["type"] == "confirmation_required")
    for key, value in binding.items():
        assert confirmation["arguments"][key] == value


@pytest.mark.asyncio
async def test_autodl_same_bound_model_and_values_remain_identical_across_requests():
    repository = FakeRepository()
    adapters = FakeAdapters(ToolResult("autodl", True, "**Prediction:** Class 1\n\n**Model score:** 82.0%"))
    service = GenAIService(repository, adapters)
    outputs = []
    for _ in range(2):
        first = [event async for event in service.stream_chat(ChatRequest(
            conversation_id="conversation-1",
            message="AutoDL predict feature_1=0.8, feature_2=2.4, feature_3=2.6",
            tools=["autodl"], tool_arguments={"autodl": {
                "action": "predict", "run_id": "run-1", "model_id": "model-1",
                "input": {"feature_1": 0.8, "feature_2": 2.4, "feature_3": 2.6},
            }},
        ), OWNER, USER)]
        confirmation = next(event for event in first if event["type"] == "confirmation_required")
        final = [event async for event in service.stream_chat(ChatRequest(
            conversation_id="conversation-1", message="confirm",
            confirmation_id=confirmation["confirmation_id"],
        ), OWNER, USER)]
        outputs.append(next(event["message"]["content"] for event in final if event["type"] == "done"))
    assert outputs[0] == outputs[1]
    assert all(arguments["run_id"] == "run-1" and arguments["model_id"] == "model-1"
               for tool, arguments in adapters.executions if tool == "autodl")


@pytest.mark.asyncio
async def test_friendly_model_switch_resolves_only_the_named_owner_model():
    adapters = GenAILabAdapters(SimpleNamespace())

    class AutoML:
        @staticmethod
        def list_models_for_owner(owner_id):
            return ["tree.pkl", "logistic.pkl"]

        @staticmethod
        def load_owned_artifact(filename, owner_id):
            name = "Decision Tree" if filename == "tree.pkl" else "Logistic Regression"
            return SimpleNamespace(
                model_name=name, target_column="churn", task="classification",
                metadata={}, original_feature_names=[],
            )

    adapters.__dict__["automl"] = AutoML()
    resolved = await adapters.resolve(
        "automl", USER,
        {"action": "predict", "_explicit_resource_switch": True},
        "Use Logistic Regression instead", [],
    )
    assert resolved["model_filename"] == "logistic.pkl"


@pytest.mark.asyncio
async def test_explicit_same_lab_attachment_replaces_pending_attachment():
    repository = FakeRepository({
        "tool": "automl", "action": "train", "attachment_ids": ["old-file"],
        "arguments": {"action": "train", "attachment_id": "old-file", "task": "classification", "target_column": "label"},
    })
    adapters = FakeAdapters()
    events = [event async for event in GenAIService(repository, adapters).stream_chat(ChatRequest(
        conversation_id="conversation-1", message="Train this AutoML model",
        tools=["automl"], attachment_ids=["new-file"],
        tool_arguments={"automl": {"action": "train"}},
    ), OWNER, USER)]
    confirmation = next(event for event in events if event["type"] == "confirmation_required")
    assert confirmation["attachment_ids"] == ["new-file"]
    assert confirmation["arguments"]["attachment_id"] == "new-file"


@pytest.mark.asyncio
async def test_bare_autodl_continuation_reuses_pending_attachment():
    repository = FakeRepository({
        "tool": "autodl", "action": "train", "attachment_ids": ["autodl-file"],
        "arguments": {
            "action": "train", "attachment_id": "autodl-file", "run_id": "run-1",
            "task": "classification", "confirmed_task": "tabular_classification",
            "target_column": "label",
        },
    })
    events = [event async for event in GenAIService(repository, FakeAdapters()).stream_chat(
        ChatRequest(conversation_id="conversation-1", message="continue"), OWNER, USER,
    )]
    confirmation = next(event for event in events if event["type"] == "confirmation_required")
    assert confirmation["attachment_ids"] == ["autodl-file"]
    assert confirmation["arguments"]["attachment_id"] == "autodl-file"


@pytest.mark.asyncio
async def test_multiple_autonlp_csv_attachments_require_one_selection():
    repository = FakeRepository()
    adapters = FakeAdapters()
    events = [event async for event in GenAIService(repository, adapters).stream_chat(ChatRequest(
        conversation_id="conversation-1", message="Predict sentiment in these CSV files",
        tools=["autonlp"], attachment_ids=["csv-1", "csv-2"],
        tool_arguments={"autonlp": {"action": "predict"}},
    ), OWNER, USER)]
    error = next(event for event in events if event["type"] == "error")
    assert error["code"] == "LAB_RESOURCE_SELECTION_REQUIRED"
    assert len(error["details"]["candidates"]) == 2
    assert adapters.executions == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "arguments"),
    (
        ("automl", {"action": "train", "task": "classification", "target_column": "label"}),
        ("autonlp", {"action": "train", "task": "sentiment_analysis", "text_column": "text", "target_column": "sentiment"}),
    ),
)
async def test_training_confirmation_retains_current_attachment(tool, arguments):
    repository = FakeRepository()
    adapters = FakeAdapters()
    service = GenAIService(repository, adapters)
    request = ChatRequest(
        conversation_id="conversation-1", message=f"Train with {tool}", tools=[tool],
        attachment_ids=["attachment-1"], tool_arguments={tool: arguments},
    )
    events = [event async for event in service.stream_chat(request, OWNER, USER)]
    confirmation = next(event for event in events if event["type"] == "confirmation_required")
    assert confirmation["attachment_ids"] == ["attachment-1"]
    assert confirmation["arguments"]["attachment_id"] == "attachment-1"
    assert repository.conversation["pending_prediction"]["action"] == "train"


@pytest.mark.asyncio
async def test_autodl_detection_persists_task_before_confirmation():
    adapters = GenAILabAdapters(SimpleNamespace())

    class Repository:
        @staticmethod
        async def read_attachment(attachment_id, owner_id):
            return {"filename": "dataset.csv"}, b"feature,label\n1,a\n2,b\n"

    class AutoDL:
        @staticmethod
        def inspect_dataset(**kwargs):
            return SimpleNamespace(
                run_id="run-1",
                task_intelligence=SimpleNamespace(
                    detected_task=SimpleNamespace(value="tabular_classification"),
                    requires_confirmation=True,
                ),
            )

    class TrainingRepository:
        @staticmethod
        def get_run(run_id, owner_id):
            return {
                "inspection": {"task_intelligence": {
                    "detected_task": "tabular_classification", "requires_confirmation": True,
                }},
                "advanced_details": {"selected_target": "label"},
            }

    adapters.__dict__["genai_repository"] = Repository()
    adapters.__dict__["autodl"] = AutoDL()
    adapters.__dict__["autodl_training"] = SimpleNamespace(repository=TrainingRepository())
    resolved = await adapters.resolve(
        "autodl", USER,
        {"action": "train", "attachment_id": "attachment-1", "task": "classification", "target_column": "label"},
        "Train a tabular classification model. Target is label.",
        [{"id": "attachment-1", "filename": "dataset.csv", "content_type": "text/csv"}],
    )
    assert resolved["run_id"] == "run-1"
    assert resolved["confirmed_task"] == "tabular_classification"
    assert resolved["confirmed_target"] == "label"


@pytest.mark.asyncio
async def test_autodl_attachment_survives_inspect_confirmation_and_submission():
    repository = FakeRepository()

    class AutoDLAdapters(FakeAdapters):
        async def resolve(self, tool, user, arguments, query, selected_attachments):
            assert selected_attachments[0]["id"] == "autodl-file"
            return {
                **arguments, "action": "train", "attachment_id": "autodl-file",
                "run_id": "run-1", "task": "classification",
                "confirmed_task": "tabular_classification", "confirmed_target": "label",
            }

    adapters = AutoDLAdapters(ToolResult("autodl", True, "native AutoDL queued"))
    service = GenAIService(repository, adapters)
    first = [event async for event in service.stream_chat(ChatRequest(
        conversation_id="conversation-1",
        message="Train an AutoDL tabular classification model. Target column is label.",
        tools=["autodl"], attachment_ids=["autodl-file"],
        tool_arguments={"autodl": {"action": "train", "task": "classification", "target_column": "label"}},
    ), OWNER, USER)]
    confirmation = next(event for event in first if event["type"] == "confirmation_required")
    assert confirmation["attachment_ids"] == ["autodl-file"]
    second = [event async for event in service.stream_chat(ChatRequest(
        conversation_id="conversation-1", message="confirm",
        confirmation_id=confirmation["confirmation_id"],
    ), OWNER, USER)]
    assert adapters.executions[0][1]["attachment_id"] == "autodl-file"
    assert adapters.executions[0][1]["run_id"] == "run-1"
    assert any(event["type"] == "done" for event in second)
    assert repository.conversation["active_lab_resources"]["autodl"]["run_id"] == "run-1"


@pytest.mark.asyncio
async def test_autodl_genuinely_ambiguous_task_uses_natural_question():
    adapters = GenAILabAdapters(SimpleNamespace())

    class TrainingRepository:
        @staticmethod
        def get_run(run_id, owner_id):
            return {
                "inspection": {"dataset_kind": "csv", "task_intelligence": {
                    "detected_task": None, "requires_confirmation": True,
                }}, "advanced_details": {},
            }

    adapters.__dict__["autodl_training"] = SimpleNamespace(repository=TrainingRepository())
    with pytest.raises(LabPredictionInputRequired) as error:
        await adapters.resolve(
            "autodl", USER, {"action": "train", "run_id": "run-1", "attachment_id": "file-1"},
            "Continue training", [{"id": "file-1"}],
        )
    assert str(error.value) == "Is this a classification or regression task?"


@pytest.mark.asyncio
async def test_pending_training_survives_missing_field_followup_and_confirmation():
    repository = FakeRepository()

    class TrainingAdapters(FakeAdapters):
        async def resolve(self, tool, user, arguments, query, selected_attachments):
            values = dict(arguments)
            values.setdefault("action", "train")
            if not values.get("target_column"):
                raise LabPredictionInputRequired(
                    "Please provide only the target column.", ["target column"], values,
                )
            return values

    adapters = TrainingAdapters()
    service = GenAIService(repository, adapters)
    first = ChatRequest(
        conversation_id="conversation-1", message="Train an AutoML classification model",
        tools=["automl"], attachment_ids=["attachment-1"],
        tool_arguments={"automl": {"action": "train", "task": "classification"}},
    )
    first_events = [event async for event in service.stream_chat(first, OWNER, USER)]
    assert any(event["type"] == "error" for event in first_events)
    assert repository.conversation["pending_prediction"]["attachment_ids"] == ["attachment-1"]

    second = ChatRequest(conversation_id="conversation-1", message="label")
    second_events = [event async for event in service.stream_chat(second, OWNER, USER)]
    confirmation = next(event for event in second_events if event["type"] == "confirmation_required")
    assert confirmation["arguments"]["target_column"] == "label"
    assert confirmation["attachment_ids"] == ["attachment-1"]

    third = ChatRequest(
        conversation_id="conversation-1", message="confirm",
        confirmation_id=confirmation["confirmation_id"],
    )
    final_events = [event async for event in service.stream_chat(third, OWNER, USER)]
    assert adapters.executions[0][0] == "automl"
    assert adapters.executions[0][1]["target_column"] == "label"
    assert any(event["type"] == "done" for event in final_events)
    assert repository.conversation["pending_prediction"] is None


@pytest.mark.asyncio
async def test_cancel_clears_pending_action_and_server_confirmation():
    repository = FakeRepository({
        "tool": "autonlp", "action": "train", "arguments": {"action": "train"},
        "attachment_ids": ["attachment-1"],
    })
    repository.confirmation = {"id": "confirm-1", "tool": "autonlp", "action": "train"}
    repository.conversation["pending_confirmation"] = dict(repository.confirmation)
    events = [event async for event in GenAIService(repository, FakeAdapters()).stream_chat(
        ChatRequest(conversation_id="conversation-1", message="cancel"), OWNER, USER,
    )]
    assert repository.conversation["pending_prediction"] is None
    assert repository.conversation["pending_confirmation"] is None
    assert any(event["type"] == "done" for event in events)


@pytest.mark.asyncio
async def test_approved_native_http_error_is_surfaced_safely():
    registry = GenAIToolRegistry()

    async def failing_handler(context, arguments):
        raise HTTPException(status_code=422, detail="Target column is invalid.")

    registry.register(ToolDefinition(
        name="automl", description="AutoML", input_schema={}, permissions=(),
        handler=failing_handler, availability=lambda: (True, None),
    ))
    result = await registry.execute(
        "automl", ToolExecutionContext(OWNER, "train", None), {"action": "train"},
    )
    assert not result.ok
    assert result.error_message == "Target column is invalid."


@pytest.mark.asyncio
async def test_native_failure_never_falls_back_to_llm_generation():
    repository = FakeRepository()
    adapters = FakeAdapters(ToolResult("automl", False, error_code="NATIVE_FAILED", error_message="native failure"))
    service = GenAIService(repository, adapters)
    request = ChatRequest(
        conversation_id="conversation-1", message="AutoML predict age=34",
        tools=["automl"], tool_arguments={"automl": {"action": "predict", "model_filename": "owned.pkl", "rows": [{"age": 34}]}},
    )
    # Grant a real server-side confirmation for the already resolved request.
    repository.confirmation = {
        "id": "confirmation-1", "tool": "automl", "action": "predict",
        "arguments": request.tool_arguments["automl"], "attachment_ids": [],
    }
    request.confirmation_id = "confirmation-1"
    events = [event async for event in service.stream_chat(request, OWNER, USER)]
    assert any(event["type"] == "error" and event["message"] == "native failure" for event in events)
    assert not any(event["type"] == "delta" for event in events)


def test_frontend_deduplicates_and_clears_native_attachment_state():
    hook = (Path(__file__).parents[2] / "frontend-main/src/hooks/useGenAIChat.ts").read_text(encoding="utf-8")
    assert "current.filter(item => item.id !== attachment.id)" in hook
    assert 'pendingResolution ? [id] : [...current, id]' in hook
    assert 'setSelectedAttachmentIds([]);' in hook
    assert 'nlpPredictionIntent' in hook


def test_frontend_never_emits_internal_provide_slot_copy():
    frontend = Path(__file__).parents[2] / "frontend-main/src"
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (frontend / "components/genai/ChatWindow.tsx", frontend / "hooks/useGenAIChat.ts")
    )
    assert "Provide: <slot>" not in source
    assert 'What text would you like me to analyze?' in source
    assert 'Which column contains the text to analyze?' in source


def test_autodl_queued_and_running_status_formatting():
    queued = _autodl_status_result({"run_id": "r1", "status": "queued", "stage": "preparing", "percentage": 0})
    running = _autodl_status_result({
        "run_id": "r1", "status": "running", "stage": "training", "percentage": 42,
        "current_epoch": 2, "total_epochs": 5, "latest_metrics": {"loss": 0.4},
    })
    assert "queued and waiting to be executed" in queued.content
    assert "AutoDL training is running" in running.content
    assert "42.0%" in running.content and "2 of 5" in running.content and "Loss" in running.content
    assert "Final results are not available yet" in running.content


def test_autodl_completed_and_failed_status_formatting_uses_native_values():
    completed = _autodl_status_result(
        {"run_id": "r1", "status": "completed", "completed_at": "2026-01-01T00:00:00Z"},
        {
            "problem": {"display_name": "Tabular Classification"},
            "best_model": {"name": "MLP"},
            "performance": {"accuracy": 0.91, "weighted_f1": 0.89},
            "prediction_ready": True,
        },
    )
    failed = _autodl_status_result({
        "run_id": "r2", "status": "failed", "failure": {"message": "Target column is invalid."},
    })
    assert "MLP" in completed.content and "91.0%" in completed.content and "89.0%" in completed.content
    assert "Ready for prediction" in completed.content
    assert "AutoDL training failed" in failed.content and "Target column is invalid" in failed.content


@pytest.mark.asyncio
async def test_autodl_result_before_completion_returns_native_progress(monkeypatch):
    adapters = GenAILabAdapters(SimpleNamespace())

    class Training:
        @staticmethod
        def get_status(run_id, owner_id):
            assert (run_id, owner_id) == ("run-1", OWNER)
            return {"run_id": run_id, "status": "running", "stage": "training", "percentage": 25}

    adapters.__dict__["autodl_training"] = Training()
    from app.modules.autodl_v2.runtime import runtime
    monkeypatch.setattr(runtime, "has_active_run", lambda run_id: True)
    result = await adapters.execute("autodl", USER, {"action": "result", "run_id": "run-1"})
    assert result.ok and "25.0%" in result.content and "Final results are not available yet" in result.content


@pytest.mark.asyncio
async def test_autodl_stale_run_and_cancellation_are_truthful(monkeypatch):
    adapters = GenAILabAdapters(SimpleNamespace())

    class Training:
        @staticmethod
        def get_status(run_id, owner_id):
            return {"run_id": run_id, "status": "queued", "stage": "preparing", "percentage": 0}

    adapters.__dict__["autodl_training"] = Training()
    from app.modules.autodl_v2.runtime import runtime
    monkeypatch.setattr(runtime, "has_active_run", lambda run_id: False)
    stale = await adapters.execute("autodl", USER, {"action": "status", "run_id": "run-1"})
    cancelled = await adapters.execute("autodl", USER, {"action": "cancel", "run_id": "run-1"})
    assert "interrupted" in stale.content and "may need to be restarted" in stale.content
    assert "cancellation is currently unsupported" in cancelled.content
    assert "cancelled" not in cancelled.content.casefold()


@pytest.mark.asyncio
async def test_conversation_bound_autodl_run_survives_refresh_and_status_never_uses_llm():
    repository = FakeRepository()
    repository.conversation["active_lab_resources"] = {"autodl": {"run_id": "bound-run"}}
    adapters = FakeAdapters(ToolResult(
        "autodl", True, "**Status:** AutoDL training completed.",
        data={"result": {"run_id": "bound-run", "status": "completed"}},
    ))
    events = [event async for event in GenAIService(repository, adapters).stream_chat(ChatRequest(
        conversation_id="conversation-1", message="What's the status of my AutoDL training?",
    ), OWNER, USER)]
    assert adapters.executions[0][0] == "autodl"
    assert adapters.executions[0][1]["run_id"] == "bound-run"
    assert adapters.executions[0][1]["action"] == "status"
    assert any(event["type"] == "done" for event in events)
    assert not any(event["type"] == "delta" for event in events)


@pytest.mark.asyncio
async def test_explicit_latest_autodl_run_is_deterministic_but_ambiguous_request_requires_selection():
    class Cursor:
        def __init__(self, rows):
            self.rows = rows

        def sort(self, *args):
            return self

        def limit(self, count):
            return self.rows[:count]

    class Runs:
        @staticmethod
        def find(query, projection):
            assert query == {"owner_id": OWNER}
            return Cursor([
                {"_id": "newest", "filename": "new.csv", "status": "running", "task": "tabular_classification"},
                {"_id": "older", "filename": "old.csv", "status": "completed", "task": "tabular_classification"},
            ])

    class Repository:
        runs = Runs()

        @staticmethod
        def get_run(run_id, owner_id):
            assert owner_id == OWNER
            return {"_id": run_id, "status": "running", "inspection": {}, "advanced_details": {}}

    adapters = GenAILabAdapters(SimpleNamespace())
    adapters.__dict__["autodl_training"] = SimpleNamespace(repository=Repository())
    latest = await adapters.resolve("autodl", USER, {"action": "status"}, "Show my latest AutoDL run", [])
    assert latest["run_id"] == "newest"
    with pytest.raises(LabResourceSelectionRequired) as error:
        await adapters.resolve("autodl", USER, {"action": "status"}, "Show AutoDL status", [])
    assert {item["run_id"] for item in error.value.candidates} == {"newest", "older"}


@pytest.mark.asyncio
async def test_explicit_foreign_autodl_run_is_not_found():
    adapters = GenAILabAdapters(SimpleNamespace())

    class Repository:
        @staticmethod
        def get_run(run_id, owner_id):
            assert (run_id, owner_id) == ("foreign-run", OWNER)
            raise LookupError("AutoDL inspection run not found.")

    adapters.__dict__["autodl_training"] = SimpleNamespace(repository=Repository())
    with pytest.raises(LookupError, match="not found"):
        await adapters.resolve(
            "autodl", USER, {"action": "status", "run_id": "foreign-run"},
            "Show status for run_id=foreign-run", [],
        )


def test_frontend_and_backend_route_autodl_status_result_without_readiness_fallback():
    status = GenAIService._tool_arguments("autodl", "Show progress of my AutoDL training", {})
    result = GenAIService._tool_arguments("autodl", "Show the result of my last AutoDL training", {})
    cancel = GenAIService._tool_arguments("autodl", "Cancel my AutoDL training", {})
    assert status["action"] == "status"
    assert result["action"] == "result"
    assert cancel["action"] == "cancel"
    hook = (Path(__file__).parents[2] / "frontend-main/src/hooks/useGenAIChat.ts").read_text(encoding="utf-8")
    assert '[/\\b(status|progress|ready|latest|last)\\b/i, "status"]' in hook
