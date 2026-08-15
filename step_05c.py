import pandas as pd
import numpy as np

from app.modules.automl.service import AutoMLService


s = AutoMLService()


# ============================================================
# DATASETS
# ============================================================

classification = pd.DataFrame({
    "age": [
        21,25,30,35,40,45,50,55,60,65,
        22,28,33,38,43,48,53,58,63,68
    ],
    "income": [
        25000,30000,40000,50000,60000,70000,80000,90000,100000,110000,
        27000,35000,45000,55000,65000,75000,85000,95000,105000,120000
    ],
    "approved": [
        "no","no","no","yes","yes","yes","yes","yes","yes","yes",
        "no","no","yes","yes","yes","yes","yes","yes","yes","yes"
    ]
})


regression = pd.DataFrame({
    "age": [
        21,22,25,28,31,35,38,42,45,48,
        52,55,58,61,64,67,70,73,76,79
    ],
    "experience": [
        0,1,2,4,6,8,10,12,15,17,
        20,22,25,28,30,32,35,38,40,42
    ],
    "education_score": [
        55,57,60,64,68,70,73,76,79,81,
        84,86,88,90,92,94,95,96,97,98
    ],
    "salary": [
        25000,27000,30000,35000,40000,46000,52000,58000,65000,72000,
        82000,90000,98000,108000,118000,128000,138000,148000,158000,170000
    ]
})


unsupervised = pd.DataFrame({
    "feature1": [
        1,1.1,0.9,1.2,1.05,0.95,1.15,1.0,1.08,0.92,
        1.1,1.02,0.98,1.12,1.06,0.94,1.03,1.07,1.01,1.09,
        50,55,60,65
    ],
    "feature2": [
        1,1.2,0.8,1.1,1.0,0.9,1.3,1.05,1.15,0.85,
        1.1,0.95,1.05,1.2,0.9,1.0,1.1,0.8,1.05,1.15,
        50,55,60,65
    ]
})


dimensionality = pd.DataFrame({
    "feature1": range(1,21),
    "feature2": range(2,41,2),
    "feature3": [
        5,4,6,3,7,2,8,1,9,2,
        10,3,11,4,12,5,13,6,14,7
    ],
    "feature4": [
        10,9,8,7,6,5,4,3,2,1,
        11,12,13,14,15,16,17,18,19,20
    ],
    "feature5": [
        3,5,2,8,1,7,4,9,6,10,
        12,11,14,13,15,17,16,19,18,20
    ]
})


tests = [
    ("classification", classification, "approved"),
    ("regression", regression, "salary"),
    ("clustering", unsupervised, None),
    ("anomaly", unsupervised, None),
    ("dimensionality", dimensionality, None),
]


print()
print("STEP 05C INFERENCE / PREDICTION VALIDATION")
print("=" * 70)

passed = 0


for task, df, target in tests:

    print()
    print(f"[{task.upper()}]")

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    result = s.train(
        df,
        target,
        task=task
    )

    assert result.task == task
    assert result.best_model is not None
    assert result.best_model.success
    assert result.best_model.model is not None

    model = result.best_model.model

    # --------------------------------------------------------
    # REMOVE TARGET COLUMN FOR SUPERVISED INFERENCE
    # --------------------------------------------------------

    if target is not None:
        prediction_df = df.drop(
            columns=[target]
        ).copy()
    else:
        prediction_df = df.copy()

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    predictions = s.predict(
        model,
        prediction_df
    )

    print(
        "  best_model        :",
        result.best_model.model_name
    )

    print(
        "  prediction_type   :",
        type(predictions).__name__
    )

    # Prediction should not be None.
    assert predictions is not None

    # --------------------------------------------------------
    # NORMALIZE OUTPUT FOR VALIDATION
    # --------------------------------------------------------

    if isinstance(predictions, pd.DataFrame):
        prediction_length = len(predictions)

    elif isinstance(predictions, pd.Series):
        prediction_length = len(predictions)

    elif isinstance(predictions, np.ndarray):
        prediction_length = len(predictions)

    elif isinstance(predictions, (list, tuple)):
        prediction_length = len(predictions)

    else:
        prediction_length = 1

    print(
        "  prediction_length :",
        prediction_length
    )

    # For all five current implementations,
    # inference should return one output per input row.
    assert prediction_length == len(prediction_df)

    # --------------------------------------------------------
    # TASK-SPECIFIC VALIDATION
    # --------------------------------------------------------

    if task == "classification":

        assert prediction_length == 20

        print(
            "  expected rows     :",
            len(prediction_df)
        )

    elif task == "regression":

        assert prediction_length == 20

        print(
            "  expected rows     :",
            len(prediction_df)
        )

    elif task == "clustering":

        assert prediction_length == 24

        print(
            "  expected rows     :",
            len(prediction_df)
        )

    elif task == "anomaly":

        assert prediction_length == 24

        print(
            "  expected rows     :",
            len(prediction_df)
        )

    elif task == "dimensionality":

        assert prediction_length == 20

        print(
            "  expected rows     :",
            len(prediction_df)
        )

    print(
        "  predict()         : PASS"
    )

    # --------------------------------------------------------
    # BATCH PREDICTION
    # --------------------------------------------------------

    batch_predictions = s.predict_batch(
        model,
        prediction_df
    )

    assert batch_predictions is not None

    if isinstance(batch_predictions, pd.DataFrame):
        batch_length = len(batch_predictions)

    elif isinstance(batch_predictions, pd.Series):
        batch_length = len(batch_predictions)

    elif isinstance(batch_predictions, np.ndarray):
        batch_length = len(batch_predictions)

    elif isinstance(batch_predictions, (list, tuple)):
        batch_length = len(batch_predictions)

    else:
        batch_length = 1

    print(
        "  batch_length      :",
        batch_length
    )

    assert batch_length == len(prediction_df)

    print(
        "  predict_batch()   : PASS"
    )

    print(
        "  INFERENCE: PASS"
    )

    passed += 1


print()
print("=" * 70)
print(f"PASSED: {passed}/5")

assert passed == 5

print("STEP 05C INFERENCE / PREDICTION PASS")
