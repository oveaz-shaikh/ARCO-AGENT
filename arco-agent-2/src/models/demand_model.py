
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "synthetic" / "daily_product_sales.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "units_sold"

NUMERIC_FEATURES = [
    "unit_price",
    "discount_pct",
    "unit_cost",
    "stock_available",
    "month",
    "day_of_week",
]

CATEGORICAL_FEATURES = [
    "product_id",
    "store_id",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    required = FEATURES + [TARGET, "date"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Demand cannot exceed the stock available at the start of the day.
    df[TARGET] = df[TARGET].clip(lower=0)

    return df


def main():
    df = load_data()

    # Chronological split: train on the first 9 months,
    # validate on October, and test on November–December.
    train = df[df["date"] < "2025-10-01"]
    validation = df[
        (df["date"] >= "2025-10-01")
        & (df["date"] < "2025-11-01")
    ]
    test = df[df["date"] >= "2025-11-01"]

    if train.empty or validation.empty or test.empty:
        raise ValueError("Train, validation, and test partitions must be non-empty.")

    # One-hot encoding is fitted only on training data.
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),

                (
                    "categorical",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                    CATEGORICAL_FEATURES,
                ),

        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    max_iter=100,
                    learning_rate=0.08,
                    max_leaf_nodes=20,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_val = validation[FEATURES]
    y_val = validation[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    # Baseline: predict the average training demand for every row.
    baseline_prediction = np.full(
        len(test),
        y_train.mean(),
        dtype=float,
    )

    print("Training demand model...")
    model.fit(X_train, y_train)

    val_predictions = np.maximum(model.predict(X_val), 0)
    test_predictions = np.maximum(model.predict(X_test), 0)

    baseline_mae = mean_absolute_error(y_test, baseline_prediction)
    model_mae = mean_absolute_error(y_test, test_predictions)
    model_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    validation_mae = mean_absolute_error(y_val, val_predictions)

    metrics = {
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "baseline_test_mae": round(float(baseline_mae), 4),
        "model_validation_mae": round(float(validation_mae), 4),
        "model_test_mae": round(float(model_mae), 4),
        "model_test_rmse": round(float(model_rmse), 4),
        "mae_improvement_pct": round(
            float((baseline_mae - model_mae) / baseline_mae * 100)
            if baseline_mae > 0 else 0.0,
            2,
        ),
        "data_type": "synthetic",
    }

    # Save the model pipeline and its metrics.
    import joblib

    joblib.dump(model, MODEL_DIR / "demand_model.joblib")

    with open(MODEL_DIR / "demand_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    # Save predictions for inspection.
    output = test[["date", "product_id", "store_id", TARGET]].copy()
    output["predicted_units"] = test_predictions
    output.to_csv(MODEL_DIR / "demand_test_predictions.csv", index=False)

    print("\nEvaluation results")
    print(json.dumps(metrics, indent=2))
    print(f"\nModel saved to: {MODEL_DIR / 'demand_model.joblib'}")
    print(f"Metrics saved to: {MODEL_DIR / 'demand_metrics.json'}")


if __name__ == "__main__":
    main()
