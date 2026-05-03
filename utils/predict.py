from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from utils.preprocessing import load_features, transform_input


def default_artifact_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    model_path = root / "model" / "model.pkl"
    features_path = root / "model" / "feature_columns.pkl"

    if not model_path.exists():
        model_path = root / "model.pkl"
    if not features_path.exists():
        features_path = root / "feature_columns.pkl"
    if not features_path.exists():
        features_path = root / "model" / "features.pkl"
    if not features_path.exists():
        features_path = root / "features.pkl"

    return model_path, features_path


def load_prediction_assets(
    model_path: str | Path | None = None,
    features_path: str | Path | None = None,
) -> tuple[Any, list[str]]:
    default_model_path, default_features_path = default_artifact_paths()
    model_path = Path(model_path) if model_path else default_model_path
    features_path = Path(features_path) if features_path else default_features_path

    model = joblib.load(model_path)
    features = load_features(features_path)
    return model, features


def predict_cancellation(model: Any, features: list[str], inputs: dict[str, Any]) -> dict[str, Any]:
    feature_frame = transform_input(inputs, features)
    probability = _positive_class_probability(model, feature_frame)
    prediction = int(probability >= 0.5)
    recommendations = generate_recommendations(inputs, probability)

    return {
        "label": "Likely to Cancel" if prediction else "Not Likely",
        "probability": probability,
        "prediction": prediction,
        "features": feature_frame,
        "recommendations": recommendations,
    }


def explain_prediction(model: Any, feature_frame: pd.DataFrame, max_features: int = 10) -> pd.DataFrame:
    """Return top SHAP contributions when shap is installed and compatible."""
    if feature_frame.empty:
        raise ValueError("SHAP input is empty.")
    if any(dtype.kind not in "biufc" for dtype in feature_frame.dtypes):
        raise ValueError("SHAP input must be fully numeric and encoded.")
    if feature_frame.isna().any().any():
        raise ValueError("SHAP input contains missing values.")

    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_frame)

    if isinstance(shap_values, list):
        values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        values = shap_values

    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]

    contribution = pd.DataFrame(
        {
            "feature": feature_frame.columns,
            "contribution": values[0],
            "value": feature_frame.iloc[0].values,
        }
    )
    contribution["impact"] = contribution["contribution"].abs()
    return contribution.sort_values("impact", ascending=False).head(max_features)


def generate_recommendations(inputs: dict[str, Any], probability: float) -> list[str]:
    high_risk = probability >= 0.5
    lead_time = _safe_float(inputs.get("lead_time"), 0)
    previous_cancellations = _safe_float(inputs.get("previous_cancellations"), 0)
    market_segment = str(inputs.get("market_segment", ""))
    deposit_type = str(inputs.get("deposit_type", ""))

    if not high_risk:
        return [
            "Mark as safe booking.",
            "Maintain the current booking policy and continue routine monitoring.",
        ]

    recommendations = [
        "Review the booking before confirmation because the predicted cancellation risk is high.",
        "Suggest reducing lead time through closer arrival-date confirmation or targeted reminders.",
        "Suggest a deposit requirement or stricter guarantee policy for this reservation.",
    ]

    if market_segment == "Online TA":
        recommendations.append("Flag Online TA as a risky segment and monitor channel behavior closely.")
    if previous_cancellations > 0:
        recommendations.append("Flag repeated cancellations from the guest history.")
    if deposit_type == "No Deposit":
        recommendations.append("Convert no-deposit booking into a refundable or non-refundable deposit option.")
    if lead_time > 90:
        recommendations.append("Long lead time detected; schedule pre-arrival confirmation checkpoints.")

    return recommendations


def _positive_class_probability(model: Any, feature_frame: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_frame)
        return float(probabilities[0][1] if probabilities.shape[1] > 1 else probabilities[0][0])

    prediction = model.predict(feature_frame)
    if isinstance(prediction, (list, tuple, np.ndarray)):
        return float(prediction[0])
    return float(prediction)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
