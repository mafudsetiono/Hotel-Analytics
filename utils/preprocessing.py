from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureOptions:
    deposit_types: list[str]
    market_segments: list[str]
    customer_types: list[str]


def load_features(path: str | Path) -> list[str]:
    """Load the trained feature columns from a joblib/pickle artifact."""
    features = joblib.load(path)
    if hasattr(features, "tolist"):
        features = features.tolist()
    return [str(feature) for feature in features]


def get_prediction_options(features: Iterable[str]) -> FeatureOptions:
    """Infer form options from one-hot columns, including dropped baselines."""
    feature_list = list(features)
    return FeatureOptions(
        deposit_types=_options_from_prefix(feature_list, "deposit_type_", "No Deposit"),
        market_segments=_options_from_prefix(feature_list, "market_segment_", "Aviation"),
        customer_types=_options_from_prefix(feature_list, "customer_type_", "Contract"),
    )


def transform_input(inputs: dict[str, Any], features: list[str]) -> pd.DataFrame:
    """Encode business inputs with get_dummies and align to training columns."""
    total_stay = max(_to_float(inputs.get("total_stay"), 0), 0)
    if total_stay == 0:
        total_stay = max(
            _to_float(inputs.get("stays_in_weekend_nights"), 0)
            + _to_float(inputs.get("stays_in_week_nights"), 0),
            0,
        )
    adr = max(_to_float(inputs.get("adr"), 0), 0)
    lead_time = max(_to_float(inputs.get("lead_time"), 0), 0)
    previous_cancellations = max(_to_float(inputs.get("previous_cancellations"), 0), 0)
    special_requests = max(_to_float(inputs.get("total_of_special_requests"), 0), 0)
    adults = max(_to_float(inputs.get("adults"), 2), 0)
    children = max(_to_float(inputs.get("children"), 0), 0)
    babies = max(_to_float(inputs.get("babies"), 0), 0)

    raw_input = {
        "lead_time": lead_time,
        "arrival_date_year": _to_float(inputs.get("arrival_date_year"), 2017),
        "adults": adults,
        "children": children,
        "babies": babies,
        "is_repeated_guest": _to_float(inputs.get("is_repeated_guest"), 0),
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": _to_float(inputs.get("previous_bookings_not_canceled"), 0),
        "booking_changes": _to_float(inputs.get("booking_changes"), 0),
        "days_in_waiting_list": _to_float(inputs.get("days_in_waiting_list"), 0),
        "adr": adr,
        "required_car_parking_spaces": _to_float(inputs.get("required_car_parking_spaces"), 0),
        "total_of_special_requests": special_requests,
        "total_stay": total_stay,
        "month_num": _to_float(inputs.get("month_num"), 7),
        "risk_score": calculate_risk_score(
            lead_time=lead_time,
            previous_cancellations=previous_cancellations,
            deposit_type=str(inputs.get("deposit_type", "No Deposit")),
        ),
        "deposit_type": str(inputs.get("deposit_type", "No Deposit")),
        "market_segment": str(inputs.get("market_segment", "Online TA")),
        "customer_type": str(inputs.get("customer_type", "Transient")),
        "hotel": str(inputs.get("hotel", "")),
        "meal": str(inputs.get("meal", "")),
        "distribution_channel": str(inputs.get("distribution_channel", "")),
        "reserved_room_type": str(inputs.get("reserved_room_type", "")),
        "lead_time_bin": bucket_lead_time(lead_time),
        "value_segment": bucket_value_segment(adr * total_stay),
        "family_type": "Family" if children > 0 or babies > 0 else "Non-Family",
        "stay_type": bucket_stay_type(total_stay),
    }

    frame = pd.DataFrame([raw_input])
    categorical_columns = [
        "deposit_type",
        "market_segment",
        "customer_type",
        "hotel",
        "meal",
        "distribution_channel",
        "reserved_room_type",
        "lead_time_bin",
        "value_segment",
        "family_type",
        "stay_type",
    ]
    encoded = pd.get_dummies(frame, columns=categorical_columns, dtype=float)
    aligned = encoded.reindex(columns=features, fill_value=0.0)
    return aligned.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)


def transform_dataset_for_model(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Vectorized dataset encoding for model evaluation."""
    frame = df.copy()
    frame["lead_time"] = _numeric_series(frame, "lead_time", 0).clip(lower=0)
    frame["adr"] = _numeric_series(frame, "adr", 0).clip(lower=0)
    frame["previous_cancellations"] = _numeric_series(frame, "previous_cancellations", 0).clip(lower=0)
    frame["total_of_special_requests"] = _numeric_series(frame, "total_of_special_requests", 0).clip(lower=0)
    frame["total_stay"] = _numeric_series(frame, "total_stay", 1).clip(lower=1)
    frame["children"] = _numeric_series(frame, "children", 0).clip(lower=0)
    frame["babies"] = _numeric_series(frame, "babies", 0).clip(lower=0)
    frame["risk_score"] = (
        frame["lead_time"] * 0.4
        + frame["previous_cancellations"] * 2
        + (frame["deposit_type"].astype(str) == "No Deposit").astype(int) * 5
    )
    frame["lead_time_bin"] = pd.cut(
        frame["lead_time"],
        bins=[-1, 7, 30, 90, 180, 365, float("inf")],
        labels=["Last Minute", "Short", "Medium", "Long", "Very Long", "Extreme"],
    ).astype(str)
    frame["value_segment"] = pd.cut(
        frame["adr"] * frame["total_stay"],
        bins=[-1, 100, 300, 700, 1500, float("inf")],
        labels=["Low", "Medium", "High", "Very High", "Luxury"],
    ).astype(str)
    frame["family_type"] = np.where((frame["children"] > 0) | (frame["babies"] > 0), "Family", "Non-Family")
    frame["stay_type"] = pd.cut(
        frame["total_stay"],
        bins=[0, 2, 5, 10, float("inf")],
        labels=["Short", "Medium", "Long", "Very Long"],
        include_lowest=True,
    ).astype(str)

    categorical_columns = [
        "deposit_type",
        "market_segment",
        "customer_type",
        "hotel",
        "meal",
        "distribution_channel",
        "reserved_room_type",
        "lead_time_bin",
        "value_segment",
        "family_type",
        "stay_type",
    ]
    available_categorical = [column for column in categorical_columns if column in frame.columns]
    encoded = pd.get_dummies(frame, columns=available_categorical, dtype=float)
    aligned = encoded.reindex(columns=features, fill_value=0.0)
    return aligned.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)


def build_feature_frame(inputs: dict[str, Any], features: list[str]) -> pd.DataFrame:
    """Backward-compatible alias for older imports."""
    return transform_input(inputs, features)


def load_hotel_dataset(path: str | Path) -> pd.DataFrame:
    """Load and prepare the dashboard dataset."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    return prepare_hotel_dataset(df)


def prepare_hotel_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "hotel",
        "is_canceled",
        "arrival_date_year",
        "arrival_date_month",
        "market_segment",
        "customer_type",
        "deposit_type",
        "adr",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    prepared = df.copy()
    prepared["is_canceled"] = pd.to_numeric(prepared["is_canceled"], errors="coerce").fillna(0).astype(int)
    prepared["adr"] = pd.to_numeric(prepared["adr"], errors="coerce").fillna(0.0)
    prepared["previous_cancellations"] = _numeric_series(prepared, "previous_cancellations", 0)
    prepared["arrival_date_year"] = pd.to_numeric(
        prepared["arrival_date_year"], errors="coerce"
    ).fillna(0).astype(int)

    if "total_stay" not in prepared.columns:
        weekend = _numeric_series(prepared, "stays_in_weekend_nights", 0)
        week = _numeric_series(prepared, "stays_in_week_nights", 0)
        prepared["total_stay"] = weekend + week

    prepared["total_stay"] = pd.to_numeric(prepared["total_stay"], errors="coerce").fillna(1).clip(lower=1)
    prepared["reservation_status_date"] = _build_booking_date(prepared)
    prepared["arrival_date_month"] = pd.Categorical(
        prepared["arrival_date_month"].astype(str),
        categories=MONTH_ORDER,
        ordered=True,
    )
    prepared["month_num"] = prepared["reservation_status_date"].dt.month
    prepared["gross_revenue"] = prepared["adr"] * prepared["total_stay"]
    prepared["real_revenue"] = np.where(prepared["is_canceled"] == 0, prepared["gross_revenue"], 0)
    prepared["revenue_loss"] = np.where(prepared["is_canceled"] == 1, prepared["gross_revenue"], 0)

    categorical_columns = ["hotel", "market_segment", "customer_type", "deposit_type"]
    for column in categorical_columns:
        prepared[column] = prepared[column].fillna("Unknown").astype(str)

    return prepared


def calculate_risk_score(lead_time: float, previous_cancellations: float, deposit_type: str) -> float:
    no_deposit_risk = 5 if deposit_type == "No Deposit" else 0
    return (lead_time * 0.4) + (previous_cancellations * 2) + no_deposit_risk


def bucket_lead_time(lead_time: float) -> str:
    if lead_time <= 7:
        return "Last Minute"
    if lead_time <= 30:
        return "Short"
    if lead_time <= 90:
        return "Medium"
    if lead_time <= 180:
        return "Long"
    if lead_time <= 365:
        return "Very Long"
    return "Extreme"


def bucket_value_segment(value: float) -> str:
    if value <= 100:
        return "Low"
    if value <= 300:
        return "Medium"
    if value <= 700:
        return "High"
    if value <= 1500:
        return "Very High"
    return "Luxury"


def bucket_stay_type(total_stay: float) -> str:
    if total_stay <= 2:
        return "Short"
    if total_stay <= 5:
        return "Medium"
    if total_stay <= 10:
        return "Long"
    return "Very Long"


def _options_from_prefix(features: list[str], prefix: str, baseline: str) -> list[str]:
    values = [feature.removeprefix(prefix) for feature in features if feature.startswith(prefix)]
    options = [baseline, *values]
    return sorted(dict.fromkeys(options))


def _to_float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def _build_booking_date(df: pd.DataFrame) -> pd.Series:
    if "reservation_status_date" in df.columns:
        parsed = pd.to_datetime(df["reservation_status_date"], errors="coerce")
    else:
        parsed = pd.Series(pd.NaT, index=df.index)

    month_lookup = {month: index + 1 for index, month in enumerate(MONTH_ORDER)}
    fallback = pd.to_datetime(
        {
            "year": pd.to_numeric(df["arrival_date_year"], errors="coerce").fillna(2017).astype(int),
            "month": df["arrival_date_month"].astype(str).map(month_lookup).fillna(1).astype(int),
            "day": _numeric_series(df, "arrival_date_day_of_month", 1).astype(int),
        },
        errors="coerce",
    )
    return parsed.fillna(fallback)


def _numeric_series(df: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in df.columns:
        values = df[column]
    else:
        values = pd.Series(default, index=df.index)
    return pd.to_numeric(values, errors="coerce").fillna(default)
