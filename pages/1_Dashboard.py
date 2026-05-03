from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score

from utils.predict import load_prediction_assets
from utils.preprocessing import (
    MONTH_ORDER,
    load_hotel_dataset,
    prepare_hotel_dataset,
)


st.set_page_config(page_title="Dashboard | Hotel Cancellation Prediction", page_icon="DB", layout="wide")

PLOTLY_TEMPLATE = "plotly_dark"
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "hotel_clean.csv"
X_TEST_PATH = ROOT / "model" / "X_test.pkl" if (ROOT / "model" / "X_test.pkl").exists() else ROOT / "data" / "X_test.pkl"
Y_TEST_PATH = ROOT / "model" / "y_test.pkl" if (ROOT / "model" / "y_test.pkl").exists() else ROOT / "data" / "y_test.pkl"


def apply_style() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
            max-width: 1240px;
        }
        [data-testid="stSidebar"] {
            background: #0f172a;
            border-right: 1px solid rgba(148, 163, 184, 0.16);
        }
        div[data-testid="stMetric"] {
            padding: 1rem;
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 14px;
            background: rgba(15, 23, 42, 0.86);
        }
        .dashboard-title {
            padding: 1.4rem 1.6rem;
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 18px;
            background: linear-gradient(135deg, #111827 0%, #0b1120 100%);
        }
        .impact-card {
            min-height: 150px;
            padding: 1.1rem 1.2rem;
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 14px;
            background: rgba(15, 23, 42, 0.86);
        }
        .impact-card h4 {
            margin: 0 0 0.45rem 0;
        }
        .risk-text {
            color: #fca5a5;
            font-weight: 700;
        }
        .safe-text {
            color: #86efac;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_dashboard_data(path: str) -> pd.DataFrame:
    return load_hotel_dataset(path)


@st.cache_data(show_spinner=False)
def compute_model_performance(x_test_path: str, y_test_path: str) -> dict[str, object]:
    model, feature_columns = load_prediction_assets()
    x_test = joblib.load(x_test_path)
    y_test = joblib.load(y_test_path)

    if not isinstance(x_test, pd.DataFrame):
        x_test = pd.DataFrame(x_test, columns=feature_columns)
    x_test = x_test.reindex(columns=feature_columns, fill_value=0.0)
    x_test = x_test.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    y_test = pd.Series(y_test).astype(int)

    if len(x_test) != len(y_test):
        raise ValueError("X_test and y_test have different lengths.")

    y_pred = np.asarray(model.predict(x_test)).astype(int)

    if hasattr(model, "predict_proba"):
        y_score = np.asarray(model.predict_proba(x_test))[:, 1]
    else:
        y_score = y_pred.astype(float)

    try:
        roc_auc = float(roc_auc_score(y_test, y_score))
    except ValueError:
        roc_auc = float("nan")

    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "matrix": matrix,
        "test_samples": int(len(y_test)),
    }


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.subheader("Dashboard Filters")

        uploaded_file = st.file_uploader("Optional CSV override", type=["csv"])
        if uploaded_file is not None:
            try:
                df = prepare_hotel_dataset(pd.read_csv(uploaded_file))
                st.success("Uploaded dataset loaded successfully.")
            except Exception as exc:
                st.error(f"Could not read uploaded file: {exc}")

        hotels = sorted(df["hotel"].dropna().unique())
        segments = sorted(df["market_segment"].dropna().unique())
        customer_types = sorted(df["customer_type"].dropna().unique())
        years = sorted(df["arrival_date_year"].dropna().unique())

        selected_hotels = st.multiselect("Hotel", hotels, default=hotels)
        selected_segments = st.multiselect("Market Segment", segments, default=segments)
        selected_customer_types = st.multiselect("Customer Type", customer_types, default=customer_types)
        selected_years = st.multiselect("Year", years, default=years)

    return df[
        df["hotel"].isin(selected_hotels)
        & df["market_segment"].isin(selected_segments)
        & df["customer_type"].isin(selected_customer_types)
        & df["arrival_date_year"].isin(selected_years)
    ].copy()


def cancellation_label(value: int) -> str:
    return "Canceled" if int(value) == 1 else "Not Canceled"


def format_currency(value: float) -> str:
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"


def format_metric(value: float) -> str:
    if np.isnan(value):
        return "N/A"
    return f"{value:.1%}"


def build_confusion_matrix_figure(matrix: np.ndarray) -> go.Figure:
    total = matrix.sum()
    percentages = matrix / total * 100 if total else np.zeros_like(matrix, dtype=float)
    labels = np.array(
        [
            ["True Negative", "False Positive"],
            ["False Negative", "True Positive"],
        ]
    )
    text = [
        [
            f"{labels[row, col]}<br>{matrix[row, col]:,}<br>{percentages[row, col]:.1f}%"
            for col in range(2)
        ]
        for row in range(2)
    ]
    colors = np.array([[0.15, 0.65], [1.0, 0.35]])
    fig = go.Figure(
        data=go.Heatmap(
            z=colors,
            x=["Predicted Not Canceled", "Predicted Canceled"],
            y=["Actual Not Canceled", "Actual Canceled"],
            text=text,
            texttemplate="%{text}",
            textfont={"size": 14, "color": "#f8fafc"},
            hovertemplate="%{y}<br>%{x}<extra>%{text}</extra>",
            colorscale=[
                [0.0, "#166534"],
                [0.35, "#0f766e"],
                [0.65, "#f97316"],
                [1.0, "#dc2626"],
            ],
            showscale=False,
        )
    )
    fig.update_layout(
        title="Confusion Matrix",
        template=PLOTLY_TEMPLATE,
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        height=430,
        margin={"l": 20, "r": 20, "t": 70, "b": 40},
    )
    return fig


apply_style()

try:
    base_df = load_dashboard_data(str(DATA_PATH))
except Exception as exc:
    st.error(f"Could not load dataset from `{DATA_PATH}`.")
    st.exception(exc)
    st.stop()

filtered_df = filter_data(base_df)

st.markdown(
    """
    <div class="dashboard-title">
        <h1>Hotel Booking Dashboard</h1>
        <p>Interactive cancellation, demand, ADR, and revenue performance from data/hotel_clean.csv.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

if filtered_df.empty:
    st.warning("No records match the selected filters.")
    st.stop()

total_bookings = len(filtered_df)
cancel_rate = filtered_df["is_canceled"].mean()
real_revenue = filtered_df["real_revenue"].sum()
revenue_loss = filtered_df["revenue_loss"].sum()

kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
kpi_1.metric("Total Bookings", f"{total_bookings:,}")
kpi_2.metric("Cancellation Rate", f"{cancel_rate:.1%}")
kpi_3.metric("Real Revenue", format_currency(real_revenue))
kpi_4.metric("Revenue Loss", format_currency(revenue_loss))

st.write("")

plot_df = filtered_df.assign(cancellation_status=filtered_df["is_canceled"].map(cancellation_label))

deposit_cancel = (
    plot_df.groupby(["deposit_type", "cancellation_status"], as_index=False)
    .size()
    .rename(columns={"size": "bookings"})
)

customer_cancel = (
    plot_df.groupby(["customer_type", "cancellation_status"], as_index=False)
    .size()
    .rename(columns={"size": "bookings"})
)

trend = (
    plot_df.assign(month=plot_df["reservation_status_date"].dt.to_period("M").dt.to_timestamp())
    .groupby("month", as_index=False)
    .agg(bookings=("is_canceled", "size"), cancellations=("is_canceled", "sum"))
)

adr_month = (
    plot_df.groupby("arrival_date_month", observed=False, as_index=False)
    .agg(adr=("adr", "mean"))
    .sort_values("arrival_date_month")
)
adr_month["arrival_date_month"] = pd.Categorical(
    adr_month["arrival_date_month"].astype(str), categories=MONTH_ORDER, ordered=True
)

previous_cancel = (
    plot_df.assign(previous_cancellations_bucket=plot_df["previous_cancellations"].clip(upper=5).astype(int))
    .groupby(["previous_cancellations_bucket", "cancellation_status"], as_index=False)
    .size()
    .rename(columns={"size": "bookings"})
)
previous_cancel["previous_cancellations_bucket"] = previous_cancel["previous_cancellations_bucket"].replace(
    {5: "5+"}
)

segment_cancel = (
    plot_df.groupby(["market_segment", "cancellation_status"], as_index=False)
    .size()
    .rename(columns={"size": "bookings"})
)

monthly_cancel = (
    plot_df.groupby(["arrival_date_month", "cancellation_status"], observed=False, as_index=False)
    .size()
    .rename(columns={"size": "bookings"})
    .sort_values("arrival_date_month")
)
monthly_cancel["arrival_date_month"] = pd.Categorical(
    monthly_cancel["arrival_date_month"].astype(str), categories=MONTH_ORDER, ordered=True
)

scatter_df = plot_df.sample(min(len(plot_df), 6000), random_state=42) if len(plot_df) > 6000 else plot_df
scatter_df = scatter_df.assign(adr_size=scatter_df["adr"].clip(lower=0.01))

left, right = st.columns(2)
with left:
    fig_deposit = px.bar(
        deposit_cancel,
        x="deposit_type",
        y="bookings",
        color="cancellation_status",
        barmode="group",
        title="Cancellation by Deposit Type",
        template=PLOTLY_TEMPLATE,
        labels={
            "deposit_type": "Deposit Type",
            "bookings": "Bookings",
            "cancellation_status": "Status",
        },
        color_discrete_map={"Canceled": "#ef4444", "Not Canceled": "#22c55e"},
    )
    st.plotly_chart(fig_deposit, use_container_width=True)

with right:
    fig_customer = px.bar(
        customer_cancel,
        x="customer_type",
        y="bookings",
        color="cancellation_status",
        barmode="group",
        title="Cancellation by Customer Type",
        template=PLOTLY_TEMPLATE,
        labels={
            "customer_type": "Customer Type",
            "bookings": "Bookings",
            "cancellation_status": "Status",
        },
        color_discrete_map={"Canceled": "#ef4444", "Not Canceled": "#22c55e"},
    )
    st.plotly_chart(fig_customer, use_container_width=True)

bottom_left, bottom_right = st.columns(2)
with bottom_left:
    fig_trend = px.line(
        trend,
        x="month",
        y=["bookings", "cancellations"],
        markers=True,
        title="Booking Trend Over Time",
        template=PLOTLY_TEMPLATE,
        labels={"value": "Bookings", "month": "Month", "variable": "Metric"},
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with bottom_right:
    fig_adr = px.line(
        adr_month,
        x="arrival_date_month",
        y="adr",
        markers=True,
        title="ADR by Month",
        template=PLOTLY_TEMPLATE,
        labels={"arrival_date_month": "Month", "adr": "Average Daily Rate"},
    )
    st.plotly_chart(fig_adr, use_container_width=True)

st.write("")

insight_left, insight_right = st.columns(2)
with insight_left:
    fig_previous = px.bar(
        previous_cancel,
        x="previous_cancellations_bucket",
        y="bookings",
        color="cancellation_status",
        barmode="group",
        title="Previous Cancellations vs Cancellation",
        template=PLOTLY_TEMPLATE,
        labels={
            "previous_cancellations_bucket": "Previous Cancellations",
            "bookings": "Bookings",
            "cancellation_status": "Status",
        },
        color_discrete_map={"Canceled": "#ef4444", "Not Canceled": "#22c55e"},
    )
    st.plotly_chart(fig_previous, use_container_width=True)

with insight_right:
    fig_segment = px.bar(
        segment_cancel,
        x="market_segment",
        y="bookings",
        color="cancellation_status",
        barmode="group",
        title="Cancellation by Market Segment",
        template=PLOTLY_TEMPLATE,
        labels={
            "market_segment": "Market Segment",
            "bookings": "Bookings",
            "cancellation_status": "Status",
        },
        color_discrete_map={"Canceled": "#ef4444", "Not Canceled": "#22c55e"},
    )
    st.plotly_chart(fig_segment, use_container_width=True)

final_left, final_right = st.columns(2)
with final_left:
    fig_stay_revenue = px.scatter(
        scatter_df,
        x="total_stay",
        y="gross_revenue",
        color="cancellation_status",
        size="adr_size",
        opacity=0.55,
        title="Length of Stay vs Revenue",
        template=PLOTLY_TEMPLATE,
        labels={
            "total_stay": "Length of Stay",
            "gross_revenue": "Gross Revenue",
            "cancellation_status": "Status",
            "adr_size": "ADR",
        },
        color_discrete_map={"Canceled": "#ef4444", "Not Canceled": "#22c55e"},
    )
    st.plotly_chart(fig_stay_revenue, use_container_width=True)

with final_right:
    fig_month_cancel = px.bar(
        monthly_cancel,
        x="arrival_date_month",
        y="bookings",
        color="cancellation_status",
        barmode="group",
        title="Cancellation by Month",
        template=PLOTLY_TEMPLATE,
        labels={
            "arrival_date_month": "Month",
            "bookings": "Bookings",
            "cancellation_status": "Status",
        },
        color_discrete_map={"Canceled": "#ef4444", "Not Canceled": "#22c55e"},
    )
    st.plotly_chart(fig_month_cancel, use_container_width=True)

st.write("")
st.divider()
st.subheader("Model Performance & Business Impact (Test Set Evaluation)")
st.caption("Evaluation is based on hold-out test data to ensure unbiased performance measurement.")

try:
    with st.spinner("Evaluating model performance..."):
        performance = compute_model_performance(str(X_TEST_PATH), str(Y_TEST_PATH))

    metric_col_1, metric_col_2, metric_col_3, metric_col_4, metric_col_5 = st.columns(5)
    metric_col_1.metric("Accuracy", format_metric(performance["accuracy"]))
    metric_col_2.metric("Precision", format_metric(performance["precision"]))
    metric_col_3.metric("Recall", format_metric(performance["recall"]))
    metric_col_4.metric("ROC-AUC", format_metric(performance["roc_auc"]))
    metric_col_5.metric("Test Samples", f"{performance['test_samples']:,}")

    matrix = performance["matrix"]
    cm_left, cm_right = st.columns([1.2, 1], gap="large")

    with cm_left:
        st.plotly_chart(build_confusion_matrix_figure(matrix), use_container_width=True)

    tn, fp, fn, tp = matrix.ravel()
    total = matrix.sum()
    fn_rate = fn / total if total else 0

    with cm_right:
        st.markdown(
            f"""
            <div class="impact-card">
                <h4>Business Interpretation</h4>
                <p><span class="safe-text">True Positive</span> - Correctly predicted cancellations, creating an opportunity to intervene.</p>
                <p><span class="risk-text">False Negative</span> - Missed cancellations that can become direct revenue loss.</p>
                <p><span class="risk-text">False Positive</span> - False alerts that create operational cost but are manageable.</p>
                <p><span class="safe-text">True Negative</span> - Stable bookings correctly identified as safe.</p>
                <h3 class="risk-text">False Negative = Revenue Loss</h3>
                <p>Current false-negative exposure: {fn:,} bookings ({fn_rate:.1%} of evaluated records).</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    impact_1, impact_2, impact_3, impact_4 = st.columns(4)
    impact_1.metric("True Negative", f"{tn:,}", "Stable bookings", delta_color="normal")
    impact_2.metric("False Positive", f"{fp:,}", "False alerts", delta_color="inverse")
    impact_3.metric("False Negative", f"{fn:,}", "Revenue loss risk", delta_color="inverse")
    impact_4.metric("True Positive", f"{tp:,}", "Intervention opportunity", delta_color="normal")

except Exception as exc:
    st.warning(f"Model performance section could not be rendered: {exc}")
