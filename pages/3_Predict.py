from __future__ import annotations

import plotly.express as px
import streamlit as st

from utils.predict import explain_prediction, load_prediction_assets, predict_cancellation
from utils.preprocessing import get_prediction_options


st.set_page_config(page_title="Predict | Hotel Cancellation Prediction", page_icon="PD", layout="wide")


def apply_style() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
            max-width: 1120px;
        }
        [data-testid="stSidebar"] {
            background: #0f172a;
            border-right: 1px solid rgba(148, 163, 184, 0.16);
        }
        .predict-hero {
            padding: 1.5rem 1.8rem;
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 18px;
            background: linear-gradient(135deg, #111827 0%, #0b1120 100%);
        }
        .result-card {
            padding: 1.5rem;
            border-radius: 16px;
            background: rgba(15, 23, 42, 0.92);
        }
        .result-high {
            border: 1px solid rgba(239, 68, 68, 0.72);
            box-shadow: 0 0 0 1px rgba(239, 68, 68, 0.16);
        }
        .result-low {
            border: 1px solid rgba(34, 197, 94, 0.72);
            box-shadow: 0 0 0 1px rgba(34, 197, 94, 0.16);
        }
        .result-card h2 {
            margin: 0 0 0.4rem 0;
        }
        .risk-high {
            color: #fca5a5;
        }
        .risk-low {
            color: #86efac;
        }
        .recommendation-card {
            padding: 1rem 1.2rem;
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 14px;
            background: rgba(15, 23, 42, 0.72);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def cached_assets():
    return load_prediction_assets()


apply_style()

st.markdown(
    """
    <div class="predict-hero">
        <h1>Cancellation Prediction</h1>
        <p>Enter booking details to estimate the probability of cancellation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

try:
    with st.spinner("Loading model assets..."):
        model, features = cached_assets()
except Exception as exc:
    st.error(
        "Model assets could not be loaded. Make sure `model/model.pkl`, "
        "`model/feature_columns.pkl`, and the required packages are available."
    )
    st.exception(exc)
    st.stop()

options = get_prediction_options(features)

with st.form("prediction_form"):
    st.subheader("Booking Input")
    left, right = st.columns(2)

    with left:
        lead_time = st.number_input("lead_time", min_value=0, max_value=1000, value=45, step=1)
        deposit_type = st.selectbox("deposit_type", options.deposit_types, index=options.deposit_types.index("No Deposit"))
        market_segment = st.selectbox(
            "market_segment",
            options.market_segments,
            index=options.market_segments.index("Online TA")
            if "Online TA" in options.market_segments
            else 0,
        )
        customer_type = st.selectbox(
            "customer_type",
            options.customer_types,
            index=options.customer_types.index("Transient")
            if "Transient" in options.customer_types
            else 0,
        )

    with right:
        previous_cancellations = st.number_input(
            "previous_cancellations", min_value=0, max_value=25, value=0, step=1
        )
        total_of_special_requests = st.number_input(
            "total_of_special_requests", min_value=0, max_value=10, value=1, step=1
        )
        adr = st.number_input("adr", min_value=0.0, max_value=5000.0, value=120.0, step=5.0)
        total_stay = st.number_input("total_stay", min_value=1, max_value=60, value=3, step=1)

    show_shap = st.checkbox("Show SHAP explanation when available", value=False)
    submitted = st.form_submit_button("Predict Cancellation")

if submitted:
    inputs = {
        "lead_time": lead_time,
        "deposit_type": deposit_type,
        "market_segment": market_segment,
        "customer_type": customer_type,
        "previous_cancellations": previous_cancellations,
        "total_of_special_requests": total_of_special_requests,
        "adr": adr,
        "total_stay": total_stay,
    }

    try:
        with st.spinner("Scoring booking..."):
            result = predict_cancellation(model, features, inputs)
    except Exception as exc:
        st.error("Prediction failed. Please check that the model artifacts and input schema are valid.")
        st.exception(exc)
        st.stop()

    probability = result["probability"]
    is_high_risk = result["prediction"] == 1
    risk_class = "risk-high" if is_high_risk else "risk-low"
    card_class = "result-high" if is_high_risk else "result-low"
    indicator = "High Risk" if is_high_risk else "Low Risk"

    st.markdown(
        f"""
        <div class="result-card {card_class}">
            <p>Prediction result</p>
            <h2 class="{risk_class}">{result["label"]}</h2>
            <h4>{indicator}</h4>
            <h3>Probability score: {probability:.2%}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(min(max(probability, 0.0), 1.0))

    st.subheader("Recommendation Engine")
    recommendation_items = "".join(f"<li>{recommendation}</li>" for recommendation in result["recommendations"])
    st.markdown(
        f"""
        <div class="recommendation-card">
            <ul>{recommendation_items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if show_shap:
        try:
            explanation = explain_prediction(model, result["features"])
            fig = px.bar(
                explanation.sort_values("contribution"),
                x="contribution",
                y="feature",
                orientation="h",
                color="contribution",
                color_continuous_scale="RdBu",
                title="Top SHAP Drivers for This Prediction",
                template="plotly_dark",
                hover_data=["value"],
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.warning(f"SHAP explanation is unavailable in this environment: {exc}")
