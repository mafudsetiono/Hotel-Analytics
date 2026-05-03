from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="Home",
    page_icon="HC",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_dark_style() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }
        [data-testid="stSidebar"] {
            background: #0f172a;
            border-right: 1px solid rgba(148, 163, 184, 0.16);
        }
        .hero {
            padding: 2.4rem;
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 18px;
            background:
                radial-gradient(circle at 15% 15%, rgba(56, 189, 248, 0.20), transparent 32%),
                linear-gradient(135deg, #111827 0%, #0b1120 70%);
        }
        .hero h1 {
            font-size: 3rem;
            line-height: 1.05;
            margin-bottom: 1rem;
        }
        .hero p {
            color: #cbd5e1;
            font-size: 1.05rem;
            max-width: 760px;
        }
        .section-card {
            min-height: 150px;
            padding: 1.15rem;
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 14px;
            background: rgba(15, 23, 42, 0.86);
        }
        .section-card h3 {
            margin-top: 0;
            color: #f8fafc;
            font-size: 1.05rem;
        }
        .section-card p {
            color: #cbd5e1;
            margin-bottom: 0;
        }
        .muted {
            color: #94a3b8;
        }
        div[data-testid="stMetric"] {
            padding: 1rem;
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 14px;
            background: rgba(15, 23, 42, 0.86);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_dark_style()

st.markdown(
    """
    <div class="hero">
        <p class="muted">Machine learning decision support</p>
        <h1>Hotel Cancellation Prediction System</h1>
        <p>
            A professional analytics application for exploring hotel booking behavior,
            monitoring cancellation risk, and predicting whether a reservation is likely
            to be canceled before arrival.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

metric_1, metric_2, metric_3 = st.columns(3)
metric_1.metric("Model Focus", "Cancellation Risk")
metric_2.metric("Workflow", "EDA to Prediction")
metric_3.metric("Interface", "Dashboard + ML Form")

st.write("")
st.subheader("Key Highlights")

col1, col2, col3, col4 = st.columns(4)
cards = [
    ("EDA", "Explore booking patterns, seasonality, customer mix, ADR, and cancellation behavior."),
    ("ML Model", "Use a trained XGBoost classifier aligned to the saved training feature schema."),
    ("SHAP", "Optional local explanation shows the strongest drivers behind a prediction."),
    ("Dashboard", "Interactive Plotly views track trend, segment risk, ADR, and revenue impact."),
]

for column, (title, body) in zip((col1, col2, col3, col4), cards):
    with column:
        st.markdown(
            f"""
            <div class="section-card">
                <h3>{title}</h3>
                <p>{body}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")
st.info("Use the sidebar to open the dashboard or run an individual booking prediction.")
