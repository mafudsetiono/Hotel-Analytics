from __future__ import annotations

import streamlit as st


st.set_page_config(page_title="Profile | Hotel Cancellation Prediction", page_icon="PR", layout="wide")


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
        .profile-hero {
            padding: 2rem;
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 18px;
            background: linear-gradient(135deg, #111827 0%, #0b1120 100%);
        }
        .profile-card {
            padding: 1.15rem;
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 14px;
            background: rgba(15, 23, 42, 0.86);
        }
        .skill-pill {
            display: inline-block;
            margin: 0.25rem 0.3rem 0.25rem 0;
            padding: 0.35rem 0.65rem;
            border-radius: 999px;
            background: rgba(56, 189, 248, 0.12);
            color: #bae6fd;
            border: 1px solid rgba(56, 189, 248, 0.24);
        }
        .muted {
            color: #94a3b8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_style()

st.markdown(
    """
    <div class="profile-hero">
        <p class="muted">Personal profile</p>
        <h1>Mafud Satrio Setiono</h1>
        <h3>Data Analyst / Data Scientist</h3>
        <p>
            Building analytics and machine learning solutions that turn hotel booking
            data into practical business decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

left, right = st.columns([1, 1.35], gap="large")

with left:
    st.markdown('<div class="profile-card">', unsafe_allow_html=True)
    st.subheader("Skills")
    for skill in ["Python", "SQL", "Power BI", "Machine Learning"]:
        st.markdown(f'<span class="skill-pill">{skill}</span>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="profile-card">', unsafe_allow_html=True)
    st.subheader("Project Description")
    st.write(
        "This project analyzes hotel reservation data to understand cancellation patterns, "
        "estimate revenue exposure, and predict whether a new booking is likely to cancel."
    )
    st.subheader("Goals")
    st.write(
        "The system is designed to support revenue teams with early warning signals, "
        "segment-level monitoring, and model-driven cancellation risk scoring."
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")
goal_1, goal_2, goal_3 = st.columns(3)
goal_1.metric("Primary Outcome", "Cancellation Risk")
goal_2.metric("Business Focus", "Revenue Protection")
goal_3.metric("Model Output", "Probability Score")
