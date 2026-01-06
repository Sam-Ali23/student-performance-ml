import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd

# -----------------------------
# Load model and metadata
# -----------------------------
model = joblib.load("model.joblib")

with open("feature_names.json", "r") as f:
    feature_names = json.load(f)

with open("class_map.json", "r") as f:
    class_map = json.load(f)

# Reverse class map (for safety)
class_map = {int(k): v for k, v in class_map.items()}

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

# -----------------------------
# Title Section
# -----------------------------
st.markdown(
    """
    <h1 style="text-align:center;">🎓 Student Performance Prediction</h1>
    <p style="text-align:center; font-size:18px;">
    Predict student academic performance using Machine Learning
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Input Section
# -----------------------------
st.subheader("📌 Student Information")

inputs = {}

col1, col2 = st.columns(2)

with col1:
    inputs["VisITedResources"] = st.slider(
        "📚 Visited Learning Resources",
        0, 100, 50
    )
    inputs["raisedhands"] = st.slider(
        "✋ Raised Hands in Class",
        0, 100, 50
    )
    inputs["AnnouncementsView"] = st.slider(
        "📢 Announcements Viewed",
        0, 100, 50
    )
    inputs["Discussion"] = st.slider(
        "💬 Participation in Discussions",
        0, 100, 50
    )

with col2:
    inputs["StudentAbsenceDays"] = st.selectbox(
        "🚫 Absence Level",
        options=[0, 1],
        format_func=lambda x: "Low Absence" if x == 0 else "High Absence"
    )
    inputs["ParentAnsweringSurvey"] = st.selectbox(
        "👨‍👩‍👧 Parent Survey Participation",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )
    inputs["ParentschoolSatisfaction"] = st.selectbox(
        "🏫 Parent School Satisfaction",
        [0, 1],
        format_func=lambda x: "Low" if x == 0 else "High"
    )
    inputs["Relation"] = st.selectbox(
        "🤝 Parent Relationship",
        [0, 1],
        format_func=lambda x: "Father" if x == 0 else "Mother"
    )

# Fill missing features with 0
for feature in feature_names:
    if feature not in inputs:
        inputs[feature] = 0

# -----------------------------
# Prediction Section
# -----------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("🔍 Prediction Result")

if st.button("🎯 Predict Performance"):
    X_input = pd.DataFrame([inputs])[feature_names]

    prediction = model.predict(X_input)[0]
    probabilities = model.predict_proba(X_input)[0]

    label = class_map[prediction]

    if label == "High":
        st.success("✅ Predicted Performance: **HIGH**")
    elif label == "Medium":
        st.warning("⚠️ Predicted Performance: **MEDIUM**")
    else:
        st.error("❌ Predicted Performance: **LOW**")

    st.markdown("### 📊 Confidence Levels")
    for i, prob in enumerate(probabilities):
        st.write(f"- **{class_map[i]}**: {prob:.2%}")

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <hr>
    <p style="text-align:center; font-size:14px;">
    Built with ❤️ using Machine Learning & Streamlit
    </p>
    """,
    unsafe_allow_html=True
)
