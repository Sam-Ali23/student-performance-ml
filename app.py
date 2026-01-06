import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import os

# -----------------------------
# Load model and metadata
# -----------------------------
model = joblib.load("model.joblib")

with open("feature_names.json", "r") as f:
    feature_names = json.load(f)

with open("class_map.json", "r") as f:
    class_map = json.load(f)

# Ensure integer keys
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
    Predict student academic performance using Machine Learning + SHAP Explainability
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["🎯 Prediction", "🧠 SHAP Explainability"])

# =========================================================
# TAB 1: Prediction
# =========================================================
with tab1:
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

    # Fill missing features with 0 (ensures the model always gets the full feature set)
    for feature in feature_names:
        if feature not in inputs:
            inputs[feature] = 0

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("🔍 Prediction Result")

    if st.button("🎯 Predict Performance"):
        # Build input row in correct feature order
        X_input = pd.DataFrame([inputs])[feature_names]

        # Predict
        prediction = int(model.predict(X_input)[0])
        probabilities = model.predict_proba(X_input)[0]

        # Label
        label = class_map.get(prediction, f"Class {prediction}")

        # Big result banner
        if label.lower() == "high":
            st.success("✅ Predicted Performance: **HIGH**")
        elif label.lower() == "medium":
            st.warning("⚠️ Predicted Performance: **MEDIUM**")
        else:
            st.error("❌ Predicted Performance: **LOW**")

        # ---------------------------------
        # Build probabilities dataframe
        # ---------------------------------
        prob_table = []
        for i, prob in enumerate(probabilities):
            prob_table.append({
                "Class": class_map.get(i, f"Class {i}"),
                "Probability": float(prob)
            })

        prob_df = pd.DataFrame(prob_table).sort_values("Probability", ascending=False)

        # ---------------------------------
        # Text output
        # ---------------------------------
        st.markdown("### 📊 Confidence Levels (Text)")
        for _, r in prob_df.iterrows():
            st.write(f"- **{r['Class']}**: {r['Probability']:.2%}")

        # ---------------------------------
        # Chart output (bar chart)
        # ---------------------------------
        st.markdown("### 📈 Confidence Chart (Bar)")
        chart_df = prob_df.set_index("Class")[["Probability"]]
        st.bar_chart(chart_df)

        # ---------------------------------
        # Progress bars (nice visual)
        # ---------------------------------
        st.markdown("### ✅ Confidence Bars")
        for _, r in prob_df.iterrows():
            st.write(f"**{r['Class']}**")
            st.progress(int(r["Probability"] * 100))

        # Optional: show as a table
        with st.expander("Show probability table"):
            show_df = prob_df.copy()
            show_df["Probability"] = show_df["Probability"].map(lambda x: f"{x:.4f}")
            st.dataframe(show_df, use_container_width=True)

    st.markdown(
        """
        <hr>
        <p style="text-align:center; font-size:14px;">
        Built using Random Forest + Streamlit
        </p>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# TAB 2: SHAP Explainability
# =========================================================
with tab2:
    st.subheader("🧠 Model Explainability (SHAP)")

    st.write(
        "SHAP (SHapley Additive Explanations) helps explain **why** the model makes a prediction. "
        "It shows how each feature contributes to increasing or decreasing the probability of a class."
    )

    st.markdown("### ✅ Global Feature Importance (All Classes)")
    if os.path.exists("shap_feature_importance.png"):
        st.image("shap_feature_importance.png", use_container_width=True)
    else:
        st.warning("Image not found: shap_feature_importance.png (Upload it to the GitHub repo).")

    st.markdown("### ✅ Beeswarm Plot (Class 2)")
    if os.path.exists("shap_beeswarm_class2.png"):
        st.image("shap_beeswarm_class2.png", use_container_width=True)
    else:
        st.warning("Image not found: shap_beeswarm_class2.png (Upload it to the GitHub repo).")

    st.markdown("### How to read the beeswarm plot")
    st.write(
        "- Each dot represents one student.\n"
        "- The x-axis shows SHAP value (impact on the model output).\n"
        "- The color indicates feature value (red = high, blue = low).\n"
        "- Features at the top have the strongest influence on predictions."
    )
