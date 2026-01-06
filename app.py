import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ---------------- Load artifacts ----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    with open("feature_names.json", "r") as f:
        feature_names = json.load(f)
    with open("class_map.json", "r") as f:
        class_map = json.load(f)
    # class_map keys might be saved as integers or strings
    class_map = {str(k): v for k, v in class_map.items()}
    return model, feature_names, class_map

model, feature_names, class_map = load_artifacts()

# ---------------- UI Header ----------------
st.title("🎓 Student Performance Prediction")
st.write(
    "This app predicts the **student performance level** using a trained **Random Forest** model. "
    "The output is shown as readable labels (Low / Medium / High) and class probabilities."
)

with st.sidebar:
    st.header("Project Info")
    st.write("**Model:** Random Forest Classifier")
    st.write("**Task:** Multi-class classification")
    st.write("**Classes:** Low / Medium / High")
    st.markdown("---")
    st.caption("Deployed using Streamlit Community Cloud")

# ---------------- Input form ----------------
st.subheader("Input Features")

with st.form("prediction_form"):
    left, right = st.columns(2)

    inputs = {}
    half = (len(feature_names) + 1) // 2

    def nice_input(col_name):
        # Numeric input with simple defaults
        return st.number_input(
            label=col_name,
            value=0.0,
            step=1.0,
            format="%.0f"
        )

    for i, feat in enumerate(feature_names):
        with (left if i < half else right):
            inputs[feat] = nice_input(feat)

    submitted = st.form_submit_button("Predict")

# ---------------- Prediction output ----------------
if submitted:
    row = pd.DataFrame([[inputs[f] for f in feature_names]], columns=feature_names)

    pred_idx = int(model.predict(row)[0])
    proba = model.predict_proba(row)[0]

    pred_label = class_map.get(str(pred_idx), f"Class {pred_idx}")

    st.markdown("---")
    st.subheader("Prediction Result")
    st.success(f"Predicted Performance Level: **{pred_label}**")

    st.subheader("Class Probabilities")
    prob_table = []
    for i, p in enumerate(proba):
        label = class_map.get(str(i), f"Class {i}")
        prob_table.append({"Class": label, "Probability": round(float(p), 4)})

    prob_df = pd.DataFrame(prob_table).sort_values("Probability", ascending=False)
    st.dataframe(prob_df, use_container_width=True)
