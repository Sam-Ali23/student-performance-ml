import streamlit as st
import joblib
import json
import pandas as pd

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

# -----------------------------
# Load model and metadata
# -----------------------------
@st.cache_resource
def load_artifacts():
    model_ = joblib.load("model.joblib")

    with open("feature_names.json", "r") as f:
        feature_names_ = json.load(f)

    with open("class_map.json", "r") as f:
        class_map_ = json.load(f)

    # Ensure keys are int
    class_map_ = {int(k): v for k, v in class_map_.items()}
    return model_, feature_names_, class_map_

model, feature_names, class_map = load_artifacts()

# -----------------------------
# Dark/Light Mode (CSS)
# -----------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

with st.sidebar:
    st.header("⚙️ Settings")
    theme_choice = st.radio("Theme", ["Light", "Dark"], index=0 if st.session_state.theme == "Light" else 1)
    st.session_state.theme = theme_choice

def apply_theme(theme: str):
    if theme == "Dark":
        st.markdown(
            """
            <style>
                .stApp { background-color: #0f1117; color: #e6e6e6; }
                h1, h2, h3, p, label, span, div { color: #e6e6e6 !important; }
                .stMarkdown, .stText, .stCaption { color: #e6e6e6 !important; }
                .stButton>button { background-color: #1f2937; color: white; border: 1px solid #374151; }
                .stButton>button:hover { background-color: #111827; border: 1px solid #4b5563; }
                .stSelectbox, .stSlider { color: #e6e6e6 !important; }
                div[data-baseweb="select"] > div { background-color: #111827 !important; border: 1px solid #374151; }
                div[data-baseweb="input"] > div { background-color: #111827 !important; border: 1px solid #374151; }
                div[data-testid="stMetric"] { background-color: #111827; border: 1px solid #374151; padding: 12px; border-radius: 12px; }
                .card { background:#111827; border:1px solid #374151; padding:16px; border-radius:16px; }
                hr { border: none; border-top: 1px solid #374151; }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
                .stApp { background-color: #ffffff; color: #111827; }
                h1, h2, h3, p, label, span, div { color: #111827 !important; }
                .stButton>button { background-color: #111827; color: white; border: 1px solid #111827; }
                .stButton>button:hover { background-color: #0b1220; }
                div[data-testid="stMetric"] { background-color: #f9fafb; border: 1px solid #e5e7eb; padding: 12px; border-radius: 12px; }
                .card { background:#f9fafb; border:1px solid #e5e7eb; padding:16px; border-radius:16px; }
                hr { border: none; border-top: 1px solid #e5e7eb; }
            </style>
            """,
            unsafe_allow_html=True
        )

apply_theme(st.session_state.theme)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <h1 style="text-align:center; margin-bottom:0;">🎓 Student Performance Prediction</h1>
    <p style="text-align:center; font-size:16px; margin-top:6px;">
    Predict student academic performance level using a trained Machine Learning model.
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("---")
    st.subheader("📌 Project Info")
    st.write("**Model:** Random Forest Classifier")
    st.write("**Output:** Low / Medium / High")
    st.write("**Explainability:** SHAP (Bonus)")
    st.caption("Deployed on Streamlit Community Cloud")

# -----------------------------
# Inputs (Few, important only)
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🧾 Student Inputs (Top Features)")

inputs = {}

c1, c2 = st.columns(2)

with c1:
    inputs["VisITedResources"] = st.slider("📚 Visited Resources", 0, 100, 50)
    inputs["raisedhands"] = st.slider("✋ Raised Hands", 0, 100, 50)
    inputs["AnnouncementsView"] = st.slider("📢 Announcements Viewed", 0, 100, 50)

with c2:
    inputs["Discussion"] = st.slider("💬 Discussion Participation", 0, 100, 50)
    inputs["ParentAnsweringSurvey"] = st.selectbox(
        "👨‍👩‍👧 Parent Answering Survey",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )
    inputs["StudentAbsenceDays"] = st.selectbox(
        "🚫 Absence Level",
        [0, 1],
        format_func=lambda x: "Low Absence" if x == 0 else "High Absence"
    )

st.markdown("</div>", unsafe_allow_html=True)

# Fill missing features with defaults (0)
for feat in feature_names:
    if feat not in inputs:
        inputs[feat] = 0

# -----------------------------
# Predict
# -----------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("🔍 Prediction")

predict_col1, predict_col2 = st.columns([1, 1])

with predict_col1:
    do_predict = st.button("🎯 Predict Performance", use_container_width=True)

with predict_col2:
    st.caption("Tip: The UI shows only the most influential features; others are set internally to default values.")

if do_predict:
    X_input = pd.DataFrame([inputs])[feature_names]

    pred_idx = int(model.predict(X_input)[0])
    proba = model.predict_proba(X_input)[0]

    label = class_map.get(pred_idx, f"Class {pred_idx}")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("✅ Result")

    if label.lower() == "high":
        st.success("Predicted Performance Level: **HIGH**")
        short_explain = "The model suggests strong engagement and overall positive learning indicators."
    elif label.lower() == "medium":
        st.warning("Predicted Performance Level: **MEDIUM**")
        short_explain = "The model suggests moderate engagement; improving participation and attendance may help."
    else:
        st.error("Predicted Performance Level: **LOW**")
        short_explain = "The model suggests low engagement and/or attendance issues that may affect performance."

    st.write(short_explain)

    st.subheader("📊 Confidence (Readable)")
    # Build a nice table
    rows = []
    for i, p in enumerate(proba):
        rows.append({"Performance Level": class_map.get(i, f"Class {i}"), "Confidence": float(p)})
    prob_df = pd.DataFrame(rows).sort_values("Confidence", ascending=False)

    # Show bars (more visual, less “numbers”)
    st.bar_chart(prob_df.set_index("Performance Level"))

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <hr>
    <p style="text-align:center; font-size:13px; opacity:0.8;">
    Built using Machine Learning & Streamlit.  
    </p>
    """,
    unsafe_allow_html=True
)
