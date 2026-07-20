"""
app.py
------
Streamlit web interface for the Disease Prediction Assistant.

Run:
    streamlit run app.py
"""

import json
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Disease Prediction Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        padding: 1.5rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%);
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 2rem; }
    .main-header p { margin: 0.3rem 0 0 0; opacity: 0.92; }

    .stat-card {
        background: #f0fdfa;
        border: 1px solid #99f6e4;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        text-align: center;
    }
    .stat-card .num { font-size: 1.6rem; font-weight: 700; color: #0f766e; }
    .stat-card .label { font-size: 0.8rem; color: #475569; }

    .result-card {
        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdfa 100%);
        border: 1px solid #6ee7b7;
        border-radius: 16px;
        padding: 1.5rem 1.8rem;
        margin-top: 0.5rem;
    }
    .result-card .disease-name {
        font-size: 1.7rem;
        font-weight: 800;
        color: #065f46;
        margin: 0.2rem 0 0.4rem 0;
    }
    .result-card .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-size: 0.75rem;
        color: #059669;
        font-weight: 700;
    }

    .symptom-chip {
        display: inline-block;
        background: #ecfeff;
        border: 1px solid #a5f3fc;
        color: #155e75;
        border-radius: 999px;
        padding: 0.2rem 0.8rem;
        margin: 0.2rem 0.3rem 0.2rem 0;
        font-size: 0.85rem;
    }

    .disclaimer-box {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        color: #78350f;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open("symptoms.pkl", "rb") as f:
        symptoms = pickle.load(f)
    try:
        with open("metrics.json", "r") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = None
    return model, label_encoder, symptoms, metrics


model, label_encoder, symptom_cols, metrics = load_artifacts()

# ---------------------------------------------------------------------
# Group symptoms into categories for a much less overwhelming UI
# ---------------------------------------------------------------------
SYMPTOM_CATEGORIES = {
    "🌡️ General": ["high_fever", "mild_fever", "chills", "sweating", "fatigue",
                    "weakness", "body_ache", "muscle_pain", "weight_loss", "dehydration", "nosebleeds"],
    "🫁 Respiratory": ["cough", "dry_cough", "sore_throat", "congestion", "runny_nose",
                        "sneezing", "shortness_of_breath", "wheezing", "chest_tightness",
                        "chest_discomfort", "chest_pain", "mucus", "loss_of_smell"],
    "🤢 Digestive": ["abdominal_pain", "nausea", "vomiting", "diarrhea", "constipation",
                      "heartburn", "regurgitation", "loss_of_appetite"],
    "🧠 Head & Neuro": ["headache", "severe_headache", "dizziness",
                         "sensitivity_to_light", "sensitivity_to_sound", "blurred_vision"],
    "🩹 Skin": ["itchy_rash", "rash", "red_spots", "pale_skin", "swelling"],
    "👁️ Eyes": ["itchy_eyes", "watery_eyes", "redness_eyes", "discharge_eyes"],
    "🚽 Urinary": ["burning_urination", "frequent_urination", "cloudy_urine", "excessive_thirst"],
    "🦴 Joints & Muscles": ["joint_pain", "joint_stiffness", "reduced_range_of_motion"],
    "🦋 Hormonal & Metabolic": ["weight_gain", "cold_intolerance", "dry_skin", "hair_loss",
                                 "heat_intolerance", "rapid_heartbeat", "tremor", "increased_appetite"],
}

def display_name(s):
    return s.replace("_", " ").title()


if "selected_symptoms" not in st.session_state:
    st.session_state.selected_symptoms = set()

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🩺 Disease Prediction Assistant</h1>
    <p>Select the symptoms you're experiencing to see the most likely conditions — powered by machine learning.</p>
</div>
""", unsafe_allow_html=True)

# Top stat row
c1, c2, c3, c4 = st.columns(4)
stats = [
    (c1, str(metrics["n_diseases"]) if metrics else str(len(label_encoder.classes_)), "Diseases Covered"),
    (c2, str(metrics["n_symptoms"]) if metrics else str(len(symptom_cols)), "Tracked Symptoms"),
    (c3, type(model).__name__, "Active Model"),
    (c4, f'{metrics["best_model_accuracy"]*100:.1f}%' if metrics else "—", "Test Accuracy"),
]
for col, num, label in stats:
    col.markdown(f'<div class="stat-card"><div class="num">{num}</div><div class="label">{label}</div></div>',
                  unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------------
# Symptom selection (tabs by body system, instead of one giant wall)
# ---------------------------------------------------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Select your symptoms")
    tabs = st.tabs(list(SYMPTOM_CATEGORIES.keys()))
    for tab, (category, symptoms_in_cat) in zip(tabs, SYMPTOM_CATEGORIES.items()):
        with tab:
            cols = st.columns(2)
            for i, symptom in enumerate(symptoms_in_cat):
                with cols[i % 2]:
                    checked = st.checkbox(
                        display_name(symptom),
                        key=f"cb_{symptom}",
                        value=symptom in st.session_state.selected_symptoms,
                    )
                    if checked:
                        st.session_state.selected_symptoms.add(symptom)
                    else:
                        st.session_state.selected_symptoms.discard(symptom)

with right:
    st.subheader("Your selections")
    selected = sorted(st.session_state.selected_symptoms)
    if selected:
        chips_html = "".join(f'<span class="symptom-chip">{display_name(s)}</span>' for s in selected)
        st.markdown(chips_html, unsafe_allow_html=True)
        st.write("")
        if st.button("🗑️ Clear all", use_container_width=True):
            st.session_state.selected_symptoms = set()
            st.rerun()
    else:
        st.caption("No symptoms selected yet. Browse the tabs on the left and check anything that applies.")

    st.write("")
    predict_clicked = st.button("🔍 Predict Disease", type="primary", use_container_width=True)

st.divider()

# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------
if predict_clicked:
    if not selected:
        st.error("Please select at least one symptom before predicting.")
    else:
        if len(selected) == 1:
            st.warning(
                "⚠️ Only 1 symptom selected. Many conditions share individual "
                "symptoms, so predictions are much more reliable with **2-3 or more** "
                "symptoms selected. Consider adding more if you have them."
            )
        input_vector = pd.DataFrame(
            [[1 if s in selected else 0 for s in symptom_cols]],
            columns=symptom_cols,
        )
        prediction = model.predict(input_vector)[0]
        predicted_disease = label_encoder.inverse_transform([prediction])[0]

        proba_df = None
        confidence_text = ""
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_vector)[0]
            proba_df = pd.DataFrame({
                "Disease": label_encoder.classes_,
                "Probability": proba,
            }).sort_values("Probability", ascending=False).reset_index(drop=True)
            top_conf = proba_df.iloc[0]["Probability"] * 100
            confidence_text = f"{top_conf:.1f}% confidence"

        res_col, chart_col = st.columns([1, 1.3])

        with res_col:
            st.markdown(f"""
            <div class="result-card">
                <div class="eyebrow">Most likely condition</div>
                <div class="disease-name">{predicted_disease}</div>
                <div style="color:#047857; font-weight:600;">{confidence_text}</div>
            </div>
            """, unsafe_allow_html=True)
            st.write("")
            st.markdown(
                f'<div class="disclaimer-box">⚠️ This is an educational demo, '
                f'<b>not medical advice</b>. Please consult a qualified doctor '
                f'for real health concerns, especially if symptoms are severe or persistent.</div>',
                unsafe_allow_html=True,
            )

        with chart_col:
            if proba_df is not None:
                st.write("**Top 5 possible conditions**")
                top5 = proba_df.head(5).copy()
                top5["Probability (%)"] = (top5["Probability"] * 100).round(1)
                st.bar_chart(top5.set_index("Disease")["Probability (%)"], color="#14b8a6")

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About this app")
    st.write(
        "Built with **Python**, **Pandas**, and **scikit-learn** "
        "(Logistic Regression, KNN, Decision Tree — best one auto-selected), "
        "deployed with **Streamlit**."
    )

    if metrics:
        st.subheader("Model comparison")
        for name, acc in metrics["all_results"].items():
            is_best = name == metrics["best_model_name"]
            label = f"**{name}** ✅" if is_best else name
            st.write(label)
            st.progress(acc, text=f"{acc*100:.1f}%")

    st.subheader("Diseases in this dataset")
    with st.expander("View all"):
        for d in sorted(label_encoder.classes_):
            st.write(f"• {d}")

    st.divider()
    st.caption("For educational purposes only. Not a diagnostic tool.")
