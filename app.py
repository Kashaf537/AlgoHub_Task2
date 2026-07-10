"""
app.py
------
Streamlit deployment app for the Student Performance Prediction project.
Updated for: feature engineering, SMOTE-trained models, and 5-model comparison
(Logistic Regression, Random Forest, SVM, Neural Network, XGBoost).

Run with:
    streamlit run app.py
"""

import pandas as pd
import joblib
import altair as alt
import streamlit as st

# set_page_config must be the very first Streamlit call in the script
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")

# ==========================================================
# Custom CSS — dark theme, gradient hero banner, colored result
# card, card-style input sections
# ==========================================================
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .hero {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        padding: 2rem 2rem 1.5rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;
    }
    .hero h1 { color: white; font-size: 2rem; margin-bottom: 0.3rem; }
    .hero p { color: rgba(255,255,255,0.85); font-size: 0.95rem; margin: 0; }
    .card {
        background-color: #1a1c24; padding: 1.3rem 1.5rem; border-radius: 14px;
        border: 1px solid #2a2d3a; margin-bottom: 1rem;
    }
    .card h3 { margin-top: 0; }
    .result-box {
        padding: 1.8rem; border-radius: 14px; text-align: center;
        margin: 1.2rem 0 0.5rem 0; box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }
    /* One gradient per performance class, matched to CLASS_CSS below */
    .result-fail { background: linear-gradient(135deg, #eb3349, #f45c43); }
    .result-average { background: linear-gradient(135deg, #f7971e, #ffd200); }
    .result-good { background: linear-gradient(135deg, #11998e, #38ef7d); }
    .result-excellent { background: linear-gradient(135deg, #667eea, #764ba2); }
    .result-box h2 { color: white; margin: 0; font-size: 2rem; }
    .result-box p { color: rgba(255,255,255,0.9); margin-top: 0.4rem; font-size: 0.95rem; }
    div.stButton > button {
        width: 100%; background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white; border: none; border-radius: 10px; padding: 0.7rem 0;
        font-weight: 600; font-size: 1rem; transition: opacity 0.15s ease-in-out;
    }
    div.stButton > button:hover { opacity: 0.85; border: none; color: white; }
    .footer-note { text-align: center; color: #6b7280; font-size: 0.8rem; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# Hero header banner
st.markdown("""
<div class="hero">
    <h1>🎓 Student Performance Predictor</h1>
    <p>Predicts performance class — Fail · Average · Good · Excellent — comparing 5 ML models</p>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# Load saved models and preprocessing objects
# @st.cache_resource keeps these in memory across reruns (Streamlit
# reruns the whole script on every widget interaction) so we don't
# reload from disk every time a slider moves.
# NOTE: if you retrain and re-save new .pkl files, you must restart
# the app (or clear the cache) for it to pick up the new files.
# ==========================================================
@st.cache_resource
def load_artifacts():
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Random Forest": "random_forest.pkl",
        "SVM": "svm.pkl",
        "Neural Network": "neural_network.pkl",
        "XGBoost": "xgboost.pkl",
    }
    models = {name: joblib.load(f"models/{file}") for name, file in model_files.items()}
    scaler = joblib.load("models/scaler.pkl")
    encoders = joblib.load("models/encoders.pkl")
    target_encoder = joblib.load("models/target_encoder.pkl")
    top_features = joblib.load("models/selected_features.pkl")
    return models, scaler, encoders, target_encoder, top_features

models, scaler, encoders, target_encoder, top_features = load_artifacts()

# XGBoost and Neural Network were trained on numeric labels (0/1/2/3) due to
# an sklearn/XGBoost quirk — their raw .predict() output must be decoded back
# to text labels ("Fail"/"Average"/...) via target_encoder before displaying.
NEEDS_DECODING = {"XGBoost", "Neural Network"}

# Read the numeric columns straight from the fitted scaler instead of
# hardcoding them. This means the app automatically adapts if the notebook's
# feature list changes (e.g. after adding/removing engineered features) —
# no manual sync required, just re-save the models and restart the app.
NUMERIC_COLS = list(scaler.feature_names_in_)

# Same ordinal mapping used in the notebook's Step 4.5 (Feature Engineering)
# to convert Low/Medium/High into 1/2/3 for Support_Score.
ORDINAL_MAP = {"Low": 1, "Medium": 2, "High": 3}

# Visual styling lookups, keyed by predicted class name
CLASS_COLORS = {"Fail": "#eb3349", "Average": "#ffd200", "Good": "#38ef7d", "Excellent": "#764ba2"}
CLASS_EMOJI = {"Fail": "❌", "Average": "📘", "Good": "✅", "Excellent": "🏆"}
CLASS_CSS = {"Fail": "result-fail", "Average": "result-average", "Good": "result-good", "Excellent": "result-excellent"}

# ==========================================================
# Input form
# We collect ALL raw fields the engineered features are derived from
# (not just whatever is currently in top_features). This way, if you
# retrain and feature selection picks a different mix of raw vs.
# engineered columns, the form still has everything it needs and
# nothing breaks.
# ==========================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 📊 Study & Attendance")
c1, c2 = st.columns(2)
with c1:
    hours_studied = st.slider("Hours Studied / week", 0, 45, 20)
    attendance = st.slider("Attendance (%)", 50, 100, 80)
    previous_scores = st.slider("Previous Scores", 40, 100, 70)
with c2:
    sleep_hours = st.slider("Sleep Hours", 3, 12, 7)
    tutoring_sessions = st.slider("Tutoring Sessions / month", 0, 8, 1)
    physical_activity = st.slider("Physical Activity (hrs/week)", 0, 6, 3)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 🏠 Background Factors")
c3, c4 = st.columns(2)
with c3:
    parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
    access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
    family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
    teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])
with c4:
    extracurricular = st.selectbox("Extracurricular Activities", ["No", "Yes"])
    peer_influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])
    parental_education = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
st.markdown('</div>', unsafe_allow_html=True)

model_choice = st.radio("Model", list(models.keys()), horizontal=True)
predict_clicked = st.button("🔮 Predict Performance")

# Raw input row, built from the widgets above — this mirrors one row
# of the original (pre-engineering) dataset
raw_input = pd.DataFrame([{
    "Hours_Studied": hours_studied, "Attendance": attendance, "Sleep_Hours": sleep_hours,
    "Previous_Scores": previous_scores, "Tutoring_Sessions": tutoring_sessions,
    "Physical_Activity": physical_activity, "Parental_Involvement": parental_involvement,
    "Access_to_Resources": access_to_resources, "Family_Income": family_income,
    "Teacher_Quality": teacher_quality, "Extracurricular_Activities": extracurricular,
    "Peer_Influence": peer_influence, "Parental_Education_Level": parental_education,
}])


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recreate the exact same engineered features as notebook Step 4.5:
    Support_Score, Study_Efficiency, Engagement_Score, Attendance_x_Previous.
    Must stay in sync with the notebook — if that cell changes, update here too.
    """
    df = df.copy()

    support_cols = ["Parental_Involvement", "Access_to_Resources", "Family_Income", "Teacher_Quality"]
    df["Support_Score"] = df[support_cols].apply(lambda col: col.map(ORDINAL_MAP)).sum(axis=1)

    df["Study_Efficiency"] = df["Hours_Studied"] / (df["Sleep_Hours"] + 1)

    extracurricular_numeric = df["Extracurricular_Activities"].map({"No": 0, "Yes": 1})
    df["Engagement_Score"] = df["Tutoring_Sessions"] + (extracurricular_numeric * 2)

    df["Attendance_x_Previous"] = df["Attendance"] * df["Previous_Scores"] / 100

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline, mirroring the notebook exactly:
    engineer features -> encode categoricals -> scale numerics -> select top_features.
    """
    df = engineer_features(df)

    # Encode each categorical column with its saved LabelEncoder.
    # Only columns present in both `encoders` and `df` get transformed.
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))

    # Scale all numeric columns at once, in the exact order the scaler expects
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

    # Return only the columns the model was actually trained on, in order
    return df[top_features]


# ==========================================================
# Predict + display results
# ==========================================================
if predict_clicked:
    X_new = preprocess(raw_input)
    model = models[model_choice]

    raw_prediction = model.predict(X_new)[0]
    probabilities = model.predict_proba(X_new)[0]

    # Decode numeric predictions back to text labels for XGBoost/Neural Network
    if model_choice in NEEDS_DECODING:
        prediction = target_encoder.inverse_transform([raw_prediction])[0]
        class_labels = target_encoder.inverse_transform(model.classes_)
    else:
        prediction = raw_prediction
        class_labels = model.classes_

    confidence = max(probabilities) * 100
    css_class = CLASS_CSS.get(prediction, "result-average")
    emoji = CLASS_EMOJI.get(prediction, "📘")

    # Color-coded result banner
    st.markdown(f"""
    <div class="result-box {css_class}">
        <h2>{emoji} {prediction}</h2>
        <p>Model confidence: {confidence:.1f}% &nbsp;|&nbsp; Model used: {model_choice}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Prediction Breakdown")
    proba_df = pd.DataFrame({"Class": class_labels, "Probability": probabilities * 100}).sort_values("Probability", ascending=False)

    # Fixed class order so bar colors/positions stay consistent across predictions
    class_order = ["Fail", "Average", "Good", "Excellent"]

    # Altair bar chart with a fixed 0-100% y-axis, so bar height always
    # matches the confidence percentage shown in the badge above
    chart = (
        alt.Chart(proba_df)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, size=55)
        .encode(
            x=alt.X("Class", sort="-y", axis=alt.Axis(labelAngle=0, title=None)),
            y=alt.Y("Probability", scale=alt.Scale(domain=[0, 100]), title="Probability (%)"),
            color=alt.Color("Class", legend=None,
                             scale=alt.Scale(domain=class_order, range=[CLASS_COLORS[c] for c in class_order])),
            tooltip=["Class", alt.Tooltip("Probability", format=".1f")],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, width="stretch")

    # Per-class probability readout as compact metric cards
    st.markdown("#### Class Probabilities")
    m1, m2, m3, m4 = st.columns(4)
    class_labels_list = list(class_labels)
    for col, cls in zip([m1, m2, m3, m4], class_order):
        prob = probabilities[class_labels_list.index(cls)] * 100 if cls in class_labels_list else 0.0
        col.metric(label=f"{CLASS_EMOJI[cls]} {cls}", value=f"{prob:.1f}%")

st.markdown(
    '<p class="footer-note"> 5-model comparison · '
    'Trained on StudentPerformanceFactors.csv with engineered features</p>',
    unsafe_allow_html=True,
)