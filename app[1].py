from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "quietspot_model.joblib"

st.set_page_config(
    page_title="QuietSpot ML",
    page_icon="📚",
    layout="centered",
)

st.title("📚 QuietSpot ML")
st.caption("A simple ML-powered library crowd predictor")

if not MODEL_PATH.exists():
    st.warning("Model not found. Run `python train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

st.subheader("Enter campus conditions")

col1, col2 = st.columns(2)

with col1:
    day = st.selectbox("Day", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    hour = st.slider("Hour", min_value=8, max_value=20, value=14)
    exam_week = st.selectbox("Exam week?", [0, 1], format_func=lambda x: "Yes" if x else "No")

with col2:
    weather = st.selectbox("Weather", ["Clear", "Cloudy", "Rainy"])
    semester_week = st.slider("Semester week", 1, 16, 8)
    nearby_events = st.slider("Nearby campus events", 0, 5, 1)

input_df = pd.DataFrame([{
    "day": day,
    "hour": hour,
    "exam_week": exam_week,
    "weather": weather,
    "semester_week": semester_week,
    "nearby_events": nearby_events,
}])

if st.button("Predict crowd level", use_container_width=True):
    prediction = float(model.predict(input_df)[0])
    prediction = max(0, min(100, prediction))

    if prediction < 35:
        label = "🟢 Quiet"
    elif prediction < 65:
        label = "🟡 Moderate"
    else:
        label = "🔴 Busy"

    st.metric("Predicted occupancy", f"{prediction:.0f}%")
    st.success(f"Expected crowd level: **{label}**")

st.divider()
st.caption(
    "Educational demo using a synthetic dataset. Predictions are illustrative, "
    "not official university occupancy data."
)
