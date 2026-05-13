import sys
from pathlib import Path

import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from config import MODEL_DIR  # noqa: E402

st.set_page_config(page_title="CampusCare AI", page_icon="🎓", layout="centered")
st.title("🎓 CampusCare AI")
st.caption("Student support message triage prototype — educational use only, not a clinical tool.")

model_path = MODEL_DIR / "tfidf_sgd_logreg_baseline.joblib"
if not model_path.exists():
    st.warning("Train the baseline first: `python src/train_baseline.py`")
    st.stop()

model = joblib.load(model_path)
text = st.text_area("Paste an anonymous student feedback message:", height=160)
if st.button("Analyze") and text.strip():
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0] if hasattr(model, "predict_proba") else None
    st.subheader(f"Predicted route: `{pred}`")
    st.write("This result should be reviewed by a trained human before any action is taken.")
    if proba is not None:
        labels = model.classes_
        st.bar_chart({labels[i]: float(proba[i]) for i in range(len(labels))})
