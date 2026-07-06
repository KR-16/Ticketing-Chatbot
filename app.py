"""Streamlit demo for the support-ticket classifier.

Run locally:
    streamlit run app.py

Uses the bundle at models/bundle (override with BUNDLE_DIR). Shows a
friendly setup message instead of crashing when no model exists yet.
"""

import os

import pandas as pd
import streamlit as st

from src.inference.predictor import TicketPredictor

st.set_page_config(
    page_title="Support Ticket Classifier",
    page_icon="🎫",
    layout="centered",
)


@st.cache_resource
def load_predictor() -> TicketPredictor:
    return TicketPredictor(os.getenv("BUNDLE_DIR", "models/bundle"))


st.title("🎫 Support Ticket Classifier")
st.caption(
    "Multilingual BERT fine-tuned on 20k German/English support tickets. "
    "Routes each ticket to **Change / Incident / Problem / Request**."
)

try:
    predictor = load_predictor()
except FileNotFoundError as error:
    st.error(f"**No trained model found.** {error}")
    st.info(
        "Train a bundle first — `python main.py` locally or "
        "`notebooks/train_colab.ipynb` on a free Colab GPU — then restart "
        "this app."
    )
    st.stop()

subject = st.text_input(
    "Subject", placeholder="e.g. Data analytics platform crashes on startup"
)
body = st.text_area(
    "Body",
    height=180,
    placeholder="Describe the issue the way a customer would...",
)

if st.button("Classify ticket", type="primary"):
    if not (subject.strip() or body.strip()):
        st.warning("Enter a subject or a body first.")
    else:
        with st.spinner("Classifying..."):
            result = predictor.predict(subject=subject, body=body)

        st.divider()
        left, right = st.columns(2)
        left.metric("Predicted type", result["predicted_type"])
        right.metric("Confidence", f"{result['confidence']:.1%}")

        probabilities = pd.DataFrame(
            {"probability": result["probabilities"]}
        ).sort_values("probability", ascending=False)
        st.bar_chart(probabilities)
