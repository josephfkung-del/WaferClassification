import json
from pathlib import Path

import numpy as np
import streamlit as st


CLASSES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "none",
]


st.set_page_config(page_title="Wafer Classification", page_icon="⚙️", layout="wide")

st.title("Wafer Defect Classification")
st.caption("Upload wafer map data and classify the defect pattern.")

with st.sidebar:
    st.header("Model")
    model_path = Path("model/wafer_classifier.keras")
    if model_path.exists():
        st.success("Model artifact found")
    else:
        st.warning("No trained model artifact found yet")
    st.markdown(
        "Save your trained Keras model as `model/wafer_classifier.keras` "
        "to enable real predictions."
    )

uploaded_file = st.file_uploader(
    "Upload a wafer map as JSON or CSV",
    type=["json", "csv"],
)

left, right = st.columns([1, 1])

with left:
    st.subheader("Input")
    if uploaded_file is None:
        st.info("Upload a wafer map file to preview it here.")
    else:
        raw = uploaded_file.getvalue().decode("utf-8")
        st.code(raw[:4000], language="text")

with right:
    st.subheader("Prediction")
    if uploaded_file is None:
        st.empty()
    elif not model_path.exists():
        st.error("Prediction is disabled until `model/wafer_classifier.keras` is added.")
        st.markdown(
            "The Colab notebook trains a CNN, but the trained model was not exported "
            "into the repository yet. Add the model artifact, then this app can load it "
            "and run inference."
        )
    else:
        st.info("Model loading code goes here once the artifact format is confirmed.")

st.divider()
st.subheader("Expected Classes")
st.write(", ".join(CLASSES))
