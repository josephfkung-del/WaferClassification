import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


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
MODEL_PATHS = [
    Path("model/wafer_classifier.keras"),
    Path("wafer_classifier.keras"),
]
TARGET_SIZE = (64, 64)


st.set_page_config(page_title="Wafer Classification", page_icon="⚙️", layout="wide")


@st.cache_resource
def import_tensorflow():
    try:
        import tensorflow as tf
    except ImportError:
        return None
    return tf


@st.cache_resource
def load_model(model_path: str):
    tf = import_tensorflow()
    if tf is None:
        return None
    return tf.keras.models.load_model(model_path)


def find_model_path():
    for path in MODEL_PATHS:
        if path.exists():
            return path
    return None


def parse_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".json":
        data = json.loads(uploaded_file.getvalue().decode("utf-8"))
        return np.array(data, dtype="float32")

    if suffix == ".csv":
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, header=None).to_numpy(dtype="float32")

    if suffix in {".png", ".jpg", ".jpeg"}:
        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("L")
        return np.array(image, dtype="float32")

    raise ValueError("Unsupported file type")


def resize_nearest(array, target_size=TARGET_SIZE):
    image = Image.fromarray(np.asarray(array, dtype="float32")).resize(target_size, Image.Resampling.NEAREST)
    return np.asarray(image, dtype="float32")


def prepare_wafer_map(array):
    array = np.asarray(array, dtype="float32")

    if array.ndim == 3:
        array = np.squeeze(array)

    if array.ndim != 2:
        raise ValueError("Expected a 2D wafer map array or grayscale image.")

    resized = resize_nearest(array)
    return resized[np.newaxis, ..., np.newaxis]


def predict(model, wafer_batch):
    probabilities = model.predict(wafer_batch, verbose=0)[0]
    order = np.argsort(probabilities)[::-1]
    return [(CLASSES[index], float(probabilities[index])) for index in order]


st.title("Wafer Defect Classification")
st.caption("Upload a wafer map as JSON, CSV, PNG, JPG, or JPEG and classify the defect pattern.")

model_path = find_model_path()
tf = import_tensorflow()

with st.sidebar:
    st.header("Model")
    if model_path is None:
        st.warning("No trained model artifact found yet")
        st.markdown("Expected `wafer_classifier.keras` or `model/wafer_classifier.keras`.")
    else:
        st.success(f"Model found: `{model_path}`")

    if tf is None:
        st.info("TensorFlow is not installed in this deployment, so live model inference is disabled.")
    else:
        st.success("TensorFlow is available")

uploaded_file = st.file_uploader(
    "Upload a wafer map",
    type=["json", "csv", "png", "jpg", "jpeg"],
)

left, right = st.columns([1, 1])

with left:
    st.subheader("Input")
    if uploaded_file is None:
        st.info("Upload a wafer map file to preview it here.")
        wafer_array = None
    else:
        try:
            wafer_array = parse_uploaded_file(uploaded_file)
            st.write(f"Shape: `{wafer_array.shape}`")
            st.image(wafer_array, caption="Uploaded wafer map", clamp=True, use_container_width=True)
        except Exception as exc:
            wafer_array = None
            st.error(f"Could not read file: {exc}")

with right:
    st.subheader("Prediction")
    if uploaded_file is None:
        st.empty()
    elif model_path is None:
        st.error("Prediction is disabled until the trained Keras model is in the repo.")
    elif tf is None:
        st.warning("The app is running, but TensorFlow is not installed on this deployment.")
        st.markdown(
            "To enable live predictions, deploy on a platform with TensorFlow support "
            "or add `tensorflow-cpu` back to `requirements.txt`."
        )
    elif wafer_array is None:
        st.empty()
    else:
        try:
            model = load_model(str(model_path))
            if model is None:
                raise RuntimeError("TensorFlow is unavailable.")
            wafer_batch = prepare_wafer_map(wafer_array)
            ranked_predictions = predict(model, wafer_batch)
            label, confidence = ranked_predictions[0]

            st.metric("Predicted defect", label, f"{confidence:.1%} confidence")
            st.dataframe(
                pd.DataFrame(ranked_predictions, columns=["Class", "Confidence"]),
                hide_index=True,
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

st.divider()
st.subheader("Expected Classes")
st.write(", ".join(CLASSES))
