# Wafer Classification

A Streamlit wafer defect classification web app based on the `Wafer_Classification.ipynb` Colab notebook.

## What Is In This Repo

- `app.py` - Streamlit app that loads the trained Keras model and runs predictions
- `wafer_classifier.keras` - trained model artifact uploaded from Colab
- `requirements.txt` - lightweight Python dependencies
- `Dockerfile` - Hugging Face Spaces deployment with `tensorflow-cpu`
- `notebooks/Wafer_Classification.ipynb` - notebook handoff with the original Colab link and export instructions

Original full notebook:

- https://github.com/josephfkung-del/DementiaBank-Classifcation/blob/main/Wafer_Classification.ipynb

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

For local live predictions, install TensorFlow too:

```bash
pip install tensorflow-cpu==2.17.1
streamlit run app.py
```

## Use The App

Upload a wafer map as one of these formats:

- JSON nested list
- CSV numeric matrix
- PNG/JPG grayscale image

The app resizes the input to `64 x 64 x 1`, loads `wafer_classifier.keras`, and shows ranked defect predictions when TensorFlow is available.

## Deploy On Hugging Face Spaces

Create a new Space and choose:

```text
SDK: Docker
Repository: josephfkung-del/WaferClassification
App file: app.py
Port: 7860
```

The `Dockerfile` installs Streamlit plus `tensorflow-cpu` and runs:

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=7860
```

Hugging Face Docker builds can take several minutes on the first deploy.
