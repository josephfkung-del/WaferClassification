# Wafer Classification

A Streamlit wafer defect classification web app based on the `Wafer_Classification.ipynb` Colab notebook.

## What Is In This Repo

- `app.py` - Streamlit app that loads the trained Keras model and runs predictions
- `wafer_classifier.keras` - trained model artifact uploaded from Colab
- `requirements.txt` - Python dependencies for local/deployment installs
- `notebooks/Wafer_Classification.ipynb` - notebook handoff with the original Colab link and export instructions

Original full notebook:

- https://github.com/josephfkung-del/DementiaBank-Classifcation/blob/main/Wafer_Classification.ipynb

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Use The App

Upload a wafer map as one of these formats:

- JSON nested list
- CSV numeric matrix
- PNG/JPG grayscale image

The app resizes the input to `64 x 64 x 1`, loads `wafer_classifier.keras`, and shows ranked defect predictions.

## Deployment Target

This is ready for a simple Streamlit deployment. Good options are Streamlit Community Cloud, Hugging Face Spaces, Render, or Railway.
