# Wafer Classification

A wafer defect classification web app based on the `Wafer_Classification.ipynb` Colab notebook.

## What Is In This Repo

- `app.py` - Streamlit web app scaffold
- `requirements.txt` - Python dependencies for local/deployment installs
- `notebooks/Wafer_Classification.ipynb` - notebook handoff with the original Colab link and export instructions

Original full notebook:

- https://github.com/josephfkung-del/DementiaBank-Classifcation/blob/main/Wafer_Classification.ipynb

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model Artifact Needed

The current Colab notebook trains the CNN but does not include a deployable model artifact in this repo yet. At the end of the Colab training notebook, run:

```python
model.save("wafer_classifier.keras")
```

Then add the saved file as:

```text
model/wafer_classifier.keras
```

Once that file exists, `app.py` can be updated to load the model and run real predictions.

## Deployment Target

This is ready for a simple Streamlit deployment once the trained model artifact is added. Good options are Streamlit Community Cloud, Hugging Face Spaces, Render, or Railway.
