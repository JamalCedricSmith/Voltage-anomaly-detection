# Voltage & Current Anomaly Detection

Unsupervised anomaly detection pipeline for multi-phase voltage and current time-series measurements using LSTM Autoencoders.

## Project Structure

- **`app.py`**: Core pipeline containing `AnomalyDetection` (base class), `Training`, and `Inference`.
- **`functions.py`**: Model architecture, sequence transformation, feature extraction, and evaluation metrics (F1, ROC-AUC, Precision, Recall).
- **`constants.py`**: Configuration constants, window sizes, and split timestamps.
- **`Training-Tutorial.ipynb`**: Walkthrough notebook for training and saving the LSTM Autoencoder model.
- **`Inference-Tutorial.ipynb`**: Walkthrough notebook for model evaluation, reconstruction loss calculation, and anomaly scoring.

## Quick Start

### Installation

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn statsmodels joblib
```

### Usage

```python
import pandas as pd
import app

# 1. Load data
df = pd.read_csv("data.csv")

# 2. Train model
trainer = app.Training(df)
trainer.preprocessing(feature_selection=False)
trainer.train(sequence_length=3, embedding_layer_size=8, epochs=10, saving_path="models/")

# 3. Inference
detector = app.Inference(df)
detector.preprocessing(feature_selection=False)
scores = detector.predictAnomalyScores(path="models/", model_name="<model_folder_name>")
```
