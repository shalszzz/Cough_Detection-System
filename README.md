# Cough-Vagal Health Monitor

**AI-Powered Cough Detection and Vagal Health Assessment Dashboard**

---

## Overview

The **Cough-Vagal Health Monitor** is an AI-powered system that detects cough events from audio recordings and assesses nervous system health using a novel **Vagal Tone Index (VTI)**.

This project was developed as a hackathon solution to build a cough detection pipeline that:

* Processes **750 Hz mono WAV audio**
* Detects cough events with timestamps and confidence
* Visualizes cough probability timeline, waveform, and spectrogram
* Provides an interactive clinical dashboard
* Assesses vagal nerve function using cough suppressibility and multimodal signals

Our system achieves:

* **Accuracy:** --%
* **AUC:** --
* **Baseline Accuracy:** 85.34%

---

## Key Features

### AI-Based Cough Detection

* Sliding-window audio analysis
* Mel-spectrogram feature extraction
* Machine learning classification
* Timestamp-level cough detection
* Confidence scoring

---

### Interactive Clinical Dashboard

The Streamlit dashboard provides:

* Full audio waveform visualization
* Cough detection probability timeline
* Spectrogram visualization
* Detected cough event list with audio snippets
* Confidence levels for each cough
* Audio playback

---

### Vagal Tone Index (VTI) — Novel Innovation

This project introduces a new metric:

**Vagal Tone Index (VTI)**

It measures nervous system health using:

* Cough suppressibility
* Respiratory patterns
* Motion sensor features

VTI provides:

* Score from 0–100
* Clinical interpretation
* Health insight visualization

---

## Dataset

**Source:**
NCSU Multimodal Cough Detection Dataset

Dryad DOI:
https://doi.org/10.5061/dryad.mkkwh717r

Dataset includes:

* 13 participants
* Multiple trials per participant
* Audio recordings
* IMU sensor data
* Expert cough annotations

---

## Project Structure

```
cough-vagal-monitor/
│
├── models/
│   ├── baseline_model.pkl
│   └── baseline_scaler.pkl
│
├── dashboard/
│   ├── dashboard.py
│   ├── audio_processing.py
│   ├── plots.py
│   └── utils.py
│
├── notebooks/
│   └── baseline_model_training.ipynb
│
├── requirements.txt
│
└── README.md
```

---

## Machine Learning Pipeline

### Feature Extraction

* Sample rate: 750 Hz
* Window size: 2 seconds
* Step size: 0.5 seconds
* Feature: Mel Spectrogram
* Feature size: 1536

---

### Baseline Model

Logistic Regression

Performance:

* Accuracy: ~87%
* AUC: ~0.91

---

### Final Model

Random Forest Classifier

Performance:

* Accuracy: 95.9%
* AUC: 0.994

---

## Dashboard Features

The Streamlit dashboard displays:

* Audio player
* Waveform
* Cough probability timeline
* Spectrogram
* Detected cough timestamps
* Confidence levels
* Vagal Tone Index

---

## Installation

Clone repository:

```
git clone https://github.com/shalszzz/Cough_Detection-System.git
```

Enter folder:

```
cd cough-vagal-monitor
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Dashboard

Run:

```
streamlit run dashboard/dashboard.py
```

Open browser at:

```
http://localhost:8501
```

---

## Technologies Used

* Python
* Streamlit
* Scikit-learn
* Librosa
* NumPy
* Plotly
* Matplotlib

---

## Results

| Model       | Accuracy | AUC   |
| ----------- | -------- | ----- |
| Baseline    | 87%      | 0.91  |
| Final Model | ---%     | ---   |

---

## Innovation

This project introduces:

**Passive vagal health assessment using cough suppressibility**

Potential applications:

* Neurological monitoring
* Long COVID assessment
* Parkinson’s disease monitoring
* Remote health monitoring

---

## Disclaimer

This tool is a hackathon prototype and is NOT a medical device. The outputs are estimates intended for research and
demonstration only. Do not use this tool to diagnose, treat, cure, or prevent any disease. If you have symptoms or
concerns, consult a qualified clinician.
---
