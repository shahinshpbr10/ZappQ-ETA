# 🏥 ETA Prediction for Hospital Appointments

A smart, ML-powered backend system to **predict Estimated Time of Arrival (ETA)** for hospital appointments using real-time and historical patient data. Designed to optimize patient flow and doctor efficiency.

---

## 🚀 Key Features

* **Smart Queue Logic**: Includes logic for doctor breaks after every 20 patients and lunch hours.
* **Day-Sensitive Adjustments**: Different ETA logic for weekdays, weekends, or busy days.
* **Token Differentiation**: Supports both **Advance (A)** and **Walk-in (W)** tokens.
* **Hybrid ETA Estimation**: Takes the **maximum of model prediction and logical estimate** for safer scheduling.

---

## 🧠 Techniques Used

* **Random Forest Regression** for time prediction.
* **Feature Engineering**: Index, token type, time of day, weekday, break rules, etc.
* **Model Validation** using realistic flow simulations (not just RMSE).
* **Data Refinement Pipeline** to preprocess and normalize noisy real-world patient logs.
* **Live ETA Prediction Script** to deploy the model in real time.

---

## 📁 Project Structure

```
.
├── backend/
│   ├── data/
│   │   ├── raw/                     # Raw input CSV files
│   │   └── processed/               # Refined data used for model training
│   ├── models/                      # Trained model and scalers
│   └── scripts/                     # All main scripts
│       ├── data_refinement.py      # Cleans and formats raw patient data
│       ├── train_model.py          # Trains the ETA prediction model
│       └── predict_eta_live.py     # Loads model and predicts ETA on new data
├── data/                           # DVC-tracked data versioning
├── main.py                         # Optional entry point (e.g., FastAPI server)
├── requirements.txt                # Project dependencies
└── README.md                       # You’re reading it!
```

---

## 📦 Setup & Installation

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd <project-folder>
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ⚙️ How to Run the Project

### ✅ Step 1: Data Preprocessing

Convert raw hospital logs to clean, refined data.

```bash
python backend/scripts/data_refinement.py
```

### ✅ Step 2: Train the ETA Model

Trains a Random Forest Regressor and saves the model & scaler.

```bash
python backend/scripts/train_model.py
```

### ✅ Step 3: Predict ETA in Real-Time

Use trained model to generate live ETA predictions.

```bash
python backend/scripts/predict_eta_live.py
```

### ✅ (Optional) Run API Server

Run a FastAPI server to serve the ETA prediction via API (if implemented in `main.py`).

```bash
uvicorn main:app --reload
```

---

## 🧾 File Breakdown & Relevance

| File                             | Description                                                   |
| -------------------------------- | ------------------------------------------------------------- |
| `data_refinement.py`             | Cleans and structures raw CSV logs into usable training data. |
| `train_model.py`                 | Trains the ETA model with preprocessed data and saves it.     |
| `predict_eta_live.py`            | Loads the trained model to predict ETA on new inputs.         |
| `eta_model.pkl`                  | Saved model file (Random Forest).                             |
| `eta_scaler.pkl`                 | Scaler used for feature normalization.                        |
| `MoideenBabuPerayil_new.csv`     | Raw patient data used for training.                           |
| `MoideenBabuPerayil_refined.csv` | Cleaned, processed version for model input.                   |

---

## 🛠 Example Features Used for Prediction

* Patient’s token number (index)
* Token type (Advance / Walk-in)
* Appointment time
* Day of the week
* Doctor break logic
* Time of day

---

## 🔍 Requirements

See [`requirements.txt`](./requirements.txt) for full dependency list.

Some key packages:

* `scikit-learn`
* `pandas`, `numpy`
* `fastapi`, `uvicorn`
* `joblib`
* `python-dateutil`

---

## ✨ Final Thoughts

This system helps **hospitals reduce waiting times**, manage patient flow effectively, and **build trust** through accurate appointment timing.

You can extend this system with:

* Real-time frontend updates
* SMS/email notifications
* Doctor-specific behavior modeling

---