from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime, timedelta

# === Load model and features ===
model = joblib.load("models/eta_model.pkl")

with open("models/eta_features.txt", "r") as f:
    feature_cols = [line.strip() for line in f.readlines()]

# === FastAPI Setup ===
app = FastAPI()

class ETAInput(BaseModel):
    Index: int
    Hour: int
    Minute: int
    Weekday: int
    ServedAfterBreak: int
    Day_Friday: int
    Specialization_Neonatology: int
    Type_walkin: int

@app.post("/predict")
def predict_eta(data: ETAInput):
    input_dict = data.dict()
    # Fill missing features with 0
    input_vector = {col: input_dict.get(col, 0) for col in feature_cols}
    X_input = pd.DataFrame([input_vector])
    
    # Predict ETA
    predicted_eta_minutes = model.predict(X_input)[0]
    
    now = datetime.now()
    estimated_time = now + timedelta(minutes=predicted_eta_minutes)
    estimated_time_str = estimated_time.strftime("%I:%M %p")
    
    return {
        "predicted_eta_minutes": round(predicted_eta_minutes, 2),
        "estimated_time": estimated_time_str
    }
