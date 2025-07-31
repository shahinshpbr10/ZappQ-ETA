from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# === Load model and dependencies ===
try:
    MODEL_PATH = "backend/models/eta_model.pkl"
    SCALER_PATH = "backend/models/eta_scaler.pkl" 
    FEATURES_PATH = "backend/models/eta_features.txt"
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    with open(FEATURES_PATH, "r") as f:
        feature_cols = [line.strip() for line in f.readlines()]
        
    print(f"✅ Model loaded successfully with {len(feature_cols)} features")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# === FastAPI Setup ===
app = FastAPI(
    title="Patient Queue ETA Prediction API",
    description="Predicts estimated waiting time for patients in a medical queue",
    version="1.0.0"
)

class ETAPredictionRequest(BaseModel):
    current_token: str  # e.g., "A5", "W3"
    current_time: str   # e.g., "10:30:00"
    test_token: str     # e.g., "A15", "W8"
    day_name: str = "Wednesday"  # Optional, defaults to Wednesday

class ETAPredictionResponse(BaseModel):
    test_token: str
    current_token: str
    patients_ahead: int
    estimated_wait_minutes: float
    estimated_consultation_time: str
    day_adjustment: str
    served_after_break: bool

def parse_token(token: str):
    """Parse token like 'A5' or 'W3' into prefix and number"""
    token = token.strip().upper()
    if len(token) < 2:
        raise ValueError("Invalid token format")
    
    prefix = token[0] if token[0] in ['A', 'W'] else ''
    try:
        number = int(token[1:] if prefix else token)
    except:
        raise ValueError("Invalid token number")
    
    return prefix, number

def build_feature_vector(prefix: str, index: int, current_time: datetime, day_name: str, weekday_num: int):
    """Build feature vector for model prediction"""
    
    # Predict if served after break
    served_after_break = 1 if (index % 20 == 0 and index > 0) or index > 25 else 0
    
    # Core features
    input_data = {
        "Index": index,
        "IndexScaled": index / 10.0,
        "IndexSqrt": np.sqrt(index),
        "Hour": current_time.hour,
        "Minute": current_time.minute, 
        "Weekday": weekday_num,
        "ServedAfterBreak": served_after_break,
        "IsLunchTime": 1 if 12 <= current_time.hour <= 13 else 0,
        "IsMorning": 1 if current_time.hour <= 11 else 0,
        "IsAfternoon": 1 if current_time.hour >= 14 else 0,
        "EveryTwentieth": 1 if index % 20 == 0 else 0,
        "AfterLunchPosition": 1 if index > 25 else 0,
        "Type_advance": 1 if prefix == 'A' else 0,
        "Type_walkin": 1 if prefix == 'W' else 0,
        "TokenPrefix_A": 1 if prefix == 'A' else 0,
        "TokenPrefix_W": 1 if prefix == 'W' else 0,
    }
    
    # Day flags
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day in days:
        input_data[f"Day_{day}"] = 1 if day_name == day else 0
    
    # Fill missing features with 0
    feature_vector = {feat: input_data.get(feat, 0) for feat in feature_cols}
    
    return feature_vector

def calculate_logical_eta(current_index: int, test_index: int, day_name: str):
    """Calculate ETA using logical queue dynamics"""
    if test_index <= current_index:
        return 0
        
    patients_ahead = test_index - current_index
    base_time = patients_ahead * 14  # 14 minutes per patient
    
    # Day adjustments
    day_multipliers = {
        'Monday': 1.2, 'Tuesday': 1.1, 'Wednesday': 1.0, 'Thursday': 1.0,
        'Friday': 1.15, 'Saturday': 0.9, 'Sunday': 0.8
    }
    
    day_multiplier = day_multipliers.get(day_name, 1.0)
    adjusted_time = base_time * day_multiplier
    
    # Add break time
    break_time = 0
    if test_index % 20 == 0:
        break_time += 15
    if test_index > 25:
        break_time += 30
        
    return adjusted_time + break_time

@app.get("/")
def read_root():
    return {
        "message": "Patient Queue ETA Prediction API",
        "status": "active",
        "endpoints": {
            "predict": "/predict - POST request for ETA prediction",
            "health": "/health - GET request for health check"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=ETAPredictionResponse)
def predict_eta(request: ETAPredictionRequest):
    try:
        # Parse tokens
        current_prefix, current_index = parse_token(request.current_token)
        test_prefix, test_index = parse_token(request.test_token)
        
        # Parse current time
        today = datetime.now().date()
        current_time = datetime.combine(
            today, 
            datetime.strptime(request.current_time, "%H:%M:%S").time()
        )
        
        # Get day info
        day_to_weekday = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        weekday_num = day_to_weekday.get(request.day_name, 2)
        
        # Calculate logical ETA
        logical_eta = calculate_logical_eta(current_index, test_index, request.day_name)
        
        # Get model prediction
        feature_vector = build_feature_vector(
            test_prefix, test_index, current_time, request.day_name, weekday_num
        )
        
        X_input = pd.DataFrame([feature_vector])
        X_input_scaled = scaler.transform(X_input)
        model_prediction = model.predict(X_input_scaled)[0]
        
        # Use the more conservative estimate
        final_eta = max(logical_eta, model_prediction)
        
        # Calculate consultation time
        if test_index <= current_index:
            consultation_time = current_time.strftime("%I:%M %p")
            final_eta = 0
        else:
            consultation_datetime = current_time + timedelta(minutes=final_eta)
            consultation_time = consultation_datetime.strftime("%I:%M %p")
        
        # Day adjustment info
        day_multipliers = {
            'Monday': 1.2, 'Tuesday': 1.1, 'Wednesday': 1.0, 'Thursday': 1.0,
            'Friday': 1.15, 'Saturday': 0.9, 'Sunday': 0.8
        }
        day_multiplier = day_multipliers.get(request.day_name, 1.0)
        
        if request.day_name in ['Monday', 'Friday']:
            day_adjustment = f"{request.day_name}s are typically busier (+{int((day_multiplier-1)*100)}%)"
        elif request.day_name in ['Saturday', 'Sunday']:
            day_adjustment = f"{request.day_name}s are typically quieter ({int((1-day_multiplier)*100)}% shorter)"
        else:
            day_adjustment = f"{request.day_name} - normal waiting times"
        
        # Check if served after break
        served_after_break = (test_index % 20 == 0 and test_index > 0) or test_index > 25
        
        return ETAPredictionResponse(
            test_token=request.test_token,
            current_token=request.current_token,
            patients_ahead=max(0, test_index - current_index),
            estimated_wait_minutes=round(final_eta, 1),
            estimated_consultation_time=consultation_time,
            day_adjustment=day_adjustment,
            served_after_break=served_after_break
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
