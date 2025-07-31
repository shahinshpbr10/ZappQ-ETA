# backend/scripts/predict_eta_live.py (FIXED VERSION)

import pandas as pd
import joblib
from datetime import datetime, timedelta
import numpy as np

# Load model, scaler, and features
MODEL_PATH = "backend/models/eta_model.pkl"
SCALER_PATH = "backend/models/eta_scaler.pkl"
FEAT_PATH = "backend/models/eta_features.txt"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(FEAT_PATH, "r") as f:
    feature_cols = [line.strip() for line in f.readlines()]

def parse_token(token):
    token = token.strip().upper()
    prefix = token[0] if token[0] in ['A','W'] else ''
    try:
        number = int(token[1:] if prefix else token)
    except:
        raise ValueError("Invalid token format. Use 'A11', 'W4' etc.")
    return prefix, number

def get_current_day_info():
    """Get current day info - HARDCODED for testing"""
    DAY_NAME = "Thursday"  # Change this to test different days
    
    day_to_weekday = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
        'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }
    
    weekday_num = day_to_weekday.get(DAY_NAME, 2)
    return DAY_NAME, weekday_num

def predict_served_after_break(index, current_time):
    """Simple break prediction logic"""
    # Every 20th patient
    if index % 20 == 0 and index > 0:
        return 1
    
    # Patients likely called after lunch (position > 25)
    if index > 25:
        return 1
    
    return 0

def calculate_logical_eta(current_index, test_index, current_time, day_name):
    """FIXED: Simple, logical ETA calculation"""
    
    if test_index <= current_index:
        return 0
    
    patients_ahead = test_index - current_index
    
    # Base calculation: 14 minutes per patient
    base_time = patients_ahead * 14
    
    # Day-of-week adjustments (more conservative)
    day_multipliers = {
        'Monday': 1.2,     # 20% longer
        'Tuesday': 1.1,    # 10% longer
        'Wednesday': 1.0,  # Normal
        'Thursday': 1.0,   # Normal
        'Friday': 1.15,    # 15% longer
        'Saturday': 0.9,   # 10% shorter
        'Sunday': 0.8      # 20% shorter
    }
    
    day_multiplier = day_multipliers.get(day_name, 1.0)
    adjusted_time = base_time * day_multiplier
    
    # Add break time
    break_time = 0
    if test_index % 20 == 0:  # Every 20th patient
        break_time += 15
    
    if test_index > 25:  # Lunch break for later patients
        break_time += 30  # Conservative lunch break estimate
    
    total_eta = adjusted_time + break_time
    return total_eta

def build_feature_vector(prefix, index, current_time, day_name, weekday_num):
    """Build feature vector for model prediction"""
    
    served_after_break = predict_served_after_break(index, current_time)
    
    # Day flags
    day_flags = {
        'Day_Monday': 1 if day_name == 'Monday' else 0,
        'Day_Tuesday': 1 if day_name == 'Tuesday' else 0,
        'Day_Wednesday': 1 if day_name == 'Wednesday' else 0,
        'Day_Thursday': 1 if day_name == 'Thursday' else 0,
        'Day_Friday': 1 if day_name == 'Friday' else 0,
        'Day_Saturday': 1 if day_name == 'Saturday' else 0,
        'Day_Sunday': 1 if day_name == 'Sunday' else 0,
    }
    
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
    
    # Add day flags
    input_data.update(day_flags)
    
    # Fill missing features with 0
    feature_vector = {feat: input_data.get(feat, 0) for feat in feature_cols}
    return feature_vector

def apply_logical_constraints(model_prediction, logical_eta, index):
    """Apply constraints to prevent unrealistic predictions"""
    
    # Constraint 1: Model prediction shouldn't be more than 3x logical estimate
    max_allowed = logical_eta * 3
    if model_prediction > max_allowed:
        constrained_prediction = max_allowed
        print(f"[CONSTRAINT] Model prediction ({model_prediction:.1f}min) capped at 3x logical ({max_allowed:.1f}min)")
        return constrained_prediction
    
    # Constraint 2: For early patients (index < 10), limit to reasonable times
    if index < 10 and model_prediction > 120:  # Max 2 hours for first 10 patients
        constrained_prediction = min(model_prediction, 120)
        print(f"[CONSTRAINT] Early patient prediction limited to {constrained_prediction:.1f}min")
        return constrained_prediction
    
    # Constraint 3: Use the more conservative estimate between model and logical
    final_prediction = max(logical_eta, model_prediction)
    return final_prediction

def main():
    # Get current day info
    day_name, weekday_num = get_current_day_info()
    
    ###########################################################################################

    # === USER INPUT AREA ===
    CURRENT_TOKEN = "A2"
    CURRENT_TIME_STR = "10:38:44"
    TEST_TOKEN = "W5"
    
    # Parse current time
    today = datetime.now().date()
    current_time = datetime.combine(today, datetime.strptime(CURRENT_TIME_STR, "%H:%M:%S").time())
    
    # Parse tokens
    current_prefix, current_index = parse_token(CURRENT_TOKEN)
    test_prefix, test_index = parse_token(TEST_TOKEN)
    
    print(f"📅 Testing Day: {day_name} (hardcoded for testing)")
    print(f"🕐 Current Time: {current_time.strftime('%I:%M %p')}")
    print(f"🎫 Current Token: {CURRENT_TOKEN} (Index: {current_index})")
    print(f"🎯 Testing Token: {TEST_TOKEN} (Index: {test_index})")
    
    # Calculate logical ETA
    logical_eta = calculate_logical_eta(current_index, test_index, current_time, day_name)
    
    # Get model prediction
    feat_vector = build_feature_vector(test_prefix, test_index, current_time, day_name, weekday_num)
    X_input = pd.DataFrame([feat_vector])
    X_input_scaled = scaler.transform(X_input)
    raw_model_prediction = model.predict(X_input_scaled)[0]
    
    # Apply logical constraints
    final_eta = apply_logical_constraints(raw_model_prediction, logical_eta, test_index)
    
    # Display results
    if test_index <= current_index:
        print("[INFO] It's your turn now or you've already been served!")
        eta_minutes = 0
        eta_time_str = current_time.strftime("%I:%M %p")
    else:
        eta_minutes = final_eta
        eta_datetime = current_time + timedelta(minutes=eta_minutes)
        eta_time_str = eta_datetime.strftime("%I:%M %p")
        
        patients_ahead = test_index - current_index
        day_multiplier = {
            'Monday': 1.2, 'Tuesday': 1.1, 'Wednesday': 1.0, 'Thursday': 1.0,
            'Friday': 1.15, 'Saturday': 0.9, 'Sunday': 0.8
        }.get(day_name, 1.0)
        
        print(f"[INFO] Patients ahead: {patients_ahead}")
        print(f"[INFO] Day adjustment: {day_name} = {day_multiplier:.1f}x multiplier")
        
        if day_name in ['Monday', 'Friday']:
            print(f"[INFO] ⚠️ {day_name}s are typically busier - expect longer waits")
        elif day_name in ['Saturday', 'Sunday']:
            print(f"[INFO] ✅ {day_name}s are typically quieter - shorter waits expected")
        
        # Show break info
        will_be_served_after_break = predict_served_after_break(test_index, current_time)
        if will_be_served_after_break:
            print(f"[INFO] 🔄 You will likely be served after a break")

    print("\n==== FIXED ETA RESULT ====")
    print(f"📅 Day: {day_name} (hardcoded for testing)")
    print(f"🎫 Token: {TEST_TOKEN}")
    print(f"⏱️ Estimated waiting time: {eta_minutes:.1f} minutes ({eta_minutes/60:.1f} hours)")
    print(f"🩺 Expected consultation start: {eta_time_str}")
    
    # Show calculation breakdown
    if test_index > current_index:
        print(f"\n[CALCULATION] Breakdown:")
        print(f"  Patients ahead: {patients_ahead}")
        print(f"  Base time (14 min/patient): {(test_index - current_index) * 14} minutes")
        print(f"  Day adjustment ({day_name}): ×{day_multiplier}")
        print(f"  Break consideration: {'Yes' if will_be_served_after_break else 'No'}")
        print(f"  Raw model prediction: {raw_model_prediction:.1f} minutes")
        print(f"  Logical calculation: {logical_eta:.1f} minutes")
        print(f"  Final ETA (constrained): {final_eta:.1f} minutes")
        
        # Show debug features
        print(f"\n[DEBUG] Key Features Used:")
        print(f"  Index: {test_index}")
        print(f"  IndexScaled: {test_index/10.0:.1f}")
        print(f"  Day_{day_name}: 1")
        print(f"  TokenPrefix_{test_prefix}: 1")
        print(f"  ServedAfterBreak: {predict_served_after_break(test_index, current_time)}")
        
        # Sanity check
        minutes_per_patient = final_eta / patients_ahead if patients_ahead > 0 else 0
        print(f"  Sanity check: {minutes_per_patient:.1f} minutes per patient ahead")
        
        if minutes_per_patient > 60:
            print(f"  ⚠️ WARNING: {minutes_per_patient:.1f} min/patient seems high!")
        elif minutes_per_patient < 5:
            print(f"  ⚠️ WARNING: {minutes_per_patient:.1f} min/patient seems low!")
        else:
            print(f"  ✅ Time per patient looks reasonable")

if __name__ == "__main__":
    main()