import pandas as pd
import joblib
from datetime import datetime, timedelta

# === STEP 1: Load model and feature list ===
model = joblib.load("models/eta_model.pkl")

with open("models/eta_features.txt", "r") as f:
    feature_cols = [line.strip() for line in f.readlines()]

# === STEP 2: Get current time ===
now = datetime.now()
hour = now.hour
minute = now.minute
weekday = now.weekday()  # Monday = 0, Sunday = 6


input_data = {
    "Index": 15,  # e.g., 15th patient in the queue
    "Hour": hour,
    "Minute": minute,
    "Weekday": weekday,
    "ServedAfterBreak": 1,
    "Day_Friday": 1,
    "Specialization_Neonatology": 1,
    "Type_walkin": 1
    # All other one-hot columns default to 0
}

# === STEP 4: Fill missing features with 0 ===
input_vector = {col: input_data.get(col, 0) for col in feature_cols}
X_input = pd.DataFrame([input_vector])

# === STEP 5: Predict ETA in minutes ===
predicted_eta_minutes = model.predict(X_input)[0]

# === STEP 6: Calculate expected time to see the doctor ===
estimated_time = now + timedelta(minutes=predicted_eta_minutes)
estimated_time_str = estimated_time.strftime("%I:%M %p")

# === STEP 7: Output ===
print(f"⏱️ Predicted Waiting Time: {predicted_eta_minutes:.2f} minutes")
print(f"🩺 You can see the doctor at: {estimated_time_str}")
