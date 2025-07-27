import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# === STEP 1: Load the dataset ===
csv_path = "data/processed/MoideenBabuPerayil_refined.csv" 
df = pd.read_csv(csv_path)

# === STEP 2: Clean column names ===
df.columns = [col.strip() for col in df.columns]

# Optional: print column names to debug if errors occur
# print("Columns:", df.columns.tolist())

# === STEP 3: Time conversions ===
try:
    df['TimeCalled'] = pd.to_datetime(df['TimeCalled'], format="%H:%M:%S")
except Exception as e:
    raise Exception(f"❌ Error in 'TimeCalled' conversion: {e}")

try:
    df['TimeTaken'] = pd.to_timedelta(df['TimeTaken'])
except Exception as e:
    raise Exception(f"❌ Error in 'TimeTaken' conversion: {e}")

# Date handling (fallback to dummy if not present)
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
else:
    df['Date'] = pd.to_datetime("2025-01-01")

df['Hour'] = df['TimeCalled'].dt.hour
df['Minute'] = df['TimeCalled'].dt.minute
df['Weekday'] = df['Date'].dt.weekday

# === STEP 4: Target value ===
df['TimeTakenMinutes'] = df['TimeTaken'].dt.total_seconds() / 60

# === STEP 5: Encode booleans and categoricals ===
if 'ServedAfterBreak' in df.columns:
    df['ServedAfterBreak'] = df['ServedAfterBreak'].astype(
        str).str.upper().map({'TRUE': 1, 'FALSE': 0})
else:
    df['ServedAfterBreak'] = 0  # fallback

# One-hot encode Day, Specialization, and Type
for cat_col in ['Day', 'Specialization', 'Type']:
    if cat_col in df.columns:
        df = pd.get_dummies(df, columns=[cat_col])
    else:
        print(
            f"⚠️ Column '{cat_col}' not found. Skipping one-hot encoding for it.")

# === STEP 6: Feature list ===
required_base_cols = ['Index', 'Hour', 'Minute', 'Weekday', 'ServedAfterBreak']
one_hot_cols = [col for col in df.columns if col.startswith(
    'Day_') or col.startswith('Specialization_') or col.startswith('Type_')]
feature_cols = required_base_cols + one_hot_cols

# Ensure all features exist in dataframe
missing_features = [col for col in feature_cols if col not in df.columns]
if missing_features:
    raise Exception(f"❌ Missing required feature columns: {missing_features}")

X = df[feature_cols]
y = df['TimeTakenMinutes']

# === STEP 7: Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# === STEP 8: Train Model ===
model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# === STEP 9: Evaluate ===
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("✅ Model trained successfully.")
print(f"📉 Mean Absolute Error (MAE): {mae:.2f} minutes")
print(f"📈 R² Score: {r2:.2f}")

# === STEP 10: Save Model & Metadata ===
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, os.path.join(model_dir, "eta_model.pkl"))
with open(os.path.join(model_dir, "eta_features.txt"), "w") as f:
    for col in feature_cols:
        f.write(col + "\n")

print(f"📦 Model saved to {model_dir}/eta_model.pkl")
print(f"📝 Features saved to {model_dir}/eta_features.txt")
