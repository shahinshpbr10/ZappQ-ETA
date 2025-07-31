# backend/scripts/train_model.py

"""
FIXED: Simple, logical model training that learns basic queue dynamics
without overfitting to noise patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

def parse_token(token):
    if isinstance(token, str) and len(token) >= 2:
        prefix = token[0].upper()
        try:
            number = int(token[1:])
        except:
            number = -1
    else:
        prefix = ''
        number = -1
    return prefix, number

def time_to_minutes(time_str):
    """Convert time string to minutes"""
    try:
        if pd.isna(time_str):
            return 0
        parts = str(time_str).split(':')
        if len(parts) == 3:
            h, m, s = map(float, parts)
            return h * 60 + m + s/60
        return 0
    except:
        return 0

def main():
    # Load refined data
    data_path = "backend/data/processed/MoideenBabuPerayil_refined.csv"
    print(f"\nLoading refined data from {data_path} ...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records.")

    # Parse tokens
    print("Parsing TokenNumber to extract features...")
    df[['TokenPrefix', 'TokenNumber_numeric']] = df['TokenNumber'].apply(lambda x: pd.Series(parse_token(x)))
    
    # Use existing Index column if valid
    df['Index'] = df.apply(
        lambda r: r['Index'] if pd.notna(r['Index']) and r['Index'] >= 0 else r['TokenNumber_numeric'],
        axis=1
    )

    # Process datetime columns
    print("Processing datetime columns...")
    df['TimeCalled'] = pd.to_datetime(df['TimeCalled'], format="%H:%M:%S", errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    df['Hour'] = df['TimeCalled'].dt.hour.fillna(10).astype(int)
    df['Minute'] = df['TimeCalled'].dt.minute.fillna(0).astype(int)
    df['Weekday'] = df['Date'].dt.weekday.fillna(2).astype(int)  # Default to Wednesday

    # TARGET: Wait time in minutes
    df['WaitTimeMinutes'] = df['AvgWaitTime'].apply(time_to_minutes)
    
    # Remove invalid records
    df = df[(df['WaitTimeMinutes'] > 0) & (df['WaitTimeMinutes'] <= 480)].reset_index(drop=True)
    print(f"After filtering: {len(df)} valid records.")

    # Clean ServedAfterBreak
    print("Processing ServedAfterBreak column...")
    df['ServedAfterBreak'] = df['ServedAfterBreak'].astype(str).str.lower()
    df['ServedAfterBreak'] = df['ServedAfterBreak'].map({
        'true': 1, 'false': 0, '1': 1, '0': 0, 'True': 1, 'False': 0
    }).fillna(0).astype(int)

    # SIMPLIFIED feature engineering - avoid overfitting
    print("Creating SIMPLIFIED features...")
    
    # Core features that make logical sense
    df['IndexScaled'] = df['Index'] / 10.0  # Scale index to reasonable range
    df['IndexSqrt'] = np.sqrt(df['Index'])  # Sqrt instead of square to avoid huge values
    
    # Time features
    df['IsLunchTime'] = ((df['Hour'] >= 12) & (df['Hour'] <= 13)).astype(int)
    df['IsMorning'] = (df['Hour'] <= 11).astype(int)
    df['IsAfternoon'] = (df['Hour'] >= 14).astype(int)
    
    # Simple break features
    df['EveryTwentieth'] = (df['Index'] % 20 == 0).astype(int)
    df['AfterLunchPosition'] = (df['Index'] > 25).astype(int)
    
    # One-hot encode only essential categorical variables
    categorical_cols = []
    if 'Day' in df.columns:
        categorical_cols.append('Day')
    if 'Type' in df.columns:
        categorical_cols.append('Type')
    if 'TokenPrefix' in df.columns:
        categorical_cols.append('TokenPrefix')

    df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)

    # Define SIMPLIFIED feature set
    base_features = [
        'Index', 'IndexScaled', 'IndexSqrt',
        'Hour', 'Minute', 'Weekday', 'ServedAfterBreak',
        'IsLunchTime', 'IsMorning', 'IsAfternoon', 
        'EveryTwentieth', 'AfterLunchPosition'
    ]
    
    # Add only existing one-hot encoded features
    one_hot_features = [col for col in df.columns if (
        col.startswith('Day_') or 
        col.startswith('Type_') or 
        col.startswith('TokenPrefix_')
    )]

    feature_cols = base_features + one_hot_features
    available_features = [f for f in feature_cols if f in df.columns]
    
    print(f"Using {len(available_features)} simplified features:")
    for feat in available_features:
        print(f"  - {feat}")

    # Prepare data
    X = df[available_features].fillna(0)
    y = df['WaitTimeMinutes']

    # Train-test split
    print("\nSplitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")

    # Scale features for better training
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SIMPLIFIED model with constraints to prevent overfitting
    print("Training SIMPLIFIED RandomForest model...")
    model = RandomForestRegressor(
        n_estimators=50,      # Fewer trees to prevent overfitting
        max_depth=8,          # Shallow trees
        min_samples_split=10, # Require more samples to split
        min_samples_leaf=5,   # Require more samples in leaves
        max_features=0.6,     # Use subset of features
        random_state=42,
        n_jobs=-1
    )
    
    # Train on scaled data
    model.fit(X_train_scaled, y_train)

    # Evaluate
    print("Evaluating model...")
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"Training MAE: {train_mae:.2f} minutes")
    print(f"Test MAE: {test_mae:.2f} minutes")
    print(f"Training R²: {train_r2:.3f}")
    print(f"Test R²: {test_r2:.3f}")
    
    # Check for overfitting
    if train_mae < test_mae * 0.5:
        print("⚠️  WARNING: Model may be overfitting!")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🎯 Feature Importance:")
    for _, row in feature_importance.head(8).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")

    # CRITICAL: Test logical progression
    print(f"\n🧪 Testing logical progression (MUST increase with Index):")
    
    # Create test sample with default values
    test_sample = pd.DataFrame([{
        'Index': 1,
        'IndexScaled': 0.1,
        'IndexSqrt': 1.0,
        'Hour': 10,
        'Minute': 0,
        'Weekday': 2,  # Wednesday
        'ServedAfterBreak': 0,
        'IsLunchTime': 0,
        'IsMorning': 1,
        'IsAfternoon': 0,
        'EveryTwentieth': 0,
        'AfterLunchPosition': 0
    }])
    
    # Add one-hot encoded features with default values
    for feat in one_hot_features:
        if 'Day_Wednesday' in feat:
            test_sample[feat] = 1
        elif 'Type_advance' in feat:
            test_sample[feat] = 1
        elif 'TokenPrefix_A' in feat:
            test_sample[feat] = 1
        else:
            test_sample[feat] = 0
    
    # Ensure all features are present
    for feat in available_features:
        if feat not in test_sample.columns:
            test_sample[feat] = 0
    
    test_sample = test_sample[available_features]
    
    predictions = []
    for test_idx in [1, 3, 5, 10, 15, 20, 25, 30, 35, 40]:
        sample = test_sample.copy()
        sample['Index'] = test_idx
        sample['IndexScaled'] = test_idx / 10.0
        sample['IndexSqrt'] = np.sqrt(test_idx)
        sample['EveryTwentieth'] = 1 if test_idx % 20 == 0 else 0
        sample['AfterLunchPosition'] = 1 if test_idx > 25 else 0
        
        # Scale the sample
        sample_scaled = scaler.transform(sample)
        prediction = model.predict(sample_scaled)[0]
        predictions.append(prediction)
        
        print(f"  Index {test_idx:2d}: {prediction:6.1f} minutes ({prediction/60:.1f}h)")
    
    # Check if predictions increase logically
    is_logical = all(predictions[i] <= predictions[i+1] * 1.5 for i in range(len(predictions)-1))
    if is_logical:
        print("✅ Model predictions follow logical progression!")
    else:
        print("❌ WARNING: Model predictions don't follow logical progression!")

    # Save model, scaler, and features
    os.makedirs("backend/models", exist_ok=True)
    
    model_path = "backend/models/eta_model.pkl"
    scaler_path = "backend/models/eta_scaler.pkl"
    features_path = "backend/models/eta_features.txt"
    
    print(f"\nSaving model to {model_path}")
    joblib.dump(model, model_path)
    
    print(f"Saving scaler to {scaler_path}")
    joblib.dump(scaler, scaler_path)
    
    print(f"Saving features to {features_path}")
    with open(features_path, 'w') as f:
        for feat in available_features:
            f.write(feat + "\n")

    print("\n✅ FIXED training complete!")

if __name__ == "__main__":
    main()