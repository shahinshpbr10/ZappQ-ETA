# backend/scripts/data_refinement.py

"""
FIXED: Simple, logical data refinement that creates realistic wait times
based on basic queue dynamics without over-complication.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def time_to_seconds(t):
    if pd.isna(t) or t == '':
        return 0
    parts = str(t).split(':')
    if len(parts) == 3:
        h, m, s = map(float, parts)
        return h*3600 + m*60 + s
    return 0

def seconds_to_timestr(seconds):
    seconds = int(round(seconds))
    h, m = divmod(seconds, 3600)
    m, s = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

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

def calculate_simple_wait_time(index, day_name='Wednesday'):
    """
    SIMPLIFIED: Calculate wait time using basic queue logic
    - Base: 12-15 minutes per patient ahead
    - Add breaks every 20 patients (15 min)
    - Add lunch break for patients after position 25
    - Day adjustments
    """
    if index <= 1:
        return np.random.uniform(60, 300)  # 1-5 minutes for first patients
    
    # Base calculation: 12-15 minutes per patient ahead
    base_minutes_per_patient = np.random.uniform(12, 15)
    base_wait_minutes = (index - 1) * base_minutes_per_patient
    
    # Add break time for every 20 patients served before this patient
    breaks_before = (index - 1) // 20
    break_time_minutes = breaks_before * 15
    
    # Add lunch break for patients likely called after 12:30 PM
    # Rough estimate: if patient is after position 25, add lunch break
    lunch_break_minutes = 60 if index > 25 else 0
    
    # Day of week adjustments
    day_multipliers = {
        'Monday': 1.2,    # 20% longer
        'Tuesday': 1.1,   # 10% longer  
        'Wednesday': 1.0, # Normal
        'Thursday': 1.0,  # Normal
        'Friday': 1.15,   # 15% longer
        'Saturday': 0.9,  # 10% shorter
        'Sunday': 0.8     # 20% shorter
    }
    
    day_multiplier = day_multipliers.get(day_name, 1.0)
    
    # Calculate total
    total_minutes = (base_wait_minutes + break_time_minutes + lunch_break_minutes) * day_multiplier
    
    # Add realistic variation (±5%)
    variation = np.random.normal(0, total_minutes * 0.05)
    total_minutes += variation
    
    # Ensure reasonable bounds
    total_minutes = max(1, min(480, total_minutes))  # 1 minute to 8 hours
    
    return total_minutes * 60  # Convert to seconds

def determine_served_after_break_simple(index):
    """
    SIMPLIFIED: Determine if served after break
    - Every 20th patient: True
    - Patients after position 25: True (lunch break)
    - Otherwise: False
    """
    if index <= 1:
        return False
    
    # Every 20th patient gets break
    if index % 20 == 0:
        return True
    
    # Patients likely called after lunch (rough estimate)
    if index > 25:
        return True
    
    return False

def main():
    # Input/output paths
    infile = "backend/data/raw/MoideenBabuPerayil_new.csv"
    outfile = "backend/data/processed/MoideenBabuPerayil_refined.csv"
    
    print(f"\n>>> Loading raw data: {infile}")
    df = pd.read_csv(infile)
    print(f"Loaded {len(df)} rows.")

    # Parse tokens
    print(">> Parsing tokens...")
    df[['TokenPrefix', 'TokenNumber']] = df['TokenNumber'].apply(lambda x: pd.Series(parse_token(x)))
    df['Index'] = df.apply(lambda r: r['Index'] if pd.notna(r['Index']) and r['Index'] >= 0 else r['TokenNumber'], axis=1)

    # Clean TimeCalled - remove invalid times
    print(">> Cleaning consultation times...")
    def is_valid_time(time_str):
        try:
            time_obj = datetime.strptime(str(time_str), '%H:%M:%S').time()
            start_time = datetime.strptime('09:00:00', '%H:%M:%S').time()
            end_time = datetime.strptime('16:30:00', '%H:%M:%S').time()
            return start_time <= time_obj <= end_time
        except:
            return False
    
    initial_count = len(df)
    df = df[df['TimeCalled'].apply(is_valid_time)].reset_index(drop=True)
    print(f"Removed {initial_count - len(df)} invalid time entries.")

    # Fix consultation times - make them realistic
    print(">> Fixing consultation times...")
    for idx, row in df.iterrows():
        consult_seconds = time_to_seconds(row['TimeTaken'])
        
        # If consultation time is unrealistic, replace it
        if consult_seconds < 300 or consult_seconds > 1800:  # Less than 5 min or more than 30 min
            if row['Type'].lower() == 'advance':
                new_seconds = np.random.normal(12 * 60, 3 * 60)  # 12±3 minutes
            else:  # walkin
                new_seconds = np.random.normal(10 * 60, 2 * 60)  # 10±2 minutes
            
            new_seconds = max(300, min(1800, new_seconds))  # 5-30 minutes
            df.at[idx, 'TimeTaken'] = seconds_to_timestr(new_seconds)

    # SIMPLIFIED wait time calculation
    print(">> Calculating SIMPLIFIED wait times...")
    
    # Group by date and doctor for session-based calculation
    for (date, doctor), session_df in df.groupby(['Date', 'DoctorID']):
        session_df = session_df.sort_values('Index').reset_index(drop=True)
        
        # Get day of week
        try:
            date_obj = pd.to_datetime(date)
            day_name = date_obj.strftime('%A')
        except:
            day_name = 'Wednesday'  # Default
        
        for _, row in session_df.iterrows():
            original_idx = row.name
            
            # Calculate simple wait time
            wait_seconds = calculate_simple_wait_time(row['Index'], day_name)
            df.at[original_idx, 'AvgWaitTime'] = seconds_to_timestr(wait_seconds)
            
            # Simple break determination
            served_after_break = determine_served_after_break_simple(row['Index'])
            df.at[original_idx, 'ServedAfterBreak'] = served_after_break

    # Save output
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    df.to_csv(outfile, index=False)
    print(f">>> SIMPLIFIED refinement complete; saved to {outfile}")
    
    # Print summary statistics
    df['WaitMinutes'] = df['AvgWaitTime'].apply(lambda x: time_to_seconds(x) / 60)
    df['ConsultMinutes'] = df['TimeTaken'].apply(lambda x: time_to_seconds(x) / 60)
    
    print(f"\n📊 Summary Statistics:")
    print(f"   Average consultation time: {df['ConsultMinutes'].mean():.1f} minutes")
    print(f"   Average wait time: {df['WaitMinutes'].mean():.1f} minutes")
    print(f"   Wait time range: {df['WaitMinutes'].min():.1f} - {df['WaitMinutes'].max():.1f} minutes")
    
    # Validate logical progression
    print(f"\n📈 Wait Time Validation by Index (should increase logically):")
    for idx_val in [1, 5, 10, 15, 20, 25, 30, 35, 40]:
        subset = df[df['Index'] == idx_val]
        if len(subset) > 0:
            avg_wait = subset['WaitMinutes'].mean()
            print(f"   Index {idx_val:2d}: {avg_wait:6.1f} minutes average")

if __name__ == "__main__":
    main()