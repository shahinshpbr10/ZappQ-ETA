# backend/scripts/run_data_refinement.py
"""
Simple script to run data refinement process
"""


import pandas as pd
import numpy as np


def refine_appointment_data():
    """
    Quick and effective data refinement for appointment data
    Focuses on fixing unrealistic consultation times
    """

    print("🏥 APPOINTMENT DATA REFINEMENT")
    print("=" * 40)

    # File paths
    input_file = 'data/raw/MoideenBabuPerayil.csv'  # Adjust path as needed
    output_file = 'data/refined/MoideenBabuPerayil_refined.csv'

    # Load data
    print(f"📁 Loading data from: {input_file}")
    try:
        df = pd.read_csv(input_file)
        print(f"✅ Loaded {len(df):,} records")
    except FileNotFoundError:
        print(f"❌ File not found: {input_file}")
        print("Please ensure the CSV file is in the same directory as this script")
        return False

    # Function to convert time string to seconds
    def time_to_seconds(time_str):
        try:
            if pd.isna(time_str) or time_str == '':
                return 0
            parts = str(time_str).split(':')
            if len(parts) == 3:
                h, m, s = map(float, parts)
                return h * 3600 + m * 60 + s
            return 0
        except:
            return 0

    # Function to convert seconds to time string
    def seconds_to_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    # Analyze current data
    df['TimeTaken_seconds'] = df['TimeTaken'].apply(time_to_seconds)
    df['AvgWaitTime_seconds'] = df['AvgWaitTime'].apply(time_to_seconds)

    # Data quality analysis
    valid_times = df[df['TimeTaken_seconds'] > 0]
    # Less than 5 minutes
    problematic_times = valid_times[valid_times['TimeTaken_seconds'] < 300]

    print(f"\n📊 Data Quality Analysis:")
    print(f"   Total records: {len(df):,}")
    print(f"   Valid consultation times: {len(valid_times):,}")
    print(
        f"   Problematic times (< 5 min): {len(problematic_times):,} ({len(problematic_times)/len(valid_times)*100:.1f}%)")

    # Define realistic consultation times by specialization and type
    realistic_times = {
        'Neonatology': {
            'advance': (8, 15, 25),    # min, typical, max minutes
            'walkin': (5, 12, 20)
        },
        'Obstetrics and Gynecology': {
            'advance': (10, 20, 35),
            'walkin': (8, 15, 25)
        }
    }

    # Refine consultation times
    print(f"\n🔧 Refining consultation times...")
    refined_count = 0

    for idx, row in df.iterrows():
        current_time_seconds = row['TimeTaken_seconds']

        # Check if time needs refinement (< 5 minutes or > 60 minutes)
        if current_time_seconds < 300 or current_time_seconds > 3600:

            # Get specialization and appointment type
            specialization = row['Specialization']
            apt_type = str(row['Type']).lower().strip()

            # Map appointment types
            if apt_type in ['advance']:
                apt_type = 'advance'
            else:
                apt_type = 'walkin'

            # Get realistic time range
            if specialization in realistic_times and apt_type in realistic_times[specialization]:
                min_time, typical_time, max_time = realistic_times[specialization][apt_type]
            else:
                # Default values
                min_time, typical_time, max_time = (8, 15, 25)

            # Calculate realistic time based on factors
            base_time = typical_time

            # Adjustments based on context
            adjustments = 0

            # Time of day (morning consultations might be longer)
            try:
                hour = int(str(row['TimeCalled']).split(':')[0])
                if hour < 10:
                    adjustments += 2
                elif hour > 16:
                    adjustments += 1
            except:
                pass

            # After break (first patient after break takes longer)
            if row['ServedAfterBreak'] in [True, 'True', 'true']:
                adjustments += 3

            # Add some realistic randomness
            random_factor = np.random.normal(0, 2)  # ±2 minutes variation
            final_time = max(min_time, min(
                max_time, base_time + adjustments + random_factor))

            # Convert to time string
            final_seconds = final_time * 60
            new_time_str = seconds_to_time(final_seconds)

            # Update the dataframe
            df.at[idx, 'TimeTaken'] = new_time_str
            df.at[idx, 'TimeTaken_seconds'] = final_seconds

            refined_count += 1

    print(f"✅ Refined {refined_count:,} consultation time records")

    # Refine wait times based on queue position
    print(f"\n⏰ Refining wait times...")

    for (date, doctor_id), group in df.groupby(['Date', 'DoctorID']):
        group = group.sort_values('Index').reset_index(drop=True)

        cumulative_time = 0
        for i, (idx, row) in enumerate(group.iterrows()):
            position = row['Index']

            # Calculate realistic wait time based on queue position
            base_wait = 3  # Base wait time in minutes
            queue_wait = position * 1.5  # 1.5 minutes per position
            consultation_backlog = cumulative_time * 0.6  # 60% of previous consultations
            break_adjustment = 8 if row['ServedAfterBreak'] in [
                True, 'True', 'true'] else 0

            # Total wait time with some randomness
            total_wait = base_wait + queue_wait + consultation_backlog + break_adjustment
            # ±10% variation
            total_wait += np.random.normal(0, total_wait * 0.1)

            # Ensure reasonable bounds (1-45 minutes)
            total_wait = max(1, min(45, total_wait))

            # Convert to time string
            wait_seconds = total_wait * 60
            new_wait_str = seconds_to_time(wait_seconds)

            # Update dataframe
            df.at[idx, 'AvgWaitTime'] = new_wait_str
            df.at[idx, 'AvgWaitTime_seconds'] = wait_seconds

            # Update cumulative time for next iteration
            cumulative_time += row['TimeTaken_seconds'] / 60

    # Generate realistic TimeCalled based on appointment flow
    print(f"\n🕒 Generating realistic call times...")

    for (date, doctor_id), group in df.groupby(['Date', 'DoctorID']):
        group = group.sort_values('Index').reset_index(drop=True)

        # Start time (parse from ConsultationFromTime)
        try:
            start_time_str = group.iloc[0]['ConsultationFromTime']
            if 'AM' in start_time_str or 'PM' in start_time_str:
                # Parse 12-hour format
                time_part = start_time_str.replace(
                    'AM', '').replace('PM', '').strip()
                hour, minute = map(int, time_part.split(':'))
                if 'PM' in start_time_str and hour != 12:
                    hour += 12
                elif 'AM' in start_time_str and hour == 12:
                    hour = 0
            else:
                # Assume 24-hour format
                hour, minute = map(int, start_time_str.split(':'))

            current_minutes = hour * 60 + minute
        except:
            current_minutes = 9 * 60  # Default to 9:00 AM

        for i, (idx, row) in enumerate(group.iterrows()):
            if i > 0:
                # Add previous consultation time + buffer
                prev_consultation_minutes = group.iloc[i -
                                                       1]['TimeTaken_seconds'] / 60
                buffer_minutes = np.random.normal(
                    2, 0.5)  # 2±0.5 minute buffer
                current_minutes += prev_consultation_minutes + buffer_minutes

            # Convert back to time string
            hours = int(current_minutes // 60)
            minutes = int(current_minutes % 60)
            new_time_called = f"{hours:02d}:{minutes:02d}:00"

            df.at[idx, 'TimeCalled'] = new_time_called

    # Final validation
    print(f"\n✅ VALIDATION RESULTS")
    print("=" * 40)

    # Recalculate statistics
    final_consultation_times = df['TimeTaken_seconds']
    valid_final = final_consultation_times[final_consultation_times > 0]

    print(f"📊 Refined Data Quality:")
    print(f"   Total records: {len(df):,}")
    print(f"   Valid consultation times: {len(valid_final):,}")
    print(f"   Mean consultation time: {valid_final.mean()/60:.1f} minutes")
    print(
        f"   Median consultation time: {valid_final.median()/60:.1f} minutes")

    # Time distribution
    time_ranges = {
        "< 5 minutes": (valid_final < 300).sum(),
        "5-15 minutes": ((valid_final >= 300) & (valid_final < 900)).sum(),
        "15-30 minutes": ((valid_final >= 900) & (valid_final < 1800)).sum(),
        "30+ minutes": (valid_final >= 1800).sum()
    }

    print(f"\n⏱️ Time Distribution:")
    for range_name, count in time_ranges.items():
        percentage = (count / len(valid_final)) * 100
        print(f"   {range_name}: {count:,} ({percentage:.1f}%)")

    # Remove temporary columns
    df_final = df.drop(columns=['TimeTaken_seconds', 'AvgWaitTime_seconds'])

    # Save refined data
    print(f"\n💾 Saving refined data to: {output_file}")
    df_final.to_csv(output_file, index=False)

    print(f"\n🎉 DATA REFINEMENT COMPLETED!")
    print(f"   Original problematic records: {len(problematic_times):,}")
    print(f"   Records refined: {refined_count:,}")
    print(
        f"   Quality improvement: {(1 - time_ranges['< 5 minutes']/len(valid_final))*100:.1f}%")
    print(f"   Output file: {output_file}")

    return True


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)

    # Run refinement
    success = refine_appointment_data()

    if success:
        print(f"\n✅ Ready for ML model training!")
        print(f"Next steps:")
        print(f"1. Use the refined CSV file for training")
        print(f"2. Run the ML model training script")
        print(f"3. Deploy the trained model")
    else:
        print(f"\n❌ Refinement failed. Please check the error messages above.")
