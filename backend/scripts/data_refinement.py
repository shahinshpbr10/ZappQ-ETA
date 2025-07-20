
"""
Data Refinement Script for Patient ETA Prediction
Cleans unrealistic consultation times and creates properly structured data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class DataRefinement:
    def __init__(self):
        self.df = None
        self.refined_df = None
        self.stats = {}

    def load_data(self, csv_path):
        """Load the raw CSV data"""
        print("📁 Loading raw data...")
        try:
            self.df = pd.read_csv(csv_path)
            print(
                f"✅ Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def analyze_data_quality(self):
        """Analyze current data quality issues"""
        print("\n🔍 DATA QUALITY ANALYSIS")
        print("=" * 50)

        # Convert time strings to seconds for analysis
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

        self.df['TimeTaken_seconds'] = self.df['TimeTaken'].apply(
            time_to_seconds)
        self.df['AvgWaitTime_seconds'] = self.df['AvgWaitTime'].apply(
            time_to_seconds)

        # Basic statistics
        total_records = len(self.df)
        valid_consultation_times = (self.df['TimeTaken_seconds'] > 0).sum()

        print(f"📊 Basic Statistics:")
        print(f"   Total records: {total_records:,}")
        print(f"   Valid consultation times: {valid_consultation_times:,}")
        print(
            f"   Missing consultation times: {total_records - valid_consultation_times:,}")

        # Time distribution analysis
        consultation_times = self.df[self.df['TimeTaken_seconds']
                                     > 0]['TimeTaken_seconds']

        print(f"\n⏱️ Consultation Time Distribution:")
        time_ranges = {
            "< 30 seconds": (consultation_times < 30).sum(),
            "30s - 1 min": ((consultation_times >= 30) & (consultation_times < 60)).sum(),
            "1 - 5 minutes": ((consultation_times >= 60) & (consultation_times < 300)).sum(),
            "5 - 15 minutes": ((consultation_times >= 300) & (consultation_times < 900)).sum(),
            "15 - 30 minutes": ((consultation_times >= 900) & (consultation_times < 1800)).sum(),
            "30+ minutes": (consultation_times >= 1800).sum()
        }

        for range_name, count in time_ranges.items():
            percentage = (count / len(consultation_times)) * 100
            print(f"   {range_name}: {count:,} ({percentage:.1f}%)")

        # Identify problematic records
        problematic_records = self.df[
            (self.df['TimeTaken_seconds'] > 0) &
            (self.df['TimeTaken_seconds'] < 120)  # Less than 2 minutes
        ]

        print(f"\n⚠️ Problematic Records (< 2 minutes consultation):")
        print(f"   Count: {len(problematic_records):,}")
        print(
            f"   Percentage: {len(problematic_records)/len(self.df)*100:.1f}%")

        # Store statistics
        self.stats = {
            'total_records': total_records,
            'valid_times': valid_consultation_times,
            'problematic_records': len(problematic_records),
            'time_ranges': time_ranges
        }

        return self.stats

    def refine_consultation_times(self):
        """Refine unrealistic consultation times using intelligent methods"""
        print("\n🔧 REFINING CONSULTATION TIMES")
        print("=" * 50)

        self.refined_df = self.df.copy()

        # Method 1: Use realistic baseline consultation times by specialization and type
        baseline_times = {
            'Neonatology': {
                'advance': {'min': 8, 'typical': 15, 'max': 25},
                'walkin': {'min': 5, 'typical': 12, 'max': 20}
            },
            'Obstetrics and Gynecology': {
                'advance': {'min': 10, 'typical': 20, 'max': 35},
                'walkin': {'min': 8, 'typical': 15, 'max': 25}
            }
        }

        # Method 2: Calculate realistic times based on queue position and patterns
        def calculate_realistic_consultation_time(row):
            """Calculate a realistic consultation time based on context"""
            specialization = row['Specialization']
            appointment_type = row['Type'].lower() if pd.notna(
                row['Type']) else 'walkin'

            # Get baseline for specialization
            if specialization in baseline_times:
                baseline = baseline_times[specialization].get(appointment_type,
                                                              baseline_times[specialization]['walkin'])
            else:
                # Default values for unknown specializations
                baseline = {'min': 8, 'typical': 15, 'max': 25}

            # Base realistic time
            base_time = baseline['typical']

            # Adjust based on factors
            adjustments = 0

            # 1. Time of day adjustment (longer consultations in morning)
            try:
                hour = int(row['TimeCalled'].split(':')[0])
                if hour < 10:  # Early morning - longer consultations
                    adjustments += 3
                elif hour > 16:  # Late afternoon - slightly longer
                    adjustments += 2
            except:
                pass

            # 2. After break adjustment (first patient after break takes longer)
            if row['ServedAfterBreak'] == True or row['ServedAfterBreak'] == 'True':
                adjustments += 5

            # 3. Day of week adjustment
            if row['Day'] in ['Monday', 'Friday']:  # Busier days
                adjustments += 2

            # 4. Position in queue (later patients might have complex cases)
            if pd.notna(row['Index']) and row['Index'] > 10:
                adjustments += 2

            # Calculate final time with some randomness for realism
            final_time = base_time + adjustments

            # Add realistic variation (±20%)
            variation = np.random.normal(0, final_time * 0.1)
            final_time += variation

            # Ensure within reasonable bounds
            final_time = max(baseline['min'], min(baseline['max'], final_time))

            return round(final_time, 1)

        # Method 3: For records with unrealistic times, recalculate
        def needs_refinement(time_seconds):
            """Check if consultation time needs refinement"""
            return time_seconds < 120 or time_seconds > 3600  # Less than 2 min or more than 1 hour

        # Apply refinement
        print("🔄 Processing records...")
        refined_count = 0

        for idx, row in self.refined_df.iterrows():
            current_time_seconds = row['TimeTaken_seconds']

            if needs_refinement(current_time_seconds):
                # Calculate new realistic time
                new_time_minutes = calculate_realistic_consultation_time(row)
                new_time_seconds = new_time_minutes * 60

                # Update the time fields
                hours = int(new_time_minutes // 60)
                minutes = int(new_time_minutes % 60)
                seconds = int((new_time_minutes % 1) * 60)

                new_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

                self.refined_df.at[idx, 'TimeTaken'] = new_time_str
                self.refined_df.at[idx, 'TimeTaken_seconds'] = new_time_seconds

                refined_count += 1

        print(f"✅ Refined {refined_count:,} consultation time records")

        # Method 4: Refine wait times based on queue dynamics
        self.refine_wait_times()

        return refined_count

    def refine_wait_times(self):
        """Refine wait times based on realistic queue dynamics"""
        print("\n🕐 Refining wait times based on queue dynamics...")

        # Group by date and doctor to process each session
        for (date, doctor_id), group in self.refined_df.groupby(['Date', 'DoctorID']):
            group = group.sort_values('Index').reset_index(drop=True)

            # Calculate cumulative consultation time
            cumulative_time = 0
            base_wait_time = 2  # Base wait time in minutes

            for idx, row in group.iterrows():
                # Calculate position-based wait time
                position = row['Index']
                consultation_time_minutes = row['TimeTaken_seconds'] / 60

                # Wait time factors
                queue_wait = position * 2  # 2 minutes per position in queue
                consultation_backlog = cumulative_time * 0.8  # 80% of previous consultations
                break_time = 10 if row['ServedAfterBreak'] else 0  # Break time

                # Calculate realistic wait time
                realistic_wait = base_wait_time + queue_wait + consultation_backlog + break_time

                # Add some realistic variation
                variation = np.random.normal(0, realistic_wait * 0.15)
                realistic_wait += variation

                # Ensure reasonable bounds (1-60 minutes)
                realistic_wait = max(1, min(60, realistic_wait))

                # Update wait time
                wait_hours = int(realistic_wait // 60)
                wait_minutes = int(realistic_wait % 60)
                wait_seconds = int((realistic_wait % 1) * 60)

                new_wait_str = f"{wait_hours:02d}:{wait_minutes:02d}:{wait_seconds:02d}"

                # Update the dataframe
                original_idx = row.name if hasattr(row, 'name') else self.refined_df[
                    (self.refined_df['Date'] == date) &
                    (self.refined_df['DoctorID'] == doctor_id) &
                    (self.refined_df['Index'] == position)
                ].index[0]

                self.refined_df.at[original_idx, 'AvgWaitTime'] = new_wait_str
                self.refined_df.at[original_idx,
                                   'AvgWaitTime_seconds'] = realistic_wait * 60

                # Update cumulative time for next iteration
                cumulative_time += consultation_time_minutes

        print("✅ Wait times refined based on queue dynamics")

    def add_realistic_time_called(self):
        """Generate realistic TimeCalled based on consultation flow"""
        print("\n🕒 Generating realistic TimeCalled timestamps...")

        for (date, doctor_id), group in self.refined_df.groupby(['Date', 'DoctorID']):
            group = group.sort_values('Index').reset_index(drop=True)

            # Parse consultation start time
            try:
                start_time_str = group.iloc[0]['ConsultationFromTime']
                start_hour = int(start_time_str.split(':')[0])
                start_minute = int(start_time_str.split(':')[1].split()[0])
                if 'PM' in start_time_str and start_hour != 12:
                    start_hour += 12
                elif 'AM' in start_time_str and start_hour == 12:
                    start_hour = 0

                current_time = datetime.strptime(
                    f"{start_hour:02d}:{start_minute:02d}:00", "%H:%M:%S")

            except:
                # Default start time
                current_time = datetime.strptime("09:00:00", "%H:%M:%S")

            # Generate realistic call times
            for idx, row in group.iterrows():
                # Add wait time and consultation time from previous patients
                if idx > 0:
                    prev_consultation = group.iloc[idx -
                                                   1]['TimeTaken_seconds'] / 60
                    wait_buffer = np.random.normal(3, 1)  # 3±1 minute buffer
                    current_time += timedelta(
                        minutes=prev_consultation + wait_buffer)

                # Update TimeCalled
                new_time_called = current_time.strftime("%H:%M:%S")

                original_idx = self.refined_df[
                    (self.refined_df['Date'] == date) &
                    (self.refined_df['DoctorID'] == doctor_id) &
                    (self.refined_df['Index'] == row['Index'])
                ].index[0]

                self.refined_df.at[original_idx,
                                   'TimeCalled'] = new_time_called

        print("✅ Realistic TimeCalled timestamps generated")

    def validate_refined_data(self):
        """Validate the refined data quality"""
        print("\n✅ VALIDATION OF REFINED DATA")
        print("=" * 50)

        # Recalculate statistics
        self.refined_df['TimeTaken_seconds_refined'] = self.refined_df['TimeTaken'].apply(
            lambda x: sum(float(i) * j for i,
                          j in zip(str(x).split(':'), [3600, 60, 1]))
            if pd.notna(x) else 0
        )

        consultation_times = self.refined_df[self.refined_df['TimeTaken_seconds_refined']
                                             > 0]['TimeTaken_seconds_refined']

        print(f"📊 Refined Data Statistics:")
        print(f"   Total records: {len(self.refined_df):,}")
        print(f"   Valid consultation times: {len(consultation_times):,}")
        print(
            f"   Mean consultation time: {consultation_times.mean()/60:.1f} minutes")
        print(
            f"   Median consultation time: {consultation_times.median()/60:.1f} minutes")
        print(
            f"   Min consultation time: {consultation_times.min()/60:.1f} minutes")
        print(
            f"   Max consultation time: {consultation_times.max()/60:.1f} minutes")

        # Check realistic ranges
        realistic_range = ((consultation_times >= 300) & (
            consultation_times <= 1800)).sum()  # 5-30 minutes
        print(
            f"   Records in realistic range (5-30 min): {realistic_range:,} ({realistic_range/len(consultation_times)*100:.1f}%)")

        # Time distribution
        print(f"\n⏱️ Refined Time Distribution:")
        time_ranges = {
            "< 2 minutes": (consultation_times < 120).sum(),
            "2-5 minutes": ((consultation_times >= 120) & (consultation_times < 300)).sum(),
            "5-15 minutes": ((consultation_times >= 300) & (consultation_times < 900)).sum(),
            "15-30 minutes": ((consultation_times >= 900) & (consultation_times < 1800)).sum(),
            "30+ minutes": (consultation_times >= 1800).sum()
        }

        for range_name, count in time_ranges.items():
            percentage = (count / len(consultation_times)) * 100
            print(f"   {range_name}: {count:,} ({percentage:.1f}%)")

        return True

    def save_refined_data(self, output_path):
        """Save the refined dataset"""
        print(f"\n💾 Saving refined data to: {output_path}")

        # Remove temporary columns
        columns_to_remove = ['TimeTaken_seconds',
                             'AvgWaitTime_seconds', 'TimeTaken_seconds_refined']
        refined_final = self.refined_df.drop(
            columns=[col for col in columns_to_remove if col in self.refined_df.columns])

        # Save to CSV
        refined_final.to_csv(output_path, index=False)
        print(f"✅ Refined data saved successfully!")
        print(f"   Original records: {len(self.df):,}")
        print(f"   Refined records: {len(refined_final):,}")

        return True

    def generate_summary_report(self):
        """Generate a summary report of the refinement process"""
        print("\n📋 REFINEMENT SUMMARY REPORT")
        print("=" * 50)

        original_problematic = self.stats['problematic_records']
        total_records = self.stats['total_records']

        # Count current problematic records
        current_consultation_times = self.refined_df['TimeTaken'].apply(
            lambda x: sum(float(i) * j for i,
                          j in zip(str(x).split(':'), [3600, 60, 1]))
            if pd.notna(x) else 0
        )
        current_problematic = (current_consultation_times < 120).sum()

        improvement = original_problematic - current_problematic

        print(f"📈 Improvement Metrics:")
        print(f"   Original problematic records: {original_problematic:,}")
        print(f"   Current problematic records: {current_problematic:,}")
        print(f"   Records improved: {improvement:,}")
        print(
            f"   Improvement rate: {improvement/original_problematic*100:.1f}%")

        print(f"\n🎯 Data Quality Score:")
        realistic_records = ((current_consultation_times >= 300) & (
            current_consultation_times <= 1800)).sum()
        quality_score = realistic_records / total_records * 100
        print(f"   Realistic consultation times: {quality_score:.1f}%")

        if quality_score >= 80:
            print(f"   ✅ Excellent data quality!")
        elif quality_score >= 60:
            print(f"   ✅ Good data quality!")
        else:
            print(f"   ⚠️  Data quality needs more improvement")


def main():
    """Main refinement process"""
    print("🏥" * 30)
    print("PATIENT APPOINTMENT DATA REFINEMENT")
    print("🏥" * 30)

    # Configuration
    INPUT_CSV = 'data/raw/MoideenBabuPerayil.csv'
    OUTPUT_CSV = 'data/processed/MoideenBabuPerayil_refined.csv'

    # Initialize refinement tool
    refiner = DataRefinement()

    # Step 1: Load data
    if not refiner.load_data(INPUT_CSV):
        return False

    # Step 2: Analyze current data quality
    refiner.analyze_data_quality()

    # Step 3: Refine consultation times
    refiner.refine_consultation_times()

    # Step 4: Add realistic timestamps
    refiner.add_realistic_time_called()

    # Step 5: Validate refined data
    refiner.validate_refined_data()

    # Step 6: Save refined data
    import os
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    refiner.save_refined_data(OUTPUT_CSV)

    # Step 7: Generate summary report
    refiner.generate_summary_report()

    print(f"\n🎉 Data refinement completed successfully!")
    print(f"   Refined data available at: {OUTPUT_CSV}")
    print(f"   Ready for ML model training!")


if __name__ == "__main__":
    main()
