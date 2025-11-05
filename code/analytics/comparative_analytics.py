import pandas as pd
import numpy as np
import os

# --- Configuration ---
# Path to your preprocessed data file
PREPROCESSED_CSV_PATH = 'data/preprocessed_data.csv'
# Column to group by
TARGET_COLUMN = 'Readmitted30'
# Directory to save the output
OUTPUT_DIR = 'results/analytics'

def run_comparative_analysis():
    """
    Loads the preprocessed dataset and computes a comparative statistical summary
    between the 'Readmitted' and 'Not Readmitted' groups, saving the output.
    """
    print("--- Starting Comparative Analysis ---")
    
    # --- 1. Load Data ---
    print(f"Loading data from {PREPROCESSED_CSV_PATH}...")
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df = pd.read_csv(PREPROCESSED_CSV_PATH, low_memory=False)
        print(f"Loaded {len(df)} records successfully.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {PREPROCESSED_CSV_PATH}")
        print("Please ensure you have run 'preprocessing.py' to generate this file.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Check if the target column exists
    if TARGET_COLUMN not in df.columns:
        print(f"Error: Target column '{TARGET_COLUMN}' not found in the dataset.")
        return

    # --- 2. Feature Engineering for Analysis ---
    # Calculate the number of ICD and CPT codes for each row
    print("Calculating number of diagnosis and procedure codes per encounter...")
    
    # Define diagnosis and procedure columns
    icd_cols = [f'I10_DX{i}' for i in range(1, 35)]
    cpt_cols = [f'CPT{i}' for i in range(1, 101)]

    # Function to count valid codes (non-null and not empty/placeholder strings)
    def count_valid_codes(row, columns):
        valid_codes = 0
        for col in columns:
            if col in row and pd.notna(row[col]) and str(row[col]).strip() not in ['', 'nan', 'missing']:
                valid_codes += 1
        return valid_codes

    df['num_icd_codes'] = df.apply(lambda row: count_valid_codes(row, icd_cols), axis=1)
    df['num_cpt_codes'] = df.apply(lambda row: count_valid_codes(row, cpt_cols), axis=1)
    
    print("Finished calculating code counts.")
    
    # --- 3. Group by Readmission Status and Aggregate ---
    print(f"Grouping data by '{TARGET_COLUMN}' and calculating mean values...")
    
    # Define columns for aggregation
    agg_cols = {
        'AGE': 'mean',
        'LOS': 'mean',
        'CCI_Score': 'mean',
        'num_icd_codes': 'mean',
        'num_cpt_codes': 'mean'
    }
    
    # Perform the aggregation
    comparative_stats = df.groupby(TARGET_COLUMN).agg(agg_cols)
    
    # Improve formatting
    comparative_stats = comparative_stats.round(2)
    comparative_stats = comparative_stats.rename(index={0: 'Not Readmitted', 1: 'Readmitted'})
    comparative_stats = comparative_stats.rename(columns={
        'AGE': 'Mean Age',
        'LOS': 'Mean Length of Stay',
        'CCI_Score': 'Mean CCI Score',
        'num_icd_codes': 'Mean # of ICD Codes',
        'num_cpt_codes': 'Mean # of CPT Codes'
    })

    # --- 4. Display the Comparative Table ---
    print("\n" + "="*60)
    print("      Comparative Analytics: Readmitted vs. Not Readmitted")
    print("="*60)
    print(comparative_stats.to_string())
    print("="*60 + "\n")

    # --- ADDED: 5. Save the Output to a CSV file ---
    output_csv_path = os.path.join(OUTPUT_DIR, 'comparative_analytics_summary.csv')
    print(f"Saving comparative analytics table to {output_csv_path}...")
    try:
        # The index=True is important here to save the 'Readmitted'/'Not Readmitted' row labels
        comparative_stats.to_csv(output_csv_path, index=True)
        print("Successfully saved summary to CSV.")
    except Exception as e:
        print(f"Error saving summary to CSV: {e}")
    # --- END ADDED SECTION ---


if __name__ == "__main__":
    run_comparative_analysis()
