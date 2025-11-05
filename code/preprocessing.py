import pandas as pd
import icd # Use the 'icd' library
import numpy as np # For NaN handling
import os # Import os for directory creation

# --- Configuration ---
INPUT_CSV_PATH = 'data/original_data.csv' # Assuming data is in a 'data' subdirectory
OUTPUT_CSV_PATH = 'data/preprocessed_data.csv' # Output to 'data' subdirectory
DIAGNOSIS_COLUMNS = [f'I10_DX{i}' for i in range(1, 35)] # ICD code columns
READMISSION_WINDOW = 30 # Define the readmission window in days
PATIENT_ID_COLUMN='N'

# --- Load Data ---
print(f"Loading data from {INPUT_CSV_PATH}...")
try:
    df = pd.read_csv(INPUT_CSV_PATH, low_memory=False)
    df[PATIENT_ID_COLUMN] = df[PATIENT_ID_COLUMN].astype(str)



except FileNotFoundError:
    print(f"Error: Input file not found at {INPUT_CSV_PATH}")
    exit()
except Exception as e:
    print(f"Error loading or initially cleaning data: {e}")
    exit()

# --- Clean ICD Codes (Ensure string, handle NaN/None for full codes) ---
print("Cleaning ICD codes (using full codes)...")
for col in DIAGNOSIS_COLUMNS:
    if col in df.columns:
        df[col] = df[col].astype(str).replace('nan', '').fillna('')
    else:
        df[col] = '' # Add missing DX columns if they don't exist
print("ICD code cleaning complete.")

# --- Calculate Comorbidity Flags using 'icd' library ---
print(f"Calculating comorbidity flags using 'icd' library (charlson10 mapping) on {len(df)} records...")
try:
    # Pass df, id_col_name, and icd_cols_list as positional arguments
    comorbidity_flags_df = icd.icd_to_comorbidities(
        df,                         # 1st positional argument: DataFrame
        PATIENT_ID_COLUMN,
        DIAGNOSIS_COLUMNS,          # 3rd positional argument: List of ICD column names
        mapping='charlson10'        # Keyword argument for mapping
    )
    print("Comorbidity flag calculation complete.")

    # --- Calculate Weighted Charlson Score ---
    print("Calculating weighted CCI score from flags...")
    charlson_weights = {
        'myocardial_infarction': 1, 'congestive_heart_failure': 1, 'periphral_vascular_disease': 1,
        'cerebrovascular_disease': 1, 'dementia': 1, 'chronic_pulmonary_disease': 1,
        'connective_tissue_disease_rheumatic_disease': 1, 'peptic_ulcer_disease': 1,
        'mild_liver_disease': 1, 'diabetes_wo_complications': 1, 'paraplegia_and_hemiplegia': 2,
        'renal_disease': 2, 'diabetes_w_complications': 2, 'cancer': 2,
        'moderate_or_sever_liver_disease': 3, 'metastitic_carcinoma': 6, 'aids_hiv': 6
    }
    comorbidity_flags_df['CCI_Score'] = 0
    for condition, weight in charlson_weights.items():
        if condition in comorbidity_flags_df.columns:
            comorbidity_flags_df['CCI_Score'] += comorbidity_flags_df[condition].fillna(False).astype(int) * weight

    # Hierarchy rules for CCI
    if 'diabetes_w_complications' in comorbidity_flags_df.columns and 'diabetes_wo_complications' in comorbidity_flags_df.columns:
         mask = comorbidity_flags_df['diabetes_w_complications'].fillna(False) & comorbidity_flags_df['diabetes_wo_complications'].fillna(False)
         comorbidity_flags_df.loc[mask, 'CCI_Score'] -= charlson_weights.get('diabetes_wo_complications', 1)
    if 'moderate_or_sever_liver_disease' in comorbidity_flags_df.columns and 'mild_liver_disease' in comorbidity_flags_df.columns:
        mask = comorbidity_flags_df['moderate_or_sever_liver_disease'].fillna(False) & comorbidity_flags_df['mild_liver_disease'].fillna(False)
        comorbidity_flags_df.loc[mask, 'CCI_Score'] -= charlson_weights.get('mild_liver_disease', 1)
    if 'metastitic_carcinoma' in comorbidity_flags_df.columns and 'cancer' in comorbidity_flags_df.columns:
        mask = comorbidity_flags_df['metastitic_carcinoma'].fillna(False) & comorbidity_flags_df['cancer'].fillna(False)
        comorbidity_flags_df.loc[mask, 'CCI_Score'] -= charlson_weights.get('cancer', 2)

    # --- CORRECTED MERGE PREPARATION ---
    # Select the necessary columns. PATIENT_ID_COLUMN is a regular column here,
    # but the index of cci_data_to_merge will still be named PATIENT_ID_COLUMN.
    cci_data_to_merge = comorbidity_flags_df[[PATIENT_ID_COLUMN, 'CCI_Score']].copy()
    
    # Reset the index of cci_data_to_merge to avoid ambiguity.
    # This makes its index a default RangeIndex and PATIENT_ID_COLUMN is only a regular column.
    cci_data_to_merge = cci_data_to_merge.reset_index(drop=True)
    # -----------------------------------

    # Ensure the PATIENT_ID_COLUMN in this new DataFrame is string type for merging
    cci_data_to_merge[PATIENT_ID_COLUMN] = cci_data_to_merge[PATIENT_ID_COLUMN].astype(str)

    # Merge CCI_Score back to the main DataFrame 'df'
    # df already has PATIENT_ID_COLUMN as string type from the loading step.
    df = pd.merge(df, cci_data_to_merge, on=PATIENT_ID_COLUMN, how='left')
    df['CCI_Score'] = df['CCI_Score'].fillna(0).astype(int)
    print("Weighted CCI score calculation and merge complete.")

except Exception as e:
    print(f"Error during comorbidity calculation: {e}")
    df['CCI_Score'] = 0 # Default CCI if calculation fails


print(f"\nSaving final preprocessed data to {OUTPUT_CSV_PATH}...")
try:
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print("File saved successfully.")
except Exception as e:
    print(f"Error saving file: {e}")
