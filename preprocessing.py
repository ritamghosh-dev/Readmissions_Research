import pandas as pd
import icd # Use the 'icd' library
import numpy as np # For NaN handling

# --- Configuration ---
INPUT_CSV_PATH = 'data/original_data.csv'
OUTPUT_CSV_PATH = 'data/preprocessed_data.csv' # Final output file name
PATIENT_ID_COLUMN = 'N' # Encounter ID
VISIT_LINK_COLUMN = 'VisitLink' # Patient identifier linking visits
READMISSION_COUNT_COLUMN = 'Readmission_Count' # Column indicating visit sequence
READMISSION_DAYS_COLUMN = 'Readmission_Days' # Column with days to readmission
TARGET_COLUMN = 'Readmitted' # New binary target column
DIAGNOSIS_COLUMNS = [f'I10_DX{i}' for i in range(1, 35)] # ICD code columns
READMISSION_WINDOW = 30 # Define the readmission window in days

# --- Load Data ---
print(f"Loading data from {INPUT_CSV_PATH}...")
try:
    # Added low_memory=False based on previous warning
    df_original = pd.read_csv(INPUT_CSV_PATH, low_memory=False)
    df_original[PATIENT_ID_COLUMN] = df_original[PATIENT_ID_COLUMN].astype(str)
    df_original[VISIT_LINK_COLUMN] = df_original[VISIT_LINK_COLUMN].astype(str) # Ensure VisitLink is string

    # Convert Readmission_Count and Readmission_Days to numeric, coercing errors to NaN
    df_original[READMISSION_COUNT_COLUMN] = pd.to_numeric(df_original[READMISSION_COUNT_COLUMN], errors='coerce')
    df_original[READMISSION_DAYS_COLUMN] = pd.to_numeric(df_original[READMISSION_DAYS_COLUMN], errors='coerce')
    print(f"Loaded {len(df_original)} records.")

except FileNotFoundError:
    print(f"Error: Input file not found at {INPUT_CSV_PATH}")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- Filter Data Based on Readmission_Count ---
print(f"Filtering records to keep only {READMISSION_COUNT_COLUMN} <= 2...")
# Drop rows where Readmission_Count is NaN or > 2
df_filtered = df_original.dropna(subset=[READMISSION_COUNT_COLUMN])
df_filtered = df_filtered[df_filtered[READMISSION_COUNT_COLUMN] <= 2].copy()
df_filtered[READMISSION_COUNT_COLUMN] = df_filtered[READMISSION_COUNT_COLUMN].astype(int) # Convert to int after filtering NaNs
print(f"Kept {len(df_filtered)} records after filtering.")

# --- Clean ICD Codes (Ensure string, handle NaN/None for full codes) ---
print("Cleaning ICD codes (using full codes)...")
for col in DIAGNOSIS_COLUMNS:
    if col in df_filtered.columns:
         # Convert to string, replace 'nan' string representations, fill actual NaN/None with empty string
        df_filtered[col] = df_filtered[col].astype(str).replace('nan', '').fillna('')
    else:
        # Add missing DX columns if they don't exist in the CSV, initialized as empty strings
        df_filtered[col] = ''
print("ICD code cleaning complete.")

# --- Calculate Comorbidity Flags using 'icd' library ---
# Calculate CCI *before* dropping rows to use all available diagnosis data
print(f"Calculating comorbidity flags using 'icd' library (charlson10 mapping) on {len(df_filtered)} records...")
try:
    comorbidity_flags = icd.icd_to_comorbidities(
        df_filtered,           # Use the filtered dataframe
        PATIENT_ID_COLUMN,     # Encounter ID (used temporarily by the library)
        DIAGNOSIS_COLUMNS,
        mapping='charlson10'
    )
    print("Comorbidity flag calculation complete.")

    # --- Calculate Weighted Charlson Score ---
    print("Calculating weighted CCI score from flags...")
    # Use the corrected weight keys matching the library's output columns
    charlson_weights = {
        'myocardial_infarction': 1, 'congestive_heart_failure': 1, 'periphral_vascular_disease': 1,
        'cerebrovascular_disease': 1, 'dementia': 1, 'chronic_pulmonary_disease': 1,
        'connective_tissue_disease_rheumatic_disease': 1, 'peptic_ulcer_disease': 1,
        'mild_liver_disease': 1, 'diabetes_wo_complications': 1, 'paraplegia_and_hemiplegia': 2,
        'renal_disease': 2, 'diabetes_w_complications': 2, 'cancer': 2,
        'moderate_or_sever_liver_disease': 3, 'metastitic_carcinoma': 6, 'aids_hiv': 6
    }
    comorbidity_flags['CCI_Score'] = 0
    for condition, weight in charlson_weights.items():
        if condition in comorbidity_flags.columns:
            comorbidity_flags['CCI_Score'] += comorbidity_flags[condition].fillna(False).astype(int) * weight

    # Apply Hierarchy Rules
    if 'diabetes_w_complications' in comorbidity_flags.columns and 'diabetes_wo_complications' in comorbidity_flags.columns:
         mask = comorbidity_flags['diabetes_w_complications'].fillna(False) & comorbidity_flags['diabetes_wo_complications'].fillna(False)
         comorbidity_flags.loc[mask, 'CCI_Score'] -= charlson_weights.get('diabetes_wo_complications', 1)
    if 'moderate_or_sever_liver_disease' in comorbidity_flags.columns and 'mild_liver_disease' in comorbidity_flags.columns:
        mask = comorbidity_flags['moderate_or_sever_liver_disease'].fillna(False) & comorbidity_flags['mild_liver_disease'].fillna(False)
        comorbidity_flags.loc[mask, 'CCI_Score'] -= charlson_weights.get('mild_liver_disease', 1)
    if 'metastitic_carcinoma' in comorbidity_flags.columns and 'cancer' in comorbidity_flags.columns:
        mask = comorbidity_flags['metastitic_carcinoma'].fillna(False) & comorbidity_flags['cancer'].fillna(False)
        comorbidity_flags.loc[mask, 'CCI_Score'] -= charlson_weights.get('cancer', 2)

    # Keep only the ID and the final score for merging
    comorbidity_flags[PATIENT_ID_COLUMN] = comorbidity_flags[PATIENT_ID_COLUMN].astype(str)
    cci_scores_final = comorbidity_flags[[PATIENT_ID_COLUMN, 'CCI_Score']].drop_duplicates(subset=[PATIENT_ID_COLUMN])
    print("Weighted CCI score calculation complete.")

    # ***** DEBUG: Show value counts of calculated CCI scores *****
    print("\nDEBUG: Value counts for calculated CCI_Score (before merge):")
    if not cci_scores_final.empty:
        print(cci_scores_final['CCI_Score'].value_counts().sort_index().to_string())
    else:
        print("cci_scores_final DataFrame is empty.")
    # ***** END DEBUG *****

except Exception as e:
    print(f"Error during comorbidity calculation with 'icd' library: {e}")
    cci_scores_final = pd.DataFrame(columns=[PATIENT_ID_COLUMN, 'CCI_Score']) # Create empty if error

# --- Merge CCI Score back ---
print("Merging CCI scores back into the filtered dataframe...")
df_filtered[PATIENT_ID_COLUMN] = df_filtered[PATIENT_ID_COLUMN].astype(str) # Ensure type match
if not cci_scores_final.empty:
    cci_scores_final = cci_scores_final.reset_index(drop=True) # Reset index before merge
    df_processed = pd.merge(df_filtered, cci_scores_final, on=PATIENT_ID_COLUMN, how='left')
else:
    df_processed = df_filtered.copy() # Start with filtered data
    df_processed['CCI_Score'] = pd.NA # Add column even if calculation failed

# Fill NaN scores with 0 and ensure integer type
df_processed['CCI_Score'] = df_processed['CCI_Score'].fillna(0).astype(int)
print("CCI Score merge complete.")


# --- Apply **NEW** Readmission Logic ---
print("Applying updated readmission logic...")

# Separate index visits (count=1) and potential readmissions (count=2)
df_index_visits = df_processed[df_processed[READMISSION_COUNT_COLUMN] == 1].copy()
df_readmission_visits = df_processed[df_processed[READMISSION_COUNT_COLUMN] == 2].copy()

print(f"Found {len(df_index_visits)} index visits and {len(df_readmission_visits)} potential readmission visits.")

# Merge readmission info onto index visits based on VisitLink
# Keep only essential info from readmission visits to avoid duplicate columns
df_merged_visits = pd.merge(
    df_index_visits,
    df_readmission_visits[[VISIT_LINK_COLUMN, READMISSION_DAYS_COLUMN]],
    on=VISIT_LINK_COLUMN,
    how='left', # Keep all index visits
    suffixes=('', '_readm') # Suffix for columns from the readmission visit df
)

# Initialize the target column
df_merged_visits[TARGET_COLUMN] = 0 # Default to 0 (not readmitted)
# Initialize the final Readmission_Days column (will overwrite later if readmitted)
# Use the original Readmission_Days from the index visit as default (likely NaN)
df_merged_visits[f"{READMISSION_DAYS_COLUMN}_final"] = df_merged_visits[READMISSION_DAYS_COLUMN]


# Identify valid readmissions (Readmission_Days_readm <= WINDOW and not NaN)
valid_readmission_mask = (df_merged_visits[f"{READMISSION_DAYS_COLUMN}_readm"].notna()) & \
                         (df_merged_visits[f"{READMISSION_DAYS_COLUMN}_readm"] <= READMISSION_WINDOW)

# Set Readmitted = 1 for valid readmissions
df_merged_visits.loc[valid_readmission_mask, TARGET_COLUMN] = 1
print(f"Marked {valid_readmission_mask.sum()} index admissions as Readmitted=1 (within {READMISSION_WINDOW} days).")

# Copy Readmission_Days from the readmission visit for valid readmissions
df_merged_visits.loc[valid_readmission_mask, f"{READMISSION_DAYS_COLUMN}_final"] = df_merged_visits.loc[valid_readmission_mask, f"{READMISSION_DAYS_COLUMN}_readm"]
print(f"Copied {READMISSION_DAYS_COLUMN} for {valid_readmission_mask.sum()} valid readmissions.")

# Final DataFrame contains only the (potentially modified) index visit rows
df_final = df_merged_visits.copy()

# --- Final Cleanup ---
# Drop temporary/original columns
cols_to_drop = [READMISSION_COUNT_COLUMN, READMISSION_DAYS_COLUMN, f"{READMISSION_DAYS_COLUMN}_readm"]
existing_cols_to_drop = [col for col in cols_to_drop if col in df_final.columns]
if existing_cols_to_drop:
    df_final = df_final.drop(columns=existing_cols_to_drop)
    print(f"Removed columns: {existing_cols_to_drop}")

# Rename the final days column
df_final = df_final.rename(columns={f"{READMISSION_DAYS_COLUMN}_final": READMISSION_DAYS_COLUMN})
print(f"Final '{READMISSION_DAYS_COLUMN}' column created.")

# Ensure target column is integer
df_final[TARGET_COLUMN] = df_final[TARGET_COLUMN].astype(int)

print(f"Final record count after applying readmission logic: {len(df_final)}")

# --- Display Results & Save ---
print("\nPreview of final preprocessed data:")
# Display relevant columns including the new target and final days
preview_cols = [PATIENT_ID_COLUMN, VISIT_LINK_COLUMN, 'CCI_Score', TARGET_COLUMN, READMISSION_DAYS_COLUMN] + DIAGNOSIS_COLUMNS[:1]
print(df_final[preview_cols].head().to_string())

# Display value counts for the new target variable
print(f"\nValue counts for final '{TARGET_COLUMN}' column:")
print(df_final[TARGET_COLUMN].value_counts().to_string())

print(f"\nSaving final preprocessed data to {OUTPUT_CSV_PATH}...")
try:
    df_final.to_csv(OUTPUT_CSV_PATH, index=False)
    print("File saved successfully.")
except Exception as e:
    print(f"Error saving file: {e}")
# 