

import pandas as pd
import re, os
import numpy as np # Import numpy for checking NaN

# Global dictionaries for mappings
diagnosis_mapping = {}
procedure_mapping = {}


# --- Define column names as constants ---
TARGET_COLUMN = 'Readmitted'

# --- END CONSTANTS ---

def load_procedure_mapping(csv_path):
    """
    Reads a CSV file for CPT mapping. Expects at least two columns: 'Code' and 'Description'.
    Populates the global dictionary: procedure_mapping.
    """
    global procedure_mapping
    try:
        df = pd.read_csv(csv_path)
        procedure_mapping = pd.Series(df.Description.values, index=df.Code.astype(str).str.strip()).to_dict()
        print("Loaded", len(procedure_mapping), "CPT mappings.")
    except FileNotFoundError:
        print(f"Warning: CPT mapping file not found at {csv_path}. Procedure lookup will not work.")
        procedure_mapping = {}
    except Exception as e:
        print(f"Error loading CPT mapping from {csv_path}: {e}")
        procedure_mapping = {}







# ICD-10 Block definitions (condensed)
icd10_blocks = [
    ("A00", "B99", "Certain infectious and parasitic diseases"),("C00", "D49", "Neoplasms"),
    ("D50", "D89", "Diseases of the blood and blood-forming organs"),("E00", "E89", "Endocrine, nutritional and metabolic diseases"),
    ("F01", "F99", "Mental, Behavioral and Neurodevelopmental disorders"),("G00", "G99", "Diseases of the nervous system"),
    ("H00", "H59", "Diseases of the eye and adnexa"),("H60", "H95", "Diseases of the ear and mastoid process"),
    ("I00", "I99", "Diseases of the circulatory system"),("J00", "J99", "Diseases of the respiratory system"),
    ("K00", "K95", "Diseases of the digestive system"),("L00", "L99", "Diseases of the skin and subcutaneous tissue"),
    ("M00", "M99", "Diseases of the musculoskeletal system and connective tissue"),("N00", "N99", "Diseases of the genitourinary system"),
    ("O00", "O9A", "Pregnancy, childbirth and the puerperium"),("P00", "P96", "Certain conditions originating in the perinatal period"),
    ("Q00", "Q99", "Congenital malformations, deformations and chromosomal abnormalities"),("R00", "R99", "Symptoms, signs and abnormal clinical and laboratory findings"),
    ("S00", "T88", "Injury, poisoning and certain other consequences of external causes"),("V00", "Y99", "External causes of morbidity"),
    ("Z00", "Z99", "Factors influencing health status and contact with health services")
]

def classify_icd_3_chars(icd_code: str) -> str:
    """ Classify ICD code using first 3 characters. """
    if not icd_code or pd.isna(icd_code): return None
    icd3 = str(icd_code).strip()[:3].upper()
    if len(icd3) < 3: return "Incomplete Code"
    for start, end, description in icd10_blocks:
        if start <= icd3 <= end: return description
    return "Unclassified Code"

def lookup_diagnosis_description(icd_code: str) -> str:
    """ Get high-level classification for ICD code. """
    return classify_icd_3_chars(icd_code)

def lookup_procedure_description(code: str) -> str:
    """ Look up CPT procedure code description. """
    if not code or pd.isna(code): return None
    code_str = str(code).strip()
    return procedure_mapping.get(code_str, None)

def get_ordinal_suffix(n):
    """ Helper function to get ordinal suffix for numbers (1st, 2nd, 3rd, 4th). """
    if 10 <= n % 100 <= 20:
        return 'th'
    else:
        return {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')

def convert_row_to_narrative(row):
    """ Generates narrative summary, now including admission sequence context. """
    narrative_parts = []
    first_part="This patient was admitted to the hospital"
    primary_dx_code = row.get('I10_DX1')
    if pd.notna(primary_dx_code) and str(primary_dx_code).strip():
        primary_dx_desc = lookup_diagnosis_description(primary_dx_code)
        if primary_dx_desc and primary_dx_desc not in ["Unclassified Code", "Incomplete Code"]: first_part += f" primarily for {primary_dx_desc}"
        else: first_part += f" with primary diagnosis code {primary_dx_code}"
    narrative_parts.append(first_part + ".")


    # --- 2. Additional Diagnoses ---
    additional_dx = []
    diag_cols = [f'I10_DX{i}' for i in range(2, 36)]
    for dx_col in diag_cols:
        if dx_col in row:
            dx_code = row[dx_col]
            if pd.notna(dx_code) and str(dx_code).strip():
                dx_desc = lookup_diagnosis_description(dx_code)
                if dx_desc and dx_desc not in ["Unclassified Code", "Incomplete Code"] and dx_desc not in additional_dx: additional_dx.append(dx_desc)
    if additional_dx: narrative_parts.append(f"Co-occurring conditions included {', '.join(additional_dx)}.")

    # --- 3. Charlson Comorbidity Index ---
    cci_score = row.get('CCI_Score', None)
    if pd.notna(cci_score):
         try: narrative_parts.append(f"The calculated Charlson Comorbidity Index score for this encounter was {int(cci_score)}.")
         except (ValueError, TypeError): pass

    # --- 4. Procedures ---
    procedures = []
    cpt_cols = [f'CPT{i}' for i in range(1, 11)]
    for cpt_col in cpt_cols:
         if cpt_col in row:
            cpt_code = row[cpt_col]
            if pd.notna(cpt_code) and str(cpt_code).strip():
                proc_desc = lookup_procedure_description(cpt_code)
                if proc_desc and proc_desc not in procedures: procedures.append(proc_desc)
    if procedures: narrative_parts.append(f"Key procedures performed included {', '.join(procedures)}.")


    # --- 7. Combine all parts ---
    narrative = " ".join(part for part in narrative_parts if part and str(part).strip())
    return narrative


# --- Main Execution Block ---
if __name__ == "__main__":
    # Define file paths
    procedure_mapping_csv = "data/cpt_code.csv"
    discharge_mapping_csv = "data/discharge_code.csv"
    preprocessed_input_csv = "data/preprocessed_data.csv" # Input from preprocessing.py
    final_output_csv = "data/narrative_data.csv" # Output file with narratives

    # Load external mappings
    print("Loading external mappings...")
    load_procedure_mapping(procedure_mapping_csv)
    print("Mappings loaded.")

    # Load the preprocessed data
    print(f"\nLoading preprocessed data from: {preprocessed_input_csv}")
    try:
        df_preprocessed = pd.read_csv(preprocessed_input_csv, low_memory=False)
        print(f"Loaded {len(df_preprocessed)} records.")
    except FileNotFoundError:
        print(f"Error: Preprocessed input file not found at {preprocessed_input_csv}")
        exit()
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        exit()

    # Generate narratives
    print("\nGenerating narratives...")
    # Ensure required columns from preprocessing.py exist for the narrative
    # READMISSION_COUNT_COLUMN and READMISSION_DAYS_COLUMN are now used in the narrative
    # TARGET_COLUMN is still used for the final label
    required_narrative_cols = ['CCI_Score', TARGET_COLUMN]
    for col in required_narrative_cols:
        if col not in df_preprocessed.columns:
             print(f"Warning: Expected input column '{col}' for narrative generation or labeling not found in {preprocessed_input_csv}.")
             # Decide on fallback: exit or add dummy column
             if col == TARGET_COLUMN: # Critical for label
                 print(f"Critical Error: Target column '{TARGET_COLUMN}' is missing. Exiting.")
                 exit()
             # For other columns, you might add a dummy, or let row.get handle it if appropriate
             # Example: df_preprocessed[col] = np.nan if col == READMISSION_DAYS_COLUMN else 0


    df_preprocessed['prompt'] = df_preprocessed.apply(convert_row_to_narrative, axis=1)
    print("Narrative generation complete.")

    # Select final columns: the generated narrative ('prompt') and the target variable
    final_columns = ['prompt']
    if TARGET_COLUMN in df_preprocessed.columns:
        final_columns.append(TARGET_COLUMN)
    else:
        print(f"Critical Error: Target column '{TARGET_COLUMN}' not found for final output. Exiting.")
        exit()

    final_df = df_preprocessed[final_columns].copy()

    # Rename TARGET_COLUMN to 'label' for downstream model training
    final_df = final_df.rename(columns={TARGET_COLUMN: 'label'})
    print(f"Renamed '{TARGET_COLUMN}' to 'label'.")

    # Display preview
    print("\nPreview of final data (prompt and label):")
    print(final_df.head().to_string())

    # Save the final data
    print(f"\nSaving final data to: {final_output_csv}")
    try:
        os.makedirs(os.path.dirname(final_output_csv), exist_ok=True)
        final_df.to_csv(final_output_csv, index=False)
        print("Final data saved successfully.")
    except Exception as e:
        print(f"Error saving final data: {e}")
