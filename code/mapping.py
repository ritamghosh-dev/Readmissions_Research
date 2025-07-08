import pandas as pd
import re, os
import numpy as np
import simple_icd_10 as icd

# Global dictionaries for mappings
procedure_mapping = {}
discharge_mapping = {}

# --- Define column names as constants ---
TARGET_COLUMN = 'Readmitted30'
READMISSION_DAYS_COLUMN = 'Days_to_be_Readmitted'
READMISSION_COUNT_COLUMN = 'Readmission_Label'
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


def load_discharge_mapping_csv(csv_path):
    """
    Reads a CSV file for discharge status mapping. Expects at least two columns: 'Code' and 'Description'.
    Populates the global dictionary: discharge_mapping.
    """
    global discharge_mapping
    try:
        df = pd.read_csv(csv_path)
        discharge_mapping = pd.Series(df.Description.values, index=df.Code.astype(str).str.strip()).to_dict()
        print("Loaded", len(discharge_mapping), "discharge status mappings (from CSV).")
    except FileNotFoundError:
        print(f"Warning: Discharge mapping file not found at {csv_path}. Disposition interpretation may be limited.")
        discharge_mapping = {}
    except Exception as e:
        print(f"Error loading discharge mapping from {csv_path}: {e}")
        discharge_mapping = {}


def interpret_disposition(code) -> str:
    """
    Interpret the discharge status code (as a string or numeric value) using the discharge_mapping.
    Returns a human-readable description or a default if not found.
    """
    if pd.isna(code):
        return "with unspecified disposition"
    try:
        code_str = str(int(code)).strip().zfill(2) if pd.notna(code) else ""
    except ValueError:
         code_str = str(code).strip()
    except Exception as e:
        print(f"Error converting disposition code {code}: {e}")
        return "with unspecified disposition"
    return discharge_mapping.get(code_str, "with unspecified disposition")

# ICD-10 Block definitions (condensed) for fallback
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
    """ Classify ICD code using first 3 characters as a fallback. """
    if not icd_code or pd.isna(icd_code): return "Unspecified diagnosis"
    icd3 = str(icd_code).strip()[:3].upper()
    if len(icd3) < 3: return "Incomplete Code"
    for start, end, description in icd10_blocks:
        if start <= icd3 <= end: return description
    return "Unclassified Code"

def lookup_diagnosis_description(icd_code: str) -> str:
    """ Get the description for a full ICD-10 code, with a fallback to 3-character classification. """
    if not icd_code or pd.isna(icd_code):
        return None
    try:
        # First, try to get the specific description
        description = icd.get_description(str(icd_code).strip())
        return description
    except ValueError:
        # If the specific code is not found, use the fallback
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

    # --- 1. Demographics and Presentation ---
    age = row.get('AGE', 'Unknown age')
    gender = "female" if row.get('FEMALE') == 1 else "male"
    race_code = row.get('RACE')
    hispanic_flag = row.get('HISPANIC')
    race_desc = ""
    if hispanic_flag == 1: race_desc = "Hispanic"
    if pd.notna(race_code):
        try:
            race_code_int = int(race_code)
            race_map = {1: "White", 2: "Black or African American", 3: "Hispanic or Latino", 4: "Asian or Pacific Islander", 5: "American Indian or Alaska Native", 6: "Other race", 7: "Unknown"}
            race_val = race_map.get(race_code_int, "")
            if race_val and not race_desc: race_desc = race_val
            elif race_val and race_desc and race_val not in ["Hispanic", "Other race", "Unknown"]: race_desc = f"Hispanic {race_val}"
        except (ValueError, TypeError): pass

    homeless = " homeless" if str(row.get('Homeless', '0')).strip() == '1' else ""
    weekend = " on a weekend" if row.get('AWEEKEND') == 1 else ""
    came_through_ed = " presented via the emergency department" if row.get('HCUP_ED') == 1 else " was admitted"
    first_part = f"A {age}-year-old{homeless}{' ' + race_desc if race_desc else ''} {gender}{came_through_ed}{weekend}"
    primary_dx_code = row.get('I10_DX1')
    if pd.notna(primary_dx_code) and str(primary_dx_code).strip():
        primary_dx_desc = lookup_diagnosis_description(primary_dx_code)
        if primary_dx_desc:
            first_part += f" primarily for {primary_dx_desc}"
        else:
            first_part += f" with primary diagnosis code {primary_dx_code}"
    narrative_parts.append(first_part + ".")

    # --- NEW: Admission Context ---
    admission_context_parts = []
    readm_count_val = row.get(READMISSION_COUNT_COLUMN)
    readm_days_val = row.get(READMISSION_DAYS_COLUMN)

    if pd.notna(readm_count_val):
        try:
            readm_count = int(readm_count_val)
            if readm_count == 1:
                admission_context_parts.append("This was an index admission.")
            elif readm_count > 1:
                ordinal_suffix = get_ordinal_suffix(readm_count)
                admission_statement = f"This was the {readm_count}{ordinal_suffix} admission for this patient"
                if pd.notna(readm_days_val):
                    try:
                        days = int(readm_days_val)
                        if days >= 0:
                             admission_statement += f", occurring {days} days after the previous discharge."
                    except (ValueError, TypeError):
                        pass
                admission_context_parts.append(admission_statement + ".")
        except (ValueError, TypeError):
            print(f"Warning: Could not parse Readmission_Count: {readm_count_val}")

    if admission_context_parts:
        narrative_parts.append(" ".join(admission_context_parts))
    # --- END NEW: Admission Context ---

    # --- 2. Additional Diagnoses ---
    additional_dx = []
    diag_cols = [f'I10_DX{i}' for i in range(2, 36)]
    for dx_col in diag_cols:
        if dx_col in row:
            dx_code = row[dx_col]
            if pd.notna(dx_code) and str(dx_code).strip():
                dx_desc = lookup_diagnosis_description(dx_code)
                if dx_desc and dx_desc not in additional_dx:
                    additional_dx.append(dx_desc)
    if additional_dx:
        narrative_parts.append(f"Co-occurring conditions included {', '.join(additional_dx)}.")

    # --- 3. Charlson Comorbidity Index ---
    cci_score = row.get('CCI_Score', None)
    if pd.notna(cci_score):
         try:
             narrative_parts.append(f"The calculated Charlson Comorbidity Index score for this encounter was {int(cci_score)}.")
         except (ValueError, TypeError):
             pass

    # --- 4. Procedures ---
    procedures = []
    cpt_cols = [f'CPT{i}' for i in range(1, 101)]
    for cpt_col in cpt_cols:
         if cpt_col in row:
            cpt_code = row[cpt_col]
            if pd.notna(cpt_code) and str(cpt_code).strip():
                proc_desc = lookup_procedure_description(cpt_code)
                if proc_desc and proc_desc not in procedures:
                    procedures.append(proc_desc)
    if procedures:
        narrative_parts.append(f"Key procedures performed included {', '.join(procedures)}.")

    # --- 5. Outcome and Disposition ---
    outcome_parts = []
    los = row.get('LOS')
    if pd.notna(los):
        try:
            los_val = int(los)
            if los_val == 0:
                outcome_parts.append("a same-day visit")
            elif los_val == 1:
                outcome_parts.append("a length of stay of 1 day")
            else:
                outcome_parts.append(f"a length of stay of {los_val} days")
        except (ValueError, TypeError):
            pass

    disp = row.get('DISPUNIFORM')
    if pd.isna(disp) and 'DISPUB04' in row:
        disp = row.get('DISPUB04')
    if pd.notna(disp):
        disp_desc = interpret_disposition(disp)
        if disp_desc != "with unspecified disposition":
            outcome_parts.append(f"with discharge {disp_desc}")

    totchg = row.get('TOTCHG')
    if pd.notna(totchg):
        try:
            outcome_parts.append(f"and total charges of ${int(totchg):,}")
        except (ValueError, TypeError):
            pass

    if outcome_parts:
        narrative_parts.append("The encounter involved " + ", ".join(outcome_parts) + ".")

    # --- 6. Social Context ---
    social_parts = []
    pay1 = row.get('PAY1')
    payer_map = {1: "Medicare", 2: "Medicaid", 3: "Private insurance", 4: "self-pay", 5: "no charge", 6: "other"}
    if pd.notna(pay1):
        try:
            pay1_int = int(pay1)
            if pay1_int in payer_map:
                payer_text = payer_map[pay1_int]
                if pay1_int in [4, 5]:
                    social_parts.append(f"listed as {payer_text}")
                else:
                    social_parts.append(f"primarily covered by {payer_text}")
        except (ValueError, TypeError):
            pass

    inc_quart_val = row.get('MEDINCSTQ')
    inc_quart_int = None
    if pd.notna(inc_quart_val):
        inc_quart_str = str(inc_quart_val).strip()
        if inc_quart_str.isdigit():
            try:
                inc_quart_int = int(inc_quart_str)
            except ValueError:
                pass
    if inc_quart_int is not None and inc_quart_int in [1, 2, 3, 4]:
         quart_map = {1: "lowest", 2: "second-lowest", 3: "second-highest", 4: "highest"}
         inc_desc = quart_map.get(inc_quart_int)
         if inc_desc:
             social_parts.append(f"residing in an area with the {inc_desc} median income quartile")

    if social_parts:
        narrative_parts.append("Social context indicates the patient was " + " and ".join(social_parts) + ".")

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
    load_discharge_mapping_csv(discharge_mapping_csv)
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
    required_narrative_cols = [READMISSION_COUNT_COLUMN, READMISSION_DAYS_COLUMN, 'CCI_Score', TARGET_COLUMN]
    for col in required_narrative_cols:
        if col not in df_preprocessed.columns:
             print(f"Warning: Expected input column '{col}' for narrative generation or labeling not found in {preprocessed_input_csv}.")
             if col == TARGET_COLUMN:
                 print(f"Critical Error: Target column '{TARGET_COLUMN}' is missing. Exiting.")
                 exit()

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