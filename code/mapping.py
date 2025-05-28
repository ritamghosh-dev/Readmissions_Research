# import pandas as pd
# import re
# import numpy as np # Import numpy for checking NaN

# # Global dictionaries for mappings
# diagnosis_mapping = {}
# procedure_mapping = {}
# discharge_mapping = {}

# # --- Define column names as constants ---
# TARGET_COLUMN = 'Readmitted' # Still needed for final output selection/renaming
# READMISSION_DAYS_COLUMN = 'Readmission_Days' # Name of the column to potentially exclude from narrative
# # --- END CONSTANTS ---

# def load_procedure_mapping(csv_path):
#     """
#     Reads a CSV file for CPT mapping. Expects at least two columns: 'Code' and 'Description'.
#     Populates the global dictionary: procedure_mapping.
#     """
#     global procedure_mapping
#     try:
#         df = pd.read_csv(csv_path)
#         procedure_mapping = pd.Series(df.Description.values, index=df.Code.astype(str).str.strip()).to_dict()
#         print("Loaded", len(procedure_mapping), "CPT mappings.")
#     except FileNotFoundError:
#         print(f"Warning: CPT mapping file not found at {csv_path}. Procedure lookup will not work.")
#         procedure_mapping = {}
#     except Exception as e:
#         print(f"Error loading CPT mapping from {csv_path}: {e}")
#         procedure_mapping = {}


# def load_discharge_mapping_csv(csv_path):
#     """
#     Reads a CSV file for discharge status mapping. Expects at least two columns: 'Code' and 'Description'.
#     Populates the global dictionary: discharge_mapping.
#     """
#     global discharge_mapping
#     try:
#         df = pd.read_csv(csv_path)
#         discharge_mapping = pd.Series(df.Description.values, index=df.Code.astype(str).str.strip()).to_dict()
#         print("Loaded", len(discharge_mapping), "discharge status mappings (from CSV).")
#     except FileNotFoundError:
#         print(f"Warning: Discharge mapping file not found at {csv_path}. Disposition interpretation may be limited.")
#         discharge_mapping = {}
#     except Exception as e:
#         print(f"Error loading discharge mapping from {csv_path}: {e}")
#         discharge_mapping = {}


# def interpret_disposition(code) -> str:
#     """
#     Interpret the discharge status code (as a string or numeric value) using the discharge_mapping.
#     Returns a human-readable description or a default if not found.
#     """
#     if pd.isna(code):
#         return "with unspecified disposition"
#     try:
#         code_str = str(int(code)).strip().zfill(2) if pd.notna(code) else ""
#     except ValueError:
#          code_str = str(code).strip()
#     except Exception as e:
#         print(f"Error converting disposition code {code}: {e}")
#         return "with unspecified disposition"
#     return discharge_mapping.get(code_str, "with unspecified disposition")

# # ICD-10 Block definitions (condensed)
# icd10_blocks = [
#     ("A00", "B99", "Certain infectious and parasitic diseases"),("C00", "D49", "Neoplasms"),
#     ("D50", "D89", "Diseases of the blood and blood-forming organs"),("E00", "E89", "Endocrine, nutritional and metabolic diseases"),
#     ("F01", "F99", "Mental, Behavioral and Neurodevelopmental disorders"),("G00", "G99", "Diseases of the nervous system"),
#     ("H00", "H59", "Diseases of the eye and adnexa"),("H60", "H95", "Diseases of the ear and mastoid process"),
#     ("I00", "I99", "Diseases of the circulatory system"),("J00", "J99", "Diseases of the respiratory system"),
#     ("K00", "K95", "Diseases of the digestive system"),("L00", "L99", "Diseases of the skin and subcutaneous tissue"),
#     ("M00", "M99", "Diseases of the musculoskeletal system and connective tissue"),("N00", "N99", "Diseases of the genitourinary system"),
#     ("O00", "O9A", "Pregnancy, childbirth and the puerperium"),("P00", "P96", "Certain conditions originating in the perinatal period"),
#     ("Q00", "Q99", "Congenital malformations, deformations and chromosomal abnormalities"),("R00", "R99", "Symptoms, signs and abnormal clinical and laboratory findings"),
#     ("S00", "T88", "Injury, poisoning and certain other consequences of external causes"),("V00", "Y99", "External causes of morbidity"),
#     ("Z00", "Z99", "Factors influencing health status and contact with health services")
# ]

# def classify_icd_3_chars(icd_code: str) -> str:
#     """ Classify ICD code using first 3 characters. """
#     if not icd_code or pd.isna(icd_code): return None
#     icd3 = str(icd_code).strip()[:3].upper()
#     if len(icd3) < 3: return "Incomplete Code"
#     for start, end, description in icd10_blocks:
#         if start <= icd3 <= end: return description
#     return "Unclassified Code"

# def lookup_diagnosis_description(icd_code: str) -> str:
#     """ Get high-level classification for ICD code. """
#     return classify_icd_3_chars(icd_code)

# def lookup_procedure_description(code: str) -> str:
#     """ Look up CPT procedure code description. """
#     if not code or pd.isna(code): return None
#     code_str = str(code).strip()
#     return procedure_mapping.get(code_str, None)

# def convert_row_to_narrative(row):
#     """ Generates narrative summary, excluding direct readmission outcome info. """
#     narrative_parts = []

#     # --- 1. Demographics and Presentation ---
#     age = row.get('AGE', 'Unknown age')
#     gender = "female" if row.get('FEMALE') == 1 else "male"
#     race_code = row.get('RACE')
#     hispanic_flag = row.get('HISPANIC')
#     race_desc = ""
#     if hispanic_flag == 1: race_desc = "Hispanic"
#     if pd.notna(race_code):
#         try:
#             race_code_int = int(race_code)
#             race_map = {1: "White", 2: "Black or African American", 3: "Hispanic or Latino", 4: "Asian or Pacific Islander", 5: "American Indian or Alaska Native", 6: "Other race", 7: "Unknown"}
#             race_val = race_map.get(race_code_int, "")
#             if race_val and not race_desc: race_desc = race_val
#             elif race_val and race_desc and race_val not in ["Hispanic", "Other race", "Unknown"]: race_desc = f"Hispanic {race_val}"
#         except (ValueError, TypeError): pass

#     homeless = " homeless" if str(row.get('Homeless', '0')).strip() == '1' else ""
#     weekend = " on a weekend" if row.get('AWEEKEND') == 1 else ""
#     came_through_ed = " presented via the emergency department" if row.get('HCUP_ED') == 1 else " was admitted"
#     first_part = f"A {age}-year-old{homeless}{' ' + race_desc if race_desc else ''} {gender}{came_through_ed}{weekend}"
#     primary_dx_code = row.get('I10_DX1')
#     if pd.notna(primary_dx_code) and str(primary_dx_code).strip():
#         primary_dx_desc = lookup_diagnosis_description(primary_dx_code)
#         if primary_dx_desc and primary_dx_desc not in ["Unclassified Code", "Incomplete Code"]: first_part += f" primarily for {primary_dx_desc}"
#         else: first_part += f" with primary diagnosis code {primary_dx_code}"
#     narrative_parts.append(first_part + ".")

#     # --- 2. Additional Diagnoses ---
#     additional_dx = []
#     diag_cols = [f'I10_DX{i}' for i in range(2, 36)]
#     for dx_col in diag_cols:
#         if dx_col in row:
#             dx_code = row[dx_col]
#             if pd.notna(dx_code) and str(dx_code).strip():
#                 dx_desc = lookup_diagnosis_description(dx_code)
#                 if dx_desc and dx_desc not in ["Unclassified Code", "Incomplete Code"] and dx_desc not in additional_dx: additional_dx.append(dx_desc)
#     if additional_dx: narrative_parts.append(f"Co-occurring conditions included {', '.join(additional_dx)}.")

#     # --- 3. Charlson Comorbidity Index ---
#     cci_score = row.get('CCI_Score', None)
#     if pd.notna(cci_score):
#          try: narrative_parts.append(f"The calculated Charlson Comorbidity Index score for this encounter was {int(cci_score)}.")
#          except (ValueError, TypeError): pass

#     # --- 4. Procedures ---
#     procedures = []
#     cpt_cols = [f'CPT{i}' for i in range(1, 4)]
#     for cpt_col in cpt_cols:
#          if cpt_col in row:
#             cpt_code = row[cpt_col]
#             if pd.notna(cpt_code) and str(cpt_code).strip():
#                 proc_desc = lookup_procedure_description(cpt_code)
#                 if proc_desc and proc_desc not in procedures: procedures.append(proc_desc)
#     if procedures: narrative_parts.append(f"Key procedures performed included {', '.join(procedures)}.")

#     # --- 5. Outcome and Disposition (REMOVED READMISSION INFO) ---
#     outcome_parts = []
#     los = row.get('LOS')
#     if pd.notna(los):
#         try:
#             los_val = int(los)
#             if los_val == 0: outcome_parts.append("a same-day visit")
#             elif los_val == 1: outcome_parts.append("a length of stay of 1 day")
#             else: outcome_parts.append(f"a length of stay of {los_val} days")
#         except (ValueError, TypeError): pass

#     if row.get('DIED') == 1:
#         outcome_parts.append("resulting in patient death")
#     else:
#         disp = row.get('DISPUNIFORM')
#         if pd.isna(disp) and 'DISPUB04' in row: disp = row.get('DISPUB04')
#         if pd.notna(disp):
#             disp_desc = interpret_disposition(disp)
#             if disp_desc != "with unspecified disposition": outcome_parts.append(f"with discharge {disp_desc}")

#     # --- !!! REMOVED READMISSION STATUS/DAYS FROM NARRATIVE !!! ---
#     # The following lines related to readmitted_flag and readmission_days
#     # have been removed from this section to prevent data leakage.
#     # --- !!! END REMOVAL !!! ---

#     totchg = row.get('TOTCHG')
#     if pd.notna(totchg):
#         try: outcome_parts.append(f"and total charges of ${int(totchg):,}")
#         except (ValueError, TypeError): pass

#     if outcome_parts: narrative_parts.append("The encounter involved " + ", ".join(outcome_parts) + ".")

#     # --- 6. Social Context ---
#     social_parts = []
#     pay1 = row.get('PAY1')
#     payer_map = {1: "Medicare", 2: "Medicaid", 3: "Private insurance", 4: "self-pay", 5: "no charge", 6: "other"}
#     if pd.notna(pay1):
#         try:
#             pay1_int = int(pay1)
#             if pay1_int in payer_map:
#                 payer_text = payer_map[pay1_int]
#                 if pay1_int == 4 or pay1_int == 5: social_parts.append(f"listed as {payer_text}")
#                 else: social_parts.append(f"primarily covered by {payer_text}")
#         except (ValueError, TypeError): pass

#     inc_quart_val = row.get('MEDINCSTQ')
#     inc_quart_int = None
#     if pd.notna(inc_quart_val):
#         inc_quart_str = str(inc_quart_val).strip()
#         if inc_quart_str.isdigit():
#             try: inc_quart_int = int(inc_quart_str)
#             except ValueError: pass
#     if inc_quart_int is not None and inc_quart_int in [1, 2, 3, 4]:
#          quart_map = {1: "lowest", 2: "second-lowest", 3: "second-highest", 4: "highest"}
#          inc_desc = quart_map.get(inc_quart_int)
#          if inc_desc: social_parts.append(f"residing in an area with the {inc_desc} median income quartile")

#     if social_parts: narrative_parts.append("Social context indicates the patient was " + " and ".join(social_parts) + ".")

#     # --- 7. Combine all parts ---
#     narrative = " ".join(part for part in narrative_parts if part)
#     return narrative


# # --- Main Execution Block ---
# if __name__ == "__main__":
#     # Define file paths
#     procedure_mapping_csv = "data/cpt_code.csv" # Adjusted path
#     discharge_mapping_csv = "data/discharge_code.csv" # Adjusted path
#     preprocessed_input_csv = "data/preprocessed_data.csv" # Adjusted path
#     final_output_csv = "data/narrative_data.csv" # Adjusted path

#     # Load external mappings
#     print("Loading external mappings...")
#     load_procedure_mapping(procedure_mapping_csv)
#     load_discharge_mapping_csv(discharge_mapping_csv)
#     print("Mappings loaded.")

#     # Load the preprocessed data
#     print(f"\nLoading preprocessed data from: {preprocessed_input_csv}")
#     try:
#         df_preprocessed = pd.read_csv(preprocessed_input_csv, low_memory=False)
#         print(f"Loaded {len(df_preprocessed)} records.")
#     except FileNotFoundError:
#         print(f"Error: Preprocessed input file not found at {preprocessed_input_csv}")
#         exit()
#     except Exception as e:
#         print(f"Error loading preprocessed data: {e}")
#         exit()

#     # Generate narratives
#     print("\nGenerating narratives...")
#     # Check for expected input columns from preprocessing.py
#     required_input_cols = [TARGET_COLUMN, READMISSION_DAYS_COLUMN, 'CCI_Score'] # Add others as needed
#     for col in required_input_cols:
#         if col not in df_preprocessed.columns:
#              print(f"Warning: Expected input column '{col}' not found in {preprocessed_input_csv}. Narratives might be incomplete.")
#              # Optionally add dummy columns if needed downstream, e.g.:
#              # if col == TARGET_COLUMN: df_preprocessed[col] = -1
#              # elif col == READMISSION_DAYS_COLUMN: df_preprocessed[col] = np.nan
#              # elif col == 'CCI_Score': df_preprocessed[col] = 0


#     df_preprocessed['prompt'] = df_preprocessed.apply(convert_row_to_narrative, axis=1)
#     print("Narrative generation complete.")

#     # Select final columns: the generated narrative ('prompt') and the target variable ('Readmitted')
#     final_columns = ['prompt']
#     if TARGET_COLUMN in df_preprocessed.columns:
#         final_columns.append(TARGET_COLUMN)
#     else:
#         # If target column wasn't found, we can't create the label
#         print(f"Error: Target column '{TARGET_COLUMN}' not found. Cannot create final dataset with labels.")
#         exit() # Exit if the label column is missing


#     final_df = df_preprocessed[final_columns].copy()

#     # Rename TARGET_COLUMN to 'label'
#     final_df = final_df.rename(columns={TARGET_COLUMN: 'label'})
#     print(f"Renamed '{TARGET_COLUMN}' to 'label'.")

#     # Display preview
#     print("\nPreview of final data (prompt and label):")
#     print(final_df.head().to_string())

#     # Save the final data
#     print(f"\nSaving final data to: {final_output_csv}")
#     try:
#         final_df.to_csv(final_output_csv, index=False)
#         print("Final data saved successfully.")
#     except Exception as e:
#         print(f"Error saving final data: {e}")

import pandas as pd
import re, os
import numpy as np # Import numpy for checking NaN

# Global dictionaries for mappings
diagnosis_mapping = {}
procedure_mapping = {}
discharge_mapping = {}

# --- Define column names as constants ---
TARGET_COLUMN = 'Readmitted'
READMISSION_DAYS_COLUMN = 'Readmission_Days'
READMISSION_COUNT_COLUMN = 'Readmission_Count' # Added constant
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
        except (ValueError, TypeError): pass # Keep race_desc as is

    homeless = " homeless" if str(row.get('Homeless', '0')).strip() == '1' else ""
    weekend = " on a weekend" if row.get('AWEEKEND') == 1 else ""
    came_through_ed = " presented via the emergency department" if row.get('HCUP_ED') == 1 else " was admitted"
    first_part = f"A {age}-year-old{homeless}{' ' + race_desc if race_desc else ''} {gender}{came_through_ed}{weekend}"
    primary_dx_code = row.get('I10_DX1')
    if pd.notna(primary_dx_code) and str(primary_dx_code).strip():
        primary_dx_desc = lookup_diagnosis_description(primary_dx_code)
        if primary_dx_desc and primary_dx_desc not in ["Unclassified Code", "Incomplete Code"]: first_part += f" primarily for {primary_dx_desc}"
        else: first_part += f" with primary diagnosis code {primary_dx_code}"
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
                        if days >= 0: # Ensure days are not negative (should be handled in preprocessing, but good check)
                             admission_statement += f", occurring {days} days after the previous discharge."
                    except (ValueError, TypeError):
                        # If Readmission_Days is not a valid number for some reason, just state the admission count
                        pass # Fall through to just the admission count statement
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
                if dx_desc and dx_desc not in ["Unclassified Code", "Incomplete Code"] and dx_desc not in additional_dx: additional_dx.append(dx_desc)
    if additional_dx: narrative_parts.append(f"Co-occurring conditions included {', '.join(additional_dx)}.")

    # --- 3. Charlson Comorbidity Index ---
    cci_score = row.get('CCI_Score', None)
    if pd.notna(cci_score):
         try: narrative_parts.append(f"The calculated Charlson Comorbidity Index score for this encounter was {int(cci_score)}.")
         except (ValueError, TypeError): pass

    # --- 4. Procedures ---
    procedures = []
    cpt_cols = [f'CPT{i}' for i in range(1, 4)]
    for cpt_col in cpt_cols:
         if cpt_col in row:
            cpt_code = row[cpt_col]
            if pd.notna(cpt_code) and str(cpt_code).strip():
                proc_desc = lookup_procedure_description(cpt_code)
                if proc_desc and proc_desc not in procedures: procedures.append(proc_desc)
    if procedures: narrative_parts.append(f"Key procedures performed included {', '.join(procedures)}.")

    # --- 5. Outcome and Disposition (Original Readmission Info Excluded) ---
    outcome_parts = []
    los = row.get('LOS')
    if pd.notna(los):
        try:
            los_val = int(los)
            if los_val == 0: outcome_parts.append("a same-day visit")
            elif los_val == 1: outcome_parts.append("a length of stay of 1 day")
            else: outcome_parts.append(f"a length of stay of {los_val} days")
        except (ValueError, TypeError): pass

    if row.get('DIED') == 1:
        outcome_parts.append("resulting in patient death")
    else:
        disp = row.get('DISPUNIFORM')
        if pd.isna(disp) and 'DISPUB04' in row: disp = row.get('DISPUB04')
        if pd.notna(disp):
            disp_desc = interpret_disposition(disp)
            if disp_desc != "with unspecified disposition": outcome_parts.append(f"with discharge {disp_desc}")

    # The TARGET_COLUMN (Readmitted) and actual readmission days are now explicitly NOT added to narrative here
    # to prevent data leakage, as per your previous instruction.
    # The target variable will be handled separately.

    totchg = row.get('TOTCHG')
    if pd.notna(totchg):
        try: outcome_parts.append(f"and total charges of ${int(totchg):,}")
        except (ValueError, TypeError): pass

    if outcome_parts: narrative_parts.append("The encounter involved " + ", ".join(outcome_parts) + ".")

    # --- 6. Social Context ---
    social_parts = []
    pay1 = row.get('PAY1')
    payer_map = {1: "Medicare", 2: "Medicaid", 3: "Private insurance", 4: "self-pay", 5: "no charge", 6: "other"}
    if pd.notna(pay1):
        try:
            pay1_int = int(pay1)
            if pay1_int in payer_map:
                payer_text = payer_map[pay1_int]
                if pay1_int == 4 or pay1_int == 5: social_parts.append(f"listed as {payer_text}")
                else: social_parts.append(f"primarily covered by {payer_text}")
        except (ValueError, TypeError): pass

    inc_quart_val = row.get('MEDINCSTQ')
    inc_quart_int = None
    if pd.notna(inc_quart_val):
        inc_quart_str = str(inc_quart_val).strip()
        if inc_quart_str.isdigit():
            try: inc_quart_int = int(inc_quart_str)
            except ValueError: pass
    if inc_quart_int is not None and inc_quart_int in [1, 2, 3, 4]:
         quart_map = {1: "lowest", 2: "second-lowest", 3: "second-highest", 4: "highest"}
         inc_desc = quart_map.get(inc_quart_int)
         if inc_desc: social_parts.append(f"residing in an area with the {inc_desc} median income quartile")

    if social_parts: narrative_parts.append("Social context indicates the patient was " + " and ".join(social_parts) + ".")

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
    # Ensure required columns from preprocessing.py exist for the narrative
    # READMISSION_COUNT_COLUMN and READMISSION_DAYS_COLUMN are now used in the narrative
    # TARGET_COLUMN is still used for the final label
    required_narrative_cols = [READMISSION_COUNT_COLUMN, READMISSION_DAYS_COLUMN, 'CCI_Score', TARGET_COLUMN]
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

