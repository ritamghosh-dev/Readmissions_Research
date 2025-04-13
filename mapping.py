# import pandas as pd
# import re
# import PyPDF2

# # Global dictionaries for mappings
# diagnosis_mapping = {}
# procedure_mapping = {}
# discharge_mapping = {}

# ##############################################
# # Load ICD-10 Diagnosis Mapping from CSV file
# ##############################################
# def load_diagnosis_mapping(csv_path):
#     """
#     Reads a CSV file for ICD-10 mapping. Expects at least two columns: 'Code' and 'Description'.
#     Populates the global dictionary: diagnosis_mapping.
#     """
#     global diagnosis_mapping
#     df = pd.read_csv(csv_path)
#     # Assuming the CSV has columns "Code" and "Description"
#     diagnosis_mapping = pd.Series(df.Description.values, index=df.Code.astype(str).str.strip()).to_dict()
#     print("Loaded", len(diagnosis_mapping), "ICD-10 mappings.")

# ##############################################
# # Load CPT Procedure Mapping from CSV file
# ##############################################
# def load_procedure_mapping(csv_path):
#     """
#     Reads a CSV file for CPT mapping. Expects at least two columns: 'Code' and 'Description'.
#     Populates the global dictionary: procedure_mapping.
#     """
#     global procedure_mapping
#     df = pd.read_csv(csv_path)
#     # Assuming the CSV columns are "Code" and "Description"
#     procedure_mapping = pd.Series(df.Description.values, index=df.Code.astype(str).str.strip()).to_dict()
#     print("Loaded", len(procedure_mapping), "CPT mappings.")

# ##############################################
# # Load Discharge Mapping from PDF file
# ##############################################
# def load_discharge_mapping_csv(csv_path):
#     """
#     Reads a CSV file for discharge status mapping. Expects at least two columns: 'Code' and 'Description'.
#     Populates the global dictionary: discharge_mapping.
#     """
#     global discharge_mapping
#     df = pd.read_csv(csv_path)
#     # Assuming the CSV columns are "Code" and "Description"
#     discharge_mapping = pd.Series(df.Description.values, index=df.Code.astype(str).str.strip()).to_dict()
#     print("Loaded", len(discharge_mapping), "discharge status mappings (from CSV).")

# ##############################################
# # Lookup functions
# ##############################################
# def lookup_diagnosis_description(code: str) -> str:
#     """
#     Look up and return the human-readable description for an ICD-10 diagnosis code.
#     Returns None if the code is not found.
#     """
#     if not code:
#         return None
#     return diagnosis_mapping.get(code.strip(), None)

# def lookup_procedure_description(code: str) -> str:
#     """
#     Look up and return the human-readable description for a CPT procedure code.
#     Returns None if the code is not found.
#     """
#     if not code:
#         return None
#     return procedure_mapping.get(code.strip(), None)

# def interpret_disposition(code) -> str:
#     """
#     Interpret the discharge status code (as a string or numeric value) using the discharge_mapping.
#     Returns a human-readable description or a default if not found.
#     """
#     try:
#         # Convert the code to a two-digit string (add leading zeros if needed)
#         code_str = str(code).strip().zfill(2)
#     except Exception as e:
#         return "with unspecified disposition"
#     return discharge_mapping.get(code_str, "with unspecified disposition")

# ##############################################
# # Example usage
# ##############################################
# if __name__ == "__main__":
#     # Replace these paths with the actual file paths on your system:
#     diagnosis_mapping_csv = "icd10.csv"
#     procedure_mapping_csv = "cpt_code.csv"
#     discharge_mapping_csv = "discharge_code.csv"  # Now using the CSV mapping

#     # Load the mappings
#     load_diagnosis_mapping(diagnosis_mapping_csv)
#     load_procedure_mapping(procedure_mapping_csv)
#     load_discharge_mapping_csv(discharge_mapping_csv)
    
#     # Test the lookup functions
#     diag_code = "R1030"
#     diag_desc = lookup_diagnosis_description(diag_code)
#     print(f"ICD-10 Code {diag_code}: {diag_desc}")
    
#     proc_code = "J2270"
#     proc_desc = lookup_procedure_description(proc_code)
#     print(f"CPT Code {proc_code}: {proc_desc}")

#     disp_code = "70"
#     disp_desc = interpret_disposition(disp_code)
#     print(f"Discharge Code {disp_code}: {disp_desc}")
import pandas as pd
import re
import PyPDF2

# Global dictionaries for mappings
diagnosis_mapping = {}
procedure_mapping = {}
discharge_mapping = {}

# ##############################################
# # Load ICD-10 Diagnosis Mapping from CSV file
# ##############################################
# def load_diagnosis_mapping(csv_path):
#     """
#     Reads a CSV file for ICD-10 mapping. Expects at least two columns: 'Code' and 'Description'.
#     Populates the global dictionary: diagnosis_mapping.
#     """
#     global diagnosis_mapping
#     df = pd.read_csv(csv_path)
#     # Assuming the CSV has columns "Code" and "Description"
#     diagnosis_mapping = pd.Series(df.Description.values, index=df.Code.astype(str).str.strip()).to_dict()
#     print("Loaded", len(diagnosis_mapping), "ICD-10 mappings.")



##############################################
# Load CPT Procedure Mapping from CSV file
##############################################
def load_procedure_mapping(csv_path):
    """
    Reads a CSV file for CPT mapping. Expects at least two columns: 'Code' and 'Description'.
    Populates the global dictionary: procedure_mapping.
    """
    global procedure_mapping
    df = pd.read_csv(csv_path)
    # Assuming the CSV columns are "Code" and "Description"
    procedure_mapping = pd.Series(df.Description.values, index=df.Code.astype(str).str.strip()).to_dict()
    print("Loaded", len(procedure_mapping), "CPT mappings.")

##############################################
# Load Discharge Mapping from PDF file
##############################################
def load_discharge_mapping_csv(csv_path):
    """
    Reads a CSV file for discharge status mapping. Expects at least two columns: 'Code' and 'Description'.
    Populates the global dictionary: discharge_mapping.
    """
    global discharge_mapping
    df = pd.read_csv(csv_path)
    # Assuming the CSV columns are "Code" and "Description"
    discharge_mapping = pd.Series(df.Description.values, index=df.Code.astype(str).str.strip()).to_dict()
    print("Loaded", len(discharge_mapping), "discharge status mappings (from CSV).")

##############################################
# Lookup functions
##############################################
# def lookup_diagnosis_description(code: str) -> str:
#     """
#     Look up and return the human-readable description for an ICD-10 diagnosis code.
#     Returns None if the code is not found.
#     """
#     if not code:
#         return None
#     return diagnosis_mapping.get(code.strip(), None)

def lookup_procedure_description(code: str) -> str:
    """
    Look up and return the human-readable description for a CPT procedure code.
    Returns None if the code is not found.
    """
    if not code:
        return None
    return procedure_mapping.get(code.strip(), None)

def interpret_disposition(code) -> str:
    """
    Interpret the discharge status code (as a string or numeric value) using the discharge_mapping.
    Returns a human-readable description or a default if not found.
    """
    try:
        # Convert the code to a two-digit string (add leading zeros if needed)
        code_str = str(code).strip().zfill(2)
    except Exception as e:
        return "with unspecified disposition"
    return discharge_mapping.get(code_str, "with unspecified disposition")

icd10_blocks = [
    # --- Chapter 1: A00–B99 ---
    ("A00", "A09", "Intestinal infectious diseases"),
    ("A15", "A19", "Tuberculosis"),
    ("A20", "A28", "Certain zoonotic bacterial diseases"),
    ("A30", "A49", "Other bacterial diseases"),
    ("A50", "A64", "Infections with a predominantly sexual mode of transmission"),
    ("A65", "A69", "Other spirochetal diseases"),
    ("A70", "A74", "Other diseases caused by chlamydiae"),
    ("A75", "A79", "Rickettsioses"),
    ("A80", "A89", "Viral and prion infections of the central nervous system"),
    ("A90", "A99", "Arthropod-borne viral fevers and viral hemorrhagic fevers"),
    ("B00", "B09", "Viral infections characterized by skin and mucous membrane lesions"),
    ("B10", "B10", "Other human herpesviruses"),
    ("B15", "B19", "Viral hepatitis"),
    ("B20", "B20", "Human immunodeficiency virus [HIV] disease"),
    ("B25", "B34", "Other viral diseases"),
    ("B35", "B49", "Mycoses"),
    ("B50", "B64", "Protozoal diseases"),
    ("B65", "B83", "Helminthiases"),
    ("B85", "B89", "Pediculosis, acariasis and other infestations"),
    ("B90", "B94", "Sequelae of infectious and parasitic diseases"),
    ("B95", "B97", "Bacterial and viral infectious agents"),
    ("B99", "B99", "Other infectious diseases"),

    # --- Chapter 2: C00–D49 ---
    ("C00", "C14", "Malignant neoplasms of lip, oral cavity and pharynx"),
    ("C15", "C26", "Malignant neoplasms of digestive organs"),
    ("C30", "C39", "Malignant neoplasms of respiratory and intrathoracic organs"),
    ("C40", "C41", "Malignant neoplasms of bone and articular cartilage"),
    ("C43", "C44", "Melanoma and other malignant neoplasms of skin"),
    ("C45", "C49", "Malignant neoplasms of mesothelial and soft tissue"),
    ("C50", "C50", "Malignant neoplasm of breast"),
    ("C51", "C58", "Malignant neoplasms of female genital organs"),
    ("C60", "C63", "Malignant neoplasms of male genital organs"),
    ("C64", "C68", "Malignant neoplasms of urinary tract"),
    ("C69", "C72", "Malignant neoplasms of eye, brain and other parts of CNS"),
    ("C73", "C75", "Malignant neoplasms of thyroid and other endocrine glands"),
    ("C7A", "C7A", "Malignant neuroendocrine tumors"),
    ("C76", "C80", "Malignant neoplasms of ill-defined, secondary and unspecified sites"),
    ("C81", "C96", "Malignant neoplasms of lymphoid, hematopoietic and related tissue"),
    ("D00", "D09", "In situ neoplasms"),
    ("D10", "D36", "Benign neoplasms, except benign neuroendocrine tumors"),
    ("D3A", "D3A", "Benign neuroendocrine tumors"),
    ("D37", "D48", "Neoplasms of uncertain behavior, polycythemia vera, MDS"),
    ("D49", "D49", "Neoplasms of unspecified behavior"),

    # --- Chapter 3: D50–D89 ---
    ("D50", "D53", "Nutritional anemias"),
    ("D55", "D59", "Hemolytic anemias"),
    ("D60", "D64", "Aplastic and other anemias and other bone marrow failure syndromes"),
    ("D65", "D69", "Coagulation defects, purpura and other hemorrhagic conditions"),
    ("D70", "D77", "Other disorders of blood and blood-forming organs"),
    ("D78", "D78", "Intraoperative and postprocedural complications of the spleen"),
    ("D80", "D89", "Certain disorders involving the immune mechanism"),

    # --- Chapter 4: E00–E89 ---
    ("E00", "E07", "Disorders of thyroid gland"),
    ("E08", "E13", "Diabetes mellitus"),
    ("E15", "E16", "Other disorders of glucose regulation and pancreatic internal secretion"),
    ("E20", "E35", "Disorders of other endocrine glands"),
    ("E36", "E36", "Intraoperative and postprocedural complications of endocrine system"),
    ("E40", "E46", "Malnutrition"),
    ("E50", "E64", "Other nutritional deficiencies"),
    ("E65", "E68", "Overweight, obesity and other hyperalimentation"),
    ("E70", "E88", "Metabolic disorders"),
    ("E89", "E89", "Postprocedural endocrine/metabolic complications and disorders"),

    # --- Chapter 5: F01–F99 ---
    ("F01", "F09", "Mental disorders due to known physiological conditions"),
    ("F10", "F19", "Mental and behavioral disorders due to psychoactive substance use"),
    ("F20", "F29", "Schizophrenia, schizotypal, delusional, other non-mood psychotic"),
    ("F30", "F39", "Mood [affective] disorders"),
    ("F40", "F48", "Anxiety, dissociative, stress-related, somatoform, other nonpsychotic"),
    ("F50", "F59", "Behavioral syndromes associated with physiological disturbances"),
    ("F60", "F69", "Disorders of adult personality and behavior"),
    ("F70", "F79", "Intellectual disabilities"),
    ("F80", "F89", "Pervasive and specific developmental disorders"),
    ("F90", "F98", "Behavioral and emotional disorders with onset in childhood/adolescence"),
    ("F99", "F99", "Unspecified mental disorder"),

    # --- Chapter 6: G00–G99 ---
    ("G00", "G09", "Inflammatory diseases of the central nervous system"),
    ("G10", "G14", "Systemic atrophies primarily affecting the central nervous system"),
    ("G20", "G26", "Extrapyramidal and movement disorders"),
    ("G30", "G32", "Other degenerative diseases of the nervous system"),
    ("G35", "G37", "Demyelinating diseases of the central nervous system"),
    ("G40", "G47", "Episodic and paroxysmal disorders"),
    ("G50", "G59", "Nerve, nerve root and plexus disorders"),
    ("G60", "G65", "Polyneuropathies and other disorders of the peripheral nervous system"),
    ("G70", "G73", "Diseases of myoneural junction and muscle"),
    ("G80", "G83", "Cerebral palsy and other paralytic syndromes"),
    ("G89", "G99", "Other disorders of the nervous system"),

    # --- Chapter 7: H00–H59 ---
    ("H00", "H05", "Disorders of eyelid, lacrimal system and orbit"),
    ("H10", "H11", "Conjunctival disorders"),
    ("H15", "H22", "Disorders of sclera, cornea, iris and ciliary body"),
    ("H25", "H28", "Disorders of lens"),
    ("H30", "H36", "Disorders of choroid and retina"),
    ("H40", "H42", "Glaucoma"),
    ("H43", "H44", "Disorders of vitreous body and globe"),
    ("H46", "H47", "Disorders of optic nerve and visual pathways"),
    ("H49", "H52", "Disorders of ocular muscles, binocular movement, refraction"),
    ("H53", "H54", "Visual disturbances and blindness"),
    ("H55", "H59", "Other disorders of eye and adnexa"),

    # --- Chapter 8: H60–H95 ---
    ("H60", "H62", "Diseases of external ear"),
    ("H65", "H75", "Diseases of middle ear and mastoid"),
    ("H80", "H83", "Diseases of inner ear"),
    ("H90", "H94", "Other disorders of ear"),
    ("H95", "H95", "Intraoperative and postprocedural complications/disorders of ear"),

    # --- Chapter 9: I00–I99 ---
    ("I00", "I02", "Acute rheumatic fever"),
    ("I05", "I09", "Chronic rheumatic heart diseases"),
    ("I10", "I16", "Hypertensive diseases"),
    ("I20", "I25", "Ischemic heart diseases"),
    ("I26", "I28", "Pulmonary heart disease and diseases of pulmonary circulation"),
    ("I30", "I52", "Other forms of heart disease"),
    ("I60", "I69", "Cerebrovascular diseases"),
    ("I70", "I79", "Diseases of arteries, arterioles and capillaries"),
    ("I80", "I89", "Diseases of veins, lymphatic vessels and lymph nodes, nec"),
    ("I95", "I99", "Other and unspecified disorders of the circulatory system"),

    # --- Chapter 10: J00–J99 ---
    ("J00", "J06", "Acute upper respiratory infections"),
    ("J09", "J18", "Influenza and pneumonia"),
    ("J20", "J22", "Other acute lower respiratory infections"),
    ("J30", "J39", "Other diseases of upper respiratory tract"),
    ("J40", "J47", "Chronic lower respiratory diseases"),
    ("J60", "J70", "Lung diseases due to external agents"),
    ("J80", "J84", "Other respiratory diseases principally affecting the interstitium"),
    ("J85", "J86", "Suppurative and necrotic conditions of lower respiratory tract"),
    ("J90", "J94", "Other diseases of pleura"),
    ("J95", "J95", "Intraoperative and postprocedural complications of respiratory system"),
    ("J96", "J99", "Other diseases of the respiratory system"),

    # --- Chapter 11: K00–K95 ---
    ("K00", "K14", "Diseases of oral cavity and salivary glands"),
    ("K20", "K31", "Diseases of esophagus, stomach and duodenum"),
    ("K35", "K38", "Diseases of appendix"),
    ("K40", "K46", "Hernia"),
    ("K50", "K52", "Noninfective enteritis and colitis"),
    ("K55", "K64", "Other diseases of intestines"),
    ("K65", "K68", "Diseases of peritoneum and retroperitoneum"),
    ("K70", "K77", "Diseases of liver"),
    ("K80", "K87", "Disorders of gallbladder, biliary tract and pancreas"),
    ("K90", "K95", "Other diseases of the digestive system"),

    # --- Chapter 12: L00–L99 ---
    ("L00", "L08", "Infections of the skin and subcutaneous tissue"),
    ("L10", "L14", "Bullous disorders"),
    ("L20", "L30", "Dermatitis and eczema"),
    ("L40", "L45", "Papulosquamous disorders"),
    ("L49", "L54", "Urticaria and erythema"),
    ("L55", "L59", "Radiation-related disorders of the skin and subcutaneous tissue"),
    ("L60", "L75", "Disorders of skin appendages"),
    ("L76", "L76", "Intraoperative/postprocedural complications of skin/subcut. tissue"),
    ("L80", "L99", "Other disorders of the skin and subcutaneous tissue"),

    # --- Chapter 13: M00–M99 ---
    ("M00", "M25", "Arthropathies"),
    ("M26", "M27", "Dentofacial anomalies and other disorders of jaw"),
    ("M30", "M36", "Systemic connective tissue disorders"),
    ("M40", "M43", "Deforming dorsopathies"),
    ("M45", "M49", "Spondylopathies"),
    ("M50", "M54", "Other dorsopathies"),
    ("M60", "M63", "Soft tissue disorders"),
    ("M65", "M67", "Disorders of synovium and tendon"),
    ("M70", "M79", "Other soft tissue disorders"),
    ("M80", "M85", "Disorders of bone density and structure"),
    ("M86", "M90", "Other osteopathies"),
    ("M91", "M94", "Chondropathies"),
    ("M95", "M95", "Other acquired deformities of musculoskeletal system"),
    ("M96", "M96", "Intraoperative/postprocedural complications of musculoskeletal system"),
    ("M99", "M99", "Biomechanical lesions, not elsewhere classified"),

    # --- Chapter 14: N00–N99 ---
    ("N00", "N08", "Glomerular diseases"),
    ("N10", "N16", "Renal tubulo-interstitial diseases"),
    ("N17", "N19", "Acute kidney failure and chronic kidney disease"),
    ("N20", "N23", "Urolithiasis"),
    ("N25", "N29", "Other disorders of kidney and ureter"),
    ("N30", "N39", "Other diseases of urinary system"),
    ("N40", "N53", "Diseases of male genital organs"),
    ("N60", "N65", "Disorders of breast"),
    ("N70", "N77", "Inflammatory diseases of female pelvic organs"),
    ("N80", "N98", "Noninflammatory disorders of female genital tract"),
    ("N99", "N99", "Intraoperative/postprocedural complications of genitourinary system"),

    # --- Chapter 15: O00–O9A ---
    ("O00", "O08", "Pregnancy with abortive outcome"),
    ("O09", "O09", "Supervision of high-risk pregnancy"),
    ("O10", "O16", "Edema, proteinuria and hypertensive disorders in pregnancy/etc."),
    ("O20", "O29", "Other maternal disorders predominantly related to pregnancy"),
    ("O30", "O48", "Maternal care related to fetus, amniotic cavity, possible delivery problems"),
    ("O60", "O77", "Complications of labor and delivery"),
    ("O80", "O82", "Encounter for delivery"),
    ("O85", "O92", "Complications predominantly related to the puerperium"),
    ("O94", "O9A", "Other obstetric conditions, not elsewhere classified"),

    # --- Chapter 16: P00–P96 ---
    ("P00", "P04", "Newborn affected by maternal factors/complications of pregnancy/labor/delivery"),
    ("P05", "P08", "Disorders of newborn related to length of gestation and fetal growth"),
    ("P09", "P09", "Abnormal findings on neonatal screening"),
    ("P10", "P15", "Birth trauma"),
    ("P19", "P29", "Respiratory and cardiovascular disorders specific to the perinatal period"),
    ("P35", "P39", "Infections specific to the perinatal period"),
    ("P50", "P61", "Hemorrhagic and hematological disorders of newborn"),
    ("P70", "P74", "Transitory endocrine and metabolic disorders specific to newborn"),
    ("P76", "P78", "Perinatal disorders of digestive system"),
    ("P80", "P83", "Conditions involving the integument and temperature regulation of newborn"),
    ("P84", "P84", "Other problems with newborn"),
    ("P90", "P96", "Other disorders originating in the perinatal period"),

    # --- Chapter 17: Q00–Q99 ---
    ("Q00", "Q07", "Congenital malformations of the nervous system"),
    ("Q10", "Q18", "Congenital malformations of eye, ear, face and neck"),
    ("Q20", "Q28", "Congenital malformations of circulatory system"),
    ("Q30", "Q34", "Congenital malformations of the respiratory system"),
    ("Q35", "Q37", "Cleft lip and cleft palate"),
    ("Q38", "Q45", "Other congenital malformations of the digestive system"),
    ("Q50", "Q56", "Congenital malformations of genitals"),
    ("Q60", "Q64", "Congenital malformations of the urinary system"),
    ("Q65", "Q79", "Congenital malformations and deformations of the musculoskeletal system"),
    ("Q80", "Q89", "Other congenital malformations"),
    ("Q90", "Q99", "Chromosomal abnormalities, not elsewhere classified"),

    # --- Chapter 18: R00–R99 ---
    ("R00", "R09", "Symptoms/signs involving the circulatory and respiratory systems"),
    ("R10", "R19", "Symptoms/signs involving the digestive system and abdomen"),
    ("R20", "R23", "Symptoms/signs involving the skin and subcutaneous tissue"),
    ("R25", "R29", "Symptoms/signs involving the nervous and musculoskeletal systems"),
    ("R30", "R39", "Symptoms/signs involving the urinary system"),
    ("R40", "R46", "Symptoms/signs involving cognition, perception, emotional state/behavior"),
    ("R47", "R49", "Symptoms/signs involving speech and voice"),
    ("R50", "R69", "General symptoms and signs"),
    ("R70", "R79", "Abnormal findings on exam of blood, without diagnosis"),
    ("R80", "R82", "Abnormal findings on exam of urine, without diagnosis"),
    ("R83", "R89", "Abnormal findings on exam of other body fluids, substances, tissues"),
    ("R90", "R94", "Abnormal findings on diagnostic imaging and in function studies"),
    ("R97", "R97", "Abnormal tumor markers"),
    ("R99", "R99", "Ill-defined and unknown cause of mortality"),

    # --- Chapter 19: S00–T88 ---
    ("S00", "S09", "Injuries to the head"),
    ("S10", "S19", "Injuries to the neck"),
    ("S20", "S29", "Injuries to the thorax"),
    ("S30", "S39", "Injuries to the abdomen, lower back, lumbar spine, pelvis, ext. genitals"),
    ("S40", "S49", "Injuries to the shoulder and upper arm"),
    ("S50", "S59", "Injuries to the elbow and forearm"),
    ("S60", "S69", "Injuries to the wrist, hand and fingers"),
    ("S70", "S79", "Injuries to the hip and thigh"),
    ("S80", "S89", "Injuries to the knee and lower leg"),
    ("S90", "S99", "Injuries to the ankle and foot"),
    ("T07", "T07", "Unspecified multiple injuries"),
    ("T14", "T14", "Injury of unspecified body region"),
    ("T15", "T19", "Effects of foreign body entering through natural orifice"),
    ("T20", "T32", "Burns and corrosions"),
    ("T33", "T34", "Frostbite"),
    ("T36", "T50", "Poisoning by/adverse effect/underdosing of drugs, medicaments, biologics"),
    ("T51", "T65", "Toxic effects of substances chiefly nonmedicinal as to source"),
    ("T66", "T78", "Other and unspecified effects of external causes"),
    ("T79", "T79", "Certain early complications of trauma"),
    ("T80", "T88", "Complications of surgical and medical care, not elsewhere classified"),

    # --- Chapter 20: V00–Y99 ---
    ("V00", "V09", "Pedestrian injured in transport accident"),
    ("V10", "V19", "Pedalcyclist injured in transport accident"),
    ("V20", "V29", "Motorcycle rider injured in transport accident"),
    ("V30", "V39", "Occupant of three-wheeled motor vehicle injured in transport accident"),
    ("V40", "V49", "Car occupant injured in transport accident"),
    ("V50", "V59", "Occupant of pick-up truck or van injured in transport accident"),
    ("V60", "V69", "Occupant of heavy transport vehicle injured in transport accident"),
    ("V70", "V79", "Bus occupant injured in transport accident"),
    ("V80", "V89", "Other land transport accidents"),
    ("V90", "V94", "Water transport accidents"),
    ("V95", "V97", "Air and space transport accidents"),
    ("V98", "V99", "Other and unspecified transport accidents"),
    ("W00", "X59", "Other external causes of accidental injury"),
    ("X60", "X84", "Intentional self-harm"),
    ("X85", "Y09", "Assault"),
    ("Y10", "Y19", "Event of undetermined intent"),
    ("Y20", "Y29", "Sequelae"),
    ("Y30", "Y39", "Sequelae"),
    ("Y40", "Y84", "Complications of medical and surgical care"),
    ("Y85", "Y89", "Sequelae of external causes of morbidity"),
    ("Y90", "Y99", "Supplementary factors related to causes of morbidity classified elsewhere"),

    # --- Chapter 21: Z00–Z99 ---
    ("Z00", "Z13", "Persons encountering health services for examinations"),
    ("Z14", "Z15", "Genetic carrier and genetic susceptibility to disease"),
    ("Z16", "Z16", "Resistance to antimicrobial drugs"),
    ("Z17", "Z17", "Estrogen receptor status"),
    ("Z18", "Z18", "Retained foreign body fragments"),
    ("Z20", "Z28", "Persons with potential health hazards related to communicable diseases"),
    ("Z30", "Z39", "Persons encountering health services in circumstances related to reproduction"),
    ("Z40", "Z53", "Encounters for other specific health care"),
    ("Z55", "Z65", "Persons with potential health hazards related to socioeconomic/psychosocial circumstances"),
    ("Z66", "Z66", "Do not resuscitate status"),
    ("Z67", "Z67", "Blood type"),
    ("Z68", "Z68", "Body mass index [BMI]"),
    ("Z69", "Z76", "Persons encountering health services in other circumstances"),
    ("Z77", "Z99", "Persons with potential health hazards related to family/personal history/etc.")
]

def classify_icd_3_chars(icd_code: str) -> str:
    """
    Classify an ICD code by using the first 3 characters,
    and return a high-level category based on ICD_RANGE_MAP.
    """
    if not icd_code:
        return None
    # Extract and normalize the first 3 characters
    icd3 = icd_code.strip()[:3].upper()
    for start, end, description in icd10_blocks:
        # Lexicographical comparison (assumes codes are uniformly 3 characters)
        if start <= icd3 <= end:
            return description
    return "Unclassified"

def lookup_diagnosis_description(icd_code: str) -> str:
    """
    Given an ICD code, return its high-level classification based on the first 3 characters.
    """
    return classify_icd_3_chars(icd_code)

##############################################
# Example usage
##############################################
if __name__ == "__main__":
    # Replace these paths with the actual file paths on your system:
    # diagnosis_mapping_csv = "icd10cm_codes_2019.csv"
    procedure_mapping_csv = "cpt_code.csv"
    discharge_mapping_csv = "discharge_code.csv"  # Now using the CSV mapping

    # Load the mappings
    # load_diagnosis_mapping(diagnosis_mapping_csv)
    load_procedure_mapping(procedure_mapping_csv)
    load_discharge_mapping_csv(discharge_mapping_csv)
    
    # Test the lookup functions
    diag_code = "C089"
    diag_desc = lookup_diagnosis_description(diag_code)
    print(f"ICD-10 Code {diag_code}: {diag_desc}")
    
    proc_code = '86985'
    proc_desc = lookup_procedure_description(proc_code)
    print(f"CPT Code {proc_code}: {proc_desc}")

    disp_code = "70"
    disp_desc = interpret_disposition(disp_code)
    print(f"Discharge Code {disp_code}: {disp_desc}")
def convert_row_to_narrative(row):
    # 1. Demographics and Presentation
    age = row['AGE']
    gender = "female" if row['FEMALE'] == 1 else "male"
    # Determine race/ethnicity description
    race_code = row.get('RACE')
    hispanic_flag = row.get('HISPANIC')
    race_desc = ""
    if hispanic_flag == 1:
        race_desc = "Hispanic"
    if race_code:  # if race code is provided
        race_map = {1: "White", 2: "Black or African American", 3: "Hispanic or Latino", 4: "Asian or Pacific Islander", 5: "American Indian or Alaska Native", 6: "Other race", 7: "Unknown"}
        # Only add race if not already described as Hispanic
        if race_code in race_map and not race_desc:
            race_desc = race_map[race_code]
        elif race_code in race_map and race_desc: 
            # If patient is Hispanic and race is also given (like White or Black),
            # combine them e.g., "Hispanic White"
            if race_map[race_code] not in ["Hispanic", "other"]:
                race_desc = f"Hispanic {race_map[race_code]}"
    # Homeless status
    homeless = True if str(row.get('Homeless')).strip() == '1' else False
    # Weekend admission
    weekend = (row.get('AWEEKEND') == 1)
    # Point of origin or ED flag
    came_through_ed = (row.get('HCUP_ED') == 1)
    
    # Construct first sentence
    first_sentence = f"{age}-year-old"
    if homeless:
        first_sentence += " homeless"
    if race_desc:
        first_sentence += f" {race_desc}"
    first_sentence += f" {gender}"
    if came_through_ed:
        first_sentence += " who presented to the emergency department"
    if weekend:
        first_sentence += " on a weekend"
    # Add primary diagnosis/presentation
    primary_dx_code = row.get('I10_DX1')
    if primary_dx_code:
        primary_dx_desc = lookup_diagnosis_description(primary_dx_code)  # use a dictionary or mapping
        # if primary_dx_desc:
        first_sentence += f" with {primary_dx_desc} "
        # else:
        #     first_sentence += f" with diagnosis code {primary_dx_code}"
    first_sentence += "."
    
    # 2. Additional Diagnoses
    additional_dx = []
    for dx_col in [col for col in row.keys() if col.startswith('I10_DX') and col != 'I10_DX1']:
        dx_code = row[dx_col]
        if pd.notnull(dx_code) and str(dx_code).strip():
            if isinstance(dx_code, str) and dx_code.strip() in ["", "."]:
                continue
            dx_desc = lookup_diagnosis_description(dx_code)
            if dx_desc:
                additional_dx.append(dx_desc)
            else:
                additional_dx.append(f"ICD-10 code {dx_code}")
    dx_sentence = ""
    if additional_dx:
        # Combine all additional diagnoses in a sentence.
        dx_list_text = ", ".join(additional_dx)
        dx_sentence = f"Additional diagnoses include {dx_list_text}."
    
    # 3. Procedures and treatments
    procedures = []
    for cpt_col in [col for col in row.keys() if col.startswith('CPT')]:
        cpt_code = str(row[cpt_col]).strip()
        if cpt_code == "" or cpt_code.lower() == "nan":
            continue
        proc_desc = lookup_procedure_description(cpt_code)
        if proc_desc:
            procedures.append(proc_desc)
        else:
            procedures.append(f"CPT {cpt_code}")
    proc_sentence = ""
    if procedures:
        # Group or join procedures logically. For simplicity, join all in one sentence here:
        proc_list_text = ", ".join(procedures)
        proc_sentence = f"Procedures and services provided during the visit included: {proc_list_text}."
    
    # 4. Outcome and Disposition
    outcome_parts = []
    los = row.get('LOS')
    if pd.notnull(los):
        if los == 0:
            outcome_parts.append("treated and released the same day")
        elif los == 1:
            outcome_parts.append("hospitalized for 1 day")
        else:
            outcome_parts.append(f"hospitalized for {int(los)} days")
    if row.get('DIED') == 1:
        outcome_parts.append("the patient died during hospitalization")
    else:
        # use disposition if available
        disp = row.get('DISPUNIFORM') or row.get('DISPUB04')
        if disp:
            disp_desc = interpret_disposition(disp)  # e.g., 1 -> discharged home, 2-> transfer, etc.
            if disp_desc:
                outcome_parts.append(f"discharged {disp_desc}")
    if row.get('TOTCHG'):
        outcome_parts.append(f"with total charges of ${row.get('TOTCHG')}")
    outcome_sentence = ""
    if outcome_parts:
        outcome_sentence = "The patient was " + ", ".join(outcome_parts) + "."
    
    # 5. Social context and readmission
    social_parts = []
    # Payer info
    raw_pay1 = str(row.get('PAY1') or "").strip()
    pay1 = int(raw_pay1) if raw_pay1.isdigit() else None
    raw_pay2 = str(row.get('PAY2') or "").strip()
    pay2 = int(raw_pay2) if raw_pay2.isdigit() else None
    raw_pay3 = str(row.get('PAY3') or "").strip()
    pay3 = int(raw_pay3) if raw_pay3.isdigit() else None
    payer_map = {1: "Medicare", 2: "Medicaid", 3: "Private insurance", 4: "self-pay", 5: "no charge", 6: "other"}
    if pay1 in payer_map:
        payer_text = payer_map[pay1]
        if pay1 == 4 or pay1 == 5:
            social_parts.append(f"no insurance ({payer_text})")
        else:
            social_parts.append(f"{payer_text} as primary insurance")
    if pay2 in payer_map:
        social_parts.append(f"{payer_map[pay2]} as secondary")
    if pay3 in payer_map:
        social_parts.append(f"{payer_map[pay3]} as tertiary")
    # Income quartile
    inc_quart = row.get('MEDINCSTQ')
    raw_inc_quart = str(row.get('MEDINCSTQ') or "").strip()
    inc_quart = int(raw_inc_quart) if raw_inc_quart.isdigit() else None
    if inc_quart in [1, 2, 3, 4]:
        quart_map = {1: "the lowest", 2: "the second", 3: "the third", 4: "the highest"}
        social_parts.append(f"resides in {quart_map[inc_quart]} income quartile area")
    # Readmission context
    readm_count = row.get('Readmission_Count')
    readm_days = row.get('Readmission_Days')
    if readm_count and pd.notnull(readm_count) and int(readm_count) > 1:
        if readm_days and int(readm_days) >= 0:
            social_parts.append(f"had a readmission {int(readm_days)} days later (total {int(readm_count)} visits in year)")
        else:
            social_parts.append(f"had {int(readm_count)} hospital visits in the year")
    social_sentence = ""
    if social_parts:
        social_sentence = "He " + ", ".join(social_parts) + "."
    
    # 6. Combine all parts
    narrative = " ".join(part for part in [first_sentence, dx_sentence, proc_sentence, outcome_sentence, social_sentence] if part)
    return narrative

df=pd.read_csv("original_data.csv")
df['prompt']= df.apply(convert_row_to_narrative, axis=1)
df.to_csv('modified_data.csv')