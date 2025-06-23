import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
# Path to your raw data file
INPUT_CSV_PATH = 'data/NY2019_Ritam.csv'
# Directory to save the output plot
OUTPUT_DIR = 'results/analytics'
OUTPUT_FILENAME = 'top_5_icd_categories.png'

# --- ICD-10 Block Mapping (as provided) ---
ICD10_BLOCKS = [
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

def classify_icd_category(icd_code: str) -> str:
    """ Classify an ICD code into its high-level block description using its first 3 characters. """
    if not isinstance(icd_code, str) or not icd_code.strip():
        return "Unknown or Invalid"
    
    icd3 = icd_code.strip()[:3].upper()
    
    if len(icd3) < 3:
        return "Unknown or Invalid"
        
    for start, end, description in ICD10_BLOCKS:
        # Special handling for 'O9A' which is not a standard numeric comparison
        if start == "O00" and end == "O9A":
            if icd3.startswith("O") and (icd3[1:].isdigit() or icd3[1] == '9' and icd3[2] == 'A'):
                if "O00" <= icd3 <= "O9A":
                     return description
        elif start <= icd3 <= end:
            return description
            
    return "Unclassified"

def analyze_top_icd_categories():
    """
    Loads data, classifies all ICD codes into high-level categories,
    and displays the top 5 most common categories.
    """
    print("--- Starting Top 5 ICD-10 Code Category Analysis ---")
    
    # --- 1. Load Data ---
    print(f"Loading data from {INPUT_CSV_PATH}...")
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df = pd.read_csv(INPUT_CSV_PATH, low_memory=False)
        print(f"Loaded {len(df)} records successfully.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_CSV_PATH}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- 2. Gather, Clean, and Classify All Diagnosis Codes ---
    print("Gathering, cleaning, and classifying all diagnosis codes...")
    
    icd_cols = [f'I10_DX{i}' for i in range(1, 35)]
    existing_icd_cols = [col for col in icd_cols if col in df.columns]
    
    if not existing_icd_cols:
        print("Error: No diagnosis columns (I10_DX*) found.")
        return
        
    # Melt the dataframe to get a single series of all diagnosis codes
    all_codes = df[existing_icd_cols].melt(value_name='ICD_Code')['ICD_Code']
    
    # Clean the series: drop nulls/NaNs and convert to string
    all_codes = all_codes.dropna().astype(str)
    
    # Map each code to its high-level category description
    all_categories = all_codes.apply(classify_icd_category)

    # Get the top 5 most common categories
    top_5_categories = all_categories.value_counts().nlargest(5)
    
    print("Found top 5 most frequent categories.")

    # --- 3. Format Output ---
    results_df = top_5_categories.reset_index()
    results_df.columns = ['ICD Category', 'Count']
    results_df['Rank'] = range(1, len(results_df) + 1)
    results_df = results_df[['Rank', 'ICD Category', 'Count']]

    # --- 4. Display Results Table ---
    print("\n" + "="*80)
    print("      Top 5 Most Common ICD-10 Diagnosis Categories in the Dataset")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80 + "\n")

    # --- 5. Generate and Save Visualization ---
    print("Generating visualization...")
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))

    ax = sns.barplot(x='Count', y='ICD Category', data=results_df, palette="crest_r", orient='h')

    for index, row in results_df.iterrows():
        ax.text(row['Count'], index, f" {row['Count']:,}", va='center', ha='left', fontsize=12, color='black')

    plt.title('Top 5 Most Frequent ICD-10 Diagnosis Categories', fontsize=18, pad=20)
    plt.xlabel('Total Count Across All Diagnosis Fields', fontsize=14)
    plt.ylabel('ICD-10 Category', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.xlim(0, top_5_categories.max() * 1.1)
    sns.despine(left=True, bottom=True)
    
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    try:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Successfully saved plot to: {output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    plt.close()


if __name__ == "__main__":
    analyze_top_icd_categories()
