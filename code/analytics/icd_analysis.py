import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# --- Global style: Times New Roman @ 18 pt ---
# If Times New Roman isn't available on your system/VM, matplotlib will fall back.
# You can switch to "Nimbus Roman No9 L" if needed.
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 18
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18

# --- Configuration ---
INPUT_CSV_PATH = 'data/NY2019_Ritam.csv'
COLUMN_TO_ANALYZE = 'I10_NDX'
OUTPUT_DIR = 'results/analytics'
OUTPUT_FILENAME = 'icd_distribution_simplified.png'  # PNG output

def analyze_ndx_distribution_simplified():
    """
    Loads the dataset, analyzes the distribution of I10_NDX with simplified
    groups (0–5 vs 6+), and saves a 300-dpi PNG bar chart (font 18, bold labels).
    """
    print(f"--- Starting Analysis of '{COLUMN_TO_ANALYZE}' Distribution (Simplified) ---")

    # 1) Load data
    print(f"Loading data from {INPUT_CSV_PATH}...")
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df = pd.read_csv(INPUT_CSV_PATH, low_memory=False)
        print(f"Loaded {len(df)} records successfully.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_CSV_PATH}")
        print("Please ensure the file path is correct.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if COLUMN_TO_ANALYZE not in df.columns:
        print(f"Error: Column '{COLUMN_TO_ANALYZE}' not found in the dataset.")
        return

    # 2) Group I10_NDX into two categories
    print(f"Grouping '{COLUMN_TO_ANALYZE}' values into two categories...")
    df[COLUMN_TO_ANALYZE] = pd.to_numeric(df[COLUMN_TO_ANALYZE], errors='coerce')

    def group_ndx_simplified(ndx):
        if pd.isna(ndx) or ndx < 0:
            return "Unknown"
        return "0–5 Diagnoses" if ndx <= 5 else "6+ Diagnoses"

    df['ndx_group'] = df[COLUMN_TO_ANALYZE].apply(group_ndx_simplified)

    # 3) Percentages and fixed order
    percentage_distribution = df['ndx_group'].value_counts(normalize=True) * 100
    percentage_distribution = percentage_distribution.reindex(
        ["0–5 Diagnoses", "6+ Diagnoses", "Unknown"]
    ).dropna()

    print("\nPercentage Distribution of Encounters by Number of ICD Diagnoses:")
    print(percentage_distribution.to_string())

    # 4) Plot
    print("\nGenerating visualization...")
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))

    ax = sns.barplot(
        x=percentage_distribution.values,
        y=percentage_distribution.index,
        palette="viridis_r",
        orient='h'
    )

    # Bold percentage labels on each bar
    for idx, value in enumerate(percentage_distribution.values):
        ax.text(
            value, idx, f' {value:.2f}%',
            va='center', ha='left',
            fontsize=18, fontweight='bold', color='black'
        )

    # Titles and labels
    ax.set_title('Distribution of Encounters by Number of ICD Diagnoses', pad=20)
    ax.set_xlabel('Percentage of Total Encounters (%)')
    ax.set_ylabel('Number of Diagnoses Recorded')

    # Keep axis as 0–100% with a bit of headroom
    plt.xlim(0, max(100, max(percentage_distribution.values) * 1.12))

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    # 5) Save as 300-dpi PNG
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    try:
        plt.savefig(output_path, bbox_inches='tight', dpi=300, format='png')
        print(f"Successfully saved plot to: {output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close()

if __name__ == "__main__":
    analyze_ndx_distribution_simplified()