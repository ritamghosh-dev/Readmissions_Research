import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# --- Global style: Times New Roman @ 18 pt ---
# If Times New Roman isn't available, matplotlib will fall back automatically.
# (You can swap to "Nimbus Roman No9 L" if needed.)
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 18
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18

# --- Configuration ---
INPUT_CSV_PATH = 'data/NY2019_Ritam.csv'      # Must contain NCPT
COLUMN_TO_ANALYZE = 'NCPT'
OUTPUT_DIR = 'results/analytics'
OUTPUT_FILENAME = 'ncpt_distribution_simplified.png'  # PNG as requested

def analyze_ncpt_distribution():
    """
    Loads the dataset, analyzes the distribution of the NCPT column,
    and saves a 300-dpi PNG bar chart with bold percentage labels (font 18).
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

    # 2) Group NCPT into two categories
    print(f"Grouping '{COLUMN_TO_ANALYZE}' values into two categories...")
    df[COLUMN_TO_ANALYZE] = pd.to_numeric(df[COLUMN_TO_ANALYZE], errors='coerce')

    def group_ncpt_simplified(ncpt):
        if pd.isna(ncpt) or ncpt < 0:
            return "Unknown"
        return "0–5 Procedures" if ncpt <= 5 else "6+ Procedures"

    df['ncpt_group'] = df[COLUMN_TO_ANALYZE].apply(group_ncpt_simplified)

    # 3) Percentages and order
    percentage_distribution = df['ncpt_group'].value_counts(normalize=True) * 100
    percentage_distribution = percentage_distribution.reindex(["0–5 Procedures", "6+ Procedures", "Unknown"]).dropna()

    print("\nPercentage Distribution of Encounters by Number of CPT Codes:")
    print(percentage_distribution.to_string())

    # 4) Plot
    print("\nGenerating visualization...")
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))  # slightly wider

    ax = sns.barplot(
        x=percentage_distribution.values,
        y=percentage_distribution.index,
        palette="viridis",
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
    ax.set_title('Distribution of Encounters by Number of CPT Codes', pad=20)
    ax.set_xlabel('Percentage of Total Encounters (%)')
    ax.set_ylabel('Number of Procedures Recorded')

    # x-axis as 0–100% with a little space for labels
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
    analyze_ncpt_distribution()