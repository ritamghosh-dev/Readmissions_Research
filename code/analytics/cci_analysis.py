import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# --- Global style: Times New Roman @ 12 pt ---
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 18
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18

# --- Configuration ---
INPUT_CSV_PATH = 'data/preprocessed_data.csv'   # Must contain CCI_Score
COLUMN_TO_ANALYZE = 'CCI_Score'
OUTPUT_DIR = 'results/analytics'
OUTPUT_FILENAME = 'cci_score_distribution_detailed.png'  # PNG as requested

def analyze_cci_distribution():
    """
    Loads the preprocessed dataset, analyzes the distribution of CCI_Score,
    and saves a 300-dpi PNG horizontal bar chart with bold percentage labels.
    """
    print(f"--- Starting Analysis of '{COLUMN_TO_ANALYZE}' Distribution (Detailed) ---")

    # 1) Load data
    print(f"Loading data from {INPUT_CSV_PATH}...")
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df = pd.read_csv(INPUT_CSV_PATH, low_memory=False)
        print(f"Loaded {len(df)} records successfully.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_CSV_PATH}")
        print("Please ensure you have run 'preprocessing.py' to generate this file.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if COLUMN_TO_ANALYZE not in df.columns:
        print(f"Error: Column '{COLUMN_TO_ANALYZE}' not found in the dataset.")
        return

    # 2) Clean and group CCI
    print(f"Grouping '{COLUMN_TO_ANALYZE}' values into descriptive categories...")
    df[COLUMN_TO_ANALYZE] = pd.to_numeric(df[COLUMN_TO_ANALYZE], errors='coerce')

    def group_cci_detailed(score):
        if pd.isna(score):
            return "Unknown Score"
        score = int(score)
        if score == 0:
            return "0 (No Comorbidities)"
        if score == 1:
            return "1"
        if score == 2:
            return "2"
        if score == 3:
            return "3"
        if score == 4:
            return "4"
        if 5 <= score <= 6:
            return "5-6"
        if 7 <= score <= 9:
            return "7-9"
        return "10+ (Very High Risk)"

    df['cci_group'] = df[COLUMN_TO_ANALYZE].apply(group_cci_detailed)

    # 3) Percentages and order
    percentage_distribution = df['cci_group'].value_counts(normalize=True) * 100
    category_order = [
        "0 (No Comorbidities)", "1", "2", "3", "4",
        "5-6", "7-9", "10+ (Very High Risk)", "Unknown Score"
    ]
    percentage_distribution = percentage_distribution.reindex(category_order).dropna()

    print("\nPercentage Distribution of Encounters by CCI Score:")
    print(percentage_distribution.to_string())

    # 4) Plot
    print("\nGenerating visualization...")
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 10))

    ax = sns.barplot(
        x=percentage_distribution.values,
        y=percentage_distribution.index,
        palette="plasma",
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
    ax.set_title('Distribution of Patient Encounters by CCI Score', pad=20)
    ax.set_xlabel('Percentage of Total Encounters (%)')
    ax.set_ylabel('CCI Score Group')

    # Give space for labels to the right
    plt.xlim(0, max(percentage_distribution.values) * 1.15)

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    # 5) Save as 300-dpi PNG
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)  # .png extension
    try:
        plt.savefig(output_path, bbox_inches='tight', dpi=300, format='png')
        print(f"Successfully saved plot to: {output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close()

if __name__ == "__main__":
    analyze_cci_distribution()