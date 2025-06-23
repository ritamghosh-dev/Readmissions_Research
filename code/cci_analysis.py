import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
# Path to your preprocessed data file, which contains the CCI_Score
INPUT_CSV_PATH = 'data/preprocessed_data.csv'
# Column to analyze
COLUMN_TO_ANALYZE = 'CCI_Score'
# Directory to save the output plot
OUTPUT_DIR = 'results/analytics'
# Output filename
OUTPUT_FILENAME = 'cci_score_distribution_detailed.png'

def analyze_cci_distribution():
    """
    Loads the preprocessed dataset, analyzes the distribution of the CCI_Score column,
    and saves the visualization as a bar chart with more descriptive risk levels.
    """
    print(f"--- Starting Analysis of '{COLUMN_TO_ANALYZE}' Distribution (Detailed) ---")
    
    # --- 1. Load Data ---
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

    # Check if the column exists
    if COLUMN_TO_ANALYZE not in df.columns:
        print(f"Error: Column '{COLUMN_TO_ANALYZE}' not found in the dataset.")
        return

    # --- 2. Group the CCI_Score values into more descriptive bins ---
    print(f"Grouping '{COLUMN_TO_ANALYZE}' values into more descriptive categories...")
    
    # Convert column to numeric, coercing errors to NaN
    df[COLUMN_TO_ANALYZE] = pd.to_numeric(df[COLUMN_TO_ANALYZE], errors='coerce')
    
    # Define a function to map scores to more granular categories
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
        else: # score >= 10
            return "10+ (Very High Risk)"

    # Apply the function to create a new category column
    df['cci_group'] = df[COLUMN_TO_ANALYZE].apply(group_cci_detailed)

    # --- 3. Calculate Percentages ---
    # Calculate the percentage of the total for each group
    percentage_distribution = df['cci_group'].value_counts(normalize=True) * 100
    # Define a specific order for the categories for logical presentation
    category_order = ["0 (No Comorbidities)", "1", "2", "3", "4", "5-6", "7-9", "10+ (Very High Risk)", "Unknown Score"]
    percentage_distribution = percentage_distribution.reindex(category_order).dropna()

    print("\nPercentage Distribution of Encounters by CCI Score:")
    print(percentage_distribution.to_string())

    # --- 4. Generate and Save Visualization ---
    print("\nGenerating visualization...")
    
    # Set plot style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 10)) # Increased height for more categories

    # Create horizontal bar plot
    ax = sns.barplot(x=percentage_distribution.values, y=percentage_distribution.index, palette="plasma", orient='h')

    # Add percentage labels to each bar
    for index, value in enumerate(percentage_distribution.values):
        ax.text(value, index, f' {value:.2f}%', va='center', ha='left', fontsize=12, color='black')

    # Set titles and labels for clarity
    plt.title(f'Distribution of Patient Encounters by CCI Score', fontsize=18, pad=20)
    plt.xlabel('Percentage of Total Encounters (%)', fontsize=14)
    plt.ylabel('CCI Score Group', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Adjust plot limits to give space for labels
    plt.xlim(0, max(percentage_distribution.values) * 1.15)
    
    # Remove unnecessary spines
    sns.despine(left=True, bottom=True)

    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    try:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Successfully saved plot to: {output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    plt.close()


if __name__ == "__main__":
    analyze_cci_distribution()
