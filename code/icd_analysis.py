# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# # --- Configuration ---
# # Path to your raw data file
# INPUT_CSV_PATH = 'data/NY2019_Ritam.csv'
# # Column to analyze - CHANGED
# COLUMN_TO_ANALYZE = 'I10_NDX'
# # Directory to save the output plot
# OUTPUT_DIR = 'results/analytics'
# # Output filename - CHANGED
# OUTPUT_FILENAME = 'icd_distribution.png'

# def analyze_ndx_distribution():
#     """
#     Loads the dataset, analyzes the distribution of the I10_NDX column,
#     and saves the visualization as a bar chart.
#     """
#     print(f"--- Starting Analysis of '{COLUMN_TO_ANALYZE}' Distribution ---")
    
#     # --- 1. Load Data ---
#     print(f"Loading data from {INPUT_CSV_PATH}...")
#     try:
#         os.makedirs(OUTPUT_DIR, exist_ok=True)
#         # Use low_memory=False for mixed-type columns, which is common in large datasets
#         df = pd.read_csv(INPUT_CSV_PATH, low_memory=False)
#         print(f"Loaded {len(df)} records successfully.")
#     except FileNotFoundError:
#         print(f"Error: Input file not found at {INPUT_CSV_PATH}")
#         print("Please ensure the file path is correct.")
#         return
#     except Exception as e:
#         print(f"Error loading data: {e}")
#         return

#     # Check if the column exists
#     if COLUMN_TO_ANALYZE not in df.columns:
#         print(f"Error: Column '{COLUMN_TO_ANALYZE}' not found in the dataset.")
#         return

#     # --- 2. Group the I10_NDX values into meaningful bins ---
#     print(f"Grouping '{COLUMN_TO_ANALYZE}' values into categories...")
    
#     # Convert column to numeric, coercing errors to NaN
#     df[COLUMN_TO_ANALYZE] = pd.to_numeric(df[COLUMN_TO_ANALYZE], errors='coerce')
    
#     # Define a function to map counts to categories appropriate for diagnoses
#     def group_ndx(ndx):
#         if pd.isna(ndx) or ndx < 0:
#             return "Unknown"
#         if ndx == 0:
#             return "0 Diagnoses"
#         if 1 <= ndx <= 3:
#             return "1-3 Diagnoses"
#         if 4 <= ndx <= 6:
#             return "4-6 Diagnoses"
#         if 7 <= ndx <= 10:
#             return "7-10 Diagnoses"
#         if 11 <= ndx <= 15:
#             return "11-15 Diagnoses"
#         else: # ndx > 15
#             return "16+ Diagnoses"

#     # Apply the function to create a new category column
#     df['ndx_group'] = df[COLUMN_TO_ANALYZE].apply(group_ndx)

#     # --- 3. Calculate Percentages ---
#     # Calculate the percentage of the total for each group
#     percentage_distribution = df['ndx_group'].value_counts(normalize=True) * 100
#     # Define a specific order for the categories for logical presentation
#     category_order = ["0 Diagnoses", "1-3 Diagnoses", "4-6 Diagnoses", "7-10 Diagnoses", "11-15 Diagnoses", "16+ Diagnoses", "Unknown"]
#     percentage_distribution = percentage_distribution.reindex(category_order).dropna()

#     print("\nPercentage Distribution of Encounters by Number of ICD Diagnoses:")
#     print(percentage_distribution.to_string())

#     # --- 4. Generate and Save Visualization ---
#     print("\nGenerating visualization...")
    
#     # Set plot style
#     sns.set_style("whitegrid")
#     plt.figure(figsize=(12, 8))

#     # Create horizontal bar plot
#     ax = sns.barplot(x=percentage_distribution.values, y=percentage_distribution.index, palette="viridis_r", orient='h')

#     # Add percentage labels to each bar
#     for index, value in enumerate(percentage_distribution.values):
#         ax.text(value, index, f' {value:.2f}%', va='center', ha='left', fontsize=12, color='black')

#     # Set titles and labels for clarity
#     plt.title(f'Distribution of Encounters by Number of ICD Diagnoses', fontsize=18, pad=20)
#     plt.xlabel('Percentage of Total Encounters (%)', fontsize=14)
#     plt.ylabel('Number of Diagnoses Recorded', fontsize=14)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
    
#     # Adjust plot limits to give space for labels
#     plt.xlim(0, max(percentage_distribution.values) * 1.1)
    
#     # Remove unnecessary spines
#     sns.despine(left=True, bottom=True)

#     # Save the plot
#     output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
#     try:
#         plt.savefig(output_path, bbox_inches='tight')
#         print(f"Successfully saved plot to: {output_path}")
#     except Exception as e:
#         print(f"Error saving plot: {e}")
    
#     plt.close()


# if __name__ == "__main__":
#     analyze_ndx_distribution()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
# Path to your raw data file
INPUT_CSV_PATH = 'data/NY2019_Ritam.csv'
# Column to analyze - CHANGED
COLUMN_TO_ANALYZE = 'I10_NDX'
# Directory to save the output plot
OUTPUT_DIR = 'results/analytics'
# Output filename - CHANGED
OUTPUT_FILENAME = 'icd_distribution_simplified.png'

def analyze_ndx_distribution_simplified():
    """
    Loads the dataset, analyzes the distribution of the I10_NDX column
    with simplified groups (0-5 and 6+), and saves the visualization.
    """
    print(f"--- Starting Analysis of '{COLUMN_TO_ANALYZE}' Distribution (Simplified) ---")
    
    # --- 1. Load Data ---
    print(f"Loading data from {INPUT_CSV_PATH}...")
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # Use low_memory=False for mixed-type columns, which is common in large datasets
        df = pd.read_csv(INPUT_CSV_PATH, low_memory=False)
        print(f"Loaded {len(df)} records successfully.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_CSV_PATH}")
        print("Please ensure the file path is correct.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Check if the column exists
    if COLUMN_TO_ANALYZE not in df.columns:
        print(f"Error: Column '{COLUMN_TO_ANALYZE}' not found in the dataset.")
        return

    # --- 2. Group the I10_NDX values into two simplified bins ---
    print(f"Grouping '{COLUMN_TO_ANALYZE}' values into two categories...")
    
    # Convert column to numeric, coercing errors to NaN
    df[COLUMN_TO_ANALYZE] = pd.to_numeric(df[COLUMN_TO_ANALYZE], errors='coerce')
    
    # Define a function to map counts to the two simplified categories
    def group_ndx_simplified(ndx):
        if pd.isna(ndx) or ndx < 0:
            return "Unknown"
        if ndx <= 5:
            return "0-5 Diagnoses"
        else: # ndx > 5
            return "6+ Diagnoses"

    # Apply the function to create a new category column
    df['ndx_group'] = df[COLUMN_TO_ANALYZE].apply(group_ndx_simplified)

    # --- 3. Calculate Percentages ---
    # Calculate the percentage of the total for each group
    percentage_distribution = df['ndx_group'].value_counts(normalize=True) * 100
    # Sort by index to maintain a consistent order
    percentage_distribution = percentage_distribution.sort_index()

    print("\nPercentage Distribution of Encounters by Number of ICD Diagnoses:")
    print(percentage_distribution.to_string())

    # --- 4. Generate and Save Visualization ---
    print("\nGenerating visualization...")
    
    # Set plot style
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    # Create horizontal bar plot
    ax = sns.barplot(x=percentage_distribution.values, y=percentage_distribution.index, palette="viridis_r", orient='h')

    # Add percentage labels to each bar
    for index, value in enumerate(percentage_distribution.values):
        ax.text(value, index, f' {value:.2f}%', va='center', ha='left', fontsize=12, color='black')

    # Set titles and labels for clarity
    plt.title(f'Distribution of Encounters by Number of ICD Diagnoses', fontsize=18, pad=20)
    plt.xlabel('Percentage of Total Encounters (%)', fontsize=14)
    plt.ylabel('Number of Diagnoses Recorded', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Adjust plot limits to give space for labels
    plt.xlim(0, 100)
    
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
    analyze_ndx_distribution_simplified()
