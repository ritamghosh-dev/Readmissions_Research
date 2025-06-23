# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# # --- Configuration ---
# # Path to your raw data file
# INPUT_CSV_PATH = 'data/NY2019_Ritam.csv'
# # Column to analyze
# COLUMN_TO_ANALYZE = 'NCPT'
# # Directory to save the output plot
# OUTPUT_DIR = 'results/analytics'
# OUTPUT_FILENAME = 'ncpt_distribution.png'

# def analyze_ncpt_distribution():
#     """
#     Loads the dataset, analyzes the distribution of the NCPT column,
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

#     # --- 2. Group the NCPT values into meaningful bins ---
#     print(f"Grouping '{COLUMN_TO_ANALYZE}' values into categories...")
    
#     # Convert column to numeric, coercing errors to NaN
#     df[COLUMN_TO_ANALYZE] = pd.to_numeric(df[COLUMN_TO_ANALYZE], errors='coerce')
    
#     # Define a function to map counts to categories
#     def group_ncpt(ncpt):
#         if pd.isna(ncpt) or ncpt < 0:
#             return "Unknown"
#         if ncpt == 0:
#             return "0 Procedures"
#         if ncpt == 1:
#             return "1 Procedure"
#         if ncpt == 2:
#             return "2 Procedures"
#         if ncpt == 3:
#             return "3 Procedures"
#         if 4 <= ncpt <= 5:
#             return "4-5 Procedures"
#         if 6 <= ncpt <= 10:
#             return "6-10 Procedures"
#         else: # ncpt > 10
#             return "11+ Procedures"

#     # Apply the function to create a new category column
#     df['ncpt_group'] = df[COLUMN_TO_ANALYZE].apply(group_ncpt)

#     # --- 3. Calculate Percentages ---
#     # Calculate the percentage of the total for each group
#     percentage_distribution = df['ncpt_group'].value_counts(normalize=True) * 100
#     percentage_distribution = percentage_distribution.sort_values() # Sort for better visualization

#     print("\nPercentage Distribution of Encounters by Number of CPT Codes:")
#     print(percentage_distribution.to_string())

#     # --- 4. Generate and Save Visualization ---
#     print("\nGenerating visualization...")
    
#     # Set plot style
#     sns.set_style("whitegrid")
#     plt.figure(figsize=(12, 8))

#     # Create horizontal bar plot
#     ax = sns.barplot(x=percentage_distribution.values, y=percentage_distribution.index, palette="viridis", orient='h')

#     # Add percentage labels to each bar
#     for index, value in enumerate(percentage_distribution.values):
#         ax.text(value, index, f' {value:.2f}%', va='center', ha='left', fontsize=12, color='black')

#     # Set titles and labels for clarity
#     plt.title(f'Distribution of Patient Encounters by Number of CPT Codes', fontsize=18, pad=20)
#     plt.xlabel('Percentage of Total Encounters (%)', fontsize=14)
#     plt.ylabel('Number of Procedures Recorded', fontsize=14)
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
#     analyze_ncpt_distribution()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
# Path to your raw data file
INPUT_CSV_PATH = 'data/NY2019_Ritam.csv'
# Column to analyze
COLUMN_TO_ANALYZE = 'NCPT'
# Directory to save the output plot
OUTPUT_DIR = 'results/analytics'
OUTPUT_FILENAME = 'ncpt_distribution_simplified.png'

def analyze_ncpt_distribution():
    """
    Loads the dataset, analyzes the distribution of the NCPT column,
    and saves the visualization as a bar chart with simplified groups.
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

    # --- 2. Group the NCPT values into two simplified bins ---
    print(f"Grouping '{COLUMN_TO_ANALYZE}' values into two categories...")
    
    # Convert column to numeric, coercing errors to NaN
    df[COLUMN_TO_ANALYZE] = pd.to_numeric(df[COLUMN_TO_ANALYZE], errors='coerce')
    
    # Define a function to map counts to the two categories
    def group_ncpt_simplified(ncpt):
        if pd.isna(ncpt) or ncpt < 0:
            return "Unknown" # Keep unknown as a separate category
        if ncpt <= 5:
            return "0-5 Procedures"
        else: # ncpt > 5
            return "6+ Procedures"

    # Apply the function to create a new category column
    df['ncpt_group'] = df[COLUMN_TO_ANALYZE].apply(group_ncpt_simplified)

    # --- 3. Calculate Percentages ---
    # Calculate the percentage of the total for each group
    percentage_distribution = df['ncpt_group'].value_counts(normalize=True) * 100
    # Sort by index to maintain a consistent order (e.g., 0-5 then 6+)
    percentage_distribution = percentage_distribution.sort_index()

    print("\nPercentage Distribution of Encounters by Number of CPT Codes:")
    print(percentage_distribution.to_string())

    # --- 4. Generate and Save Visualization ---
    print("\nGenerating visualization...")
    
    # Set plot style
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    # Create horizontal bar plot
    ax = sns.barplot(x=percentage_distribution.values, y=percentage_distribution.index, palette="viridis", orient='h')

    # Add percentage labels to each bar
    for index, value in enumerate(percentage_distribution.values):
        ax.text(value, index, f' {value:.2f}%', va='center', ha='left', fontsize=12, color='black')

    # Set titles and labels for clarity
    plt.title(f'Distribution of Encounters by Number of CPT Codes', fontsize=18, pad=20)
    plt.xlabel('Percentage of Total Encounters (%)', fontsize=14)
    plt.ylabel('Number of Procedures Recorded', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Adjust plot limits to give space for labels
    plt.xlim(0, 100) # Set x-limit to 100 for percentages
    
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
    analyze_ncpt_distribution()
