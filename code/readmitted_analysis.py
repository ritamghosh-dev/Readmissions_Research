import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
# Path to your preprocessed data file, which contains the final 'Readmitted' target column
INPUT_CSV_PATH = 'data/preprocessed_data.csv'
# Column to analyze
TARGET_COLUMN = 'Readmitted30'
# Directory to save the output plot
OUTPUT_DIR = 'results/analytics'
# Output filename
OUTPUT_FILENAME = 'readmission_ratio_pie_chart.png'

def analyze_readmission_ratio():
    """
    Loads the preprocessed dataset, analyzes the distribution of the 
    readmission target variable, and saves the visualization as a pie chart.
    """
    print(f"--- Starting Analysis of '{TARGET_COLUMN}' Distribution ---")
    
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
    if TARGET_COLUMN not in df.columns:
        print(f"Error: Column '{TARGET_COLUMN}' not found in the dataset.")
        return

    # --- 2. Calculate Value Counts and Percentages ---
    print(f"Calculating distribution for '{TARGET_COLUMN}'...")
    
    class_counts = df[TARGET_COLUMN].value_counts()
    
    # Define labels for the pie chart
    labels = {
        0: 'Not Readmitted',
        1: 'Readmitted'
    }
    class_labels = class_counts.index.map(labels)
    
    print("\nDistribution of Encounters by Readmission Status:")
    print(class_counts)

    # --- 3. Generate and Save Visualization ---
    print("\nGenerating pie chart...")
    
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8))

    # Define colors and explode properties for a beautiful chart
    colors = sns.color_palette('pastel')[0:2]
    explode = (0, 0.1)  # Explode the 'Readmitted' slice slightly to highlight it

    # Create the pie chart
    plt.pie(
        class_counts, 
        labels=class_labels, 
        colors=colors, 
        autopct='%.1f%%', # Format percentages to one decimal place
        shadow=True, 
        startangle=140,
        explode=explode,
        textprops={'fontsize': 14}
    )

    # Set title
    plt.title('Ratio of 30-Day Hospital Readmissions', fontsize=20, pad=20)
    
    # Ensure the pie chart is a circle
    plt.axis('equal')  

    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    try:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Successfully saved plot to: {output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    plt.close()


if __name__ == "__main__":
    analyze_readmission_ratio()
