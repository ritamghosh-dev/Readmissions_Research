import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# --- Global style: Times New Roman @ 18 pt ---
# If Times New Roman isn't available, matplotlib will fall back automatically.
# (Swap to "Nimbus Roman No9 L" if needed.)
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 18
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18

# --- Configuration ---
INPUT_CSV_PATH = 'data/preprocessed_data.csv'    # Must contain Readmitted30
TARGET_COLUMN   = 'Readmitted30'
OUTPUT_DIR      = 'results/analytics'
OUTPUT_FILENAME = 'readmission_ratio_pie_chart.png'  # PNG as requested

def analyze_readmission_ratio():
    """
    Loads the preprocessed dataset, analyzes the distribution of the 
    readmission target variable, and saves a 300-dpi PNG pie chart
    with Times New Roman (18 pt) and bold percentage numbers.
    """
    print(f"--- Starting Analysis of '{TARGET_COLUMN}' Distribution ---")

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

    if TARGET_COLUMN not in df.columns:
        print(f"Error: Column '{TARGET_COLUMN}' not found in the dataset.")
        return

    # 2) Counts & labels
    print(f"Calculating distribution for '{TARGET_COLUMN}'...")
    class_counts = df[TARGET_COLUMN].value_counts()
    labels_map = {0: 'Not Readmitted', 1: 'Readmitted'}
    class_labels = class_counts.index.map(labels_map)

    print("\nDistribution of Encounters by Readmission Status:")
    print(class_counts.to_string())

    # 3) Pie chart
    print("\nGenerating pie chart...")
    sns.set_style("whitegrid")  # style (fonts come from rcParams)
    plt.figure(figsize=(10, 8))

    colors  = sns.color_palette('pastel')[0:2]
    explode = (0, 0.1)  # emphasize the 'Readmitted' slice

    # Draw and capture handles so we can style percentages only
    wedges, texts, autotexts = plt.pie(
        class_counts,
        labels=class_labels,
        colors=colors,
        autopct='%1.1f%%',      # percentage text
        shadow=True,
        startangle=140,
        explode=explode,
        textprops={'fontsize': 18}  # label text (not bold)
    )

    # Make just the percentages bold (leave labels normal)
    for at in autotexts:
        at.set_fontweight('bold')
        at.set_fontsize(18)

    # Title
    plt.title('Ratio of 30-Day Hospital Readmissions', pad=20)

    # Keep aspect ratio as a circle
    plt.axis('equal')

    # 4) Save PNG @ 300 dpi
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    try:
        plt.savefig(output_path, bbox_inches='tight', dpi=300, format='png')
        print(f"Successfully saved plot to: {output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.close()

if __name__ == "__main__":
    analyze_readmission_ratio()