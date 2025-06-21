import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay)
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from collections import Counter

# --- Configuration ---
# Assuming scripts are run from the 'code/' directory
PREPROCESSED_CSV_PATH = 'data/preprocessed_data.csv'
OUTPUT_DIR = 'results/CatBoost_5'

# --- 1. Load Data and Select Features ---
print(f"Loading data from {PREPROCESSED_CSV_PATH}...")
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(PREPROCESSED_CSV_PATH, low_memory=False)
    print(f"Loaded {len(df)} records successfully.")
except FileNotFoundError:
    print(f"Error: Preprocessed input file not found at {PREPROCESSED_CSV_PATH}")
    print("Please ensure preprocessing.py has been run and the path is correct.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Define the columns to be used as features
ICD_COLUMNS = [f'I10_DX{i}' for i in range(1, 6)]
CPT_COLUMNS = [f'CPT{i}' for i in range(1, 6)]
NUMERICAL_FEATURES = ['CCI_Score']
CATEGORICAL_FEATURES = ICD_COLUMNS + CPT_COLUMNS
TARGET_COLUMN = 'Readmitted'

# Ensure all specified feature columns exist in the DataFrame
all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
missing_cols = [col for col in all_features + [TARGET_COLUMN] if col not in df.columns]
if missing_cols:
    print(f"Error: The following required columns are missing from the input CSV: {missing_cols}")
    exit()

# Keep only the necessary columns
df_selected = df[all_features + [TARGET_COLUMN]].copy()
print(f"Selected {len(df_selected.columns)} columns for the CatBoost model.")

# --- 2. Data Preparation for CatBoost ---
# CatBoost can handle NaN in numerical features, but it's better to fill categorical NaNs with a placeholder.
print("Preparing data for CatBoost...")
for col in CATEGORICAL_FEATURES:
    df_selected[col] = df_selected[col].astype(str).fillna('missing')

print("Filled NaN values in categorical features with 'missing'.")

# --- 3. Data Splitting and Bootstrapping ---
print("\nSplitting data into 80:20 train/test sets...")
X = df_selected.drop(TARGET_COLUMN, axis=1)
y = df_selected[TARGET_COLUMN]

# Create the initial 80:20 split. The test set will be held out and untouched.
X_train_orig, X_test, y_train_orig, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Original training samples: {len(X_train_orig)}, Test samples: {len(X_test)}")
original_train_dist = Counter(y_train_orig)
print(f"Original class distribution in training set: {original_train_dist}")

# --- Bootstrapping a Balanced Training Set ---
print("\n--- Creating a balanced training set via bootstrapping ---")
try:
    # Combine original training features and labels for sampling
    train_df_orig = pd.concat([X_train_orig, y_train_orig], axis=1)

    # Separate the DataFrame by class
    df_class_0 = train_df_orig[train_df_orig[TARGET_COLUMN] == 0]
    df_class_1 = train_df_orig[train_df_orig[TARGET_COLUMN] == 1]

    # Undersample the majority class and oversample the minority class
    target_sample_size = 15000
    sampled_class_0 = df_class_0.sample(n=target_sample_size, replace=False, random_state=42)
    print(f"Undersampled majority class (0) from {len(df_class_0)} to {len(sampled_class_0)}.")

    sampled_class_1 = df_class_1.sample(n=target_sample_size, replace=True, random_state=42)
    print(f"Oversampled minority class (1) from {len(df_class_1)} to {len(sampled_class_1)}.")

    # Combine and shuffle to create the final training set
    bootstrapped_df = pd.concat([sampled_class_0, sampled_class_1])
    bootstrapped_df = bootstrapped_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Separate the final bootstrapped features and labels
    X_train = bootstrapped_df.drop(TARGET_COLUMN, axis=1)
    y_train = bootstrapped_df[TARGET_COLUMN]
    final_train_dist = Counter(y_train)
    
    print(f"Created new balanced training set with {len(X_train)} total samples.")
    print(f"Final class distribution in new training set: {final_train_dist}")

except Exception as e:
    print(f"Error during bootstrapping: {e}")
    exit()

# --- 4. Train the CatBoost Model ---
print("\n--- Training CatBoost Classifier ---")
# Initialize the CatBoost model
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    loss_function='Logloss',
    eval_metric='F1',
    random_seed=42,
    verbose=100, # Print training info every 100 iterations
    early_stopping_rounds=50 # Stop if F1 on eval set doesn't improve for 50 rounds
)

# Identify the indices of categorical features for CatBoost
categorical_features_indices = [X_train.columns.get_loc(col) for col in CATEGORICAL_FEATURES]

# Train the model
model.fit(
    X_train, y_train,
    cat_features=categorical_features_indices,
    eval_set=(X_test, y_test),
    use_best_model=True # Uses the model from the best iteration
)
print("CatBoost model training complete.")

# --- 5. Evaluate the Model ---
print("\n--- Evaluating Model on Untouched Test Set ---")
# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1] # Get probability of the positive class

# Calculate and print metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba) # Calculate AUC-ROC Score

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"AUC-ROC:   {roc_auc:.4f}")

# Generate and save a confusion matrix
print("\nGenerating Confusion Matrix...")
try:
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("CatBoost Confusion Matrix - Test Set")
    cm_path = os.path.join(OUTPUT_DIR, "catboost_confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    plt.close()
except Exception as e:
    print(f"Error generating confusion matrix: {e}")

# Generate and save an ROC Curve plot
print("\nGenerating ROC Curve...")
try:
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, y_pred_proba, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Chance Level (AUC = 0.5)")
    plt.title("CatBoost ROC Curve - Test Set")
    plt.legend()
    roc_path = os.path.join(OUTPUT_DIR, "catboost_roc_curve.png")
    plt.savefig(roc_path)
    print(f"ROC Curve saved to {roc_path}")
    plt.close()
except Exception as e:
    print(f"Error generating ROC curve: {e}")

# --- ADDED: Save Evaluation Summary to a Text File ---
print("\nSaving evaluation summary to a text file...")
try:
    summary_path = os.path.join(OUTPUT_DIR, "evaluation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("--- CatBoost Model Evaluation Summary ---\n\n")
        f.write(f"Timestamp: {pd.Timestamp.now()}\n")
        f.write(f"Output Directory: {os.path.abspath(OUTPUT_DIR)}\n\n")
        
        f.write("--- Data Information ---\n")
        f.write(f"Input Data Source: {PREPROCESSED_CSV_PATH}\n")
        f.write(f"Total Features Used: {len(all_features)}\n")
        f.write("Encoding Method: CatBoost Internal (for categorical features)\n")
        f.write(f"Train/Test Split: 80/20\n")
        f.write(f"Original Training Set Size: {len(X_train_orig)} records\n")
        f.write(f"Original Training Class Distribution: {dict(original_train_dist)}\n")
        f.write(f"Test Set Size: {len(X_test)} records\n\n")
        
        f.write("--- Training Strategy: Bootstrapping ---\n")
        f.write(f"Final Balanced Training Set Size: {len(X_train)} records\n")
        f.write(f"Final Training Class Distribution: {dict(final_train_dist)}\n\n")
        
        f.write("--- Final Evaluation Metrics on Test Set ---\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n")
        f.write(f"AUC-ROC:   {roc_auc:.4f}\n\n")
        
        f.write("--- Confusion Matrix (Test Set) ---\n")
        tn, fp, fn, tp = cm.ravel()
        f.write(f"True Negatives (TN):  {tn}\n")
        f.write(f"False Positives (FP): {fp}\n")
        f.write(f"False Negatives (FN): {fn}\n")
        f.write(f"True Positives (TP):  {tp}\n")

    print(f"Evaluation summary saved to: {summary_path}")
except Exception as e:
    print(f"Error saving evaluation summary file: {e}")
# --- END ADDED SECTION ---

print("\n--- CatBoost Experiment Finished ---")
