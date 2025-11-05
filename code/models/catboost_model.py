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
PREPROCESSED_CSV_PATH = 'data/preprocessed_data.csv'
OUTPUT_DIR = 'results/CatBoost'

# --- 1. Load Data and Select Features ---
print(f"Loading data from {PREPROCESSED_CSV_PATH}...")
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(PREPROCESSED_CSV_PATH, low_memory=False)
    print(f"Loaded {len(df)} records successfully.")
except FileNotFoundError:
    print(f"Error: Preprocessed input file not found at {PREPROCESSED_CSV_PATH}")
    exit()

# Define feature columns
ICD_COLUMNS = [f'I10_DX{i}' for i in range(1, 35)]
CPT_COLUMNS = [f'CPT{i}' for i in range(1, 101)]
NUMERICAL_FEATURES = ['CCI_Score']
CATEGORICAL_FEATURES = ICD_COLUMNS + CPT_COLUMNS
TARGET_COLUMN = 'Readmitted30'
all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
df_selected = df[all_features + [TARGET_COLUMN]].copy()

# --- 2. Data Preparation for CatBoost ---
print("Preparing data for CatBoost...")
for col in CATEGORICAL_FEATURES:
    df_selected[col] = df_selected[col].astype(str).fillna('missing')

# --- 3. Data Splitting and Bootstrapping ---
print("\nSplitting data into 80:20 train/test sets...")
X = df_selected.drop(TARGET_COLUMN, axis=1)
y = df_selected[TARGET_COLUMN]
X_train_orig, X_test, y_train_orig, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
original_train_dist = Counter(y_train_orig)

print("\n--- Creating a balanced training set via bootstrapping ---")
train_df_orig = pd.concat([X_train_orig, y_train_orig], axis=1)
df_class_0 = train_df_orig[train_df_orig[TARGET_COLUMN] == 0]
df_class_1 = train_df_orig[train_df_orig[TARGET_COLUMN] == 1]
target_sample_size = 15000
sampled_class_0 = df_class_0.sample(n=target_sample_size, replace=False, random_state=42)
sampled_class_1 = df_class_1.sample(n=target_sample_size, replace=True, random_state=42)
bootstrapped_df = pd.concat([sampled_class_0, sampled_class_1])
bootstrapped_df = bootstrapped_df.sample(frac=1, random_state=42).reset_index(drop=True)
X_train = bootstrapped_df.drop(TARGET_COLUMN, axis=1)
y_train = bootstrapped_df[TARGET_COLUMN]
final_train_dist = Counter(y_train)

# --- 4. Train the CatBoost Model ---
print("\n--- Training CatBoost Classifier ---")
model = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3, loss_function='Logloss', eval_metric='F1', random_seed=42, verbose=100, early_stopping_rounds=50)
categorical_features_indices = [X_train.columns.get_loc(col) for col in CATEGORICAL_FEATURES]
model.fit(X_train, y_train, cat_features=categorical_features_indices, eval_set=(X_test, y_test), use_best_model=True)
print("CatBoost model training complete.")

# --- 5. Evaluate the Model ---
print("\n--- Evaluating Model on Untouched Test Set ---")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"AUC-ROC:   {roc_auc:.4f}")

# --- ADDED: Full Evaluation Reporting ---
try:
    # 1. Save Predictions CSV
    results_df = pd.DataFrame({'true_label': y_test, 'predicted_probability': y_pred_proba})
    output_csv_path = os.path.join(OUTPUT_DIR, "catboost_predictions.csv")
    results_df.to_csv(output_csv_path, index=False)
    print(f"\nPredictions saved successfully to: {output_csv_path}")

    # 2. Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("CatBoost Confusion Matrix - Test Set")
    cm_path = os.path.join(OUTPUT_DIR, "catboost_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")
    
    # 3. Save Evaluation Summary
    summary_path = os.path.join(OUTPUT_DIR, "catboost_evaluation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("--- CatBoost Model Evaluation Summary ---\n\n")
        f.write(f"Timestamp: {pd.Timestamp.now()}\n\n")
        f.write("--- Data Info ---\n")
        f.write(f"Total Features Used: {len(all_features)}\n")
        f.write(f"Original Training Set Size: {len(X_train_orig)}\n")
        f.write(f"Test Set Size: {len(X_test)}\n")
        f.write(f"Original Training Class Distribution: {dict(original_train_dist)}\n")
        f.write(f"Training Class Distribution (Bootstrapped): {dict(final_train_dist)}\n\n")
        f.write("--- Final Evaluation Metrics ---\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n")
        f.write(f"AUC-ROC:   {roc_auc:.4f}\n")
    print(f"Evaluation summary saved to: {summary_path}")

except Exception as e:
    print(f"Error during final reporting: {e}")
# --- END ADDED SECTION ---

print("\n--- CatBoost Experiment Finished ---")
