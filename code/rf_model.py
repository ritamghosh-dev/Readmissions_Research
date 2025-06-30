# import pandas as pd
# import numpy as np
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
#                              confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay)
# import matplotlib.pyplot as plt
# from collections import Counter

# # --- Configuration ---
# # Assuming scripts are run from the 'code/' directory
# PREPROCESSED_CSV_PATH = 'data/preprocessed_data.csv'
# OUTPUT_DIR = 'results/RandomForest_5'

# # --- 1. Load Data and Select Features ---
# print(f"Loading data from {PREPROCESSED_CSV_PATH}...")
# try:
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     df = pd.read_csv(PREPROCESSED_CSV_PATH, low_memory=False)
#     print(f"Loaded {len(df)} records successfully.")
# except FileNotFoundError:
#     print(f"Error: Preprocessed input file not found at {PREPROCESSED_CSV_PATH}")
#     print("Please ensure preprocessing.py has been run and the path is correct.")
#     exit()
# except Exception as e:
#     print(f"Error loading data: {e}")
#     exit()

# # Define the columns to be used as features
# ICD_COLUMNS = [f'I10_DX{i}' for i in range(1, 6)]
# CPT_COLUMNS = [f'CPT{i}' for i in range(1, 6)]
# NUMERICAL_FEATURES = ['CCI_Score']
# CATEGORICAL_FEATURES = ICD_COLUMNS + CPT_COLUMNS
# TARGET_COLUMN = 'Readmitted'

# # Ensure all specified feature columns exist in the DataFrame
# all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
# missing_cols = [col for col in all_features + [TARGET_COLUMN] if col not in df.columns]
# if missing_cols:
#     print(f"Error: The following required columns are missing from the input CSV: {missing_cols}")
#     exit()

# # Keep only the necessary columns
# df_selected = df[all_features + [TARGET_COLUMN]].copy()
# print(f"Selected {len(df_selected.columns)} columns for the Random Forest model.")

# # --- 2. Data Preparation for Random Forest ---
# # Fill NaN values before one-hot encoding
# print("Preparing data for Random Forest...")
# for col in CATEGORICAL_FEATURES:
#     df_selected[col] = df_selected[col].astype(str).fillna('missing')
# df_selected[NUMERICAL_FEATURES] = df_selected[NUMERICAL_FEATURES].fillna(0) # Fill numerical NaNs with 0

# # Perform One-Hot Encoding for all categorical features
# print("Performing one-hot encoding on CPT and ICD columns...")
# df_one_hot = pd.get_dummies(df_selected, columns=CATEGORICAL_FEATURES, prefix=CATEGORICAL_FEATURES, dummy_na=False)
# print(f"Data transformed. Number of features after one-hot encoding: {len(df_one_hot.columns) - 1}")

# # --- 3. Data Splitting and Bootstrapping ---
# print("\nSplitting data into 80:20 train/test sets...")
# X = df_one_hot.drop(TARGET_COLUMN, axis=1)
# y = df_one_hot[TARGET_COLUMN]

# # Create the initial 80:20 split. The test set will be held out and untouched.
# X_train_orig, X_test, y_train_orig, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
# print(f"Original training samples: {len(X_train_orig)}, Test samples: {len(X_test)}")
# original_train_dist = Counter(y_train_orig)
# print(f"Original class distribution in training set: {original_train_dist}")

# # --- Bootstrapping a Balanced Training Set ---
# print("\n--- Creating a balanced training set via bootstrapping ---")
# try:
#     # Combine original training features and labels for sampling
#     train_df_orig = pd.concat([X_train_orig, y_train_orig], axis=1)
    
#     df_class_0 = train_df_orig[train_df_orig[TARGET_COLUMN] == 0]
#     df_class_1 = train_df_orig[train_df_orig[TARGET_COLUMN] == 1]

#     target_sample_size = 15000
#     sampled_class_0 = df_class_0.sample(n=target_sample_size, replace=False, random_state=42)
#     print(f"Undersampled majority class (0) from {len(df_class_0)} to {len(sampled_class_0)}.")
    
#     sampled_class_1 = df_class_1.sample(n=target_sample_size, replace=True, random_state=42)
#     print(f"Oversampled minority class (1) from {len(df_class_1)} to {len(sampled_class_1)}.")

#     bootstrapped_df = pd.concat([sampled_class_0, sampled_class_1])
#     bootstrapped_df = bootstrapped_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
#     X_train = bootstrapped_df.drop(TARGET_COLUMN, axis=1)
#     y_train = bootstrapped_df[TARGET_COLUMN]
    
#     final_train_dist = Counter(y_train)
#     print(f"Created new balanced training set with {len(X_train)} total samples.")
#     print(f"Final class distribution in new training set: {final_train_dist}")

# except Exception as e:
#     print(f"Error during bootstrapping: {e}")
#     exit()

# # --- 4. Train the Random Forest Model ---
# print("\n--- Training Random Forest Classifier ---")
# # Initialize the Random Forest model.
# # n_jobs=-1 uses all available CPU cores for faster training.
# # class_weight='balanced_subsample' is a good practice for RF even with balanced data.
# model = RandomForestClassifier(
#     n_estimators=200,       # Number of trees in the forest
#     max_depth=20,           # Maximum depth of the tree
#     min_samples_leaf=5,     # Minimum number of samples required at a leaf node
#     class_weight='balanced',# Also give weight to classes
#     random_state=42,
#     n_jobs=-1,              # Use all available CPU cores
#     verbose=1               # Print progress
# )

# # Train the model
# model.fit(X_train, y_train)
# print("Random Forest model training complete.")

# # --- 5. Evaluate the Model ---
# print("\n--- Evaluating Model on Untouched Test Set ---")
# # Make predictions on the test set
# y_pred = model.predict(X_test)
# y_pred_proba = model.predict_proba(X_test)[:, 1] # Get probability of the positive class

# # Calculate and print metrics
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# roc_auc = roc_auc_score(y_test, y_pred_proba)

# print(f"Accuracy:  {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall:    {recall:.4f}")
# print(f"F1-Score:  {f1:.4f}")
# print(f"AUC-ROC:   {roc_auc:.4f}")

# # Generate and save a confusion matrix
# print("\nGenerating Confusion Matrix...")
# try:
#     cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
#     disp.plot(cmap=plt.cm.Blues)
#     plt.title("Random Forest Confusion Matrix - Test Set")
#     cm_path = os.path.join(OUTPUT_DIR, "randomforest_confusion_matrix.png")
#     plt.savefig(cm_path)
#     print(f"Confusion matrix saved to {cm_path}")
#     plt.close()
# except Exception as e:
#     print(f"Error generating confusion matrix: {e}")

# # Generate and save an ROC Curve plot
# print("\nGenerating ROC Curve...")
# try:
#     fig, ax = plt.subplots()
#     RocCurveDisplay.from_predictions(y_test, y_pred_proba, ax=ax)
#     ax.plot([0, 1], [0, 1], "k--", label="Chance Level (AUC = 0.5)")
#     plt.title("Random Forest ROC Curve - Test Set")
#     plt.legend()
#     roc_path = os.path.join(OUTPUT_DIR, "randomforest_roc_curve.png")
#     plt.savefig(roc_path)
#     print(f"ROC Curve saved to {roc_path}")
#     plt.close()
# except Exception as e:
#     print(f"Error generating ROC curve: {e}")

# # --- ADDED: Save Evaluation Summary to a Text File ---
# print("\nSaving evaluation summary to a text file...")
# try:
#     summary_path = os.path.join(OUTPUT_DIR, "evaluation_summary.txt")
#     with open(summary_path, 'w') as f:
#         f.write("--- Random Forest Model Evaluation Summary ---\n\n")
#         f.write(f"Timestamp: {pd.Timestamp.now()}\n")
#         f.write(f"Output Directory: {os.path.abspath(OUTPUT_DIR)}\n\n")
        
#         f.write("--- Data Information ---\n")
#         f.write(f"Input Data Source: {PREPROCESSED_CSV_PATH}\n")
#         f.write(f"Total Features Used (before one-hot encoding): {len(all_features)}\n")
#         f.write(f"Total Features Used (after one-hot encoding): {len(X.columns)}\n")
#         f.write(f"Train/Test Split: 80/20\n")
#         f.write(f"Original Training Set Size: {len(X_train_orig)} records\n")
#         f.write(f"Original Training Class Distribution: {dict(original_train_dist)}\n")
#         f.write(f"Test Set Size: {len(X_test)} records\n\n")
        
#         f.write("--- Training Strategy: Bootstrapping ---\n")
#         f.write(f"Final Balanced Training Set Size: {len(X_train)} records\n")
#         f.write(f"Final Training Class Distribution: {dict(final_train_dist)}\n\n")
        
#         f.write("--- Final Evaluation Metrics on Test Set ---\n")
#         f.write(f"Accuracy:  {accuracy:.4f}\n")
#         f.write(f"Precision: {precision:.4f}\n")
#         f.write(f"Recall:    {recall:.4f}\n")
#         f.write(f"F1-Score:  {f1:.4f}\n")
#         f.write(f"AUC-ROC:   {roc_auc:.4f}\n\n")
        
#         f.write("--- Confusion Matrix (Test Set) ---\n")
#         tn, fp, fn, tp = cm.ravel()
#         f.write(f"True Negatives (TN):  {tn}\n")
#         f.write(f"False Positives (FP): {fp}\n")
#         f.write(f"False Negatives (FN): {fn}\n")
#         f.write(f"True Positives (TP):  {tp}\n")

#     print(f"Evaluation summary saved to: {summary_path}")
# except Exception as e:
#     print(f"Error saving evaluation summary file: {e}")
# # --- END ADDED SECTION ---

# print("\n--- Random Forest Experiment Finished ---")
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay)
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
from collections import Counter

# --- Configuration ---
PREPROCESSED_CSV_PATH = 'data/preprocessed_data.csv'
OUTPUT_DIR = 'results/RandomForest'

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

# --- 2. Data Preparation ---
print("Preparing data for Random Forest...")
for col in CATEGORICAL_FEATURES:
    df_selected[col] = df_selected[col].astype(str).fillna('missing')
df_selected[NUMERICAL_FEATURES] = df_selected[NUMERICAL_FEATURES].fillna(0)

print("Performing Ordinal Encoding on CPT and ICD columns...")
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df_selected[CATEGORICAL_FEATURES] = encoder.fit_transform(df_selected[CATEGORICAL_FEATURES])

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

# --- 4. Hyperparameter Tuning with GridSearchCV ---
print("\n--- Starting Hyperparameter Tuning for Random Forest ---")
param_grid = { 'n_estimators': [200, 300], 'max_depth': [20, 30], 'min_samples_split': [5, 10], 'min_samples_leaf': [2, 4], 'class_weight': ['balanced', 'balanced_subsample'] }
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='f1', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_
print("\nHyperparameter tuning complete.")
print("Best Parameters Found:")
print(grid_search.best_params_)

# --- 5. Evaluate the Best Model ---
print("\n--- Evaluating Best Model on Untouched Test Set ---")
y_pred = best_rf_model.predict(X_test)
y_pred_proba = best_rf_model.predict_proba(X_test)[:, 1]

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
    output_csv_path = os.path.join(OUTPUT_DIR, "rf_predictions.csv")
    results_df.to_csv(output_csv_path, index=False)
    print(f"\nPredictions saved successfully to: {output_csv_path}")

    # 2. Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=best_rf_model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_rf_model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Tuned Random Forest Confusion Matrix - Test Set")
    cm_path = os.path.join(OUTPUT_DIR, "tuned_randomforest_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")
    
    # 3. Save Evaluation Summary
    summary_path = os.path.join(OUTPUT_DIR, "tuned_randomforest_evaluation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("--- Random Forest Model Evaluation Summary ---\n\n")
        f.write(f"Timestamp: {pd.Timestamp.now()}\n\n")
        f.write("--- Data Info ---\n")
        f.write(f"Total Features Used (before one-hot encoding): {len(all_features)}\n")
        f.write(f"Total Features Used (after one-hot encoding): {len(X.columns)}\n")
        f.write(f"Original Training Set Size: {len(X_train_orig)}\n")
        f.write(f"Test Set Size: {len(X_test)}\n")
        f.write(f"Original Training Class Distribution: {dict(original_train_dist)}\n")
        f.write(f"Training Class Distribution (Bootstrapped): {dict(final_train_dist)}\n\n")
        f.write("--- Best Hyperparameters Found ---\n")
        f.write(str(grid_search.best_params_) + "\n\n")
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

print("\n--- Random Forest Experiment Finished ---")


