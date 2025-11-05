import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- Configuration ---
PREPROCESSED_CSV_PATH = 'data/preprocessed_data.csv'
OUTPUT_DIR = 'results/MLP_5'
EPOCHS = 15
BATCH_SIZE = 256
LEARNING_RATE = 0.001

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
ICD_COLUMNS = [f'I10_DX{i}' for i in range(1, 6)]
CPT_COLUMNS = [f'CPT{i}' for i in range(1, 6)]
NUMERICAL_FEATURES = ['CCI_Score']
CATEGORICAL_FEATURES = ICD_COLUMNS + CPT_COLUMNS
TARGET_COLUMN = 'Readmitted30'

all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
df_selected = df[all_features + [TARGET_COLUMN]].copy()

# --- 2. Data Preparation for Deep Learning ---
print("Preparing data for MLP model...")
for col in CATEGORICAL_FEATURES:
    df_selected[col] = df_selected[col].astype(str).fillna('missing')
df_selected[NUMERICAL_FEATURES] = df_selected[NUMERICAL_FEATURES].fillna(0)

categorical_mappings = {}
for col in CATEGORICAL_FEATURES:
    df_selected[col] = df_selected[col].astype('category')
    categorical_mappings[col] = dict(enumerate(df_selected[col].cat.categories))
    df_selected[col] = df_selected[col].cat.codes

embedding_sizes = []
for col in CATEGORICAL_FEATURES:
    num_categories = len(df_selected[col].unique())
    embedding_dim = min(50, (num_categories + 1) // 2)
    embedding_sizes.append((num_categories, embedding_dim))

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

scaler = StandardScaler()
X_train[NUMERICAL_FEATURES] = scaler.fit_transform(X_train[NUMERICAL_FEATURES])
X_test[NUMERICAL_FEATURES] = scaler.transform(X_test[NUMERICAL_FEATURES])

# --- 4. PyTorch Dataset and DataLoader ---
class TabularDataset(Dataset):
    def __init__(self, features_df, labels_series, cat_cols, num_cols):
        self.features_cat = features_df[cat_cols].values.astype(np.int64)
        self.features_num = features_df[num_cols].values.astype(np.float32)
        self.labels = labels_series.values.astype(np.float32)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return [self.features_cat[idx], self.features_num[idx], self.labels[idx]]

train_dataset = TabularDataset(X_train, y_train, CATEGORICAL_FEATURES, NUMERICAL_FEATURES)
test_dataset = TabularDataset(X_test, y_test, CATEGORICAL_FEATURES, NUMERICAL_FEATURES)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 5. Define the MLP Model ---
class MLPWithEmbeddings(nn.Module):
    def __init__(self, embedding_sizes, n_numeric_features, hidden_layers_sizes, output_size, dropout_rate=0.4):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(num_categories, embedding_dim) for num_categories, embedding_dim in embedding_sizes])
        n_embedding_dims = sum(embedding_dim for _, embedding_dim in embedding_sizes)
        self.n_numeric_features = n_numeric_features
        all_input_size = n_embedding_dims + n_numeric_features
        layer_sizes = [all_input_size] + hidden_layers_sizes
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
            self.layers.append(nn.Dropout(dropout_rate))
        self.output_layer = nn.Linear(layer_sizes[-1], output_size)
    def forward(self, x_categorical, x_numerical):
        embedding_outputs = []
        for i, emb_layer in enumerate(self.embeddings):
            embedding_outputs.append(emb_layer(x_categorical[:, i]))
        x = torch.cat(embedding_outputs, 1)
        x = torch.cat([x, x_numerical], 1)
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

model = MLPWithEmbeddings(embedding_sizes, n_numeric_features=len(NUMERICAL_FEATURES), hidden_layers_sizes=[256, 128], output_size=1)

# --- 6. Training Loop ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nTraining on device: {device}")
model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(f"Starting training for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    model.train()
    for cats, nums, labels in train_loader:
        cats, nums, labels = cats.to(device), nums.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(cats, nums).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
print("Training finished.")

# --- 7. Final Evaluation and Reporting ---
print("\n--- Evaluating Final Model on Untouched Test Set ---")
model.eval()
all_preds, all_labels, all_probas = [], [], []
with torch.no_grad():
    for cats, nums, labels in test_loader:
        cats, nums, labels = cats.to(device), nums.to(device), labels.to(device)
        outputs = model(cats, nums).squeeze()
        probas = torch.sigmoid(outputs)
        preds = (probas > 0.5).long()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probas.extend(probas.cpu().numpy())

y_test_final, y_pred_final, y_prob_final = np.array(all_labels), np.array(all_preds), np.array(all_probas)
accuracy = accuracy_score(y_test_final, y_pred_final)
precision = precision_score(y_test_final, y_pred_final)
recall = recall_score(y_test_final, y_pred_final)
f1 = f1_score(y_test_final, y_pred_final)
roc_auc = roc_auc_score(y_test_final, y_prob_final)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"AUC-ROC:   {roc_auc:.4f}")

# --- ADDED: Save Predictions, Plots, and Summary ---
try:
    # 1. Save Predictions CSV
    results_df = pd.DataFrame({'true_label': y_test_final, 'predicted_probability': y_prob_final})
    output_csv_path = os.path.join(OUTPUT_DIR, "mlp_predictions.csv")
    results_df.to_csv(output_csv_path, index=False)
    print(f"\nPredictions saved successfully to: {output_csv_path}")

    # 2. Save Confusion Matrix
    cm = confusion_matrix(y_test_final, y_pred_final)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("MLP Confusion Matrix - Test Set")
    cm_path = os.path.join(OUTPUT_DIR, "mlp_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # 3. Save Evaluation Summary
    summary_path = os.path.join(OUTPUT_DIR, "evaluation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("--- MLP Model Evaluation Summary ---\n\n")
        f.write(f"Timestamp: {pd.Timestamp.now()}\n\n")
        f.write("--- Data Info ---\n")
        f.write(f"Total Features Used: {len(all_features)}\n")
        f.write(f"Original Training Set Size: {len(X_train_orig)}\n")
        f.write(f"Test Set Size: {len(X_test)}\n")
        f.write(f"Training Class Distribution (Original): {dict(original_train_dist)}\n")
        f.write(f"Training Class Distribution (Bootstrapped): {dict(final_train_dist)}\n\n")
        f.write("--- Model Hyperparameters ---\n")
        f.write(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, Learning Rate: {LEARNING_RATE}\n\n")
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

print("\n--- MLP Experiment Finished ---")

