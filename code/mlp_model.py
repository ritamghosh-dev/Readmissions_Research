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
# Assuming scripts are run from the 'code/' directory
PREPROCESSED_CSV_PATH = 'data/preprocessed_data.csv'
OUTPUT_DIR = 'results/MLP'
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
ICD_COLUMNS = [f'I10_DX{i}' for i in range(1, 35)]
CPT_COLUMNS = [f'CPT{i}' for i in range(1, 101)]
NUMERICAL_FEATURES = ['CCI_Score']
CATEGORICAL_FEATURES = ICD_COLUMNS + CPT_COLUMNS
TARGET_COLUMN = 'Readmitted'

all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
df_selected = df[all_features + [TARGET_COLUMN]].copy()
print(f"Selected {len(df_selected.columns)} columns for the MLP model.")

# --- 2. Data Preparation for Deep Learning ---
print("Preparing data for MLP model...")
# Fill NaNs
for col in CATEGORICAL_FEATURES:
    df_selected[col] = df_selected[col].astype(str).fillna('missing')
df_selected[NUMERICAL_FEATURES] = df_selected[NUMERICAL_FEATURES].fillna(0)

# Convert categorical features to integer codes using Ordinal Encoding
for col in CATEGORICAL_FEATURES:
    df_selected[col] = df_selected[col].astype('category')
    df_selected[col] = df_selected[col].cat.codes

print("Converted categorical features to integer codes for embeddings.")

# --- CORRECTED: Calculate Embedding Sizes from the FULL dataset BEFORE splitting ---
embedding_sizes = []
for col in CATEGORICAL_FEATURES:
    # Get the number of unique categories from the entire column
    num_categories = len(df_selected[col].unique())
    # Rule of thumb for embedding dimension
    embedding_dim = min(50, (num_categories + 1) // 2)
    embedding_sizes.append((num_categories, embedding_dim))
print("Calculated embedding layer sizes based on the full dataset.")
# --- END CORRECTION ---

# --- 3. Data Splitting and Bootstrapping ---
print("\nSplitting data into 80:20 train/test sets...")
X = df_selected.drop(TARGET_COLUMN, axis=1)
y = df_selected[TARGET_COLUMN]

X_train_orig, X_test, y_train_orig, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Original training samples: {len(X_train_orig)}, Test samples: {len(X_test)}")
original_train_dist = Counter(y_train_orig)
print(f"Original class distribution in training set: {original_train_dist}")

print("\n--- Creating a balanced training set via bootstrapping ---")
try:
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
    print(f"Created new balanced training set with {len(X_train)} total samples.")
except Exception as e:
    print(f"Error during bootstrapping: {e}")
    exit()
    
scaler = StandardScaler()
X_train[NUMERICAL_FEATURES] = scaler.fit_transform(X_train[NUMERICAL_FEATURES])
X_test[NUMERICAL_FEATURES] = scaler.transform(X_test[NUMERICAL_FEATURES])
print("Scaled numerical features (CCI_Score).")

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
print("Created PyTorch DataLoaders.")

# --- 5. Define the MLP Model with Entity Embeddings ---
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
print("\nMLP Model Architecture:")
print(model)

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
        
    model.eval()
    all_preds_epoch, all_labels_epoch = [], []
    with torch.no_grad():
        for cats, nums, labels in test_loader:
            cats, nums, labels = cats.to(device), nums.to(device), labels.to(device)
            outputs = model(cats, nums).squeeze()
            preds = (torch.sigmoid(outputs) > 0.5).long()
            all_preds_epoch.extend(preds.cpu().numpy())
            all_labels_epoch.extend(labels.cpu().numpy())
            
    f1 = f1_score(all_labels_epoch, all_preds_epoch)
    print(f"Epoch {epoch+1}/{EPOCHS} -> Eval F1: {f1:.4f}")

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

y_test, y_pred, y_pred_proba = np.array(all_labels), np.array(all_preds), np.array(all_probas)
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

# Generate and save a confusion matrix
print("\nGenerating Confusion Matrix...")
try:
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("MLP Confusion Matrix - Test Set")
    cm_path = os.path.join(OUTPUT_DIR, "mlp_confusion_matrix.png")
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
    plt.title("MLP ROC Curve - Test Set")
    plt.legend()
    roc_path = os.path.join(OUTPUT_DIR, "mlp_roc_curve.png")
    plt.savefig(roc_path)
    print(f"ROC Curve saved to {roc_path}")
    plt.close()
except Exception as e:
    print(f"Error generating ROC curve: {e}")
    
# --- Save Evaluation Summary to a Text File ---
print("\nSaving evaluation summary to a text file...")
try:
    summary_path = os.path.join(OUTPUT_DIR, "evaluation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("--- MLP with Entity Embeddings Model Evaluation Summary ---\n\n")
        f.write(f"Timestamp: {pd.Timestamp.now()}\n")
        f.write(f"Output Directory: {os.path.abspath(OUTPUT_DIR)}\n\n")
        f.write("--- Data Information ---\n")
        f.write(f"Input Data Source: {PREPROCESSED_CSV_PATH}\n")
        f.write(f"Total Features Used: {len(all_features)}\n")
        f.write("Encoding Method: Entity Embeddings via Ordinal Encoding\n")
        f.write(f"Train/Test Split: 80/20\n")
        f.write(f"Original Training Set Size: {len(X_train_orig)} records\n")
        f.write(f"Original Training Class Distribution: {dict(original_train_dist)}\n")
        f.write(f"Test Set Size: {len(X_test)} records\n\n")
        f.write("--- Training Strategy: Bootstrapping ---\n")
        f.write(f"Final Balanced Training Set Size: {len(X_train)} records\n")
        f.write(f"Final Training Class Distribution: {dict(final_train_dist)}\n\n")
        f.write("--- Model Hyperparameters ---\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write("Hidden Layers: [256, 128]\n\n")
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

print("\n--- MLP Experiment Finished ---")
