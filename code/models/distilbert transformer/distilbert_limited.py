import os
import numpy as np
import pandas as pd
import evaluate
from transformers import DistilBertForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset, load_from_disk
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import Counter
from google.colab import userdata

# Retrieve your Hugging Face access token
hf_token = userdata.get('HF_ACCESS_TOKEN')

# --- Configuration ---
TOKENIZED_DATASET_DIR = '/content/drive/My Drive/Readmissions_Research/Data/tokenized_hf_dataset_variable_cpt'
MODEL_CHECKPOINT = "distilbert-base-uncased"
OUTPUT_DIR = "/content/drive/My Drive/Readmissions_Research/Results/bootstrapped_5_cpt_icd"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOGGING_DIR = os.path.join(OUTPUT_DIR, 'logs')
NUM_LABELS = 2
NUM_TRAIN_EPOCHS = 10

# --- Load Tokenized Dataset from Disk ---
print(f"Loading tokenized dataset from directory: {TOKENIZED_DATASET_DIR}...")
try:
    os.makedirs(LOGGING_DIR, exist_ok=True)
    dataset = load_from_disk(TOKENIZED_DATASET_DIR)
    print(f"Loaded dataset with {len(dataset)} records.")
except FileNotFoundError:
    print(f"Error: Tokenized dataset directory not found at {TOKENIZED_DATASET_DIR}")
    exit()
except Exception as e:
    print(f"Error loading tokenized dataset from disk: {e}")
    exit()

# --- Prepare Dataset for Training ---
print("\nSplitting the dataset into 80:20 train/eval sets...")
split_dataset = dataset.train_test_split(test_size=0.20, seed=42)
train_dataset_original_hf = split_dataset["train"]
eval_dataset = split_dataset["test"]
print(f"Original training samples: {len(train_dataset_original_hf)}, Evaluation samples: {len(eval_dataset)}")
print(f"Original class distribution in training set: {Counter(train_dataset_original_hf['label'])}")


# --- NEW: Bootstrapping a Balanced Training Set ---
print("\n--- Creating a balanced training set via bootstrapping ---")
try:
    train_df = train_dataset_original_hf.to_pandas()
    df_class_0 = train_df[train_df['label'] == 0]
    df_class_1 = train_df[train_df['label'] == 1]

    sampled_class_0 = df_class_0.sample(n=15000, replace=False, random_state=42)
    print(f"Undersampled majority class (0) from {len(df_class_0)} to {len(sampled_class_0)}.")

    sampled_class_1 = df_class_1.sample(n=15000, replace=True, random_state=42)
    print(f"Oversampled minority class (1) from {len(df_class_1)} to {len(sampled_class_1)}.")

    bootstrapped_df = pd.concat([sampled_class_0, sampled_class_1])
    bootstrapped_df = bootstrapped_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Created new balanced training set with {len(bootstrapped_df)} total samples.")
    print(f"Final class distribution in new training set: {Counter(bootstrapped_df['label'])}")

    train_dataset = Dataset.from_pandas(bootstrapped_df, features=train_dataset_original_hf.features)
    print("Successfully replaced original training set with the new bootstrapped set.")

except Exception as e:
    print(f"Error during bootstrapping: {e}")
    print("Falling back to using the original imbalanced training data.")
    train_dataset = train_dataset_original_hf
# --- END NEW SECTION ---


# --- Initialize Model ---
print(f"\nInitializing model from {MODEL_CHECKPOINT}...")
try:
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=NUM_LABELS
    )
    model.config.label2id = {"NOT_READMITTED": 0, "READMITTED": 1}
    model.config.id2label = {0: "NOT_READMITTED", 1: "READMITTED"}
    print("Model initialized successfully.")
except Exception as e:
    print(f"Error initializing model: {e}")
    exit()

# --- Define Metrics ---
print("\nLoading evaluation metrics...")
try:
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
except Exception as e:
    print(f"Error loading metrics: {e}")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    try:
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
        precision = precision_metric.compute(predictions=predictions, references=labels, average="binary")["precision"]
        recall = recall_metric.compute(predictions=predictions, references=labels, average="binary")["recall"]
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")["f1"]
    except Exception as e:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# --- Training Setup ---
print("\nSetting up training arguments and trainer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
except Exception as e:
    print(f"Error initializing tokenizer for DataCollator: {e}")
    exit()

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_dir=LOGGING_DIR,
    logging_steps=100,
    push_to_hub=False,
)

# --- Using standard Trainer with the new balanced training data ---
print("Initializing standard Trainer with bootstrapped balanced training data.")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
print("Trainer initialized.")

# --- Train ---
print("\nStarting training...")
try:
    trainer.train()
    print("Training finished.")
except Exception as e:
    print(f"Error during training: {e}")
    exit()

# --- Evaluate the Best Model ---
print("\nEvaluating the best model on the evaluation set...")
eval_results = None
try:
    eval_results = trainer.evaluate()
    print("\nEvaluation Results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
except Exception as e:
    print(f"Error during evaluation: {e}")

# --- Generate Detailed Predictions, Scores, and Confusion Matrix ---
if eval_results:
    print("\n--- Generating Detailed Predictions, Scores, and Confusion Matrix ---")
    try:
        predictions_output = trainer.predict(eval_dataset)
        logits = predictions_output.predictions
        true_labels = predictions_output.label_ids
        predicted_labels = np.argmax(logits, axis=-1)

        # Calculate probabilities
        probabilities_all_classes = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        probs_positive_class = probabilities_all_classes[:, 1]

        print(f"Successfully got probabilities for {len(probs_positive_class)} evaluation samples.")

        # Generate and Save Confusion Matrix
        print("\nGenerating Confusion Matrix...")
        cm = confusion_matrix(true_labels, predicted_labels, labels=list(model.config.id2label.keys()))
        display_labels = list(model.config.id2label.values())
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix - Evaluation Set")
        cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_eval.png")
        plt.savefig(cm_path)
        print(f"Confusion matrix saved to {cm_path}")
        plt.close()

        # Prepare DataFrame with scores and save to CSV
        print("\nPreparing evaluation data with scores for CSV output...")
        eval_df_with_scores = eval_dataset.to_pandas()
        eval_df_with_scores['predicted_label'] = predicted_labels
        eval_df_with_scores['readmission_probability'] = probs_positive_class

        # Select and order columns for the final output
        output_columns = []
        if 'prompt' in eval_df_with_scores.columns:
            output_columns.append('prompt')
        output_columns.extend(['label', 'predicted_label', 'readmission_probability'])

        output_csv_path = os.path.join(OUTPUT_DIR, "evaluation_predictions_with_scores.csv")
        eval_df_with_scores[output_columns].to_csv(output_csv_path, index=False)
        print(f"Evaluation data with prediction probabilities saved to: {output_csv_path}")

    except Exception as e:
        print(f"Error during detailed prediction analysis or confusion matrix generation: {e}")
else:
    print("Skipping detailed predictions and confusion matrix due to evaluation failure.")


print("\nScript finished.")
