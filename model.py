import os
import numpy as np
import pandas as pd # Keep for potential future use, though not needed for loading
import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import precision_recall_fscore_support
from datasets import Dataset, load_from_disk # Import load_from_disk

# Optional: Set environment variable for MPS memory management if needed
# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Retrieve your Hugging Face access token from your environment (likely not needed for DistilBERT)
hf_token = os.getenv("HF_ACCESS_TOKEN")

# --- Configuration ---
# Path to the directory where the tokenized dataset was saved by tokenizer.py
TOKENIZED_DATASET_DIR = 'data/tokenized_hf_dataset'
# Model checkpoint should match the one used in tokenizer.py
MODEL_CHECKPOINT = "distilbert-base-uncased"
# Output directory for saving training results and checkpoints
OUTPUT_DIR = "./results" # Matches the user's project structure
NUM_LABELS = 2 # Binary classification (Readmitted: 0 or 1)

# --- Load Tokenized Dataset from Disk ---
print(f"Loading tokenized dataset from directory: {TOKENIZED_DATASET_DIR}...")
try:
    # Load the dataset directly from the saved directory using datasets library
    dataset = load_from_disk(TOKENIZED_DATASET_DIR)
    print(f"Loaded dataset with {len(dataset)} records.")
    print("\nDataset Info:")
    print(dataset) # Print info to verify structure (columns, features)

    # Verify required columns are present after loading
    required_cols = ['input_ids', 'attention_mask', 'label']
    if not all(col in dataset.column_names for col in required_cols):
        missing_cols = [col for col in required_cols if col not in dataset.column_names]
        raise ValueError(f"Dataset loaded from disk is missing required columns: {missing_cols}")

except FileNotFoundError:
    print(f"Error: Tokenized dataset directory not found at {TOKENIZED_DATASET_DIR}")
    print("Ensure you have run the updated tokenizer.py successfully and the directory exists.")
    exit()
except Exception as e:
    print(f"Error loading tokenized dataset from disk: {e}")
    exit()

# --- Prepare Dataset for Training ---
# Dataset is already loaded in the correct Hugging Face Dataset format

# Split the dataset into training (e.g., 75%) and evaluation (e.g., 25%).
print("\nSplitting the dataset...")
# Using 75/25 split as mentioned in the initial project description
split_dataset = dataset.train_test_split(test_size=0.25, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]
print(f"Split data into {len(train_dataset)} training and {len(eval_dataset)} evaluation samples.")

# --- Initialize Model ---
print(f"\nInitializing model from {MODEL_CHECKPOINT}...")
try:
    # Load the pre-trained model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=NUM_LABELS
        # token=hf_token # Not usually needed for public models like DistilBERT
    )
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
    print("Metrics loaded.")
except Exception as e:
    print(f"Error loading metrics: {e}")

# Function to compute metrics during evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Get predictions by finding the index with the highest logit value
    predictions = np.argmax(logits, axis=-1)

    # Compute each metric
    try:
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
        # Specify average='binary' for binary classification tasks
        precision = precision_metric.compute(predictions=predictions, references=labels, average="binary")["precision"]
        recall = recall_metric.compute(predictions=predictions, references=labels, average="binary")["recall"]
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")["f1"]
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {} # Return empty dict if metrics fail

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# --- Training Setup ---
print("\nSetting up training arguments and trainer...")
try:
    # Initialize the tokenizer again (needed for the DataCollator)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT) # No token needed
except Exception as e:
    print(f"Error initializing tokenizer for DataCollator: {e}")
    exit()

# Data collator handles dynamic padding within batches
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define training arguments based on initial project description and best practices
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,          # Directory to save checkpoints and results
    num_train_epochs=3,             # Number of training epochs
    per_device_train_batch_size=4,  # Batch size for training (kept small for low RAM)
    per_device_eval_batch_size=4,   # Batch size for evaluation (kept small for low RAM)
    learning_rate=2e-5,             # Learning rate
    weight_decay=0.01,              # Weight decay for regularization
    eval_strategy="epoch",          # Use corrected argument name for evaluation timing
    save_strategy="epoch",          # Save a checkpoint at the end of each epoch
    load_best_model_at_end=True,    # Load the best model checkpoint at the end of training
    metric_for_best_model="f1",     # Use F1 score to determine the best model
    greater_is_better=True,         # Higher F1 is better
    logging_dir='./logs',           # Directory for logs
    logging_steps=100,              # Log training loss every 100 steps
    # bf16=False,                   # Disable bf16 unless hardware supports it well
    # fp16=True,                    # Consider fp16 if you have a compatible GPU (NVIDIA)
    push_to_hub=False,              # Set to True to push model to Hugging Face Hub
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,            # Pass tokenizer for padding/saving purposes
    data_collator=data_collator,    # Use the data collator for dynamic padding
)
print("Trainer initialized.")

# --- Train and Evaluate ---
print("\nStarting training...")
try:
    trainer.train()
    print("Training finished.")
except Exception as e:
    print(f"Error during training: {e}")
    exit()

print("\nEvaluating the best model on the evaluation set...")
try:
    eval_results = trainer.evaluate()
    print("\nEvaluation Results:")
    # Print results in a formatted way
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
except Exception as e:
    print(f"Error during evaluation: {e}")

# --- Save Final Model (Optional) ---
# The best model is automatically saved in a checkpoint directory within OUTPUT_DIR
# based on save_strategy and load_best_model_at_end=True.
# You can explicitly save the final trained model state if desired:
# final_model_path = os.path.join(OUTPUT_DIR, "final_model_state")
# trainer.save_model(final_model_path)
# tokenizer.save_pretrained(final_model_path) # Save tokenizer alongside model
# print(f"Final model state saved to {final_model_path}")

print("\nScript finished.")
