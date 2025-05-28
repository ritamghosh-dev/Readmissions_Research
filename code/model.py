import os
import numpy as np
import pandas as pd
import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import precision_recall_fscore_support # Keep for reference, but evaluate handles it
from datasets import Dataset, load_from_disk
import torch
from torch import nn

# Optional: Set environment variable for MPS memory management if needed
# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Retrieve your Hugging Face access token from your environment (likely not needed for DistilBERT)
hf_token = os.getenv("HF_ACCESS_TOKEN")

# --- Configuration ---
TOKENIZED_DATASET_DIR = 'data/tokenized_hf_dataset'
MODEL_CHECKPOINT = "distilbert-base-uncased"
OUTPUT_DIR = "./results"
NUM_LABELS = 2

# --- Class Weights Calculation ---
count_class_0 = 28798
count_class_1 = 3588
total_samples = count_class_0 + count_class_1
weight_class_0 = total_samples / (NUM_LABELS * count_class_0)
weight_class_1 = total_samples / (NUM_LABELS * count_class_1)
class_weights = torch.tensor([weight_class_0, weight_class_1], dtype=torch.float)
print(f"Calculated class weights: {class_weights.tolist()}")

# --- Custom Trainer with Weighted Loss ---
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs): # Added **kwargs
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# --- Load Tokenized Dataset from Disk ---
print(f"Loading tokenized dataset from directory: {TOKENIZED_DATASET_DIR}...")
try:
    dataset = load_from_disk(TOKENIZED_DATASET_DIR)
    print(f"Loaded dataset with {len(dataset)} records.")
    print("\nDataset Info:")
    print(dataset)
    required_cols = ['input_ids', 'attention_mask', 'label']
    if not all(col in dataset.column_names for col in required_cols):
        missing_cols = [col for col in required_cols if col not in dataset.column_names]
        raise ValueError(f"Dataset loaded from disk is missing required columns: {missing_cols}")
except FileNotFoundError:
    print(f"Error: Tokenized dataset directory not found at {TOKENIZED_DATASET_DIR}")
    exit()
except Exception as e:
    print(f"Error loading tokenized dataset from disk: {e}")
    exit()

# --- Prepare Dataset for Training ---
print("\nSplitting the dataset...")
split_dataset = dataset.train_test_split(test_size=0.25, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]
print(f"Split data into {len(train_dataset)} training and {len(eval_dataset)} evaluation samples.")

# --- Initialize Model ---
print(f"\nInitializing model from {MODEL_CHECKPOINT}...")
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=NUM_LABELS
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

# --- MODIFIED compute_metrics function ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    try:
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
        # Removed zero_division=0 from the .compute() calls
        precision = precision_metric.compute(predictions=predictions, references=labels, average="binary")["precision"]
        recall = recall_metric.compute(predictions=predictions, references=labels, average="binary")["recall"]
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")["f1"]
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {} # Return empty dict if metrics computation fails
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
# --------------------------------------

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
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_dir='./logs',
    logging_steps=100,
    push_to_hub=False,
)

# Use the Custom WeightedTrainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
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
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
except Exception as e:
    print(f"Error during evaluation: {e}")

print("\nScript finished.")
