import os
import numpy as np
import pandas as pd
import evaluate
from transformers import DistilBertForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset, load_from_disk
import torch
from torch import nn
from collections import Counter 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
import matplotlib.pyplot as plt 


# --- Configuration ---
TOKENIZED_DATASET_DIR = '/content/drive/My Drive/Readmissions_Research/Data/tokenized_hf_dataset_cpt_icd'
MODEL_CHECKPOINT = "distilbert-base-uncased"
OUTPUT_DIR = "/content/drive/My Drive/Readmissions_Research/Results/ClassWeight_8E_DetailedEval" 
LOGGING_DIR = os.path.join(OUTPUT_DIR, 'logs')
NUM_LABELS = 2

class_weights = None

# --- Custom Trainer with Weighted Loss ---
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        global class_weights
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if class_weights is None:
            print("Warning: Class weights not set for WeightedTrainer. Using unweighted loss.")
            loss_fct = nn.CrossEntropyLoss()
        else:
            loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(model.device))

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# --- Load Tokenized Dataset from Disk ---
print(f"Loading tokenized dataset from directory: {TOKENIZED_DATASET_DIR}...")
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)

    dataset = load_from_disk(TOKENIZED_DATASET_DIR)
    print(f"Loaded dataset with {len(dataset)} records.")
    print("\nDataset Info:")
    print(dataset)
    required_cols = ['input_ids', 'attention_mask', 'label']
    if 'prompt' in dataset.column_names:
        required_cols.append('prompt')
        print("'prompt' column found in dataset and will be included in output CSV.")
    else:
        print("Warning: 'prompt' column not found in dataset. It will not be in the output CSV with predictions.")

    if not all(col in dataset.column_names for col in ['input_ids', 'attention_mask', 'label']):
        missing_core_cols = [col for col in ['input_ids', 'attention_mask', 'label'] if col not in dataset.column_names]
        raise ValueError(f"Dataset loaded from disk is missing core required columns: {missing_core_cols}")

except FileNotFoundError:
    print(f"Error: Tokenized dataset directory not found at {TOKENIZED_DATASET_DIR}")
    exit()
except Exception as e:
    print(f"Error loading tokenized dataset from disk: {e}")
    exit()

# --- Prepare Dataset for Training ---
print("\nSplitting the dataset...")
split_dataset = dataset.train_test_split(test_size=0.20, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"] 
print(f"Split data into {len(train_dataset)} training and {len(eval_dataset)} evaluation samples.")



labels_in_train = train_dataset['label']
class_counts = Counter(labels_in_train)
count_class_0 = class_counts.get(0, 0)
count_class_1 = class_counts.get(1, 0)

if count_class_0 == 0 or count_class_1 == 0:
    print("Warning: One of the classes has zero samples in the training set. Using equal weights.")
    class_weights = torch.tensor([1.0, 1.0], dtype=torch.float)
else:
    total_train_samples = len(labels_in_train)
    weight_class_0 = total_train_samples / (NUM_LABELS * count_class_0)
    weight_class_1 = total_train_samples / (NUM_LABELS * count_class_1)
    class_weights = torch.tensor([weight_class_0, weight_class_1], dtype=torch.float)

print(f"Dynamically calculated class counts in training set: Class 0: {count_class_0}, Class 1: {count_class_1}")
print(f"Dynamically calculated class weights: {class_weights.tolist()}")

# --- Initialize Model ---
print(f"\nInitializing model from {MODEL_CHECKPOINT}...")
try:
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=NUM_LABELS
    )
    if model.config.label2id is None:
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
    print("Metrics loaded.")
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
        print(f"Error computing metrics: {e}")
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
    num_train_epochs=20, 
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

print("Initializing WeightedTrainer with dynamically calculated class weights.")
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

# --- Train ---
print("\nStarting training...")
try:
    trainer.train()
    print("Training finished.")
except Exception as e:
    print(f"Error during training: {e}")
    exit()


print("\nEvaluating the best model on the evaluation set (default threshold 0.5)...")
eval_results = None
try:
    eval_results = trainer.evaluate() # trainer.model is already the best model
    print("\nStandard Evaluation Results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
except Exception as e:
    print(f"Error during standard evaluation: {e}")


if eval_results: 
    print("\n--- Generating Detailed Predictions, Scores, and Confusion Matrix ---")
    try:
        predictions_output = trainer.predict(eval_dataset)
        logits = predictions_output.predictions
        true_labels = predictions_output.label_ids

        predicted_labels_argmax = np.argmax(logits, axis=-1)

        probabilities_all_classes = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        probs_positive_class = probabilities_all_classes[:, 1]

        print(f"Successfully got probabilities for {len(probs_positive_class)} evaluation samples.")


        cm = confusion_matrix(true_labels, predicted_labels_argmax, labels=[0, 1])

        display_labels = ["Not Readmitted (0)", "Readmitted (1)"]
        if model.config.id2label:
            try:
                display_labels = [model.config.id2label[0], model.config.id2label[1]]
            except KeyError:
                print("Warning: Could not get display labels from model.config.id2label. Using default labels.")

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix - Evaluation Set (Default Threshold)")
        cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_eval.png")
        plt.savefig(cm_path)
        print(f"Confusion matrix saved to {cm_path}")


        eval_df_with_scores = eval_dataset.to_pandas()


        eval_df_with_scores['Readmission_Predicted'] = predicted_labels_argmax

        eval_df_with_scores['Readmission_Probability'] = [f"{p*100:.2f}%" for p in probs_positive_class]


        output_columns = []
        if 'prompt' in eval_df_with_scores.columns:
            output_columns.append('prompt')


        output_columns.extend(['label', 'Readmission_Predicted', 'Readmission_Probability'])

        final_output_columns = [col for col in output_columns if col in eval_df_with_scores.columns]

        output_csv_path = os.path.join(OUTPUT_DIR, "evaluation_predictions_with_scores.csv")
        eval_df_with_scores[final_output_columns].to_csv(output_csv_path, index=False)
        print(f"Evaluation data with prediction scores saved to: {output_csv_path}")

    except Exception as e:
        print(f"Error during detailed prediction analysis or confusion matrix generation: {e}")
else:
    print("Skipping detailed predictions and confusion matrix due to standard evaluation failure.")


