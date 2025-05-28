import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset, Features, Value, Sequence # Import Features for schema definition
import os

hf_token = os.getenv("HF_ACCESS_TOKEN")

# --- Configuration ---
INPUT_CSV_PATH = 'data/narrative_data.csv'
OUTPUT_DATASET_DIR = 'data/tokenized_hf_dataset_explainability'
MODEL_CHECKPOINT = "distilbert-base-uncased"
MAX_LENGTH = 512

# --- Load Data ---
print(f"Loading data from {INPUT_CSV_PATH}...")
try:
    df = pd.read_csv(INPUT_CSV_PATH)
    if 'prompt' not in df.columns or 'label' not in df.columns:
         raise ValueError("Input CSV must contain 'prompt' and 'label' columns.")
    df = df.reset_index(drop=True)
    df['label'] = df['label'].astype(int)
    # Ensure prompt is string
    df['prompt'] = df['prompt'].astype(str)
    print(f"Loaded {len(df)} records.")
except FileNotFoundError:
    print(f"Error: Input file not found at {INPUT_CSV_PATH}")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Define initial features including prompt
initial_features = Features({
    'label': Value('int64'),
    'prompt': Value('string')
})
dataset = Dataset.from_pandas(df, features=initial_features)
print("Converted DataFrame to Hugging Face Dataset.")

# --- Initialize Tokenizer ---
print(f"Initializing tokenizer from {MODEL_CHECKPOINT}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    # Ensure tokenizer has a pad token if not already set (DistilBERT usually has [PAD])
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print("Added pad_token to tokenizer.")
except Exception as e:
    print(f"Error initializing tokenizer: {e}")
    exit()

# --- Define Tokenization Function ---
def tokenize_function(examples):
    # The tokenizer will return input_ids, attention_mask
    # The 'prompt' field will be passed through if not in remove_columns
    return tokenizer(examples['prompt'], padding='max_length', truncation=True, max_length=MAX_LENGTH)

# --- Tokenize the Dataset ---
print("Tokenizing the dataset...")
try:
    # Define the features *after* tokenization, now including prompt
    tokenized_features_with_prompt = Features({
        'label': Value('int64'),
        'prompt': Value('string'), # Keep prompt
        'input_ids': Sequence(feature=Value(dtype='int32'), length=MAX_LENGTH),
        'attention_mask': Sequence(feature=Value(dtype='int8'), length=MAX_LENGTH)
    })
    # Apply tokenization. Remove_columns is NOT removing 'prompt' here.
    # If 'prompt' is not explicitly removed, it's usually kept if it's part of the input features.
    # To be safe, let's ensure map doesn't remove it by default or explicitly handle its presence.
    # The map function will add new columns ('input_ids', 'attention_mask').
    # Original columns not specified in remove_columns are typically kept.
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        # remove_columns=['some_other_column_if_any'] # We want to keep 'prompt' and 'label'
                                                    # Tokenizer output will be added.
        features=tokenized_features_with_prompt # Ensure new schema is applied
        )
    # Ensure 'prompt' is still there, and other columns like 'label'
    print("Tokenization complete.")
    print("\nTokenized Dataset Info (should include 'prompt'):")
    print(tokenized_dataset)
except Exception as e:
    print(f"Error during tokenization: {e}")
    exit()

# --- Save Tokenized Dataset to Disk ---
print(f"\nSaving tokenized dataset to directory: {OUTPUT_DATASET_DIR}...")
try:
    os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)
    tokenized_dataset.save_to_disk(OUTPUT_DATASET_DIR)
    print("Tokenized dataset saved successfully.")
except Exception as e:
    print(f"Error saving tokenized dataset: {e}")

