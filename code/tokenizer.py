import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset, Features, Value, Sequence # Import Features for schema definition
import os
import ast

# Retrieve your Hugging Face access token from environment variable
hf_token = os.getenv("HF_ACCESS_TOKEN") # Keep if needed, though DistilBERT is public

# --- Configuration ---
INPUT_CSV_PATH = 'data/narrative_data.csv' # Path based on project structure
OUTPUT_DATASET_DIR = 'data/tokenized_hf_dataset' # Output directory
# --- Use DistilBERT ---
MODEL_CHECKPOINT = "distilbert-base-uncased"
# --------------------
MAX_LENGTH = 512 # DistilBERT's default max length is 512

# --- Load Data ---
print(f"Loading data from {INPUT_CSV_PATH}...")
try:
    df = pd.read_csv(INPUT_CSV_PATH)
    if 'prompt' not in df.columns or 'label' not in df.columns:
         raise ValueError("Input CSV must contain 'prompt' and 'label' columns.")
    df = df.reset_index(drop=True)
    df['label'] = df['label'].astype(int)
    print(f"Loaded {len(df)} records.")
except FileNotFoundError:
    print(f"Error: Input file not found at {INPUT_CSV_PATH}")
    print("Ensure 'narrative_data.csv' is inside the 'data' directory.")
    exit()
except ValueError as ve:
     print(f"Error: {ve}")
     exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()


# --- Convert to Hugging Face Dataset ---
features = Features({
    'label': Value('int64'),
    'prompt': Value('string')
})
dataset = Dataset.from_pandas(df, features=features)
print("Converted DataFrame to Hugging Face Dataset.")

# --- Initialize Tokenizer ---
print(f"Initializing tokenizer from {MODEL_CHECKPOINT}...")
try:
    # DistilBERT doesn't typically require a token
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
except Exception as e:
    print(f"Error initializing tokenizer: {e}")
    exit()

# --- Define Tokenization Function ---
def tokenize_function(examples):
    prompts = [str(p) if pd.notna(p) else "" for p in examples['prompt']]
    # Use padding='max_length' if you want all sequences padded to 512,
    # or 'longest' to pad to the longest in the batch (usually more efficient)
    return tokenizer(prompts, padding='longest', truncation=True, max_length=MAX_LENGTH)

# --- Tokenize the Dataset ---
print("Tokenizing the dataset...")
try:
    tokenized_features = Features({
        'label': Value('int64'),
        'input_ids': Sequence(feature=Value(dtype='int32')),
        'attention_mask': Sequence(feature=Value(dtype='int8'))
    })
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['prompt'],
        features=tokenized_features
        )
    print("Tokenization complete.")
    print("\nTokenized Dataset Info:")
    print(tokenized_dataset)
except Exception as e:
    print(f"Error during tokenization: {e}")
    exit()

# --- Save Tokenized Dataset to Disk ---
print(f"\nSaving tokenized dataset to directory: {OUTPUT_DATASET_DIR}...")
try:
    os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)
    # Overwrite existing dataset if it exists
    tokenized_dataset.save_to_disk(OUTPUT_DATASET_DIR)
    print("Tokenized dataset saved successfully.")
except Exception as e:
    print(f"Error saving tokenized dataset: {e}")

