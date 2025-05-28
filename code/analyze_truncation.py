import pandas as pd
from transformers import AutoTokenizer
import numpy as np
import os

NARRATIVE_CSV_PATH = 'data/narrative_data.csv'
PROMPT_COLUMN_NAME = 'prompt'
MODEL_CHECKPOINT = "distilbert-base-uncased"
MAX_MODEL_LENGTH = 512

def run_truncation_analysis():
    print(f"Loading narratives from {NARRATIVE_CSV_PATH}...")
    if not os.path.exists(NARRATIVE_CSV_PATH):
        print(f"Error: File not found at {NARRATIVE_CSV_PATH}")
        return

    try:
        df = pd.read_csv(NARRATIVE_CSV_PATH)
        if PROMPT_COLUMN_NAME not in df.columns:
            print(f"Error: Column '{PROMPT_COLUMN_NAME}' not found in {NARRATIVE_CSV_PATH}.")
            print(f"Available columns are: {df.columns.tolist()}")
            return
        narratives = df[PROMPT_COLUMN_NAME].astype(str).tolist()
        print(f"Loaded {len(narratives)} narratives.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if not narratives:
        print("No narratives found to analyze.")
        return

    print(f"\nInitializing tokenizer from {MODEL_CHECKPOINT}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    except Exception as e:
        print(f"Error initializing tokenizer: {e}")
        return

    num_special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    effective_max_content_length = MAX_MODEL_LENGTH - num_special_tokens

    print(f"\nTokenizer: {MODEL_CHECKPOINT}")
    print(f"Model Max Sequence Length (including special tokens): {MAX_MODEL_LENGTH}")
    print(f"Number of special tokens added by tokenizer (e.g., [CLS], [SEP]): {num_special_tokens}")
    print(f"Effective max length for narrative content (excluding special tokens): {effective_max_content_length}")

    truncated_count = 0
    total_tokens_lost = 0
    max_tokens_lost_for_one_sample = 0
    original_content_lengths = []
    
    print("\nAnalyzing narratives for truncation (this may take a moment for many narratives)...")
    for i, narrative_text in enumerate(narratives):
        if pd.isna(narrative_text) or not narrative_text.strip():
            original_content_lengths.append(0)
            continue

        original_token_ids = tokenizer.encode(narrative_text, add_special_tokens=False)
        original_content_token_count = len(original_token_ids)
        original_content_lengths.append(original_content_token_count)

        if original_content_token_count > effective_max_content_length:
            truncated_count += 1
            tokens_lost = original_content_token_count - effective_max_content_length
            total_tokens_lost += tokens_lost
            if tokens_lost > max_tokens_lost_for_one_sample:
                max_tokens_lost_for_one_sample = tokens_lost
            
    print("\n--- Truncation Analysis Results ---")
    total_narratives_analyzed = len(narratives)
    if total_narratives_analyzed == 0:
        print("No valid narratives were analyzed.")
        return

    percentage_truncated = (truncated_count / total_narratives_analyzed) * 100
    avg_tokens_lost_overall = total_tokens_lost / total_narratives_analyzed if total_narratives_analyzed > 0 else 0
    avg_tokens_lost_for_truncated = total_tokens_lost / truncated_count if truncated_count > 0 else 0

    print(f"Total narratives analyzed: {total_narratives_analyzed}")
    print(f"Number of narratives truncated: {truncated_count} ({percentage_truncated:.2f}%)")
    if truncated_count > 0:
        print(f"  Average tokens lost across ALL samples: {avg_tokens_lost_overall:.2f}")
        print(f"  Average tokens lost FOR TRUNCATED samples: {avg_tokens_lost_for_truncated:.2f}")
        print(f"  Maximum tokens lost for a single sample: {max_tokens_lost_for_one_sample}")

    if original_content_lengths:
        original_lengths_series = pd.Series(original_content_lengths)
        print("\nStatistics for original content token lengths (excluding special tokens):")
        print(original_lengths_series.describe().to_string())
        print(f"Number of narratives with content tokens exceeding effective max length ({effective_max_content_length}): "
              f"{(original_lengths_series > effective_max_content_length).sum()}")
    else:
        print("No valid original lengths recorded for statistical analysis.")

if __name__ == "__main__":
    run_truncation_analysis()
