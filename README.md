# Readmissions_Research

This project is designed to analyze and model hospital readmission data. Below is a brief description of each file and folder in the repository.

## File Structure

- **data/**
  - **cpt_code.csv**: Contains CPT procedure codes and their descriptions used for procedure lookups in [`mapping.py`](mapping.py).
  - **discharge_code.csv**: Maps discharge status codes to descriptive text for use in narratives, referenced in [`mapping.py`](mapping.py).
  - **narrative_data.csv**: The final output file with narrative summaries generated from patient data in [`mapping.py`](mapping.py).
  - **original_data.csv**: The raw input data file which is preprocessed in [`preprocessing.py`](preprocessing.py).
  - **preprocessed_data.csv**: The cleaned and processed data generated from [`preprocessing.py`](preprocessing.py).
  - **tokenized_hf_dataset/**: Contains the Hugging Face tokenized dataset files (arrow file, dataset_info.json, and state.json), produced by [`tokenizer.py`](tokenizer.py).

- **mapping.py**: Handles loading of external mappings (CPT, discharge codes), converting rows of patient data into narrative summaries, and providing lookup functions for procedure and diagnosis codes.

- **preprocessing.py**: Preprocesses the raw data including filtering, ICD code cleaning, comorbidity flag calculation for the Charlson Comorbidity Index, applying readmission logic, and ultimately generating the `preprocessed_data.csv`.

- **model.py**: Defines and trains a classification model using a transformer-based architecture. It includes:
  - Custom trainer with weighted loss (`WeightedTrainer`)
  - Metrics computation functions for model evaluation
  - Code to load a tokenized dataset from disk and split it for training and evaluation.

- **results/**: Contains checkpoint directories which store intermediate and final model artifacts such as configuration files, model weights (`model.safetensors`), optimizer states, and training logs.

## Setup and Usage

1. **Preprocessing Data:**  
   Run [`preprocessing.py`](preprocessing.py) to load raw data from `data/original_data.csv`, clean and process it, and save the final preprocessed data and computed features.

2. **Generating Narratives:**  
   Use [`mapping.py`](mapping.py) after preprocessing to generate narrative summaries from the preprocessed data and output the result to `data/narrative_data.csv`.

3. **Tokenization:**  
   The [`tokenizer.py`](tokenizer.py) script reads `data/narrative_data.csv`, tokenizes the text using DistilBERT, and saves a tokenized dataset in `data/tokenized_hf_dataset`.

4. **Model Training & Evaluation:**  
   Run [`model.py`](model.py) to load the tokenized dataset, set up the model, and train/evaluate it. Model checkpoints are saved in the `results/` folder.

This repository is structured to progress from raw data to a trained and evaluated model, with clear separation of concerns between preprocessing, narrative generation, tokenization, and model handling.