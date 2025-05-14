
```
Readmissions_Research
├─ data
│  ├─ cpt_code.csv
│  ├─ discharge_code.csv
│  ├─ narrative_data.csv
│  ├─ original_data.csv
│  ├─ preprocessed_data.csv
│  └─ tokenized_hf_dataset
│     ├─ data-00000-of-00001.arrow
│     ├─ dataset_info.json
│     └─ state.json
├─ mapping.py
├─ model.py
├─ preprocessing.py
├─ results
│  ├─ checkpoint-1000
│  │  ├─ config.json
│  │  ├─ model.safetensors
│  │  ├─ optimizer.pt
│  │  ├─ rng_state.pth
│  │  ├─ scheduler.pt
│  │  ├─ special_tokens_map.json
│  │  ├─ tokenizer.json
│  │  ├─ tokenizer_config.json
│  │  ├─ trainer_state.json
│  │  ├─ training_args.bin
│  │  └─ vocab.txt
│  ├─ checkpoint-1500
│  │  ├─ config.json
│  │  ├─ model.safetensors
│  │  ├─ optimizer.pt
│  │  ├─ rng_state.pth
│  │  ├─ scheduler.pt
│  │  ├─ special_tokens_map.json
│  │  ├─ tokenizer.json
│  │  ├─ tokenizer_config.json
│  │  ├─ trainer_state.json
│  │  ├─ training_args.bin
│  │  └─ vocab.txt
│  ├─ checkpoint-2000
│  │  ├─ config.json
│  │  ├─ model.safetensors
│  │  ├─ optimizer.pt
│  │  ├─ rng_state.pth
│  │  ├─ scheduler.pt
│  │  ├─ special_tokens_map.json
│  │  ├─ tokenizer.json
│  │  ├─ tokenizer_config.json
│  │  ├─ trainer_state.json
│  │  ├─ training_args.bin
│  │  └─ vocab.txt
│  ├─ checkpoint-2500
│  │  ├─ config.json
│  │  ├─ model.safetensors
│  │  ├─ optimizer.pt
│  │  ├─ rng_state.pth
│  │  ├─ scheduler.pt
│  │  ├─ special_tokens_map.json
│  │  ├─ tokenizer.json
│  │  ├─ tokenizer_config.json
│  │  ├─ trainer_state.json
│  │  ├─ training_args.bin
│  │  └─ vocab.txt
│  ├─ checkpoint-3000
│  │  ├─ config.json
│  │  ├─ model.safetensors
│  │  ├─ optimizer.pt
│  │  ├─ rng_state.pth
│  │  ├─ scheduler.pt
│  │  ├─ special_tokens_map.json
│  │  ├─ tokenizer.json
│  │  ├─ tokenizer_config.json
│  │  ├─ trainer_state.json
│  │  ├─ training_args.bin
│  │  └─ vocab.txt
│  ├─ checkpoint-3500
│  │  ├─ config.json
│  │  ├─ model.safetensors
│  │  ├─ optimizer.pt
│  │  ├─ rng_state.pth
│  │  ├─ scheduler.pt
│  │  ├─ special_tokens_map.json
│  │  ├─ tokenizer.json
│  │  ├─ tokenizer_config.json
│  │  ├─ trainer_state.json
│  │  ├─ training_args.bin
│  │  └─ vocab.txt
│  ├─ checkpoint-4000
│  │  ├─ config.json
│  │  ├─ model.safetensors
│  │  ├─ optimizer.pt
│  │  ├─ rng_state.pth
│  │  ├─ scheduler.pt
│  │  ├─ special_tokens_map.json
│  │  ├─ tokenizer.json
│  │  ├─ tokenizer_config.json
│  │  ├─ trainer_state.json
│  │  ├─ training_args.bin
│  │  └─ vocab.txt
│  ├─ checkpoint-4179
│  │  ├─ config.json
│  │  ├─ model.safetensors
│  │  ├─ optimizer.pt
│  │  ├─ rng_state.pth
│  │  ├─ scheduler.pt
│  │  ├─ special_tokens_map.json
│  │  ├─ tokenizer.json
│  │  ├─ tokenizer_config.json
│  │  ├─ trainer_state.json
│  │  ├─ training_args.bin
│  │  └─ vocab.txt
│  ├─ checkpoint-4500
│  │  ├─ config.json
│  │  ├─ model.safetensors
│  │  ├─ optimizer.pt
│  │  ├─ rng_state.pth
│  │  ├─ scheduler.pt
│  │  ├─ special_tokens_map.json
│  │  ├─ tokenizer.json
│  │  ├─ tokenizer_config.json
│  │  ├─ trainer_state.json
│  │  ├─ training_args.bin
│  │  └─ vocab.txt
│  ├─ checkpoint-500
│  │  ├─ config.json
│  │  ├─ model.safetensors
│  │  ├─ optimizer.pt
│  │  ├─ rng_state.pth
│  │  ├─ scheduler.pt
│  │  ├─ special_tokens_map.json
│  │  ├─ tokenizer.json
│  │  ├─ tokenizer_config.json
│  │  ├─ trainer_state.json
│  │  ├─ training_args.bin
│  │  └─ vocab.txt
│  ├─ checkpoint-5000
│  │  ├─ config.json
│  │  ├─ model.safetensors
│  │  ├─ optimizer.pt
│  │  ├─ rng_state.pth
│  │  ├─ scheduler.pt
│  │  ├─ special_tokens_map.json
│  │  ├─ tokenizer.json
│  │  ├─ tokenizer_config.json
│  │  ├─ trainer_state.json
│  │  ├─ training_args.bin
│  │  └─ vocab.txt
│  └─ checkpoint-5013
│     ├─ config.json
│     ├─ model.safetensors
│     ├─ optimizer.pt
│     ├─ rng_state.pth
│     ├─ scheduler.pt
│     ├─ special_tokens_map.json
│     ├─ tokenizer.json
│     ├─ tokenizer_config.json
│     ├─ trainer_state.json
│     ├─ training_args.bin
│     └─ vocab.txt
└─ tokenizer.py

```