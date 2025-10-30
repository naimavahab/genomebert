import optuna
import evaluate
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
)
from datasets import load_dataset
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score
import numpy as np
import os

# Disable WandB and other loggers for a clean run
os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_HOME"] = "/opt/home/s4021545/MosaicBert/modernbert/interpret"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

TASK_NAME = "promoter_all"


# === 1. Load and preprocess data only ONCE ===
print("Step 1: Loading and preprocessing data...")
full_dataset = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks")

# Filter for the specific task across all splits
print(f"Filtering dataset for task: {TASK_NAME}")
filtered_dataset = {}
for split in full_dataset.keys():
    split_ds = full_dataset[split].filter(lambda example: example["task"] == TASK_NAME)
    split_ds = split_ds.remove_columns(["task"])
    # select a small subset for quick testing
   # filtered_dataset[split] = split_ds.select(range(min(10, len(split_ds))))
    data = []
    for i in split_ds.select(range(min(500, len(split_ds)))):
      if i['label'] == 1 and i['name'].split('|')[1] == 'TATA':
          data.append(i)
          print(i)
    filtered_dataset[split] = data
    
# === 2. Write sequences to FASTA file ===
output_fasta = f"{TASK_NAME}_sequences.fasta"
print(f"Writing sequences to {output_fasta}...")

with open(output_fasta, "w") as fasta_file:
    seq_count = 0
    for split_name, dataset_split in filtered_dataset.items():
        for i, example in enumerate(dataset_split):
            print(example)
            seq = example.get("sequence", example.get("input", ""))
            fasta_file.write(f">{split_name}_seq_{i}\n{seq}\n")
            seq_count += 1

print(f"✅ Done! Wrote {seq_count} sequences to {output_fasta}.")

