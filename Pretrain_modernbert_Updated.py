# train_modernbert_genome.py

# Install necessary packages
# !pip install -q datasets transformers wandb

# !pip show transformers (have use Version: 4.53.0)

# !pip install Bio
import torch
torch.cuda.empty_cache()
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import os
os.environ["PYTORCH_ENABLE_DYNAMO"] = "0"

import math
import time
import pandas as pd
import numpy as np
import wandb
from datetime import datetime
from datasets import load_dataset

from datasets import Dataset
from transformers import (
    PreTrainedTokenizerFast,
    ModernBertConfig,
    ModernBertForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    EvalPrediction
)


import torch
if torch.cuda.is_available():
    print("CUDA is available. Number of GPUs:", torch.cuda.device_count())
else:
    print("CUDA is not available. No GPUs detected.")
    
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_time = time.time()

def load_data(csv_path, max_samples=4377233):
    df = pd.read_csv(csv_path)
    df['length'] = df['sequence'].str.len()
    print("Dataset loaded. Sequence length stats:\n", df['length'].describe())
    print('\n')

    dataset = Dataset.from_pandas(df)
    dataset = dataset.rename_column("sequence", "text")
    return dataset.select(range(min(max_samples, len(dataset))))

# def load_data_from_hf(max_samples=None):
#     # Load full dataset from Hugging Face hub
#     print("Load full dataset from Hugging Face hub \n")
#     dataset = load_dataset("InstaDeepAI/multi_species_genomes", split="train[:500]")
    
#     # Convert to pandas for stats (optional)
#     df = dataset.to_pandas()
#     df['length'] = df['sequence'].str.len()
#     print("Dataset loaded from Hugging Face.")
#     print("Sequence length stats:\n", df['length'].describe())
#     print('\n')

#     # Convert back to Dataset if needed, rename "sequence" to "text"
#     dataset = Dataset.from_pandas(df)
#     dataset = dataset.rename_column("sequence", "text")
    
#     if max_samples:
#         return dataset.select(range(min(max_samples, len(dataset))))
#     else:
#         return dataset


def load_tokenizer(tokenizer_path):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        special_tokens=["<unk>", "<pad>", "<cls>", "<sep>", "<mask>"],
        unk_token="<unk>",
        pad_token="<pad>",
        cls_token="<cls>",
        sep_token="<sep>",
        mask_token="<mask>",
    )
    required_tokens = ["<pad>", "<cls>", "<sep>", "<mask>", "<unk>", "<s>", "</s>"]
    for token in required_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            print(f"Missing token: {token}")
            print('\n')
        else:
            print(f"{token} ID: {token_id}")
            print('\n')
    return tokenizer


def tokenize_dataset(dataset, tokenizer, max_len=128):
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_len
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
        remove_columns=[col for col in dataset.column_names if col != "text"]
    )
    print("Tokenization complete.")
    print('\n')
    return tokenized


def compute_metrics(eval_preds: EvalPrediction):
    predictions, labels = eval_preds.predictions, eval_preds.label_ids
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    logits = torch.tensor(predictions)
    labels = torch.tensor(labels)

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    perplexity = torch.exp(loss).item() if loss.item() < 100 else float("inf")

    # wandb.log({
    #     "eval_loss": loss.item(),
    #     "eval_perplexity": perplexity
    # })
    
    # print(f"[Eval] Loss: {loss.item():.4f}, Perplexity: {perplexity:.4f}")
    print('\n')

    return {"eval_loss": loss.item(), "perplexity": perplexity}


def main():
    # Configuration
    csv_path ="subset_full_dataset_1000seq.csv" #subset_100_truncated_len3000.csv"
    tokenizer_path ="4k_unigram_tokenizer.json"
    output_dir = "./model_files_4k_unigram"
    save_model_path = "/home/ec2-user/Genomic_multi/model_files_4k_unigram"
    
    # Login to Weights & Biases

    wandb.login(key="e7e01f84083cf5898f0c4607d529b71cb7c89f73")
    print(" Logged into Weights & Biases.")
    print('\n')

    dataset = load_data(csv_path)
    # Load data from HF dataset instead of CSV
    # dataset = load_data_from_hf(max_samples=500) 
    tokenizer = load_tokenizer(tokenizer_path)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    tokenized_dataset.save_to_disk("./my_4k_tokenized_dataset") # Specify a directory to save to

    # Train/Test Split
    split = tokenized_dataset # tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    # print(f"Train size: {len(split['train'])}, Test size: {len(split['test'])}")

    # Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # Model Config and Initialization
    config = ModernBertConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        mask_token_id=tokenizer.mask_token_id,
        global_rope_theta=10000,
        reference_compile=False
    )

    model = ModernBertForMaskedLM(config=config)
    print("Model initialized.")
    print('\n')

    # # Training Arguments
    # training_args = TrainingArguments(
    #     output_dir=output_dir,
    #     overwrite_output_dir=True,
    #     num_train_epochs=20,
    #     per_device_train_batch_size=96,             
    #     per_device_eval_batch_size=96,
    #     gradient_accumulation_steps=2,
    #     eval_accumulation_steps=8,
    #     learning_rate=5e-5,                           
    #     adam_beta1=0.9,
    #     adam_beta2=0.98,
    #     adam_epsilon=1e-6,
    #     weight_decay=0.01,
    #     lr_scheduler_type="linear",                   
    #     warmup_ratio=0.06,
    #     save_steps=1000,
    #     save_total_limit=2,
    #     logging_steps=10,
	#     logging_dir='./logs',
    #     eval_strategy="steps",
    #     eval_steps=10,
    #     fp16=False,
	#     torch_compile=False,
    #     no_cuda=False,
    #     report_to=["wandb"],
    #     run_name="modernbert-pretrainmodel-logs-Xrun"
    # )
    
    training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=50,
    per_device_train_batch_size=128,             
    #per_device_eval_batch_size=64,
    gradient_accumulation_steps=8,
    #eval_accumulation_steps=2,
    learning_rate=5e-5,                           
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-6,
    weight_decay=0.01,
    lr_scheduler_type="linear",                   
    warmup_ratio=0.06,
    save_strategy="epoch",       # save at the end of each epoch
    save_total_limit=2,          # keep only the last 2 checkpoints
    logging_steps=10,
    logging_dir='./logs',
   # eval_strategy="epoch", # also evaluate at the end of each epoch
    fp16=True,
    torch_compile=False,
    no_cuda=False,
    report_to=["wandb"],
    run_name="pretrained_modernBERT"
)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split,
       # eval_dataset=split["test"].shuffle(seed=42),
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    print("Starting training...")
    print('\n')
        # Access the number of GPUs configured for the Trainer
    if hasattr(trainer.args, '_n_gpu'):
        print(f"Hugging Face Trainer is configured to use {trainer.args._n_gpu} GPUs.")
    else:
        print("Could not determine GPU usage from Trainer arguments directly (e.g., using DistributedDataParallel).")

        
    trainer.train()
    

    trainer.save_model(save_model_path)
    print(f"Model saved to: {save_model_path}")
    print('\n')


    #Final Evaluation

    # print("\nRunning final evaluation on test set...")
    # eval_dataset = split["test"]

    # final_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    # print("Final Evaluation Results:")
    # print(final_metrics)
    # wandb.log(final_metrics)
    # print(f"Final Evaluation Loss: {final_metrics.get('eval_loss', 0):.4f}, "
    #     f"Perplexity: {final_metrics.get('eval_perplexity', float('nan')):.4f}")

    # Record end time
    end_time = time.time()

    # Print total training time
    total_time_sec = end_time - start_time
    hours = int(total_time_sec // 3600)
    minutes = int((total_time_sec % 3600) // 60)
    seconds = int(total_time_sec % 60)
    print(f"Total training time: {hours}h {minutes}m {seconds}s")


if __name__ == "__main__":
    
    main()

