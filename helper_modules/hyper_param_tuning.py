import optuna
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments,PreTrainedTokenizerFast,AutoModelForSequenceClassification
from datasets import load_dataset, load_metric
from datasets import load_dataset, load_metric,load_from_disk


# Load Dataset and Metric
def get_data():
    # Use a demo dataset (replace with your dataset as needed)
    tokeniser_path = '4k_vocab_dna.json'
    special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
    tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokeniser_path,
            special_tokens=special_tokens,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            sep_token="<sep>",
            pad_token="<pad>",
            cls_token="<cls>",
            mask_token="<mask>",
            )
    dataset = load_from_disk('./new_runs/final_ds') #./species/final_ds') #lncRNA/lnc_dataset') #miRNA/mirna_dataset')

    train_dataset = dataset['train'].select(range(500))
    eval_dataset = dataset['train'].select(range(1000,1300))
    
    def preprocess_function(examples):
    # Tokenize each RNA sequence and prepare for model input
      tokenized_inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=40)
      tokenized_inputs["label"] = examples["label"]  # Adjust label key if different in your dataset
      return tokenized_inputs
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True)
    
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return train_dataset,eval_dataset
    
# Objective function for Optuna
def objective(trial):
    # Hyperparameter search space
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    num_epochs = trial.suggest_int("num_epochs", 2, 7)

    # Load Data
    train_dataset, eval_dataset = get_data()
    
    # Define Model
    model = AutoModelForSequenceClassification.from_pretrained('./genome_bert_base')

    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=1.00e-06,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1.00e-06,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=lambda p: {"f1": load_metric("f1").compute(predictions=p.predictions.argmax(-1), references=p.label_ids)["f1"]},
    )

    # Train and evaluate
    trainer.train()
    eval_results = trainer.evaluate()
    return eval_results["eval_f1"]

# Run Optuna
def tune_model():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print(f"Params: {trial.params}")
    return trial

if __name__ == "__main__":
    best_trial = tune_model()

