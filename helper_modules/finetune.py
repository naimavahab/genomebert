import torch
from transformers import Trainer, TrainingArguments,  AutoModel,  PreTrainedTokenizerFast,AutoModelForSequenceClassification
from multimolecule import RnaTokenizer, RiNALMoForSequencePrediction
from datasets import load_dataset, load_metric,load_from_disk
from sklearn.metrics import f1_score, matthews_corrcoef
# Initialize tokenizer and model
import numpy as np
import evaluate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support,matthews_corrcoef

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

#tokenizer = RnaTokenizer.from_pretrained('multimolecule/rinalmo')
model = AutoModelForSequenceClassification.from_pretrained('./genome_bert_base')
print(model)
# Load a Hugging Face dataset (e.g., a hypothetical RNA classification dataset)
# Replace 'rna_dataset' with your dataset name and split with your data split
dataset = load_from_disk('./new_runs/final_ds') #/species/final_ds') #lncRNA/lnc_dataset') #miRNA/mirna_dataset')
# You might want to create a validation split if it's not already in the dataset
#dataset = dataset.train_test_split(test_size=0.1)  # 90% train, 10% validation
print(dataset)


train_dataset = dataset['train']#.select(range(10:200))
eval_dataset = dataset['test']#.select(range(200:270))
print(train_dataset)
# Preprocess the dataset
def preprocess_function(examples):
    # Tokenize each RNA sequence and prepare for model input
    tokenized_inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=40)
    tokenized_inputs["label"] = examples["label"]  # Adjust label key if different in your dataset
    return tokenized_inputs

# Apply preprocessing to datasets
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

print(train_dataset[0])

# Convert datasets to torch tensor format
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results_mirna",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=6,
    weight_decay=1.0e-6,
    load_best_model_at_end=True,  # Automatically loads the best model after training
    logging_dir='./logs',  # Directory for storing logs
    learning_rate=1.0e-5,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1.0e-6

)
accuracy_metric = evaluate.load("accuracy")
mcc_metric = evaluate.load("matthews_correlation")
f1_metric = evaluate.load("f1")
# Define custom compute_metrics function with F1 and MCC



def compute_metrics(eval_preds):
    metrics = dict()
    accuracy_metric = load_metric('accuracy')
    precision_metric = load_metric('precision')
    recall_metric = load_metric('recall')
    f1_metric = load_metric('f1')
    matthews_metric = evaluate.load("matthews_correlation")
    logits = eval_preds.predictions
    labels = eval_preds.label_ids
    print(labels)
    preds = np.argmax(logits, axis=-1)
    metrics.update(accuracy_metric.compute(predictions=preds, references=labels))
    metrics.update(precision_metric.compute(predictions=preds, references=labels, average='weighted'))
    metrics.update(recall_metric.compute(predictions=preds, references=labels, average='weighted'))
    metrics.update(f1_metric.compute(predictions=preds, references=labels, average='weighted'))
    metrics.update(matthews_metric.compute(predictions=preds, references=labels, average='weighted'))
    return metrics
    #labels = pred.label_ids
   # print('labels',labels)
   # logits=pred.predictions
   # print('logits',logits)
   # preds = pred.predictions.argmax(-1)
   # precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
   # acc = accuracy_score(labels, preds)
   # mcc = matthews_corrcoef(labels, preds)
   # return {'accuracy': acc,'f1': f1,'precision': precision,'recall': recall,'mcc':mcc}
   
 
# Initialize evaluation metric
# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model on the evaluation dataset
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Save the model and tokenizer
model_save_path = "./tfbs2_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"Model and tokenizer saved to {model_save_path}")

