import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import PreTrainedTokenizerFast

# Read the train and test CSV files
df = pd.read_csv('mortality_data_30.csv') #prom_nonpromdata/train.csv')
#test_df = pd.read_csv('prom_nonpromdata/test.csv')
df = df[['feature', 'labels']]
df.rename(columns={'feature': 'sequence','labels':'label'}, inplace=True)
# Concatenate the train and test DataFrames
combined_df = df #pd.concat([train_df, test_df], ignore_index=True)

# Convert the DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(combined_df)



tokeniser_path='../dnavocab.json'
special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
print("\nUSING EXISTING TOKENISER:", tokeniser_path)
tokeniser = PreTrainedTokenizerFast(
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

dataset = dataset.map(lambda data: tokeniser(data['sequence'],padding='max_length',truncation=True, max_length=512), batched=True)
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)

# Save the dataset to disk (optional)
dataset.save_to_disk('health_dataset_30')

print("Dataset created and saved successfully.")

