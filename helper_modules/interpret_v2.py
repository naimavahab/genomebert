import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer,PreTrainedTokenizerFast
from transformers_interpret import SequenceClassificationExplainer
from Bio import SeqIO
from html import escape

# Install necessary libraries
# !pip install transformers transformers-interpret biopython

# Load the model and tokenizer
model_name = "./genome_bert_healthdata"
model = AutoModelForSequenceClassification.from_pretrained(model_name)

#tokenizer = AutoTokenizer.from_pretrained(model_name)
tokeniser_path = 'dnavocab.json'
special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
print("USING EXISTING TOKENISER:", tokeniser_path)
tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokeniser_path,
            special_tokens=special_tokens,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            sep_token="<sep>",
            pad_token="<pad>",
            cls_token="<cls>",
            mask_token="<mask>",truncation=True, max_length=512,padding='max_length'
            )
model.resize_token_embeddings(len(tokenizer))
#model = model.to(device)
print(model)

tokenizer_settings = {
    'truncation': True,
    'max_length': 512,
    'padding': 'max_length'
}
# Initialize the explainer
explainer = SequenceClassificationExplainer(model, tokenizer)

# Function to read sequences from a FASTA file
def read_fasta(file_path):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
    return sequences

# Function to generate HTML for highlighted sequences
def generate_html(sequences, explanations, output_file):
    with open(output_file, "w") as f:
        f.write("<html><body>\n")
        for seq, explanation in zip(sequences, explanations):
            highlighted_seq = ""
            for word, importance in explanation:
                color = f"background-color: rgba(255,0,0,{importance});" if importance > 0 else ""
                highlighted_seq += f"<span style='{color}'>{escape(word)}</span>"
            f.write(f"<p>{highlighted_seq}</p>\n")
        f.write("</body></html>\n")

# Read sequences from a FASTA file
fasta_file = "./input_data/health.fasta"
sequences = read_fasta(fasta_file)

inputs = tokenizer(sequences[0], truncation=True, max_length=512, padding='max_length', return_tensors='pt')
print(inputs)
# Get explanations for each sequence
explanations = explainer(inputs['input_ids'], inputs['attention_mask'])#[explainer(seq,**tokenizer_settings) for seq in sequences]

# Generate HTML file
output_html_file = "health.html"
generate_html(sequences, explanations, output_html_file)

print(f"HTML file generated: {output_html_file}")

