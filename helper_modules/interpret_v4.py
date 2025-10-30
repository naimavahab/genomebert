import csv
from transformers import AutoModelForSequenceClassification, AutoTokenizer,PreTrainedTokenizerFast
from transformers_interpret import SequenceClassificationExplainer
from html import escape
from Bio import SeqIO

# Install necessary libraries
# !pip install transformers transformers-interpret biopython

# Load the model and tokenizer
model_name = "./prom_hf" #distilbert-base-uncased-finetuned-sst-2-english"
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
#model.resize_token_embeddings(len(tokenizer))
#

# Initialize the explainer
explainer = SequenceClassificationExplainer(model, tokenizer)

# Increase the field size limit
import sys
csv.field_size_limit(sys.maxsize)

# Function to read sequences from a CSV file
def read_sequences_from_csv(csv_file):
    sequences = []
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            seq_id = row['id']
            sequence = row['sequence']
            sequences.append((seq_id, sequence))
    return sequences

def read_fasta(file_path):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
    return sequences

# Function to truncate or pad sequences to a maximum of 512 tokens
def preprocess_sequence(sequence, tokenizer, max_length=512):
    encoded = tokenizer.encode_plus(
        sequence,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
    return tokenizer.convert_tokens_to_string(tokens)

# Function to generate HTML for highlighted sequences
def generate_html(sequences, explanations, output_file):
    with open(output_file, "w") as f:
        f.write("<html><body>\n")
        for  seq, explanation in zip(sequences, explanations):
            highlighted_seq = ""
            for word, importance in explanation:
                color = f"background-color: rgba(255,0,0,{importance});" if importance > 0 else ""
                highlighted_seq += f"<span style='{color}'>{escape(word)}</span>"
            f.write(f"\n<p>{highlighted_seq}</p>\n")
        f.write("</body></html>\n")

# CSV file path
#csv_file = 'input.csv'  # Replace with your CSV file path
fasta_file = "./prom_fasta/promoter.fasta"
# Read sequences from CSV
sequences = read_fasta(fasta_file)
#print(sequences)
# Preprocess and get explanations for each sequence
explanations = []
for sequence in sequences[100]:
    preprocessed_sequence = sequence[:2600]
   # preprocessed_sequence = preprocess_sequence(sequence, tokenizer)
    explanation = explainer(preprocessed_sequence)
    explanations.append(explanation)

print(explanations)

# Generate HTML file
output_html_file = 'prom_feat.html'  # Replace with your desired output HTML file path
generate_html(sequences, explanations, output_html_file)

print(f"HTML file generated: {output_html_file}")

