from datasets import load_from_disk
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

# Load the Hugging Face dataset
# Replace 'your_dataset_name' and 'split' with the appropriate dataset name and split
dataset = load_from_disk('health_dataset')

# Function to extract sequences and save to a FASTA file
def save_sequences_to_fasta(dataset, fasta_file):
    sequences = []
    i = 0
    for record in dataset:
        seq_id =  i  # Adjust the key if necessary
        sequence = record['sequence']  # Adjust the key if necessary
        sequences.append(SeqRecord(Seq(sequence), id=str(seq_id), description=""))
        i+=1
    # Write sequences to FASTA file
    with open(fasta_file, 'w') as file:
        SeqIO.write(sequences, file, 'fasta')

# Path to the output FASTA file
fasta_file = 'health_data.fasta'  # Replace with your desired output FASTA file path

# Save sequences to FASTA
save_sequences_to_fasta(dataset, fasta_file)

print(f"FASTA file generated: {fasta_file}")

