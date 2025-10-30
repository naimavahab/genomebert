from Bio import SeqIO

def filter_and_save_sequences(input_fasta, output_fasta, min_length=10000, max_length=15000):
    with open(output_fasta, 'w') as output_handle:
        for record in SeqIO.parse(input_fasta, "fasta"):
            if min_length <= len(record.seq) <= max_length:
                SeqIO.write(record, output_handle, "fasta")
    print(f"Sequences between {min_length} and {max_length} bases have been saved to {output_fasta}")

# Example usage
input_fasta = "health.fasta"  # Replace with your input FASTA file path
output_fasta = "health_small.fasta"  # Replace with your desired output file path

filter_and_save_sequences(input_fasta, output_fasta)
def count_sequences(fasta_file):
    count = sum(1 for _ in SeqIO.parse(fasta_file, "fasta"))
    return count

print(count_sequences(output_fasta))
