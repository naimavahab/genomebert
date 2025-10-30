#!/usr/bin/python
# make sure utils can be imported
import inspect
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.dirname(inspect.getfile(lambda: None)))
from bs4 import BeautifulSoup
# conduct interpretation step
import argparse
import screed
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer,PreTrainedTokenizerFast
from transformers_interpret import SequenceClassificationExplainer

def main():
    parser = argparse.ArgumentParser(
         description='Take a file of sequences, model, tokeniser and interpret.'
        )
    # parser.add_argument('seqs_file', type=str,
    #                     help='path to fasta file')
    # parser.add_argument('model', type=str,
    #                     help='path to model files from transformers/pytorch')
    # parser.add_argument('tokeniser_path', type=str,
    #                     help='path to tokeniser.json dir to load data from')
    # parser.add_argument('-o', '--output_dir', type=str, default="./vis_out",
    #                     help='specify path for output (DEFAULT: ./vis_out)')
    # parser.add_argument('-l', '--', type=str, default=None, nargs="+",
    #                     help='provide label names (DEFAULT: "").')
    # parser.add_argument('-a', '--attribution', type=str, default=None,
    #                     help='provide attribution matching label (DEFAULT: "").')
    # parser.add_argument('-t', '--true_class', type=str, default=None,
    #                     help='provide label of the true class (DEFAULT: "").')

    args = parser.parse_args()
    seqs_file = 'promoter_all_sequences.fasta' #./promoter/promoter.fasta' #TFBS_human/pos_2000_sequences_cleaned.fasta' #'./promoter/promoter.fasta'
    # model = "/Users/sunitaverma/Desktop/genomenlp-main/results/model_files"

    output_dir = "./interpret_out_promoterall"
    model_path = "Modern_bert_model" #./promoter_model" #"./tfbs_model" #"./promoter_model"
    label_names = [0,1]
    attribution = ""
    true_class = ""

    print("\n\nARGUMENTS:\n", args, "\n\n")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True
        )
    # tokeniser_path = 'dnavocab.json'
    special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]

    tokeniser = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    #tokeniser = AutoTokenizer.from_pretrained(
    #tokeniser_path, local_files_only=True
     #   )
    explainer = SequenceClassificationExplainer(
        model, tokeniser
        )
    # attri= explainer()
    # print(attri.word_attributions)
    
    explainers = []
    count=0
    maxc=5000
    rows = []
    with screed.open(seqs_file) as infile:
        for read in tqdm(infile, desc="Parsing seqs"):
            # print(read)
            if count>maxc:
                break
            ex = explainer(read.sequence.upper(), class_name=attribution if attribution else None)
            # print(ex)
            sorted_data = sorted(ex, key=lambda x: x[1], reverse=True)
            # Add the sequence and sorted data as a row in the CSV
            rows.append({'sequence': read.sequence.upper(), 'sorted_data': sorted_data,'Attribution labels': explainer.predicted_class_name})
            print("labels",explainer.predicted_class_name)
            html = explainer.visualize(true_class)
            data =html.data
            explainers.append(data)
            count+=1
    soup = BeautifulSoup('<html><head><title>My HTML File</title></head><body></body></html>', 'html.parser')
    df = pd.DataFrame(rows)
    # print(rows)
    df.to_csv('word_attr_promoterall.csv', index=False)
# Add the HTML objects to the body
    body = soup.body
    for obj in explainers:
      body.append(BeautifulSoup(obj, 'html.parser'))
# Write the complete HTML to a file
    with open('promoter_all.html', 'w', encoding='utf-8') as file:
      file.write(str(soup.prettify()))
  # explainers.write('feature.html')
if __name__ == "__main__":
    main()
    
    

