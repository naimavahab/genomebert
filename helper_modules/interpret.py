#!/usr/bin/python
# make sure utils can be imported
import inspect
import os
import sys
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
    parser.add_argument('seqs_file', type=str,
                        help='path to fasta file')
    parser.add_argument('model', type=str,
                        help='path to model files from transformers/pytorch')
    parser.add_argument('tokeniser_path', type=str,
                        help='path to tokeniser.json dir to load data from')
    parser.add_argument('-o', '--output_dir', type=str, default="./vis_out",
                        help='specify path for output (DEFAULT: ./vis_out)')
    parser.add_argument('-l', '--label_names', type=str, default=None, nargs="+",
                        help='provide label names (DEFAULT: "").')
    parser.add_argument('-a', '--attribution', type=str, default=None,
                        help='provide attribution matching label (DEFAULT: "").')
    parser.add_argument('-t', '--true_class', type=str, default=None,
                        help='provide label of the true class (DEFAULT: "").')

    args = parser.parse_args()
    seqs_file = args.seqs_file
    model = args.model
    tokeniser_path = args.tokeniser_path
    output_dir = args.output_dir
    label_names = args.label_names
    attribution = args.attribution
    true_class = args.true_class

    print("\n\nARGUMENTS:\n", args, "\n\n")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    model = AutoModelForSequenceClassification.from_pretrained(
        model, local_files_only=True
        )
    tokeniser_path = 'dnavocab.json'
    special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
    print("USING EXISTING TOKENISER:", tokeniser_path)
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
#
    #tokeniser = AutoTokenizer.from_pretrained(
    #tokeniser_path, local_files_only=True
     #   )

    count=0
    maxc=20
    with screed.open(seqs_file) as infile:
        for read in tqdm(infile, desc="Parsing seqs"):
            if count>maxc:
                break
            
            example_seq = read.sequence.upper()[2:]
            print(example_seq)
            model_inputs = tokeniser(example_seq)

            tokens = tokeniser.tokenize(example_seq)
            ids = tokeniser.convert_tokens_to_ids(tokens)
            print("Sample tokenised:", ids)

            for i in ids:
              print("Token::k-mer map:", i, "\t::", tokeniser.decode(i))
            count+=1
    explainer = SequenceClassificationExplainer(
        model, tokeniser #, custom_labels=label_names
        )
    explainers = []
    count=0
    maxc=20
    with screed.open(seqs_file) as infile:
        for read in tqdm(infile, desc="Parsing seqs"):
            if count>maxc:
                break
            ex = explainer(read.sequence.upper()[2:2500], class_name=attribution)
       #     explainers.append(ex)
       #  print(explainers)

            html = explainer.visualize(true_class)
            data =html.data
            explainers.append(data)
            count+=1
    soup = BeautifulSoup('<html><head><title>My HTML File</title></head><body></body></html>', 'html.parser')

# Add the HTML objects to the body
    body = soup.body
    for obj in explainers:
      body.append(BeautifulSoup(obj, 'html.parser'))

# Write the complete HTML to a file
    with open('feature_healthdata.html', 'w', encoding='utf-8') as file:
      file.write(str(soup.prettify()))
  # explainers.write('feature.html')

if __name__ == "__main__":
    main()
