import torch
import transformers
from transformers import AutoModelForMaskedLM, BertTokenizer, pipeline
from transformers import BertTokenizer, BertConfig

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # MosaicBERT uses the standard BERT tokenizer

config = transformers.BertConfig.from_pretrained('hf_models') # the config needs to be passed in
mosaicbert = AutoModelForMaskedLM.from_pretrained('hf_models',config=config)

# To use this model directly for masked language modeling
mosaicbert_classifier = pipeline('fill-mask', model=mosaicbert, tokenizer=tokenizer,device="cpu")
mosaicbert_classifier("I [MASK] to the store yesterday.")

