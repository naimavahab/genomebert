import torch
import transformers
from transformers import AutoModelForMaskedLM, BertTokenizer, pipeline
from transformers import BertTokenizer, BertConfig

#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # MosaicBERT uses the standard BERT tokenizer
model = 'prom_hf'
config = transformers.BertConfig.from_pretrained(model) # the config needs to be passed in
mosaicbert = AutoModelForMaskedLM.from_pretrained(model,config=config)




model_size = sum(t.numel() for t in mosaicbert.parameters())
print(f"\nBert size: {model_size/1000**2:.1f}M parameters")

