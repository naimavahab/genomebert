from transformers import PreTrainedTokenizerFast
from datasets import load_from_disk
special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]

tokeniser = PreTrainedTokenizerFast(
            tokenizer_file='dnavocab.json',
            special_tokens=special_tokens,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            sep_token="<sep>",
            pad_token="<pad>",
            cls_token="<cls>",
            mask_token="<mask>",
            )

dataset = load_from_disk('input_data/health_dataset/')

:wq

