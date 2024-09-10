import torch
from transformers import BertTokenizer, BertForSequenceClassification,PreTrainedTokenizerFast,BertConfig
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt

# Load your model and tokenizer
model_name = "bert-base-uncased"

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
            mask_token="<mask>",
            )
#model = BertForSequenceClassification.from_pretrained(model_name)
#tokenizer = BertTokenizer.from_pretrained(model_name)

#config = BertForSequenceClassification.from_pretrained('/mosaicml-examples/examples/benchmarks/bert/hf_models') # thei config needs to be passed in
model = BertForSequenceClassification.from_pretrained('/mosaicml-examples/examples/benchmarks/bert/hf_models',num_labels=2)

#checkpoint_path = 'mosaic-bert-base-uncasedepoch20/ckpt/latest-rank0.pt'
#model.load_state_dict(torch.load(checkpoint_path))
# Sample input text
text = "This is a great movie!"

# Tokenize input text
inputs = tokenizer(text, return_tensors="pt")

# Get model prediction
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()

# Integrated Gradients
ig = IntegratedGradients(model)

# Calculate attributions
def forward_func(inputs):
    outputs = model(**inputs)
    return outputs.logits

attributions, delta = ig.attribute(inputs['input_ids'], target=predicted_class, return_convergence_delta=True)

# Visualize attributions
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
attributions = attributions.squeeze().cpu().detach().numpy()

# Plot the attributions
plt.figure(figsize=(12, 8))
plt.bar(range(len(tokens)), attributions, align='center')
plt.xticks(range(len(tokens)), tokens, rotation='vertical')
plt.xlabel('Tokens')
plt.ylabel('Attribution')
plt.title('Token Importance Visualization')

# Save the plot to a file
plt.savefig('token_importance_visualization.png', bbox_inches='tight')
plt.show()

