from torchmetrics.classification import MulticlassAccuracy
from composer.metrics import CrossEntropy
from composer.models import HuggingFaceModel

# Note: this does not load the weights, just the right model/tokenizer class and config.
# The weights will be loaded by the Composer trainer
model, tokenizer = HuggingFaceModel.hf_from_composer_checkpoint(
    f'outputs_finetune/prom_nonprom/latest-rank0.pt',
    model_instantiation_class='transformers.AutoModelForSequenceClassification',
    model_config_kwargs={'num_labels': 2})

metrics = [CrossEntropy(), MulticlassAccuracy(num_classes=2, average='micro')]
composer_model = HuggingFaceModel(model, tokenizer=tokenizer, metrics=metrics, use_logits=True)
