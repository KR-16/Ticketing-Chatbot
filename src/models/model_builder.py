from transformers import AutoModelForSequenceClassification
import torch.nn as nn
import yaml

class ModelBuilder:
    def __init__(self, config_path: str):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
    def build_model(self) -> nn.Module:
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-multilingual-cased",
            num_labels = self.config["model"]["num_classes"]
        )
        return model