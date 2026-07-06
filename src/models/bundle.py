"""Save and load a complete, self-contained inference bundle.

A bundle directory holds everything needed to classify a new ticket:

    bundle_dir/
        config.json, model.safetensors, ...   # model  (save_pretrained)
        tokenizer_config.json, vocab.txt, ... # tokenizer (save_pretrained)
        label_map.json                        # target column + class names
        training_config.yaml                  # config the model was trained with

The model's id2label/label2id are also set from the label encoder, so the
saved HF config maps logits to human-readable class names on its own.
"""

import json
import logging
import os
from dataclasses import dataclass

import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

LABEL_MAP_FILE = "label_map.json"
TRAINING_CONFIG_FILE = "training_config.yaml"


@dataclass
class InferenceBundle:
    model: object
    tokenizer: object
    classes: list
    target_column: str
    config: dict


def save_bundle(model, tokenizer, label_encoder, config, bundle_dir,
                target_column="type"):
    """Persist model, tokenizer, label mapping, and config to bundle_dir."""
    os.makedirs(bundle_dir, exist_ok=True)

    classes = [str(cls) for cls in label_encoder.classes_]
    model.config.id2label = {i: cls for i, cls in enumerate(classes)}
    model.config.label2id = {cls: i for i, cls in enumerate(classes)}

    model.save_pretrained(bundle_dir)
    tokenizer.save_pretrained(bundle_dir)

    with open(os.path.join(bundle_dir, LABEL_MAP_FILE), "w") as file:
        json.dump({"target_column": target_column, "classes": classes},
                  file, indent=2)

    with open(os.path.join(bundle_dir, TRAINING_CONFIG_FILE), "w") as file:
        yaml.safe_dump(config, file)

    logger.info(f"Inference bundle saved to {bundle_dir}")


def load_bundle(bundle_dir) -> InferenceBundle:
    """Reconstruct everything needed for inference from a bundle directory."""
    label_map_path = os.path.join(bundle_dir, LABEL_MAP_FILE)
    if not os.path.isfile(label_map_path):
        raise FileNotFoundError(
            f"No inference bundle at {bundle_dir}: {LABEL_MAP_FILE} missing. "
            "Train a model first (python main.py) or point to a valid bundle."
        )

    with open(label_map_path, "r") as file:
        label_map = json.load(file)

    with open(os.path.join(bundle_dir, TRAINING_CONFIG_FILE), "r") as file:
        config = yaml.safe_load(file)

    model = AutoModelForSequenceClassification.from_pretrained(bundle_dir)
    tokenizer = AutoTokenizer.from_pretrained(bundle_dir)
    model.eval()

    logger.info(f"Inference bundle loaded from {bundle_dir}")
    return InferenceBundle(
        model=model,
        tokenizer=tokenizer,
        classes=label_map["classes"],
        target_column=label_map["target_column"],
        config=config,
    )
