"""Single-ticket inference against a saved bundle.

Loads the bundle once (model, tokenizer, class names, training config) and
applies the same text preparation the model saw during training, so there is
no train/serve skew.

CLI smoke test:
    python -m src.inference.predictor --subject "Server down" --body "..."
"""

import argparse
import json
import logging

import torch

from src.data.data_preprocessor import DataPreprocessor, _ensure_nltk_data
from src.models.bundle import load_bundle

logger = logging.getLogger(__name__)


class TicketPredictor:
    def __init__(self, bundle_dir: str = "models/bundle"):
        self.bundle = load_bundle(bundle_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bundle.model.to(self.device)

        preprocessing = self.bundle.config.get("preprocessing", {})
        self.max_length = preprocessing.get("max_length", 200)
        self.strip_stopwords = preprocessing.get("remove_stopwords", False)
        if self.strip_stopwords:
            _ensure_nltk_data()

        logger.info(
            f"Predictor ready on {self.device}: "
            f"classes={self.bundle.classes}"
        )

    @property
    def classes(self) -> list:
        return self.bundle.classes

    def predict(self, subject: str = "", body: str = "") -> dict:
        """Classify one ticket; returns predicted type, confidence, and the
        full per-class probability distribution."""
        text = (
            f"{DataPreprocessor.clean_text(subject)} "
            f"{DataPreprocessor.clean_text(body)}"
        ).strip()
        if not text:
            raise ValueError(
                "Ticket is empty after cleaning; provide a subject or body"
            )

        # Mirror the training-time pipeline (config flags travel in the bundle)
        if self.strip_stopwords:
            language = DataPreprocessor.detect_language(text)
            text = DataPreprocessor.remove_stopwords(text, language)

        encoding = self.bundle.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.bundle.model(**encoding).logits

        probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
        best = int(torch.tensor(probabilities).argmax())

        return {
            "predicted_type": self.bundle.classes[best],
            "confidence": round(probabilities[best], 4),
            "probabilities": {
                cls: round(prob, 4)
                for cls, prob in zip(self.bundle.classes, probabilities)
            },
        }


def main():
    parser = argparse.ArgumentParser(description="Classify a support ticket")
    parser.add_argument("--subject", default="", help="Ticket subject")
    parser.add_argument("--body", default="", help="Ticket body")
    parser.add_argument("--bundle-dir", default="models/bundle",
                        help="Path to the inference bundle")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    predictor = TicketPredictor(args.bundle_dir)
    result = predictor.predict(subject=args.subject, body=args.body)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
