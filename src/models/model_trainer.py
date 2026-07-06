import json
import logging
import os
import random
import urllib.request

import matplotlib
matplotlib.use("Agg")  # render figures without a display (servers, Colab)
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm
import yaml

REPORTS_DIR = "reports"

def set_seed(seed: int):
    """Seed every RNG that affects training so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ModelTrainer:
    def __init__(self, config_path: str):
        with open(config_path,"r") as file:
            self.config =  yaml.safe_load(file)
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Point MLflow at the configured server, falling back to local file
        tracking (./mlruns) when no server is configured or reachable, so the
        pipeline runs on machines without MLflow infrastructure (e.g. Colab)."""
        mlflow_config = self.config.get("mlflow", {})
        tracking_uri = mlflow_config.get("tracking_uri") or "file:./mlruns"

        if tracking_uri.startswith("http"):
            try:
                urllib.request.urlopen(
                    f"{tracking_uri.rstrip('/')}/health", timeout=3
                )
            except (urllib.error.URLError, OSError):
                self.logger.warning(
                    f"MLflow server {tracking_uri} unreachable; "
                    "falling back to local file tracking in ./mlruns"
                )
                tracking_uri = "file:./mlruns"

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(mlflow_config.get("experiment_name", "default"))
        self.logger.info(f"MLflow tracking: {tracking_uri}")
    
    def create_data_loader(self, inputs, labels) -> DataLoader:
        dataset = TensorDataset(
            inputs["input_ids"],
            inputs["attention_mask"],
            labels
        )
        return DataLoader(
            dataset,
            batch_size=self.config["model"]["batch_size"],
            shuffle=True
        )
    def train(self, model, train_loader, val_loader, class_names=None):
        seed = self.config["data"]["random_state"]
        set_seed(seed)
        self.logger.info(f"Training on {self.device} (seed={seed})")

        model.to(self.device)
        optimizer = AdamW(
            model.parameters(),
            lr = self.config["model"]["learning_rate"]
        )
        with mlflow.start_run():
            mlflow.log_params(self.config["model"])
            mlflow.log_params({"device": str(self.device), "seed": seed})

            for epoch in range(self.config["model"]["epochs"]):
                model.train()
                train_loss = 0

                for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                    input_ids, attention_mask, labels = (
                        tensor.to(self.device) for tensor in batch
                    )
                    outputs = model(
                        input_ids = input_ids,
                        attention_mask = attention_mask,
                        labels = labels
                    )

                    loss = outputs.loss
                    train_loss += loss.item()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                # Validation
                validation_loss, val_labels, val_preds = self._evaluate(
                    model, val_loader
                )
                val_accuracy = accuracy_score(val_labels, val_preds)
                val_macro_f1 = f1_score(val_labels, val_preds, average="macro")

                mlflow.log_metrics({
                    "train_loss": train_loss/len(train_loader),
                    "validation_loss": validation_loss,
                    "validation_accuracy": val_accuracy,
                    "validation_macro_f1": val_macro_f1,
                }, step=epoch)
                self.logger.info(
                    f"Epoch {epoch + 1}: val_loss={validation_loss:.4f} "
                    f"accuracy={val_accuracy:.4f} macro_f1={val_macro_f1:.4f}"
                )

            self._log_final_report(val_labels, val_preds, class_names)
        return model

    def _evaluate(self, model, data_loader):
        """Run the model over a loader; return mean loss, labels, predictions."""
        model.eval()
        total_loss = 0
        all_labels, all_preds = [], []

        with torch.no_grad():
            for batch in data_loader:
                input_ids, attention_mask, labels = (
                    tensor.to(self.device) for tensor in batch
                )
                outputs = model(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    labels = labels
                )
                total_loss += outputs.loss.item()
                all_preds.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        return total_loss / len(data_loader), all_labels, all_preds

    def _log_final_report(self, labels, preds, class_names):
        """Write per-class report + confusion matrix to reports/ and MLflow."""
        os.makedirs(REPORTS_DIR, exist_ok=True)

        report_text = classification_report(
            labels, preds, target_names=class_names, zero_division=0
        )
        self.logger.info(f"Final validation report:\n{report_text}")

        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
            "per_class": classification_report(
                labels, preds, target_names=class_names,
                output_dict=True, zero_division=0
            ),
        }
        metrics_path = os.path.join(REPORTS_DIR, "metrics.json")
        with open(metrics_path, "w") as file:
            json.dump(metrics, file, indent=2)

        matrix = confusion_matrix(labels, preds)
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(image)
        ticks = class_names if class_names is not None else range(len(matrix))
        ax.set_xticks(range(len(matrix)), ticks, rotation=45, ha="right")
        ax.set_yticks(range(len(matrix)), ticks)
        for row in range(len(matrix)):
            for col in range(len(matrix)):
                ax.text(col, row, str(matrix[row, col]), ha="center",
                        va="center",
                        color="white" if matrix[row, col] > matrix.max() / 2
                        else "black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Validation confusion matrix")
        fig.tight_layout()
        matrix_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
        fig.savefig(matrix_path, dpi=150)
        plt.close(fig)

        mlflow.log_artifact(metrics_path)
        mlflow.log_artifact(matrix_path)
        self.logger.info(f"Evaluation artifacts written to {REPORTS_DIR}/")