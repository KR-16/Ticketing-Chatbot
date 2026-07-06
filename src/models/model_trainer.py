import logging
import random
import urllib.request

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm
import yaml

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
    def train(self, model, train_loader, val_loader):
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
                model.eval()
                validation_loss = 0

                with torch.no_grad():
                    for batch in val_loader:
                        input_ids, attention_mask, labels = (
                            tensor.to(self.device) for tensor in batch
                        )
                        outputs = model(
                            input_ids = input_ids,
                            attention_mask = attention_mask,
                            labels = labels
                        )
                        validation_loss += outputs.loss.item()
                mlflow.log_metrics({
                    "train_loss": train_loss/len(train_loader),
                    "validation_loss": validation_loss/len(val_loader)
                }, step=epoch)
        return model