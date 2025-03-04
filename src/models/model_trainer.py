import mlflow
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW
from tqdm import tqdm
import yaml

class ModelTrainer:
    def __init__(self, config_path: str):
        with open(config_path,"r") as file:
            self.config =  yaml.safe_load(file)
        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])
    
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
        optimizer = AdamW(
            model.parameters(),
            lr = self.config["model"]["learning_rate"]
        )
        with mlflow.start_run():
            mlflow.log_params(self.config["model"])

            for epoch in range(self.config["model"]["epochs"]):
                model.train()
                train_loss = 0
                
                for batch in tqdm(train_loader):
                    input_ids = attention_mask, labels = batch
                    outputs = model(
                        input_ids = input_ids,
                        attention_mask = attention_mask,
                        labels = labels
                    )

                    loss = outputs.loss
                    train_loss += loss.items()

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Validation
                model.eval()
                validation_loss = 0

                with torch.no_grad():
                    for batch in val_loader:
                        input_ids, attention_mask, labels = batch
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