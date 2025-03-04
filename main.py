import logging
from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.models.model_builder import ModelBuilder
from src.models.model_trainer import ModelTrainer
import torch

def setup_logging():
    logging.basicConfig(
        level = logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def main():
    config_path = "config/config.yaml"
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Data Loading
        logger.info("Loading Data...")
        data_loader = DataLoader(config_path)
        data = data_loader.load_data()
        train_data, test_data = data_loader.split_data(data)

        # Preprocessing
        logger.info("Preprocessing Data...")
        preprocessor = DataPreprocessor(config_path)
        train_data = preprocessor.preprocess_text(train_data)
        test_data = preprocessor.preprocess_text(test_data)

        # Tokenization
        train_inputs = preprocessor.tokenize(train_data["text"].tolist())
        test_inputs = preprocessor.tokenize(test_data["text"].tolist())

        # Converting Labels to tensors
        train_labels = torch.tensor(train_data["label"].values)
        test_labels = torch.tensor(test_data["label"].values)

        # Model Building
        logger.info("Building Model...")
        model_builder = ModelBuilder(config_path)
        model = model_builder.build_model()

        # Model Training
        logger.info("Training Model...")
        trainer = ModelTrainer(config_path)
        train_loader = trainer.create_data_loader(train_inputs, train_labels)
        val_loader = trainer.create_data_loader(test_inputs, test_labels)

        trained_model = trainer.train(model, train_loader, val_loader)

        # Save Model
        torch.save(trained_model.state_dict(), "models/customer_support_model.pth")
        logger.info("Pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()