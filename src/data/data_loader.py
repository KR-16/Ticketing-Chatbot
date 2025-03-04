import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
import yaml
import logging

class DataLoader:
    def __init__(self, config_path: str):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.logger = logging.getLogger(__name__)
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset
        """
        try:
            data = pd.read_csv(self.config["data"]["raw_data_path"])
            self.logger.info(f"Data Loaded Successfully with shape {data.shape}")
            return data
        except Exception as e:
            self.logger.error(f"Error Loading data: {str(e)}")
            raise
    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data into train and test sets
        """
        train_data, test_data = train_test_split(
            data,
            test_size = self.config["data"]["test_size"],
            random_state = self.config["data"]["random_state"]
        )
        return train_data, test_data