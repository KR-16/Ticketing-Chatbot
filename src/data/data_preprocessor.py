from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from langdetect import detect
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import yaml
import re
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from typing import Tuple, Dict
import logging

nltk.download("stopwords")
nltk.download("punkt")

class DataPreprocessor:
    def __init__(self, config_path: str):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.logger = logging.getLogger(__name__)
        self.label_encoders = {}
        self.mlb = MultiLabelBinarizer()
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data structure and content"""
        required_columns = ['subject', 'body', 'type', 'priority', 'language']
        
        # Check required columns
        if not all(col in data.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in data.columns]
            self.logger.error(f"Missing required columns: {missing_cols}")
            return False
            
        # Check for minimum data requirements
        if len(data) == 0:
            self.logger.error("Empty dataset provided")
            return False
            
        # Validate language codes
        valid_languages = ['en', 'de', 'unknown']
        invalid_langs = data['language'].unique().tolist()
        invalid_langs = [lang for lang in invalid_langs if lang not in valid_languages]
        if invalid_langs:
            self.logger.warning(f"Invalid language codes found: {invalid_langs}")
            
        return True

    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning"""
        if not isinstance(text, str):
            return ""
            
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '[URL]', text)
        
        # Remove phone numbers
        text = re.sub(r'\+?[\d\s-]{10,}', '[PHONE]', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def prepare_text_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced text preparation"""
        df = data.copy()
        
        # Clean subject and body separately
        df['subject'] = df['subject'].fillna('').apply(self.clean_text)
        df['body'] = df['body'].fillna('').apply(self.clean_text)
        
        # Combine cleaned text
        df['text'] = df['subject'] + ' ' + df['body']
        
        # Add text length features
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].apply(lambda x: len(word_tokenize(x)))
        
        return df

    def prepare_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced label preparation with hierarchical encoding"""
        df = data.copy()
        
        # Encode categorical variables
        categorical_columns = ['type', 'priority', 'queue']
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[f'{col}_label'] = self.label_encoders[col].fit_transform(df[col])
        
        # Create priority level numeric mapping
        priority_map = {'low': 0, 'medium': 1, 'high': 2}
        df['priority_level'] = df['priority'].map(priority_map)
        
        return df

    def process_tags(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced tag processing with multi-label encoding"""
        df = data.copy()
        tag_cols = ['tag_1', 'tag_2', 'tag_3', 'tag_4', 'tag_5']
        
        # Combine tags
        df['tags'] = df[tag_cols].fillna('').agg(' '.join, axis=1)
        df['tags'] = df['tags'].str.strip()
        
        # Create list of tags for each row
        df['tag_list'] = df['tags'].apply(lambda x: [tag for tag in x.split() if tag])
        
        # Multi-label encode tags
        tag_matrix = self.mlb.fit_transform(df['tag_list'])
        tag_df = pd.DataFrame(tag_matrix, columns=self.mlb.classes_)
        
        # Add encoded tags back to dataframe
        df = pd.concat([df, tag_df.add_prefix('tag_encoded_')], axis=1)
        
        return df

    def preprocess_text(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced main preprocessing pipeline"""
        # Validate input data
        if not self.validate_data(data):
            raise ValueError("Data validation failed")
            
        try:
            # Apply preprocessing steps
            df = self.prepare_text_data(data)
            df = self.prepare_labels(df)
            df = self.process_tags(df)
            
            if self.config["preprocessing"]["language_detection"]:
                df["language"] = df["text"].apply(self.detect_language)

            if self.config["preprocessing"]["remove_stopwords"]:
                df["text"] = df.apply(
                    lambda x: self.remove_stopwords(x["text"], x["language"]),
                    axis=1
                )
                
            # Add preprocessing metadata
            df['preprocessed_timestamp'] = pd.Timestamp.now()
            
            self.logger.info("Preprocessing completed successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {str(e)}")
            raise

    def get_feature_names(self) -> Dict[str, list]:
        """Get names of all features created during preprocessing"""
        return {
            'categorical_features': list(self.label_encoders.keys()),
            'tag_features': self.mlb.classes_.tolist(),
            'numeric_features': ['text_length', 'word_count', 'priority_level']
        }

    def tokenize(self, texts: list) -> dict:
        """Enhanced tokenization with length validation"""
        if not texts:
            raise ValueError("Empty text list provided for tokenization")
            
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config["preprocessing"]["max_length"],
            return_tensors="pt"
        )

    def save_encoders(self, path: str):
        """Save label encoders and binarizer for later use"""
        import joblib
        joblib.dump({
            'label_encoders': self.label_encoders,
            'multilabel_binarizer': self.mlb
        }, path)

    def load_encoders(self, path: str):
        """Load saved encoders"""
        import joblib
        encoders = joblib.load(path)
        self.label_encoders = encoders['label_encoders']
        self.mlb = encoders['multilabel_binarizer']

    def detect_language(self, text:str) -> str:
        try:
            return detect(text)
        except:
            return "unknown"
    def remove_stopwords(self, text: str, language: str) -> str:
        if language in stopwords.fileids():
            stop_words = set(stopwords.words(language))
            words = text.split()
            return " ".join([word for word in words if word.lower() not in stop_words])
        return text