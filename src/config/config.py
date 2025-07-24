"""
Configuration file for ESCI Challenge project
"""
import os
from pathlib import Path

class Config:
    # Base paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    FEATURES_DIR = DATA_DIR / "features"
    RESULTS_DIR = BASE_DIR / "results"
    EXPERIMENTS_DIR = BASE_DIR / "experiments"
    
    # Raw data files
    EXAMPLES_FILE = RAW_DATA_DIR / "shopping_queries_dataset_examples.parquet"
    PRODUCTS_FILE = RAW_DATA_DIR / "shopping_queries_dataset_products.parquet"
    SOURCES_FILE = RAW_DATA_DIR / "shopping_queries_dataset_sources.csv"
    
    # Task-specific settings
    TASKS = {
        1: {
            "name": "Query-Product Ranking",
            "version_filter": "small_version",
            "target_col": "esci_label",
            "eval_metric": "ndcg"
        },
        2: {
            "name": "Multi-class Product Classification",
            "version_filter": "large_version", 
            "target_col": "esci_label",
            "eval_metric": "accuracy"
        },
        3: {
            "name": "Product Substitute Identification",
            "version_filter": "large_version",
            "target_col": "substitute_label", 
            "eval_metric": "f1"
        }
    }
    
    # Model settings
    LGBM_PARAMS = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": 0,
        "num_threads": -1,
        "random_state": 42
    }
    
    # Feature engineering settings
    TEXT_FEATURES = {
        "max_features": 10000,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.95
    }
    
    # Training settings
    TRAIN_PARAMS = {
        "num_boost_round": 1000,
        "early_stopping_rounds": 100,
        "verbose_eval": 50
    }
    
    # Language filter (English only)
    LANGUAGE = "us"  # product_locale filter
    
    # ESCI label mapping
    ESCI_MAPPING = {
        "E": 3,  # Exact
        "S": 2,  # Substitute  
        "C": 1,  # Complement
        "I": 0   # Irrelevant
    }
    
    # Random seed for reproducibility
    RANDOM_STATE = 42
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.FEATURES_DIR,
            cls.RESULTS_DIR,
            cls.EXPERIMENTS_DIR,
            cls.PROCESSED_DATA_DIR / "task_1",
            cls.PROCESSED_DATA_DIR / "task_2", 
            cls.PROCESSED_DATA_DIR / "task_3",
            cls.FEATURES_DIR / "basic_features",
            cls.FEATURES_DIR / "text_features",
            cls.FEATURES_DIR / "advanced_features",
            cls.RESULTS_DIR / "models",
            cls.RESULTS_DIR / "predictions",
            cls.RESULTS_DIR / "evaluation"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print("Directory structure created successfully!")