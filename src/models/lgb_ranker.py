"""
LightGBM Ranker model for ESCI Challenge
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import List, Dict, Tuple, Optional
import pickle
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.config.config import Config

logger = logging.getLogger(__name__)

class LGBRanker:
    """LightGBM Ranker for ESCI Challenge"""
    
    def __init__(self, task_id: int, params: Optional[Dict] = None):
        self.task_id = task_id
        self.model = None
        self.feature_columns = []
        self.params = params or self._get_default_params()
        
    def _get_default_params(self) -> Dict:
        """Get default LightGBM parameters for ranking"""
        return {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
    
    def prepare_ranking_data(self, df: pd.DataFrame, feature_columns: List[str]) -> Tuple:
        """Prepare data for LightGBM ranking"""
        logger.info("Preparing ranking data...")
        
        # Sort by query_id for proper grouping
        df_sorted = df.sort_values('query_id').reset_index(drop=True)
        
        # Get features and target
        X = df_sorted[feature_columns].fillna(0)
        y = df_sorted['esci_score']
        
        # Create group information (number of items per query)
        groups = df_sorted.groupby('query_id').size().values
        
        logger.info(f"Data shape: {X.shape}")
        logger.info(f"Groups: {len(groups)} queries")
        logger.info(f"Group sizes - min: {min(groups)}, max: {max(groups)}, mean: {np.mean(groups):.1f}")
        
        return X, y, groups, df_sorted
    
    def train(self, df_train: pd.DataFrame, df_val: pd.DataFrame = None, 
              feature_columns: List[str] = None) -> Dict:
        """Train LightGBM ranker model"""
        logger.info(f"Training LightGBM ranker for Task {self.task_id}...")
        
        self.feature_columns = feature_columns or []
        
        # Prepare training data
        X_train, y_train, train_groups, _ = self.prepare_ranking_data(df_train, self.feature_columns)
        
        # Create LightGBM dataset
        train_dataset = lgb.Dataset(X_train, label=y_train, group=train_groups)
        
        valid_sets = [train_dataset]
        valid_names = ['train']
        
        # Prepare validation data if provided
        if df_val is not None:
            X_val, y_val, val_groups, _ = self.prepare_ranking_data(df_val, self.feature_columns)
            val_dataset = lgb.Dataset(X_val, label=y_val, group=val_groups, reference=train_dataset)
            valid_sets.append(val_dataset)
            valid_names.append('valid')
        
        # Train model
        self.model = lgb.train(
            self.params,
            train_dataset,
            valid_sets=valid_sets,
            valid_names=valid_names,
            num_boost_round=500,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=25)
            ]
        )
        
        # Return training results
        return {
            'best_iteration': self.model.best_iteration,
            'best_score': self.model.best_score,
            'feature_importance': self.get_feature_importance()
        }
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X = df[self.feature_columns].fillna(0)
        predictions = self.model.predict(X, num_iteration=self.model.best_iteration)
        return predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        importance = self.model.feature_importance(importance_type='gain')
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: Path):
        """Save trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'params': self.params,
            'task_id': self.task_id
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Path):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.params = model_data['params']
        self.task_id = model_data['task_id']
        
        logger.info(f"Model loaded from {filepath}")