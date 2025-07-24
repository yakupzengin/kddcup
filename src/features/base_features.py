"""
Basic feature engineering for ESCI Challenge
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import re
import string
from collections import Counter
import logging

from src.config.config import Config

logger = logging.getLogger(__name__)

class BaseFeatureEngineer:
    """Basic feature engineering for ESCI Challenge"""
    
    def __init__(self):
        self.feature_columns = []
        
    def clean_text(self, text):
        """Clean text data"""
        if pd.isna(text):
            return ""
        
        # Convert to string and strip
        text = str(text).strip()
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def create_basic_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic text features"""
        logger.info("Creating basic text features...")
        features = df.copy()
        
        # Query features
        features['query_len'] = features['query'].str.len()
        features['query_word_count'] = features['query'].str.split().str.len()
        features['query_unique_words'] = features['query'].apply(
            lambda x: len(set(str(x).lower().split()))
        )
        
        # Product title features
        features['title_len'] = features['product_title'].str.len()
        features['title_word_count'] = features['product_title'].str.split().str.len()
        features['title_unique_words'] = features['product_title'].apply(
            lambda x: len(set(str(x).lower().split()))
        )
        
        # Product description features
        features['description_len'] = features['product_description'].str.len()
        features['description_word_count'] = features['product_description'].str.split().str.len()
        
        # Brand and color features
        features['has_brand'] = (features['product_brand'].str.len() > 0).astype(int)
        features['has_color'] = (features['product_color'].str.len() > 0).astype(int)
        
        # Add to feature columns list
        basic_features = [
            'query_len', 'query_word_count', 'query_unique_words',
            'title_len', 'title_word_count', 'title_unique_words',
            'description_len', 'description_word_count',
            'has_brand', 'has_color'
        ]
        self.feature_columns.extend(basic_features)
        
        return features
    
    def create_similarity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create query-product similarity features"""
        logger.info("Creating similarity features...")
        features = df.copy()
        
        # Exact matches
        features['query_in_title'] = features.apply(
            lambda x: 1 if str(x['query']).lower() in str(x['product_title']).lower() else 0, 
            axis=1
        )
        
        features['title_in_query'] = features.apply(
            lambda x: 1 if str(x['product_title']).lower() in str(x['query']).lower() else 0,
            axis=1
        )
        
        # Word overlap features
        def word_overlap_ratio(text1, text2):
            words1 = set(str(text1).lower().split())
            words2 = set(str(text2).lower().split())
            if len(words1) == 0 or len(words2) == 0:
                return 0
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return intersection / union if union > 0 else 0
        
        def word_jaccard_similarity(text1, text2):
            words1 = set(str(text1).lower().split())
            words2 = set(str(text2).lower().split())
            if len(words1) == 0 and len(words2) == 0:
                return 1
            if len(words1) == 0 or len(words2) == 0:
                return 0
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return intersection / union
        
        features['query_title_word_overlap'] = features.apply(
            lambda x: word_overlap_ratio(x['query'], x['product_title']), axis=1
        )
        
        features['query_title_jaccard'] = features.apply(
            lambda x: word_jaccard_similarity(x['query'], x['product_title']), axis=1
        )
        
        # Brand matching
        features['brand_in_query'] = features.apply(
            lambda x: 1 if str(x['product_brand']).lower() in str(x['query']).lower() 
            and len(str(x['product_brand'])) > 0 else 0, axis=1
        )
        
        # Add to feature columns list
        similarity_features = [
            'query_in_title', 'title_in_query', 'query_title_word_overlap',
            'query_title_jaccard', 'brand_in_query'
        ]
        self.feature_columns.extend(similarity_features)
        
        return features
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features"""
        logger.info("Creating statistical features...")
        features = df.copy()
        
        # Query frequency features
        query_counts = features.groupby('query')['query_id'].nunique()
        features['query_frequency'] = features['query'].map(query_counts)
        
        # Product frequency features
        product_counts = features.groupby('product_id')['example_id'].count()
        features['product_frequency'] = features['product_id'].map(product_counts)
        
        # ESCI label encoding
        features['esci_score'] = features['esci_label'].map(Config.ESCI_MAPPING)
        
        # Add to feature columns list
        statistical_features = ['query_frequency', 'product_frequency', 'esci_score']
        self.feature_columns.extend(statistical_features)
        
        return features
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all basic features"""
        logger.info("Creating all basic features...")
        
        # Clean text data first
        df_clean = df.copy()
        text_columns = ['query', 'product_title', 'product_description', 'product_brand']
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(self.clean_text)
        
        # Create features
        df_features = self.create_basic_text_features(df_clean)
        df_features = self.create_similarity_features(df_features)
        df_features = self.create_statistical_features(df_features)
        
        logger.info(f"Created {len(self.feature_columns)} features")
        return df_features
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns"""
        return [col for col in self.feature_columns 
                if col.endswith(('_len', '_count', '_words', '_overlap', '_jaccard', '_in_', 'has_', '_frequency'))]