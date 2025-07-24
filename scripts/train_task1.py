"""
Training script for Task 1: Query-Product Ranking
KDD Cup 2022 ESCI Challenge

This script implements a LightGBM ranker for ranking products given a query.
Target: Start with basic features and achieve incremental improvements.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def load_data():
    """Load and prepare data for Task 1"""
    print("Loading data...")
    
    # Load datasets
    data_path = Path(__file__).parent.parent / "data" / "raw"
    
    df_examples = pd.read_parquet(data_path / "shopping_queries_dataset_examples.parquet")
    df_products = pd.read_parquet(data_path / "shopping_queries_dataset_products.parquet")
    
    # Merge examples with products
    df_merged = pd.merge(
        df_examples, 
        df_products, 
        how='left', 
        left_on=['product_locale', 'product_id'], 
        right_on=['product_locale', 'product_id']
    )
    
    # Filter for Task 1 (small version)
    df_task1 = df_merged[df_merged["small_version"] == 1].copy()
    
    # Filter for English only
    df_task1 = df_task1[df_task1["product_locale"] == "us"].copy()
    
    print(f"Task 1 data shape: {df_task1.shape}")
    print(f"Train examples: {len(df_task1[df_task1['split'] == 'train'])}")
    print(f"Test examples: {len(df_task1[df_task1['split'] == 'test'])}")
    
    return df_task1

def create_basic_features(df):
    """Create basic features for the model"""
    print("Creating basic features...")
    
    features = df.copy()
    
    # Text length features
    features['query_len'] = features['query'].str.len()
    features['title_len'] = features['product_title'].fillna('').str.len()
    features['description_len'] = features['product_description'].fillna('').str.len()
    
    # Word count features
    features['query_word_count'] = features['query'].str.split().str.len()
    features['title_word_count'] = features['product_title'].fillna('').str.split().str.len()
    
    # Simple text matching features
    features['query_in_title'] = features.apply(
        lambda x: 1 if str(x['query']).lower() in str(x['product_title']).lower() else 0, axis=1
    )
    
    # Brand matching
    features['has_brand'] = features['product_brand'].notna().astype(int)
    
    # ESCI label encoding for training
    esci_mapping = {'E': 3, 'S': 2, 'C': 1, 'I': 0}
    features['esci_score'] = features['esci_label'].map(esci_mapping)
    
    # Select feature columns
    feature_cols = [
        'query_len', 'title_len', 'description_len',
        'query_word_count', 'title_word_count',
        'query_in_title', 'has_brand'
    ]
    
    return features, feature_cols

def create_text_similarity_features(df):
    """Create TF-IDF based similarity features"""
    print("Creating text similarity features...")
    
    # Prepare text data
    queries = df['query'].fillna('')
    titles = df['product_title'].fillna('')
    descriptions = df['product_description'].fillna('')
    
    # Create TF-IDF features for query-title similarity
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    
    # Combine queries and titles for fitting
    all_text = pd.concat([queries, titles])
    tfidf.fit(all_text)
    
    # Transform queries and titles
    query_tfidf = tfidf.transform(queries)
    title_tfidf = tfidf.transform(titles)
    
    # Calculate cosine similarity
    similarity_scores = []
    for i in range(len(queries)):
        sim = cosine_similarity(query_tfidf[i], title_tfidf[i])[0][0]
        similarity_scores.append(sim)
    
    df['query_title_similarity'] = similarity_scores
    
    return df

def train_lgb_ranker(X_train, y_train, group_train, X_val, y_val, group_val):
    """Train LightGBM ranker"""
    print("Training LightGBM ranker...")
    
    # LightGBM parameters for ranking
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'random_state': 42
    }
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
    val_data = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_data)
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=50)
        ]
    )
    
    return model

def prepare_ranking_data(df, feature_cols):
    """Prepare data for ranking (group by query_id)"""
    print("Preparing ranking data...")
    
    # Sort by query_id for proper grouping
    df_sorted = df.sort_values('query_id').reset_index(drop=True)
    
    # Create features and target
    X = df_sorted[feature_cols].fillna(0)
    y = df_sorted['esci_score']
    
    # Create group information (number of items per query)
    group_info = df_sorted.groupby('query_id').size().values
    
    return X, y, group_info, df_sorted

def evaluate_model(model, X_test, y_test, group_test):
    """Evaluate the trained model"""
    print("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate NDCG using LightGBM's built-in metric
    test_data = lgb.Dataset(X_test, label=y_test, group=group_test)
    eval_result = model.eval(test_data, feval=None)
    
    print(f"Test NDCG: {eval_result}")
    
    return y_pred

def main():
    """Main training pipeline"""
    print("Starting Task 1 training pipeline...")
    print("=" * 50)
    
    # Load data
    df = load_data()
    
    # Create basic features
    df_features, feature_cols = create_basic_features(df)
    
    # Add text similarity features
    df_features = create_text_similarity_features(df_features)
    feature_cols.append('query_title_similarity')
    
    print(f"Total features: {len(feature_cols)}")
    print(f"Features: {feature_cols}")
    
    # Split train and test
    train_df = df_features[df_features['split'] == 'train'].copy()
    test_df = df_features[df_features['split'] == 'test'].copy()
    
    # Prepare ranking data
    X_train, y_train, group_train, train_sorted = prepare_ranking_data(train_df, feature_cols)
    X_test, y_test, group_test, test_sorted = prepare_ranking_data(test_df, feature_cols)
    
    # Create validation split from training data
    # Split by queries to avoid data leakage
    unique_queries = train_sorted['query_id'].unique()
    train_queries, val_queries = train_test_split(unique_queries, test_size=0.2, random_state=42)
    
    train_mask = train_sorted['query_id'].isin(train_queries)
    val_mask = train_sorted['query_id'].isin(val_queries)
    
    X_train_split = X_train[train_mask]
    y_train_split = y_train[train_mask]
    group_train_split = train_sorted[train_mask].groupby('query_id').size().values
    
    X_val_split = X_train[val_mask]
    y_val_split = y_train[val_mask]
    group_val_split = train_sorted[val_mask].groupby('query_id').size().values
    
    # Train model
    model = train_lgb_ranker(
        X_train_split, y_train_split, group_train_split,
        X_val_split, y_val_split, group_val_split
    )
    
    # Evaluate on test set
    y_pred = evaluate_model(model, X_test, y_test, group_test)
    
    # Save model and results
    output_dir = Path(__file__).parent.parent / "results" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_model(str(output_dir / "task1_baseline_model.txt"))
    
    # Save predictions
    pred_df = test_sorted[['example_id', 'query_id', 'product_id', 'esci_label']].copy()
    pred_df['predicted_score'] = y_pred
    
    pred_output_dir = Path(__file__).parent.parent / "results" / "predictions"
    pred_output_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(pred_output_dir / "task1_baseline_predictions.csv", index=False)
    
    print("=" * 50)
    print("Training completed successfully!")
    print(f"Model saved to: {output_dir / 'task1_baseline_model.txt'}")
    print(f"Predictions saved to: {pred_output_dir / 'task1_baseline_predictions.csv'}")

if __name__ == "__main__":
    main()