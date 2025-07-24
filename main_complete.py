"""
Main training pipeline for KDD Cup 2022 ESCI Challenge
Orchestrates data loading, feature engineering, training, and evaluation
"""
import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
import json
import warnings
import sys
from datetime import datetime

warnings.filterwarnings('ignore')

# Add src to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

from src.config.config import Config
from src.data.data_loader import DataLoader
from src.features.base_features import BaseFeatureEngineer
from src.models.lgb_ranker import LGBRanker
from src.evaluation.metrics import ESCIEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ESCITrainingPipeline:
    """Main training pipeline for ESCI Challenge"""
    
    def __init__(self, task_id: int, experiment_name: str = None):
        self.task_id = task_id
        self.experiment_name = experiment_name or f"baseline_task_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize components
        self.data_loader = DataLoader()
        self.feature_engineer = BaseFeatureEngineer()
        self.model = LGBRanker(task_id)
        self.evaluator = ESCIEvaluator()
        
        # Create experiment directory
        self.experiment_dir = Config.EXPERIMENTS_DIR / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized pipeline for Task {task_id}")
        logger.info(f"Experiment: {self.experiment_name}")
    
    def load_data(self):
        """Load and prepare data"""
        logger.info("Loading data...")
        
        # Try to load processed data first
        try:
            datasets = self.data_loader.load_processed_data(self.task_id)
            self.train_data = datasets['train']
            self.test_data = datasets['test']
            logger.info("Loaded processed data")
        except FileNotFoundError:
            # Load raw data and process
            logger.info("Processed data not found, loading raw data...")
            df_examples, df_products, df_sources = self.data_loader.load_raw_data()
            
            # Merge and filter for task
            df_merged = self.data_loader.merge_examples_products(df_examples, df_products)
            df_task = self.data_loader.filter_for_task(df_merged, self.task_id)
            
            # Split train/test
            self.train_data, self.test_data = self.data_loader.split_train_test(df_task)
    
    def engineer_features(self):
        """Create features"""
        logger.info("Engineering features...")
        
        # Create features for train and test
        self.train_data = self.feature_engineer.create_all_features(self.train_data)
        self.test_data = self.feature_engineer.create_all_features(self.test_data)
        
        # Get feature columns
        self.feature_columns = self.feature_engineer.get_feature_columns()
        logger.info(f"Created {len(self.feature_columns)} features")
    
    def train_model(self):
        """Train model"""
        logger.info("Training model...")
        
        # Split train data for validation
        from sklearn.model_selection import train_test_split
        train_queries = self.train_data['query_id'].unique()
        val_queries, train_queries_final = train_test_split(
            train_queries, test_size=0.8, random_state=42
        )
        
        train_final = self.train_data[self.train_data['query_id'].isin(train_queries_final)]
        val_data = self.train_data[self.train_data['query_id'].isin(val_queries)]
        
        # Train model
        self.training_results = self.model.train(
            train_final, val_data, self.feature_columns
        )
        
        logger.info(f"Training completed. Best iteration: {self.training_results['best_iteration']}")
    
    def evaluate_model(self):
        """Evaluate model"""
        logger.info("Evaluating model...")
        
        # Make predictions
        predictions = self.model.predict(self.test_data)
        
        # Evaluate based on task type
        task_config = Config.TASKS[self.task_id]
        if task_config['type'] == 'ranking':
            self.evaluation_results = self.evaluator.evaluate_ranking(self.test_data, predictions)
        else:
            task_type = 'binary' if self.task_id == 3 else 'multiclass'
            target_col = task_config['target_column']
            self.evaluation_results = self.evaluator.evaluate_classification(
                self.test_data[target_col], predictions, task_type
            )
        
        # Create evaluation report
        self.evaluation_report = self.evaluator.create_evaluation_report(
            self.evaluation_results, self.task_id, self.experiment_name
        )
        
        logger.info(f"Evaluation completed: {self.evaluation_report['summary']['primary_metric']}")
    
    def save_results(self):
        """Save all results"""
        logger.info("Saving results...")
        
        # Save model
        model_path = self.experiment_dir / "model.pkl"
        self.model.save_model(model_path)
        
        # Save evaluation report
        report_path = self.experiment_dir / "evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.evaluation_report, f, indent=2, default=str)
        
        # Save feature importance
        feature_importance = self.model.get_feature_importance()
        importance_path = self.experiment_dir / "feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)
        
        logger.info(f"Results saved to {self.experiment_dir}")
    
    def run_pipeline(self):
        """Run complete training pipeline"""
        logger.info(f"Starting pipeline for Task {self.task_id}")
        
        try:
            self.load_data()
            self.engineer_features()
            self.train_model()
            self.evaluate_model()
            self.save_results()
            
            logger.info("Pipeline completed successfully!")
            return self.evaluation_report
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='KDD Cup 2022 ESCI Challenge Training Pipeline')
    parser.add_argument('--task', type=int, choices=[1, 2, 3], help='Task ID (1, 2, or 3)')
    parser.add_argument('--all-tasks', action='store_true', help='Run all tasks')
    parser.add_argument('--experiment', type=str, help='Experiment name')
    
    args = parser.parse_args()
    
    if args.all_tasks:
        # Run all tasks
        results = {}
        for task_id in [1, 2, 3]:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running Task {task_id}")
            logger.info(f"{'='*60}")
            
            pipeline = ESCITrainingPipeline(task_id, args.experiment)
            results[task_id] = pipeline.run_pipeline()
        
        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        for task_id, result in results.items():
            print(f"Task {task_id}: {result['summary']['primary_metric']}")
    
    elif args.task:
        # Run single task
        pipeline = ESCITrainingPipeline(args.task, args.experiment)
        result = pipeline.run_pipeline()
        
        print("\n" + "="*60)
        print("EXPERIMENT RESULT")
        print("="*60)
        print(f"Task {args.task}: {result['summary']['primary_metric']}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()