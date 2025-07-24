"""
Training script for Task 1: Query-Product Ranking
KDD Cup 2022 ESCI Challenge

This script implements a LightGBM ranker for ranking products given a query.
Target: Start with basic features and achieve incremental improvements.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from main_complete import ESCITrainingPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Train Task 1: Query-Product Ranking"""
    logger.info("Starting Task 1 training: Query-Product Ranking")
    
    # Initialize and run pipeline
    pipeline = ESCITrainingPipeline(
        task_id=1,
        experiment_name="task1_ranking_baseline"
    )
    
    result = pipeline.run_pipeline()
    
    print("\n" + "="*60)
    print("TASK 1 TRAINING COMPLETED")
    print("="*60)
    print(f"Primary Metric: {result['summary']['primary_metric']}")
    print(f"NDCG@1: {result['summary']['ndcg@1']:.4f}")
    print(f"NDCG@5: {result['summary']['ndcg@5']:.4f}")
    print(f"NDCG@10: {result['summary']['ndcg@10']:.4f}")
    print(f"Total Queries: {result['summary']['total_queries']}")
    print("="*60)

if __name__ == "__main__":
    main()