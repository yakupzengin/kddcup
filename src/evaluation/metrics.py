"""
Evaluation metrics for ESCI Challenge
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, ndcg_score
import logging

logger = logging.getLogger(__name__)

class ESCIEvaluator:
    """Evaluation metrics for ESCI Challenge"""
    
    @staticmethod
    def calculate_ndcg_by_query(y_true: pd.Series, y_pred: np.ndarray, 
                               groups: np.ndarray, k_values: List[int] = [1, 5, 10]) -> Dict:
        """Calculate NDCG@k for each query and return average"""
        ndcg_scores = {f'ndcg@{k}': [] for k in k_values}
        
        start_idx = 0
        for group_size in groups:
            end_idx = start_idx + group_size
            
            # Get true and predicted relevance for this query
            y_true_query = y_true.iloc[start_idx:end_idx].values
            y_pred_query = y_pred[start_idx:end_idx]
            
            # Calculate NDCG@k for this query
            for k in k_values:
                if len(y_true_query) >= k:
                    ndcg_k = ndcg_score([y_true_query], [y_pred_query], k=k)
                    ndcg_scores[f'ndcg@{k}'].append(ndcg_k)
            
            start_idx = end_idx
        
        # Calculate average NDCG@k
        avg_ndcg = {}
        for metric, scores in ndcg_scores.items():
            avg_ndcg[metric] = np.mean(scores) if scores else 0.0
        
        return avg_ndcg
    
    @staticmethod
    def evaluate_ranking(df_test: pd.DataFrame, predictions: np.ndarray) -> Dict:
        """Comprehensive evaluation for ranking task"""
        logger.info("Evaluating ranking performance...")
        
        # Prepare data
        df_sorted = df_test.sort_values('query_id').reset_index(drop=True)
        y_true = df_sorted['esci_score']
        groups = df_sorted.groupby('query_id').size().values
        
        # Calculate NDCG scores
        ndcg_results = ESCIEvaluator.calculate_ndcg_by_query(y_true, predictions, groups)
        
        # Performance by ESCI label
        df_with_pred = df_sorted.copy()
        df_with_pred['pred_score'] = predictions
        
        esci_performance = {}
        for label in ['E', 'S', 'C', 'I']:
            label_mask = df_with_pred['esci_label'] == label
            if label_mask.sum() > 0:
                label_data = df_with_pred[label_mask]
                esci_performance[label] = {
                    'count': len(label_data),
                    'avg_true': label_data['esci_score'].mean(),
                    'avg_pred': label_data['pred_score'].mean()
                }
        
        return {
            'ndcg_scores': ndcg_results,
            'esci_performance': esci_performance,
            'total_queries': len(groups),
            'avg_query_size': np.mean(groups)
        }
    
    @staticmethod
    def evaluate_classification(y_true: pd.Series, y_pred: np.ndarray, 
                               task_type: str = 'multiclass') -> Dict:
        """Evaluate classification performance"""
        logger.info(f"Evaluating {task_type} classification performance...")
        
        if task_type == 'binary':
            # Binary classification (Task 3)
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            results = {
                'accuracy': accuracy_score(y_true, y_pred_binary),
                'precision': precision_score(y_true, y_pred_binary),
                'recall': recall_score(y_true, y_pred_binary),
                'f1_score': f1_score(y_true, y_pred_binary),
                'confusion_matrix': confusion_matrix(y_true, y_pred_binary).tolist()
            }
        else:
            # Multi-class classification (Task 2)
            unique_labels = sorted(y_true.unique())
            y_pred_class = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
            
            results = {
                'accuracy': accuracy_score(y_true, y_pred_class),
                'precision': precision_score(y_true, y_pred_class, average='weighted'),
                'recall': recall_score(y_true, y_pred_class, average='weighted'),
                'f1_score': f1_score(y_true, y_pred_class, average='weighted'),
                'classification_report': classification_report(y_true, y_pred_class, 
                                                              target_names=[str(x) for x in unique_labels],
                                                              output_dict=True)
            }
        
        return results
    
    @staticmethod
    def create_evaluation_report(results: Dict, task_id: int, experiment_name: str) -> Dict:
        """Create comprehensive evaluation report"""
        logger.info(f"Creating evaluation report for Task {task_id}")
        
        report = {
            'task_id': task_id,
            'experiment_name': experiment_name,
            'task_type': Config.TASKS[task_id]['type'],
            'metrics': results,
            'summary': {}
        }
        
        # Add task-specific summary
        if task_id == 1:  # Ranking
            report['summary'] = {
                'primary_metric': f"NDCG@10: {results['ndcg_scores']['ndcg@10']:.4f}",
                'ndcg@1': results['ndcg_scores']['ndcg@1'],
                'ndcg@5': results['ndcg_scores']['ndcg@5'],
                'ndcg@10': results['ndcg_scores']['ndcg@10'],
                'total_queries': results['total_queries']
            }
        elif task_id in [2, 3]:  # Classification
            report['summary'] = {
                'primary_metric': f"F1-Score: {results['f1_score']:.4f}",
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1_score': results['f1_score']
            }
        
        return report
    
    @staticmethod
    def compare_experiments(reports: List[Dict]) -> pd.DataFrame:
        """Compare multiple experiment results"""
        logger.info("Comparing experiment results...")
        
        comparison_data = []
        for report in reports:
            row = {
                'experiment': report['experiment_name'],
                'task_id': report['task_id'],
                'task_type': report['task_type']
            }
            
            # Add metrics based on task type
            if report['task_id'] == 1:
                row.update({
                    'ndcg@1': report['metrics']['ndcg_scores']['ndcg@1'],
                    'ndcg@5': report['metrics']['ndcg_scores']['ndcg@5'],
                    'ndcg@10': report['metrics']['ndcg_scores']['ndcg@10']
                })
            else:
                row.update({
                    'accuracy': report['metrics']['accuracy'],
                    'precision': report['metrics']['precision'],
                    'recall': report['metrics']['recall'],
                    'f1_score': report['metrics']['f1_score']
                })
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)