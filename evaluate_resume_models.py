#!/usr/bin/env python3
"""
Comprehensive evaluation script for resume scoring models.
Compares v4 and v5 LLM models against ground truth using NDCG, correlation, MAE, and RMSE.
"""

import math
import numpy as np
from zbench.utils import load_jsonl, ndcg


def pearson_correlation(x, y):
    """Calculate Pearson correlation coefficient"""
    x = np.array(x)
    y = np.array(y)
    
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    num = np.sum((x - mean_x) * (y - mean_y))
    den = np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2))
    
    if den == 0:
        return 0.0
    
    return num / den


def mean_absolute_error(y_true, y_pred):
    """Calculate mean absolute error"""
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def root_mean_squared_error(y_true, y_pred):
    """Calculate root mean squared error"""
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))


def load_resume_scores(file_path):
    """Load resume scores from JSONL file and return as dict mapping query_id -> {doc_id: score}"""
    data = load_jsonl(file_path)
    scores = {}
    
    for entry in data:
        query_id = entry['query']['id']
        doc_scores = {}
        for doc in entry['documents']:
            doc_scores[doc['id']] = doc['score']
        scores[query_id] = doc_scores
    
    return scores


def calculate_ndcg_at_k(ground_truth_scores, predicted_scores, k=None):
    """Calculate NDCG@k for a single query"""
    if k is None:
        k = len(ground_truth_scores)
    
    # Limit to k documents
    limited_gt = ground_truth_scores[:k]
    limited_pred = predicted_scores[:k]
    
    # Convert ELO scores to relevance scores for ground truth
    gt_relevance = [math.exp(score) for score in limited_gt]
    
    return ndcg(gt_relevance, limited_pred)


def evaluate_model(ground_truth_file, predicted_file, model_name):
    """Evaluate a single model against ground truth"""
    
    # Load scores
    gt_scores = load_resume_scores(ground_truth_file)
    pred_scores = load_resume_scores(predicted_file)
    
    results = {
        'model': model_name,
        'ndcg_full': [],
        'ndcg_10': [],
        'ndcg_5': [],
        'correlations': [],
        'maes': [],
        'rmses': []
    }
    
    # Process each query
    for query_id in gt_scores:
        if query_id not in pred_scores:
            print(f"Warning: Query {query_id} not found in predictions for {model_name}")
            continue
            
        gt_query = gt_scores[query_id]
        pred_query = pred_scores[query_id]
        
        # Get common documents
        common_docs = set(gt_query.keys()) & set(pred_query.keys())
        if not common_docs:
            continue
            
        # Extract scores for common documents
        gt_vals = [gt_query[doc_id] for doc_id in common_docs]
        pred_vals = [pred_query[doc_id] for doc_id in common_docs]
        
        # Sort by predicted scores (descending)
        sorted_indices = sorted(range(len(pred_vals)), key=lambda i: pred_vals[i], reverse=True)
        sorted_gt = [gt_vals[i] for i in sorted_indices]
        sorted_pred = [pred_vals[i] for i in sorted_indices]
        
        # Calculate NDCG metrics
        ndcg_full = calculate_ndcg_at_k(sorted_gt, sorted_pred, k=None)
        ndcg_10 = calculate_ndcg_at_k(sorted_gt, sorted_pred, k=min(10, len(sorted_gt)))
        ndcg_5 = calculate_ndcg_at_k(sorted_gt, sorted_pred, k=min(5, len(sorted_gt)))
        
        results['ndcg_full'].append(ndcg_full)
        results['ndcg_10'].append(ndcg_10)
        results['ndcg_5'].append(ndcg_5)
        
        # Calculate correlation, MAE, RMSE
        if len(gt_vals) > 1:  # Need at least 2 points for correlation
            correlation = pearson_correlation(gt_vals, pred_vals)
            results['correlations'].append(correlation)
        
        mae = mean_absolute_error(gt_vals, pred_vals)
        rmse = root_mean_squared_error(gt_vals, pred_vals)
        
        results['maes'].append(mae)
        results['rmses'].append(rmse)
    
    # Calculate averages
    avg_results = {
        'model': model_name,
        'ndcg_full': np.mean(results['ndcg_full']) if results['ndcg_full'] else 0,
        'ndcg_10': np.mean(results['ndcg_10']) if results['ndcg_10'] else 0,
        'ndcg_5': np.mean(results['ndcg_5']) if results['ndcg_5'] else 0,
        'correlation': np.mean(results['correlations']) if results['correlations'] else 0,
        'mae': np.mean(results['maes']) if results['maes'] else 0,
        'rmse': np.mean(results['rmses']) if results['rmses'] else 0
    }
    
    return avg_results


def main():
    """Main evaluation function"""
    
    # File paths
    ground_truth_file = "annotated_resumes_combined.jsonl"
    v4_file = "annotated_v4_02_llm_resume_data.jsonl"
    v5_file = "annotated_v5_02_llm_resume_data.jsonl"
    
    print("Starting comprehensive model evaluation...")
    print("=" * 60)
    
    # Evaluate both models
    v4_results = evaluate_model(ground_truth_file, v4_file, "V4 LLM")
    v5_results = evaluate_model(ground_truth_file, v5_file, "V5 LLM")
    
    # Print results table
    print("\nNDCG Score Comparison Results\n")
    print("| Dataset        | NDCG@Full | NDCG@10  | NDCG@5   | Correlation | MAE   | RMSE  |")
    print("|----------------|-----------|----------|----------|-------------|-------|-------|")
    
    # Print V4 results
    print(f"| {v4_results['model']:<14} | {v4_results['ndcg_full']:.6f}  | {v4_results['ndcg_10']:.6f} | {v4_results['ndcg_5']:.6f} | {v4_results['correlation']:.6f}    | {v4_results['mae']:.3f} | {v4_results['rmse']:.3f} |")
    
    # Print V5 results
    print(f"| {v5_results['model']:<14} | {v5_results['ndcg_full']:.6f}  | {v5_results['ndcg_10']:.6f} | {v5_results['ndcg_5']:.6f} | {v5_results['correlation']:.6f}    | {v5_results['mae']:.3f} | {v5_results['rmse']:.3f} |")
    
    print("\nKey Findings:\n")
    
    # Determine best performing model
    if v5_results['ndcg_full'] > v4_results['ndcg_full']:
        best_ndcg = "V5"
        best_ndcg_score = v5_results['ndcg_full']
    else:
        best_ndcg = "V4"
        best_ndcg_score = v4_results['ndcg_full']
    
    if v4_results['mae'] < v5_results['mae']:
        best_mae = "V4"
        best_mae_score = v4_results['mae']
    else:
        best_mae = "V5"
        best_mae_score = v5_results['mae']
    
    if v4_results['correlation'] > v5_results['correlation']:
        best_corr = "V4"
        best_corr_score = v4_results['correlation']
    else:
        best_corr = "V5"
        best_corr_score = v5_results['correlation']
    
    print(f"1. Best Overall Performance: {best_ndcg} LLM achieves the highest NDCG scores (~{best_ndcg_score:.3f})")
    print(f"2. Ranking Quality: {'V5 > V4' if v5_results['ndcg_full'] > v4_results['ndcg_full'] else 'V4 > V5'}")
    print(f"3. Score Accuracy: {best_mae} has the lowest MAE ({best_mae_score:.3f}), indicating most accurate absolute scores")
    print(f"4. Correlation: {best_corr} has the highest correlation ({best_corr_score:.3f}) with ground truth")
    
    # Calculate performance gap
    ndcg_gap = abs(v5_results['ndcg_full'] - v4_results['ndcg_full'])
    print(f"5. Performance Gap: {ndcg_gap:.3f} NDCG difference between V4 and V5 models")
    
    if best_ndcg == "V5":
        print(f"\nThe V5 LLM model demonstrates the best ranking performance while maintaining good score accuracy and correlation with ground truth.")
    else:
        print(f"\nThe V4 LLM model demonstrates the best ranking performance while maintaining good score accuracy and correlation with ground truth.")


if __name__ == "__main__":
    main()