"""
Compare classification accuracy across LDA, NMF, and BERTopic topic modeling methods.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
import logging
from config import (
    RESULTS_DIR, PROCESSED_DATA_DIR, MIN_CATEGORY_COUNT, TOP_CATEGORIES_FOR_EVAL
)
from utils import (
    ensure_dir, load_pickle, load_json, save_json, load_dataframe, logger
)
from evaluation import (
    analyze_topic_category_alignment, compute_predicted_categories,
    compute_classification_metrics
)

# Set style
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")


def evaluate_method(document_topics, topic_words_json_path, method_name, df):
    """
    Evaluate a single topic modeling method.
    
    Args:
        document_topics: Document-topic distribution matrix
        topic_words_json_path: Path to topic words JSON file
        method_name: Name of the method ('lda', 'nmf', 'bertopic')
        df: DataFrame with category information
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"\nEvaluating {method_name.upper()}...")
    
    # Load topic words
    topic_words_json = load_json(topic_words_json_path)
    
    # Convert topic words to list format
    topic_words = []
    for topic_key in sorted(topic_words_json.keys(), key=lambda x: int(x.split('_')[1])):
        words_dict = topic_words_json[topic_key]
        words_list = [(word, prob) for word, prob in words_dict.items()]
        words_list.sort(key=lambda x: x[1], reverse=True)
        topic_words.append(words_list)
    
    # Analyze topic-category alignment
    alignment_results, df_analysis = analyze_topic_category_alignment(
        document_topics, df, topic_words
    )
    
    # Compute predicted categories
    predicted_categories, topic_to_predicted_cat = compute_predicted_categories(
        alignment_results, df_analysis
    )
    df_analysis['predicted_category'] = predicted_categories
    
    # Compute classification metrics
    classification_metrics = compute_classification_metrics(
        df_analysis['primary_category'].values,
        predicted_categories,
        min_support=MIN_CATEGORY_COUNT
    )
    
    # Compute top categories metrics
    top_categories = df_analysis['primary_category'].value_counts().head(TOP_CATEGORIES_FOR_EVAL).index.tolist()
    top_cat_mask = df_analysis['primary_category'].isin(top_categories)
    classification_metrics_top = compute_classification_metrics(
        df_analysis.loc[top_cat_mask, 'primary_category'].values,
        predicted_categories[top_cat_mask],
        categories=top_categories
    )
    
    return {
        'method_name': method_name,
        'alignment_results': alignment_results,
        'classification_metrics': classification_metrics,
        'classification_metrics_top': classification_metrics_top,
        'df_analysis': df_analysis,
        'topic_to_predicted_cat': topic_to_predicted_cat,
        'num_topics': alignment_results['num_topics']
    }


def compare_all_methods():
    """
    Compare classification accuracy across all topic modeling methods.
    
    Returns:
        Dictionary with comparison results
    """
    logger.info("=" * 70)
    logger.info("COMPARING ALL TOPIC MODELING METHODS")
    logger.info("=" * 70)
    
    # Load data
    df = load_dataframe(f"{PROCESSED_DATA_DIR}/processed_data.csv")
    
    results = {}
    
    # Evaluate LDA
    try:
        document_topics_lda = load_pickle(f"{RESULTS_DIR}/document_topics.pkl")
        results['lda'] = evaluate_method(
            document_topics_lda,
            f"{RESULTS_DIR}/topic_words.json",
            'lda',
            df
        )
        logger.info(f"✅ LDA evaluated: Accuracy = {results['lda']['classification_metrics']['accuracy']:.4f}")
    except Exception as e:
        logger.warning(f"Could not evaluate LDA: {e}")
        results['lda'] = None
    
    # Evaluate NMF
    nmf_file = f"{RESULTS_DIR}/nmf_document_topics.pkl"
    if os.path.exists(nmf_file):
        try:
            document_topics_nmf = load_pickle(nmf_file)
            results['nmf'] = evaluate_method(
                document_topics_nmf,
                f"{RESULTS_DIR}/nmf_topic_words.json",
                'nmf',
                df
            )
            logger.info(f"✅ NMF evaluated: Accuracy = {results['nmf']['classification_metrics']['accuracy']:.4f}")
        except Exception as e:
            logger.warning(f"Could not evaluate NMF: {e}")
            results['nmf'] = None
    else:
        logger.warning(f"NMF results not found at {nmf_file}. Run NMF topic modeling first.")
        results['nmf'] = None
    
    # Evaluate BERTopic
    bertopic_file = f"{RESULTS_DIR}/bertopic_document_topics.pkl"
    if os.path.exists(bertopic_file):
        try:
            document_topics_bertopic = load_pickle(bertopic_file)
            results['bertopic'] = evaluate_method(
                document_topics_bertopic,
                f"{RESULTS_DIR}/bertopic_topic_words.json",
                'bertopic',
                df
            )
            logger.info(f"✅ BERTopic evaluated: Accuracy = {results['bertopic']['classification_metrics']['accuracy']:.4f}")
        except Exception as e:
            logger.warning(f"Could not evaluate BERTopic: {e}")
            results['bertopic'] = None
    else:
        logger.warning(f"BERTopic results not found at {bertopic_file}. Run BERTopic topic modeling first.")
        results['bertopic'] = None
    
    # Create comparison summary
    comparison_summary = create_comparison_summary(results)
    
    # Only save and visualize if we have at least one method
    if len(comparison_summary['methods']) > 0:
        save_json(comparison_summary, f"{RESULTS_DIR}/method_comparison_summary.json")
        
        # Visualize comparison
        visualize_method_comparison(results, f"{RESULTS_DIR}/method_comparison.png")
    else:
        logger.warning("No methods available for comparison. Skipping summary and visualization.")
        logger.info("Make sure to run topic modeling for at least one method first.")
    
    logger.info("\n" + "=" * 70)
    logger.info("METHOD COMPARISON COMPLETE")
    logger.info("=" * 70)
    
    return results


def create_comparison_summary(results):
    """Create summary comparison of all methods."""
    summary = {
        'methods': {},
        'best_method': None,
        'best_accuracy': 0.0,
        'available_methods': []
    }
    
    for method_name, result in results.items():
        if result is not None:
            metrics = result['classification_metrics']
            summary['methods'][method_name] = {
                'accuracy': metrics['accuracy'],
                'f1_weighted': metrics['f1_weighted'],
                'f1_macro': metrics['f1_macro'],
                'num_topics': result['num_topics'],
                'top_accuracy': result['classification_metrics_top']['accuracy'],
                'top_f1_weighted': result['classification_metrics_top']['f1_weighted']
            }
            summary['available_methods'].append(method_name)
            
            if metrics['accuracy'] > summary['best_accuracy']:
                summary['best_accuracy'] = metrics['accuracy']
                summary['best_method'] = method_name
    
    if len(summary['methods']) == 0:
        logger.warning("No methods available for comparison!")
    
    return summary


def visualize_method_comparison(results, output_path=None):
    """Visualize comparison of all methods."""
    methods = []
    accuracies = []
    f1_weighted = []
    f1_macro = []
    num_topics_list = []
    top_accuracies = []
    
    for method_name, result in results.items():
        if result is not None:
            methods.append(method_name.upper())
            accuracies.append(result['classification_metrics']['accuracy'])
            f1_weighted.append(result['classification_metrics']['f1_weighted'])
            f1_macro.append(result['classification_metrics']['f1_macro'])
            num_topics_list.append(result['num_topics'])
            top_accuracies.append(result['classification_metrics_top']['accuracy'])
    
    if not methods:
        logger.warning("No methods to compare")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Overall Accuracy Comparison
    ax1 = axes[0, 0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars1 = ax1.bar(methods, accuracies, color=colors[:len(methods)], alpha=0.8)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Classification Accuracy Comparison\n(All Categories)', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, axis='y')
    for i, (bar, acc) in enumerate(zip(bars1, accuracies)):
        ax1.text(bar.get_x() + bar.get_width()/2, acc + 0.02, 
                f'{acc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. Top Categories Accuracy Comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(methods, top_accuracies, color=colors[:len(methods)], alpha=0.8)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'Classification Accuracy Comparison\n(Top {TOP_CATEGORIES_FOR_EVAL} Categories)', 
                  fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, axis='y')
    for i, (bar, acc) in enumerate(zip(bars2, top_accuracies)):
        ax2.text(bar.get_x() + bar.get_width()/2, acc + 0.02, 
                f'{acc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 3. F1-Score Comparison
    ax3 = axes[1, 0]
    x = np.arange(len(methods))
    width = 0.35
    ax3.bar(x - width/2, f1_weighted, width, label='Weighted F1', alpha=0.8)
    ax3.bar(x + width/2, f1_macro, width, label='Macro F1', alpha=0.8)
    ax3.set_ylabel('F1-Score', fontsize=12)
    ax3.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods)
    ax3.legend()
    ax3.set_ylim([0, 1])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Number of Topics
    ax4 = axes[1, 1]
    bars4 = ax4.bar(methods, num_topics_list, color=colors[:len(methods)], alpha=0.8)
    ax4.set_ylabel('Number of Topics', fontsize=12)
    ax4.set_title('Number of Topics per Method', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    for i, (bar, num) in enumerate(zip(bars4, num_topics_list)):
        ax4.text(bar.get_x() + bar.get_width()/2, num + 0.5, 
                f'{num}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle('Topic Modeling Methods Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if output_path:
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved method comparison plot to {output_path}")
    
    plt.close()


if __name__ == "__main__":
    results = compare_all_methods()
    
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    summary = load_json(f"{RESULTS_DIR}/method_comparison_summary.json")
    
    print("\nClassification Accuracy (All Categories):")
    for method, metrics in summary['methods'].items():
        print(f"  {method.upper()}: {metrics['accuracy']:.4f}")
    
    print(f"\n🏆 Best Method: {summary['best_method'].upper()} "
          f"(Accuracy: {summary['best_accuracy']:.4f})")

