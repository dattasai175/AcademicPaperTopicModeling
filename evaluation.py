"""
Evaluation and comparison framework for topic modeling and classification.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import logging
from config import RESULTS_DIR, PROCESSED_DATA_DIR
from utils import (
    ensure_dir, load_pickle, load_json, save_json, load_dataframe,
    get_dominant_topic, logger
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


def compare_topic_clusters_vs_classification(document_topics, classification_labels, 
                                            topic_to_subfield_mapping):
    """
    Compare unsupervised topic clusters with supervised classifications.
    
    Args:
        document_topics: Topic distributions for documents
        classification_labels: Labels from supervised classification
        topic_to_subfield_mapping: Mapping from topics to subfields
        
    Returns:
        Dictionary with comparison metrics
    """
    logger.info("Comparing topic clusters with supervised classifications...")
    
    # Get dominant topic for each document
    dominant_topics = np.array([get_dominant_topic(dist) for dist in document_topics])
    
    # Map topics to subfields
    topic_labels = np.array([
        topic_to_subfield_mapping.get(topic_id, 'Other')
        for topic_id in dominant_topics
    ])
    
    # Calculate alignment metrics
    # Adjusted Rand Index
    ari = adjusted_rand_score(classification_labels, topic_labels)
    
    # Normalized Mutual Information
    nmi = normalized_mutual_info_score(classification_labels, topic_labels)
    
    # Agreement percentage
    agreement = np.mean(classification_labels == topic_labels)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'topic_label': topic_labels,
        'classification_label': classification_labels,
        'agreement': classification_labels == topic_labels
    })
    
    # Per-class agreement
    per_class_agreement = {}
    for label in np.unique(classification_labels):
        mask = classification_labels == label
        if np.sum(mask) > 0:
            per_class_agreement[label] = np.mean(topic_labels[mask] == label)
    
    results = {
        'adjusted_rand_index': float(ari),
        'normalized_mutual_info': float(nmi),
        'overall_agreement': float(agreement),
        'per_class_agreement': per_class_agreement,
        'comparison_df': comparison_df
    }
    
    logger.info(f"Adjusted Rand Index: {ari:.4f}")
    logger.info(f"Normalized Mutual Information: {nmi:.4f}")
    logger.info(f"Overall Agreement: {agreement:.4f}")
    
    return results


def visualize_confusion_matrix(confusion_matrix, class_names, output_path=None):
    """
    Visualize confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 10))
    
    # Normalize confusion matrix
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Frequency'}
    )
    
    plt.title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if output_path:
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {output_path}")
    
    plt.close()


def visualize_topic_classification_alignment(comparison_df, output_path=None):
    """
    Visualize alignment between topic clusters and classifications.
    
    Args:
        comparison_df: DataFrame with topic and classification labels
        output_path: Path to save the plot
    """
    # Create cross-tabulation
    crosstab = pd.crosstab(
        comparison_df['topic_label'],
        comparison_df['classification_label'],
        normalize='index'
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        crosstab,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Proportion'}
    )
    
    plt.title('Topic Labels vs Classification Labels', fontsize=14, fontweight='bold')
    plt.ylabel('Topic Label', fontsize=12)
    plt.xlabel('Classification Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if output_path:
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved alignment plot to {output_path}")
    
    plt.close()


def visualize_topic_distribution(document_topics, labels=None, output_path=None):
    """
    Visualize topic distribution across documents.
    
    Args:
        document_topics: Topic distributions
        labels: Optional labels for coloring
        output_path: Path to save the plot
    """
    # Calculate average topic distribution
    avg_topic_dist = np.mean(document_topics, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(avg_topic_dist)), avg_topic_dist)
    plt.xlabel('Topic ID', fontsize=12)
    plt.ylabel('Average Probability', fontsize=12)
    plt.title('Average Topic Distribution Across Documents', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if output_path:
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved topic distribution plot to {output_path}")
    
    plt.close()


def create_error_analysis(df, predictions, true_labels, output_path=None):
    """
    Create error analysis report.
    
    Args:
        df: DataFrame with abstracts
        predictions: Predicted labels
        true_labels: True labels
        output_path: Path to save the report
    """
    error_mask = predictions != true_labels
    
    if np.sum(error_mask) == 0:
        logger.info("No errors found!")
        return
    
    error_data = {
        'true_label': true_labels[error_mask],
        'predicted_label': predictions[error_mask]
    }
    
    if df is not None:
        if 'title' in df.columns:
            error_data['title'] = df['title'].values[error_mask]
        if 'abstract' in df.columns:
            error_data['abstract'] = df['abstract'].values[error_mask]
    
    error_df = pd.DataFrame(error_data)
    
    # Error statistics by class
    error_stats = pd.DataFrame({
        'true_label': true_labels[error_mask],
        'predicted_label': predictions[error_mask]
    }).value_counts().reset_index(name='count')
    
    logger.info("\n=== Error Analysis ===")
    logger.info(f"Total errors: {len(error_df)}")
    logger.info(f"Error rate: {len(error_df) / len(predictions):.4f}")
    logger.info("\nError patterns:")
    logger.info(error_stats.to_string())
    
    if output_path:
        ensure_dir(os.path.dirname(output_path))
        error_df.to_csv(output_path, index=False)
        logger.info(f"Saved error analysis to {output_path}")


def generate_evaluation_report():
    """
    Generate comprehensive evaluation report.
    
    Returns:
        Dictionary with all evaluation results
    """
    import os
    
    logger.info("Generating comprehensive evaluation report...")
    
    # Load data
    try:
        df = load_dataframe(f"{PROCESSED_DATA_DIR}/processed_data.csv")
    except Exception as e:
        logger.warning(f"Could not load processed data: {e}")
        df = None
    
    document_topics = load_pickle(f"{RESULTS_DIR}/document_topics.pkl")
    classification_metrics = load_json(f"{RESULTS_DIR}/classification_metrics.json")
    topic_to_subfield = load_json(f"{RESULTS_DIR}/topic_to_subfield_mapping.json")
    predictions_df = load_dataframe(f"{RESULTS_DIR}/predictions.csv")
    
    # Get labels
    true_labels = predictions_df['true_label'].values
    predicted_labels = predictions_df['predicted_label'].values
    
    # Load test set indices
    try:
        test_indices = load_pickle(f"{RESULTS_DIR}/test_indices.pkl")
        # Get document topics for test set only
        test_document_topics = document_topics[test_indices]
    except Exception as e:
        logger.warning(f"Could not load test indices: {e}. Using all documents for comparison.")
        test_document_topics = document_topics
        test_indices = np.arange(len(document_topics))
    
    # Compare topic clusters with classification on test set
    comparison_results = compare_topic_clusters_vs_classification(
        test_document_topics, true_labels, topic_to_subfield
    )
    
    # Save comparison results
    save_json({
        'adjusted_rand_index': comparison_results['adjusted_rand_index'],
        'normalized_mutual_info': comparison_results['normalized_mutual_info'],
        'overall_agreement': comparison_results['overall_agreement'],
        'per_class_agreement': comparison_results['per_class_agreement']
    }, f"{RESULTS_DIR}/comparison_results.json")
    
    # Visualizations
    visualize_confusion_matrix(
        np.array(classification_metrics['confusion_matrix']),
        classification_metrics['class_names'],
        f"{RESULTS_DIR}/confusion_matrix.png"
    )
    
    visualize_topic_classification_alignment(
        comparison_results['comparison_df'],
        f"{RESULTS_DIR}/topic_classification_alignment.png"
    )
    
    visualize_topic_distribution(
        document_topics,
        output_path=f"{RESULTS_DIR}/topic_distribution.png"
    )
    
    # Error analysis (only if we have the full dataframe)
    try:
        create_error_analysis(
            df,
            predicted_labels,
            true_labels,
            f"{RESULTS_DIR}/error_analysis.csv"
        )
    except Exception as e:
        logger.warning(f"Could not create error analysis: {e}")
    
    # Print summary
    logger.info("\n=== Evaluation Summary ===")
    logger.info(f"Classification Accuracy: {classification_metrics['accuracy']:.4f}")
    logger.info(f"Classification F1-score: {classification_metrics['f1_weighted']:.4f}")
    logger.info(f"Topic-Classification Agreement: {comparison_results['overall_agreement']:.4f}")
    logger.info(f"Adjusted Rand Index: {comparison_results['adjusted_rand_index']:.4f}")
    logger.info(f"Normalized Mutual Information: {comparison_results['normalized_mutual_info']:.4f}")
    
    return {
        'classification_metrics': classification_metrics,
        'comparison_results': comparison_results
    }


if __name__ == "__main__":
    import os
    results = generate_evaluation_report()
    print("\nEvaluation complete!")

