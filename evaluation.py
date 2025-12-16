"""
Evaluation and analysis framework for comparing LDA topics with arXiv categories.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import logging
from config import (
    RESULTS_DIR, PROCESSED_DATA_DIR, TOP_WORDS_PER_TOPIC, TOP_CATEGORIES_PER_TOPIC,
    MIN_CATEGORY_COUNT, TOP_CATEGORIES_FOR_EVAL
)
from utils import (
    ensure_dir, load_pickle, load_json, save_json, load_dataframe,
    get_dominant_topic, logger
)
from visualization import (
    create_topic_word_interactive_html,
    create_document_topic_interactive_html,
    create_topic_dashboard_html
)
from generate_report import generate_pdf_report

# Set style
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")


def get_dominant_topics(document_topics):
    """
    Get dominant topic for each document.
    
    Args:
        document_topics: Array of topic distributions (n_documents x n_topics)
        
    Returns:
        Array of dominant topic indices
    """
    return np.array([get_dominant_topic(dist) for dist in document_topics])


def analyze_category_distribution(df):
    """
    Analyze distribution of arXiv categories in the dataset.
    
    Args:
        df: DataFrame with category information
        
    Returns:
        Dictionary with category statistics
    """
    logger.info("Analyzing category distribution...")
    
    # Primary category distribution
    primary_cat_counts = df['primary_category'].value_counts()
    
    # All categories distribution (flatten categories column)
    all_categories = []
    for cats_str in df['categories'].dropna():
        if isinstance(cats_str, str):
            cats = [cat.strip() for cat in cats_str.split(',')]
            all_categories.extend(cats)
    
    all_cat_counts = Counter(all_categories)
    
    results = {
        'primary_category_counts': primary_cat_counts.to_dict(),
        'all_category_counts': dict(all_cat_counts),
        'num_unique_primary_categories': len(primary_cat_counts),
        'num_unique_all_categories': len(all_cat_counts),
        'total_papers': len(df)
    }
    
    logger.info(f"Found {results['num_unique_primary_categories']} unique primary categories")
    logger.info(f"Found {results['num_unique_all_categories']} unique categories (including secondary)")
    
    return results


def analyze_topic_category_alignment(document_topics, df, topic_words=None):
    """
    Analyze alignment between LDA topics and arXiv categories.
    
    Args:
        document_topics: Array of topic distributions (n_documents x n_topics)
        df: DataFrame with category information
        topic_words: Optional list of topic words for labeling
        
    Returns:
        Dictionary with alignment analysis results
    """
    logger.info("Analyzing topic-category alignment...")
    
    # Get dominant topics
    dominant_topics = get_dominant_topics(document_topics)
    
    # Ensure DataFrame has the right length
    if len(df) != len(dominant_topics):
        logger.warning(f"DataFrame length ({len(df)}) != document_topics length ({len(dominant_topics)})")
        min_len = min(len(df), len(dominant_topics))
        df = df.iloc[:min_len]
        dominant_topics = dominant_topics[:min_len]
    
    # Add dominant topic to DataFrame
    df_analysis = df.copy()
    df_analysis['dominant_topic'] = dominant_topics
    
    # Analyze primary category distribution per topic
    topic_primary_cat_dist = {}
    topic_all_cat_dist = {}
    
    num_topics = document_topics.shape[1]
    
    for topic_id in range(num_topics):
        topic_mask = dominant_topics == topic_id
        topic_papers = df_analysis[topic_mask]
        
        if len(topic_papers) > 0:
            # Primary category distribution
            primary_cats = topic_papers['primary_category'].value_counts(normalize=True)
            topic_primary_cat_dist[topic_id] = primary_cats.to_dict()
            
            # All categories distribution
            all_cats_list = []
            for cats_str in topic_papers['categories'].dropna():
                if isinstance(cats_str, str):
                    cats = [cat.strip() for cat in cats_str.split(',')]
                    all_cats_list.extend(cats)
            
            all_cats_counter = Counter(all_cats_list)
            total = sum(all_cats_counter.values())
            if total > 0:
                topic_all_cat_dist[topic_id] = {
                    cat: count / total 
                    for cat, count in all_cats_counter.items()
                }
            else:
                topic_all_cat_dist[topic_id] = {}
        else:
            topic_primary_cat_dist[topic_id] = {}
            topic_all_cat_dist[topic_id] = {}
    
    # Calculate purity metrics
    # Purity: for each topic, what fraction of papers belong to the most common category
    topic_purities = {}
    for topic_id in range(num_topics):
        topic_mask = dominant_topics == topic_id
        topic_papers = df_analysis[topic_mask]
        
        if len(topic_papers) > 0:
            primary_cats = topic_papers['primary_category'].value_counts()
            if len(primary_cats) > 0:
                purity = primary_cats.iloc[0] / len(topic_papers)
                topic_purities[topic_id] = purity
            else:
                topic_purities[topic_id] = 0.0
        else:
            topic_purities[topic_id] = 0.0
    
    # Calculate overall metrics
    # Convert primary categories to numeric labels for ARI/NMI
    unique_cats = df_analysis['primary_category'].unique()
    cat_to_label = {cat: idx for idx, cat in enumerate(unique_cats)}
    category_labels = np.array([cat_to_label[cat] for cat in df_analysis['primary_category']])
    
    # Calculate Adjusted Rand Index and Normalized Mutual Information
    ari = adjusted_rand_score(category_labels, dominant_topics)
    nmi = normalized_mutual_info_score(category_labels, dominant_topics)
    
    results = {
        'topic_primary_category_distribution': topic_primary_cat_dist,
        'topic_all_category_distribution': topic_all_cat_dist,
        'topic_purities': topic_purities,
        'adjusted_rand_index': float(ari),
        'normalized_mutual_info': float(nmi),
        'average_purity': float(np.mean(list(topic_purities.values()))),
        'num_topics': num_topics,
        'num_categories': len(unique_cats)
    }
    
    logger.info(f"Adjusted Rand Index: {ari:.4f}")
    logger.info(f"Normalized Mutual Information: {nmi:.4f}")
    logger.info(f"Average Topic Purity: {results['average_purity']:.4f}")
    
    return results, df_analysis


def compute_predicted_categories(alignment_results, df_analysis):
    """
    Compute predicted category for each paper based on dominant topic's most common category.
    
    Args:
        alignment_results: Dictionary from analyze_topic_category_alignment
        df_analysis: DataFrame with dominant_topic column
        
    Returns:
        Array of predicted categories
    """
    topic_cat_dist = alignment_results['topic_primary_category_distribution']
    
    # For each topic, get the most common category (predicted category)
    topic_to_predicted_cat = {}
    for topic_id, dist in topic_cat_dist.items():
        if len(dist) > 0:
            predicted_cat = max(dist.items(), key=lambda x: x[1])[0]
            topic_to_predicted_cat[topic_id] = predicted_cat
        else:
            topic_to_predicted_cat[topic_id] = 'Unknown'
    
    # Assign predicted category to each paper
    predicted_categories = np.array([
        topic_to_predicted_cat[topic_id] 
        for topic_id in df_analysis['dominant_topic']
    ])
    
    return predicted_categories, topic_to_predicted_cat


def compute_classification_metrics(y_true, y_pred, categories=None, min_support=None):
    """
    Compute classification metrics comparing predicted vs ground truth categories.
    
    Args:
        y_true: Ground truth categories
        y_pred: Predicted categories
        categories: List of all unique categories (for consistent ordering)
        min_support: Minimum number of papers per category to include
        
    Returns:
        Dictionary with metrics
    """
    if categories is None:
        categories = sorted(list(set(y_true) | set(y_pred)))
    
    # Filter categories by minimum support if specified
    if min_support is not None:
        category_counts = pd.Series(y_true).value_counts()
        valid_categories = category_counts[category_counts >= min_support].index.tolist()
        categories = [c for c in categories if c in valid_categories]
        
        # Filter data to only include valid categories
        mask = pd.Series(y_true).isin(categories)
        y_true_filtered = np.array(y_true)[mask]
        y_pred_filtered = np.array(y_pred)[mask]
        
        logger.info(f"Filtered to {len(categories)} categories with >= {min_support} papers")
        logger.info(f"Evaluating on {len(y_true_filtered)} papers (out of {len(y_true)})")
    else:
        y_true_filtered = y_true
        y_pred_filtered = y_pred
    
    # Overall accuracy
    accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
    
    # Per-category metrics
    precision_per_cat = precision_score(
        y_true_filtered, y_pred_filtered, labels=categories, average=None, zero_division=0
    )
    recall_per_cat = recall_score(
        y_true_filtered, y_pred_filtered, labels=categories, average=None, zero_division=0
    )
    f1_per_cat = f1_score(
        y_true_filtered, y_pred_filtered, labels=categories, average=None, zero_division=0
    )
    
    # Macro averages (unweighted mean across categories)
    precision_macro = precision_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
    recall_macro = recall_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
    f1_macro = f1_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
    
    # Weighted averages (weighted by support)
    precision_weighted = precision_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=categories)
    
    # Per-category support (number of true instances per category)
    category_support = {}
    for cat in categories:
        category_support[cat] = int(np.sum(y_true_filtered == cat))
    
    # Create per-category metrics dictionary
    per_category_metrics = {}
    for idx, cat in enumerate(categories):
        per_category_metrics[cat] = {
            'precision': float(precision_per_cat[idx]),
            'recall': float(recall_per_cat[idx]),
            'f1': float(f1_per_cat[idx]),
            'support': category_support[cat]
        }
    
    results = {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'per_category_metrics': per_category_metrics,
        'confusion_matrix': cm.tolist(),
        'categories': categories
    }
    
    return results


def compute_per_topic_accuracy(df_analysis, topic_to_predicted_cat):
    """
    Compute accuracy for each topic (what percentage of papers in each topic have correct category).
    
    Args:
        df_analysis: DataFrame with dominant_topic and primary_category columns
        topic_to_predicted_cat: Dictionary mapping topic_id to predicted category
        
    Returns:
        Dictionary with per-topic accuracies
    """
    per_topic_accuracy = {}
    num_topics = df_analysis['dominant_topic'].max() + 1
    
    for topic_id in range(num_topics):
        topic_mask = df_analysis['dominant_topic'] == topic_id
        topic_papers = df_analysis[topic_mask]
        
        if len(topic_papers) > 0:
            predicted_cat = topic_to_predicted_cat[topic_id]
            correct = (topic_papers['primary_category'] == predicted_cat).sum()
            accuracy = correct / len(topic_papers)
            per_topic_accuracy[topic_id] = {
                'accuracy': float(accuracy),
                'correct': int(correct),
                'total': int(len(topic_papers)),
                'predicted_category': predicted_cat
            }
        else:
            per_topic_accuracy[topic_id] = {
                'accuracy': 0.0,
                'correct': 0,
                'total': 0,
                'predicted_category': 'Unknown'
            }
    
    return per_topic_accuracy


def visualize_confusion_matrix(metrics_results, output_path=None):
    """
    Visualize confusion matrix for predicted vs ground truth categories.
    
    Args:
        metrics_results: Dictionary from compute_classification_metrics
        output_path: Path to save the plot
    """
    cm = np.array(metrics_results['confusion_matrix'])
    categories = metrics_results['categories']
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=categories,
        yticklabels=categories,
        cbar_kws={'label': 'Count'},
        ax=axes[0]
    )
    axes[0].set_title('Confusion Matrix (Counts)\nPredicted vs Ground Truth Categories', 
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted Category', fontsize=12)
    axes[0].set_ylabel('True Category', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)
    
    # Normalized
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=categories,
        yticklabels=categories,
        cbar_kws={'label': 'Normalized Proportion'},
        ax=axes[1]
    )
    axes[1].set_title('Confusion Matrix (Normalized)\nPredicted vs Ground Truth Categories', 
                       fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted Category', fontsize=12)
    axes[1].set_ylabel('True Category', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    
    if output_path:
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {output_path}")
    
    plt.close()


def visualize_per_category_metrics(metrics_results, output_path=None, top_n=20):
    """
    Visualize precision, recall, and F1 per category.
    
    Args:
        metrics_results: Dictionary from compute_classification_metrics
        output_path: Path to save the plot
        top_n: Number of top categories to show
    """
    per_cat_metrics = metrics_results['per_category_metrics']
    
    # Sort categories by support (number of papers)
    categories_sorted = sorted(
        per_cat_metrics.items(),
        key=lambda x: x[1]['support'],
        reverse=True
    )[:top_n]
    
    categories = [cat for cat, _ in categories_sorted]
    precision = [per_cat_metrics[cat]['precision'] for cat in categories]
    recall = [per_cat_metrics[cat]['recall'] for cat in categories]
    f1 = [per_cat_metrics[cat]['f1'] for cat in categories]
    support = [per_cat_metrics[cat]['support'] for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(max(14, len(categories) * 0.6), 8))
    
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Per-Category Classification Metrics\n(Top {top_n} Categories by Support)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add support annotations
    for i, (cat, supp) in enumerate(zip(categories, support)):
        ax.text(i, 1.05, f'n={supp}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved per-category metrics plot to {output_path}")
    
    plt.close()


def create_comparison_summary_table(alignment_results, classification_metrics, output_path=None):
    """
    Create a clear summary table comparing topics with categories.
    
    Args:
        alignment_results: Dictionary from analyze_topic_category_alignment
        classification_metrics: Dictionary from compute_classification_metrics
        output_path: Path to save the visualization
    """
    num_topics = alignment_results['num_topics']
    topic_cat_dist = alignment_results['topic_primary_category_distribution']
    
    # Create summary data
    summary_data = []
    for topic_id in range(num_topics):
        dist = topic_cat_dist.get(topic_id, {})
        if len(dist) > 0:
            top_cat = max(dist.items(), key=lambda x: x[1])
            summary_data.append({
                'Topic': topic_id,
                'Top Category': top_cat[0],
                'Top Category %': f"{top_cat[1]:.1%}",
                'Purity': f"{alignment_results['topic_purities'].get(topic_id, 0):.1%}",
                'Top 3 Categories': ', '.join([f"{cat}({prob:.1%})" for cat, prob in 
                                              sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]])
            })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, max(8, num_topics * 0.5)))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df_summary.values,
                    colLabels=df_summary.columns,
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.1, 0.2, 0.15, 0.15, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(df_summary.columns)):
        table[(0, i)].set_facecolor('#667eea')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code by purity
    for i in range(1, len(df_summary) + 1):
        purity = float(df_summary.iloc[i-1]['Purity'].rstrip('%')) / 100
        color_intensity = purity * 0.5 + 0.5
        for j in range(len(df_summary.columns)):
            table[(i, j)].set_facecolor(plt.cm.Greens(color_intensity))
    
    plt.title('Topic-Category Alignment Summary\n(Top Category per Topic)', 
              fontsize=16, fontweight='bold', pad=20)
    
    if output_path:
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison summary table to {output_path}")
    
    plt.close()
    
    return df_summary


def visualize_topic_category_comparison(alignment_results, classification_metrics, output_path=None):
    """
    Create a clear side-by-side comparison visualization.
    
    Args:
        alignment_results: Dictionary from analyze_topic_category_alignment
        classification_metrics: Dictionary from compute_classification_metrics
        output_path: Path to save the plot
    """
    num_topics = alignment_results['num_topics']
    topic_cat_dist = alignment_results['topic_primary_category_distribution']
    
    # Get top categories
    top_categories = classification_metrics.get('top_categories_list', 
                                                list(classification_metrics['per_category_metrics'].keys())[:10])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Topic-Category Heatmap (Top categories only)
    ax1 = fig.add_subplot(gs[0, 0])
    matrix = np.zeros((num_topics, len(top_categories)))
    for topic_id, dist in topic_cat_dist.items():
        for cat_idx, cat in enumerate(top_categories):
            if cat in dist:
                matrix[topic_id, cat_idx] = dist[cat]
    
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=top_categories, yticklabels=[f"Topic {i}" for i in range(num_topics)],
                cbar_kws={'label': 'Proportion'}, ax=ax1)
    ax1.set_title('Topic-Category Alignment\n(Top Categories)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('arXiv Category', fontsize=10)
    ax1.set_ylabel('LDA Topic', fontsize=10)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Per-Category Performance (Top categories)
    ax2 = fig.add_subplot(gs[0, 1])
    top_metrics = classification_metrics.get('top_categories_metrics', classification_metrics)
    per_cat = top_metrics['per_category_metrics']
    
    cats = top_categories[:10]
    precision = [per_cat[cat]['precision'] for cat in cats]
    recall = [per_cat[cat]['recall'] for cat in cats]
    f1 = [per_cat[cat]['f1'] for cat in cats]
    
    x = np.arange(len(cats))
    width = 0.25
    ax2.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax2.bar(x, recall, width, label='Recall', alpha=0.8)
    ax2.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    ax2.set_xlabel('Category', fontsize=10)
    ax2.set_ylabel('Score', fontsize=10)
    ax2.set_title('Classification Performance\n(Top Categories)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(cats, rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylim([0, 1.1])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Topic Purity Distribution
    ax3 = fig.add_subplot(gs[1, 0])
    purities = [alignment_results['topic_purities'][i] for i in range(num_topics)]
    colors = plt.cm.RdYlGn(purities)
    bars = ax3.bar(range(num_topics), purities, color=colors)
    ax3.set_xlabel('Topic ID', fontsize=10)
    ax3.set_ylabel('Purity', fontsize=10)
    ax3.set_title('Topic Purity\n(% Papers in Most Common Category)', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(num_topics))
    ax3.set_xticklabels([f"T{i}" for i in range(num_topics)])
    ax3.set_ylim([0, 1])
    ax3.grid(True, alpha=0.3, axis='y')
    for i, (bar, purity) in enumerate(zip(bars, purities)):
        ax3.text(bar.get_x() + bar.get_width()/2, purity + 0.02, 
                f'{purity:.1%}', ha='center', va='bottom', fontsize=8)
    
    # 4. Overall Metrics Summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    metrics_text = f"""
    Overall Performance Metrics:
    
    Classification Accuracy: {classification_metrics['accuracy']:.2%}
    Weighted F1-Score: {classification_metrics['f1_weighted']:.2%}
    Macro F1-Score: {classification_metrics['f1_macro']:.2%}
    
    Alignment Metrics:
    Adjusted Rand Index: {alignment_results['adjusted_rand_index']:.3f}
    Normalized Mutual Info: {alignment_results['normalized_mutual_info']:.3f}
    Average Topic Purity: {alignment_results['average_purity']:.2%}
    
    Top Categories Performance:
    """
    
    if 'top_categories_metrics' in classification_metrics:
        top_met = classification_metrics['top_categories_metrics']
        metrics_text += f"""
    Accuracy (Top {len(top_categories)}): {top_met['accuracy']:.2%}
    Weighted F1 (Top {len(top_categories)}): {top_met['f1_weighted']:.2%}
    """
    
    ax4.text(0.1, 0.9, metrics_text, fontsize=11, verticalalignment='top',
            family='monospace', transform=ax4.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Topic-Category Comparison Summary', fontsize=16, fontweight='bold', y=0.98)
    
    if output_path:
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved topic-category comparison to {output_path}")
    
    plt.close()


def visualize_per_topic_accuracy(per_topic_accuracy, output_path=None):
    """
    Visualize accuracy for each topic.
    
    Args:
        per_topic_accuracy: Dictionary from compute_per_topic_accuracy
        output_path: Path to save the plot
    """
    topics = sorted(per_topic_accuracy.keys())
    accuracies = [per_topic_accuracy[topic]['accuracy'] for topic in topics]
    totals = [per_topic_accuracy[topic]['total'] for topic in topics]
    predicted_cats = [per_topic_accuracy[topic]['predicted_category'] for topic in topics]
    
    fig, ax = plt.subplots(figsize=(max(12, len(topics) * 0.8), 6))
    
    bars = ax.bar(range(len(topics)), accuracies, alpha=0.8)
    
    # Color bars by accuracy
    colors = plt.cm.RdYlGn(accuracies)
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add value labels
    for i, (acc, total, pred_cat) in enumerate(zip(accuracies, totals, predicted_cats)):
        ax.text(i, acc + 0.02, f'{acc:.2%}\n(n={total})', 
                ha='center', va='bottom', fontsize=9)
        ax.text(i, -0.05, f'→{pred_cat}', 
                ha='center', va='top', fontsize=8, rotation=45)
    
    ax.set_xlabel('Topic ID', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Per-Topic Classification Accuracy\n(Predicted Category vs Ground Truth)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(topics)))
    ax.set_xticklabels([f"Topic {t}" for t in topics])
    ax.set_ylim([-0.15, 1.1])
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved per-topic accuracy plot to {output_path}")
    
    plt.close()


def visualize_category_distribution(category_stats, output_path=None):
    """
    Visualize distribution of arXiv categories.
    
    Args:
        category_stats: Dictionary from analyze_category_distribution
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Primary categories
    primary_cats = category_stats['primary_category_counts']
    top_primary = dict(sorted(primary_cats.items(), key=lambda x: x[1], reverse=True)[:15])
    
    axes[0].barh(range(len(top_primary)), list(top_primary.values()))
    axes[0].set_yticks(range(len(top_primary)))
    axes[0].set_yticklabels(list(top_primary.keys()))
    axes[0].set_xlabel('Number of Papers', fontsize=12)
    axes[0].set_title('Top 15 Primary Categories', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # All categories
    all_cats = category_stats['all_category_counts']
    top_all = dict(sorted(all_cats.items(), key=lambda x: x[1], reverse=True)[:15])
    
    axes[1].barh(range(len(top_all)), list(top_all.values()))
    axes[1].set_yticks(range(len(top_all)))
    axes[1].set_yticklabels(list(top_all.keys()))
    axes[1].set_xlabel('Number of Papers', fontsize=12)
    axes[1].set_title('Top 15 All Categories (Including Secondary)', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if output_path:
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved category distribution plot to {output_path}")
    
    plt.close()


def visualize_topic_category_heatmap(alignment_results, output_path=None):
    """
    Create heatmap showing topic-category alignment.
    
    Args:
        alignment_results: Dictionary from analyze_topic_category_alignment
        output_path: Path to save the plot
    """
    topic_cat_dist = alignment_results['topic_primary_category_distribution']
    num_topics = alignment_results['num_topics']
    
    # Collect all unique categories
    all_categories = set()
    for dist in topic_cat_dist.values():
        all_categories.update(dist.keys())
    all_categories = sorted(list(all_categories))
    
    # Create matrix
    matrix = np.zeros((num_topics, len(all_categories)))
    for topic_id, dist in topic_cat_dist.items():
        for cat_idx, cat in enumerate(all_categories):
            if cat in dist:
                matrix[topic_id, cat_idx] = dist[cat]
    
    # Create heatmap
    plt.figure(figsize=(max(12, len(all_categories) * 0.8), max(8, num_topics * 0.6)))
    sns.heatmap(
        matrix,
        xticklabels=all_categories,
        yticklabels=[f"Topic {i}" for i in range(num_topics)],
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Proportion'}
    )
    plt.title('Topic-Category Alignment Heatmap\n(Proportion of Primary Categories per Topic)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('arXiv Primary Category', fontsize=12)
    plt.ylabel('LDA Topic', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if output_path:
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved topic-category heatmap to {output_path}")
    
    plt.close()


def visualize_topic_category_stacked_bar(alignment_results, output_path=None, top_n=10):
    """
    Create stacked bar chart showing category proportions per topic.
    
    Args:
        alignment_results: Dictionary from analyze_topic_category_alignment
        output_path: Path to save the plot
        top_n: Number of top categories to show per topic
    """
    topic_cat_dist = alignment_results['topic_primary_category_distribution']
    num_topics = alignment_results['num_topics']
    
    # Get top categories across all topics
    all_cat_counts = Counter()
    for dist in topic_cat_dist.values():
        all_cat_counts.update(dist)
    top_categories = [cat for cat, _ in all_cat_counts.most_common(top_n)]
    
    # Create matrix
    matrix = np.zeros((num_topics, len(top_categories)))
    for topic_id, dist in topic_cat_dist.items():
        for cat_idx, cat in enumerate(top_categories):
            if cat in dist:
                matrix[topic_id, cat_idx] = dist[cat]
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(max(12, num_topics * 0.8), 8))
    
    x = np.arange(num_topics)
    width = 0.8
    bottom = np.zeros(num_topics)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_categories)))
    
    for cat_idx, cat in enumerate(top_categories):
        ax.bar(x, matrix[:, cat_idx], width, label=cat, bottom=bottom, color=colors[cat_idx])
        bottom += matrix[:, cat_idx]
    
    ax.set_xlabel('LDA Topic', fontsize=12)
    ax.set_ylabel('Proportion of Papers', fontsize=12)
    ax.set_title(f'Category Distribution per Topic\n(Top {top_n} Categories)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Topic {i}" for i in range(num_topics)])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved stacked bar chart to {output_path}")
    
    plt.close()


def visualize_topic_words(topic_words, output_path=None, words_per_topic=10):
    """
    Visualize top words for each topic.
    
    Args:
        topic_words: List of topic words (from get_topic_words)
        output_path: Path to save the plot
        words_per_topic: Number of words to show per topic
    """
    num_topics = len(topic_words)
    
    fig, axes = plt.subplots((num_topics + 2) // 3, 3, figsize=(15, 5 * ((num_topics + 2) // 3)))
    axes = axes.flatten() if num_topics > 1 else [axes]
    
    for topic_id, words in enumerate(topic_words):
        if topic_id >= len(axes):
            break
        
        top_words = words[:words_per_topic]
        words_list = [w[0] for w in top_words]
        probs_list = [w[1] for w in top_words]
        
        axes[topic_id].barh(range(len(words_list)), probs_list)
        axes[topic_id].set_yticks(range(len(words_list)))
        axes[topic_id].set_yticklabels(words_list)
        axes[topic_id].set_xlabel('Probability', fontsize=10)
        axes[topic_id].set_title(f'Topic {topic_id}', fontsize=12, fontweight='bold')
        axes[topic_id].invert_yaxis()
        axes[topic_id].grid(True, alpha=0.3, axis='x')
    
    # Hide extra subplots
    for idx in range(num_topics, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Top Words per Topic', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if output_path:
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved topic words visualization to {output_path}")
    
    plt.close()


def visualize_topic_distribution_over_time(df_analysis, output_path=None):
    """
    Visualize topic distribution over time (by year).
    
    Args:
        df_analysis: DataFrame with dominant_topic and year columns
        output_path: Path to save the plot
    """
    if 'year' not in df_analysis.columns:
        logger.warning("Year column not found, skipping time-based visualization")
        return
    
    # Group by year and topic
    topic_year_counts = df_analysis.groupby(['year', 'dominant_topic']).size().unstack(fill_value=0)
    
    # Normalize to proportions
    topic_year_props = topic_year_counts.div(topic_year_counts.sum(axis=1), axis=0)
    
    # Create stacked area chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    topic_year_props.plot(kind='area', ax=ax, stacked=True, alpha=0.7)
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Proportion of Papers', fontsize=12)
    ax.set_title('Topic Distribution Over Time', fontsize=14, fontweight='bold')
    ax.legend(title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved topic distribution over time plot to {output_path}")
    
    plt.close()


def generate_analysis_report():
    """
    Generate comprehensive analysis report comparing LDA topics with arXiv categories.
    
    Returns:
        Dictionary with all analysis results
    """
    logger.info("Generating comprehensive analysis report...")
    
    # Load data
    try:
        df = load_dataframe(f"{PROCESSED_DATA_DIR}/processed_data.csv")
    except Exception as e:
        logger.error(f"Could not load processed data: {e}")
        return None
    
    # Load topic modeling results
    document_topics = load_pickle(f"{RESULTS_DIR}/document_topics.pkl")
    topic_words_json = load_json(f"{RESULTS_DIR}/topic_words.json")
    
    # Convert topic words to list format
    topic_words = []
    for topic_key in sorted(topic_words_json.keys(), key=lambda x: int(x.split('_')[1])):
        words_dict = topic_words_json[topic_key]
        words_list = [(word, prob) for word, prob in words_dict.items()]
        words_list.sort(key=lambda x: x[1], reverse=True)
        topic_words.append(words_list)
    
    # Analyze category distribution
    category_stats = analyze_category_distribution(df)
    save_json(category_stats, f"{RESULTS_DIR}/category_statistics.json")
    
    # Analyze topic-category alignment
    alignment_results, df_analysis = analyze_topic_category_alignment(
        document_topics, df, topic_words
    )
    save_json(alignment_results, f"{RESULTS_DIR}/topic_category_alignment.json")
    
    # Compute predicted categories (based on dominant topic's most common category)
    predicted_categories, topic_to_predicted_cat = compute_predicted_categories(
        alignment_results, df_analysis
    )
    df_analysis['predicted_category'] = predicted_categories
    
    # Compute classification metrics (predicted vs ground truth)
    # Focus on major categories for clearer evaluation
    classification_metrics = compute_classification_metrics(
        df_analysis['primary_category'].values,
        predicted_categories,
        min_support=MIN_CATEGORY_COUNT
    )
    
    # Also compute metrics for top categories only
    top_categories = df_analysis['primary_category'].value_counts().head(TOP_CATEGORIES_FOR_EVAL).index.tolist()
    top_cat_mask = df_analysis['primary_category'].isin(top_categories)
    classification_metrics_top = compute_classification_metrics(
        df_analysis.loc[top_cat_mask, 'primary_category'].values,
        predicted_categories[top_cat_mask],
        categories=top_categories
    )
    classification_metrics['top_categories_metrics'] = classification_metrics_top
    classification_metrics['top_categories_list'] = top_categories
    save_json(classification_metrics, f"{RESULTS_DIR}/classification_metrics.json")
    
    # Compute per-topic accuracy
    per_topic_accuracy = compute_per_topic_accuracy(df_analysis, topic_to_predicted_cat)
    save_json(per_topic_accuracy, f"{RESULTS_DIR}/per_topic_accuracy.json")
    
    # Save analysis DataFrame
    df_analysis.to_csv(f"{RESULTS_DIR}/analysis_data.csv", index=False)
    
    # Create simplified CSV with id, predicted_category, and ground_truth
    df_predictions = pd.DataFrame({
        'id': df_analysis['id'],
        'predicted_category': df_analysis['predicted_category'],
        'ground_truth': df_analysis['primary_category']
    })
    df_predictions.to_csv(f"{RESULTS_DIR}/predictions_vs_ground_truth.csv", index=False)
    logger.info(f"Saved predictions vs ground truth CSV to {RESULTS_DIR}/predictions_vs_ground_truth.csv")
    
    # Create visualizations
    visualize_category_distribution(
        category_stats,
        f"{RESULTS_DIR}/category_distribution.png"
    )
    
    visualize_topic_category_heatmap(
        alignment_results,
        f"{RESULTS_DIR}/topic_category_heatmap.png"
    )
    
    visualize_topic_category_stacked_bar(
        alignment_results,
        f"{RESULTS_DIR}/topic_category_stacked_bar.png"
    )
    
    visualize_topic_words(
        topic_words,
        f"{RESULTS_DIR}/topic_words_visualization.png"
    )
    
    visualize_topic_distribution_over_time(
        df_analysis,
        f"{RESULTS_DIR}/topic_distribution_over_time.png"
    )
    
    # Create classification metrics visualizations
    visualize_confusion_matrix(
        classification_metrics,
        f"{RESULTS_DIR}/confusion_matrix.png"
    )
    
    visualize_per_category_metrics(
        classification_metrics,
        f"{RESULTS_DIR}/per_category_metrics.png"
    )
    
    visualize_per_topic_accuracy(
        per_topic_accuracy,
        f"{RESULTS_DIR}/per_topic_accuracy.png"
    )
    
    # Create clearer comparison visualizations
    create_comparison_summary_table(
        alignment_results,
        classification_metrics,
        f"{RESULTS_DIR}/comparison_summary_table.png"
    )
    
    visualize_topic_category_comparison(
        alignment_results,
        classification_metrics,
        f"{RESULTS_DIR}/topic_category_comparison.png"
    )
    
    # Create interactive HTML visualizations
    try:
        logger.info("Creating interactive HTML visualizations...")
        topic_word_df = pd.read_csv(f"{RESULTS_DIR}/topic_word_distribution.csv")
        doc_topic_df = pd.read_csv(f"{RESULTS_DIR}/document_topic_distribution.csv")
        
        create_topic_word_interactive_html(topic_word_df)
        create_document_topic_interactive_html(doc_topic_df)
        create_topic_dashboard_html(topic_word_df, doc_topic_df)
        logger.info("Interactive HTML visualizations created successfully!")
    except Exception as e:
        logger.warning(f"Could not create interactive HTML visualizations: {e}")
    
    # Generate PDF report
    try:
        logger.info("Generating PDF report...")
        generate_pdf_report()
        logger.info("PDF report generated successfully!")
    except Exception as e:
        logger.warning(f"Could not generate PDF report: {e}")
    
    # Print summary
    logger.info("\n=== Analysis Summary ===")
    logger.info(f"Total papers analyzed: {len(df_analysis)}")
    logger.info(f"Number of topics: {alignment_results['num_topics']}")
    logger.info(f"Number of unique categories: {alignment_results['num_categories']}")
    logger.info(f"Adjusted Rand Index: {alignment_results['adjusted_rand_index']:.4f}")
    logger.info(f"Normalized Mutual Information: {alignment_results['normalized_mutual_info']:.4f}")
    logger.info(f"Average Topic Purity: {alignment_results['average_purity']:.4f}")
    
    logger.info("\n" + "="*70)
    logger.info("CLASSIFICATION METRICS (Predicted vs Ground Truth)")
    logger.info("="*70)
    logger.info(f"\nOverall Performance:")
    logger.info(f"  Accuracy: {classification_metrics['accuracy']:.2%}")
    logger.info(f"  Weighted F1-Score: {classification_metrics['f1_weighted']:.2%}")
    logger.info(f"  Macro F1-Score: {classification_metrics['f1_macro']:.2%}")
    logger.info(f"\nDetailed Metrics:")
    logger.info(f"  Precision (weighted): {classification_metrics['precision_weighted']:.2%}")
    logger.info(f"  Recall (weighted): {classification_metrics['recall_weighted']:.2%}")
    logger.info(f"  Precision (macro): {classification_metrics['precision_macro']:.2%}")
    logger.info(f"  Recall (macro): {classification_metrics['recall_macro']:.2%}")
    
    if 'top_categories_metrics' in classification_metrics:
        top_met = classification_metrics['top_categories_metrics']
        top_cats = classification_metrics['top_categories_list']
        logger.info(f"\nTop {len(top_cats)} Categories Performance:")
        logger.info(f"  Accuracy: {top_met['accuracy']:.2%}")
        logger.info(f"  Weighted F1-Score: {top_met['f1_weighted']:.2%}")
        logger.info(f"  Categories: {', '.join(top_cats)}")
    
    logger.info("\n=== Per-Topic Accuracy ===")
    for topic_id in sorted(per_topic_accuracy.keys()):
        acc_info = per_topic_accuracy[topic_id]
        logger.info(f"Topic {topic_id}: {acc_info['accuracy']:.2%} "
                   f"(Predicted: {acc_info['predicted_category']}, "
                   f"Correct: {acc_info['correct']}/{acc_info['total']})")
    
    logger.info("\n" + "="*70)
    logger.info("TOP CATEGORIES BY PERFORMANCE")
    logger.info("="*70)
    per_cat_metrics = classification_metrics['per_category_metrics']
    
    # Filter to categories with reasonable support
    valid_cats = {k: v for k, v in per_cat_metrics.items() if v['support'] >= MIN_CATEGORY_COUNT}
    sorted_by_f1 = sorted(
        valid_cats.items(),
        key=lambda x: x[1]['f1'],
        reverse=True
    )[:15]
    
    logger.info(f"\n{'Category':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    logger.info("-" * 70)
    for cat, metrics in sorted_by_f1:
        logger.info(f"{cat:<15} {metrics['precision']:>10.2%} {metrics['recall']:>10.2%} "
                   f"{metrics['f1']:>10.2%} {metrics['support']:>10}")
    
    logger.info("\n" + "="*70)
    logger.info("TOPIC-CATEGORY ALIGNMENT SUMMARY")
    logger.info("="*70)
    topic_cat_dist = alignment_results['topic_primary_category_distribution']
    logger.info(f"\n{'Topic':<8} {'Top Category':<15} {'Top %':<10} {'Purity':<10} {'Top 3 Categories'}")
    logger.info("-" * 80)
    for topic_id in range(alignment_results['num_topics']):
        dist = topic_cat_dist.get(topic_id, {})
        if len(dist) > 0:
            top_cat = max(dist.items(), key=lambda x: x[1])
            top3 = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
            top3_str = ', '.join([f"{cat}({prob:.0%})" for cat, prob in top3])
            purity = alignment_results['topic_purities'].get(topic_id, 0)
            logger.info(f"Topic {topic_id:<4} {top_cat[0]:<15} {top_cat[1]:>8.1%} "
                       f"{purity:>8.1%} {top3_str}")
    
    logger.info("\n=== Top Categories per Topic ===")
    for topic_id in range(alignment_results['num_topics']):
        dist = alignment_results['topic_primary_category_distribution'][topic_id]
        top_cats = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:TOP_CATEGORIES_PER_TOPIC]
        logger.info(f"\nTopic {topic_id}:")
        for cat, prop in top_cats:
            logger.info(f"  {cat}: {prop:.2%}")
    
    return {
        'category_stats': category_stats,
        'alignment_results': alignment_results,
        'classification_metrics': classification_metrics,
        'per_topic_accuracy': per_topic_accuracy,
        'df_analysis': df_analysis
    }


if __name__ == "__main__":
    results = generate_analysis_report()
    print("\nAnalysis complete!")
