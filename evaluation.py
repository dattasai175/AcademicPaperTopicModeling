"""
Evaluation and analysis framework for comparing LDA topics with arXiv categories.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import logging
from config import (
    RESULTS_DIR, PROCESSED_DATA_DIR, TOP_WORDS_PER_TOPIC, TOP_CATEGORIES_PER_TOPIC
)
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
    
    # Save analysis DataFrame
    df_analysis.to_csv(f"{RESULTS_DIR}/analysis_data.csv", index=False)
    
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
    
    # Print summary
    logger.info("\n=== Analysis Summary ===")
    logger.info(f"Total papers analyzed: {len(df_analysis)}")
    logger.info(f"Number of topics: {alignment_results['num_topics']}")
    logger.info(f"Number of unique categories: {alignment_results['num_categories']}")
    logger.info(f"Adjusted Rand Index: {alignment_results['adjusted_rand_index']:.4f}")
    logger.info(f"Normalized Mutual Information: {alignment_results['normalized_mutual_info']:.4f}")
    logger.info(f"Average Topic Purity: {alignment_results['average_purity']:.4f}")
    
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
        'df_analysis': df_analysis
    }


if __name__ == "__main__":
    results = generate_analysis_report()
    print("\nAnalysis complete!")
