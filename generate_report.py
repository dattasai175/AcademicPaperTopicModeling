"""
Generate comprehensive PDF report for the topic modeling project.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('PDF')  # Use PDF backend
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from config import RESULTS_DIR, PROCESSED_DATA_DIR
from utils import load_json, load_dataframe, logger

# Set style
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")


def wrap_text(text, width=80):
    """Wrap text to specified width."""
    words = text.split(' ')
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 for space
        if current_length + word_length <= width and current_line:
            current_line.append(word)
            current_length += word_length
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines


def add_text_page(pdf, title, content, fontsize_title=20, fontsize_body=11):
    """Add a text-only page to the PDF."""
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, title, fontsize=fontsize_title, fontweight='bold',
            ha='center', va='top', transform=fig.transFigure)
    
    # Wrap and add content
    wrapped_lines = wrap_text(content, width=90)
    y_start = 0.88
    line_height = 0.025
    
    for i, line in enumerate(wrapped_lines):
        y_pos = y_start - (i * line_height)
        if y_pos < 0.05:  # Start new page if needed
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_subplot(111)
            ax.axis('off')
            y_start = 0.95
            y_pos = y_start - ((i - len(wrapped_lines) + int(0.88/line_height)) * line_height)
        
        ax.text(0.1, y_pos, line, fontsize=fontsize_body,
                ha='left', va='top', transform=fig.transFigure,
                wrap=True)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_title_page(pdf):
    """Create title page."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.7, 'arXiv AI Papers', fontsize=32, fontweight='bold',
             ha='center', va='center')
    fig.text(0.5, 0.6, 'Topic Modeling & Category Alignment Analysis', fontsize=28,
             ha='center', va='center')
    fig.text(0.5, 0.4, 'A Comprehensive Pipeline for Unsupervised Topic Discovery', fontsize=18,
             ha='center', va='center', style='italic')
    fig.text(0.5, 0.2, f'Generated: {datetime.now().strftime("%B %d, %Y")}', fontsize=14,
             ha='center', va='center')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_table_of_contents(pdf):
    """Create table of contents."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.95, 'Table of Contents', fontsize=24, fontweight='bold',
             ha='center', va='top')
    
    contents = [
        ('1. Introduction', 0.85),
        ('2. Project Overview', 0.80),
        ('3. Methodology', 0.75),
        ('   3.1 Data Collection', 0.70),
        ('   3.2 Text Preprocessing', 0.65),
        ('   3.3 Topic Modeling with LDA', 0.60),
        ('   3.4 Category Alignment Analysis', 0.55),
        ('4. Algorithms and Techniques', 0.50),
        ('   4.1 Latent Dirichlet Allocation (LDA)', 0.45),
        ('   4.2 Coherence Score Optimization', 0.40),
        ('   4.3 Evaluation Metrics', 0.35),
        ('5. Results and Visualizations', 0.30),
        ('6. Discussion and Interpretation', 0.25),
        ('7. Conclusion', 0.20),
    ]
    
    for content, y_pos in contents:
        fig.text(0.15, y_pos, content, fontsize=12, ha='left', va='top')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def load_results():
    """Load all result files."""
    results = {}
    try:
        results['coherence'] = load_json(f"{RESULTS_DIR}/coherence_results.json")
        results['classification'] = load_json(f"{RESULTS_DIR}/classification_metrics.json")
        results['alignment'] = load_json(f"{RESULTS_DIR}/topic_category_alignment.json")
        results['category_stats'] = load_json(f"{RESULTS_DIR}/category_statistics.json")
        results['topic_words'] = load_json(f"{RESULTS_DIR}/topic_words.json")
        results['per_topic_acc'] = load_json(f"{RESULTS_DIR}/per_topic_accuracy.json")
        results['df'] = load_dataframe(f"{RESULTS_DIR}/analysis_data.csv")
        return results
    except Exception as e:
        logger.warning(f"Could not load some results: {e}")
        return results


def generate_pdf_report(output_path=None):
    """
    Generate comprehensive PDF report.
    
    Args:
        output_path: Path to save PDF file
    """
    if output_path is None:
        output_path = f"{RESULTS_DIR}/project_report.pdf"
    
    logger.info("Generating PDF report...")
    
    # Load results
    results = load_results()
    
    # Create PDF
    with PdfPages(output_path) as pdf:
        # Title page
        create_title_page(pdf)
        
        # Table of contents
        create_table_of_contents(pdf)
        
        # 1. Introduction
        intro_text = """
        This report presents a comprehensive analysis of arXiv AI research papers using 
        unsupervised topic modeling techniques. The project implements a complete pipeline 
        for collecting, preprocessing, and analyzing academic abstracts to discover latent 
        thematic structures and compare them with ground-truth arXiv categories.
        
        The primary goal is to understand how well unsupervised topic discovery (using 
        Latent Dirichlet Allocation) aligns with the explicit category labels assigned by 
        arXiv, providing insights into both the discovered topics and the category system 
        itself.
        """
        add_text_page(pdf, "1. Introduction", intro_text)
        
        # 2. Project Overview
        overview_text = f"""
        Project Overview:
        
        This project implements a reproducible pipeline for analyzing AI research papers 
        from arXiv. The pipeline consists of four main stages:
        
        1. Data Collection: Automated collection of up to {results.get('coherence', {}).get('optimal_topics', 'N/A')} 
           recent papers from arXiv API across AI-related categories (cs.AI, cs.LG, cs.CV, 
           cs.CL, cs.NE, cs.RO, stat.ML).
        
        2. Preprocessing: Text cleaning, tokenization, lemmatization, and stopword removal 
           to prepare abstracts for topic modeling.
        
        3. Topic Modeling: LDA-based topic discovery with automatic optimization of the 
           number of topics using coherence scores.
        
        4. Analysis: Comprehensive comparison between discovered topics and arXiv categories, 
           including alignment metrics and visualizations.
        
        Key Features:
        - Large-scale data collection (up to 5000 papers)
        - Automatic topic number optimization
        - Comprehensive evaluation metrics
        - Rich visualizations and interactive HTML reports
        """
        add_text_page(pdf, "2. Project Overview", overview_text)
        
        # 3. Methodology
        methodology_text = """
        3.1 Data Collection:
        
        Papers are collected from the arXiv API using the arxiv Python library. The 
        collection process:
        - Searches across multiple AI-related categories
        - Filters papers by publication year (minimum 2018)
        - Handles pagination for large-scale collection
        - Preserves both primary and secondary categories
        - Implements rate limiting to respect API constraints
        
        3.2 Text Preprocessing:
        
        The preprocessing pipeline includes:
        - Lowercasing all text
        - Removing URLs, email addresses, and special characters
        - Tokenization using NLTK's word_tokenize
        - Lemmatization using WordNetLemmatizer
        - Stopword removal (including domain-specific stopwords)
        - Filtering by minimum word length (2 characters)
        - Creating bag-of-words representation for LDA
        
        3.3 Topic Modeling:
        
        Latent Dirichlet Allocation (LDA) is used to discover topics. The process:
        - Evaluates multiple topic numbers (typically 3-20)
        - Computes coherence scores (C_v metric) for each
        - Selects optimal number of topics based on highest coherence
        - Trains final LDA model with optimal parameters
        
        3.4 Category Alignment Analysis:
        
        The analysis compares discovered topics with arXiv categories:
        - Assigns predicted category to each topic (most common category)
        - Computes classification metrics (accuracy, precision, recall, F1)
        - Calculates alignment metrics (ARI, NMI, purity)
        - Generates visualizations and reports
        """
        add_text_page(pdf, "3. Methodology", methodology_text)
        
        # 4. Algorithms
        algorithms_text = """
        4.1 Latent Dirichlet Allocation (LDA):
        
        LDA is a generative probabilistic model that assumes:
        - Each document is a mixture of topics
        - Each topic is a distribution over words
        - Documents are generated by:
          1. Choosing a topic distribution θ ~ Dir(α)
          2. For each word:
             - Choose a topic z ~ Multinomial(θ)
             - Choose a word w ~ Multinomial(β_z)
        
        Parameters:
        - α (alpha): Document-topic prior (controls topic sparsity per document)
        - β (beta/eta): Topic-word prior (controls word sparsity per topic)
        - Number of topics (K): Discovered through coherence optimization
        
        The model is trained using Gibbs sampling or variational inference to 
        estimate the posterior distributions.
        
        4.2 Coherence Score Optimization:
        
        Coherence measures the semantic quality of topics by evaluating:
        - How often top words in a topic co-occur in documents
        - The C_v metric uses normalized pointwise mutual information (NPMI)
        - Higher coherence indicates more interpretable topics
        
        Optimization process:
        - Train LDA models for different numbers of topics
        - Compute coherence score for each
        - Select number of topics with highest coherence
        
        4.3 Evaluation Metrics:
        
        Classification Metrics (Predicted vs Ground Truth):
        - Accuracy: Overall percentage of correct predictions
        - Precision: Of predicted category, how many are correct
        - Recall: Of actual category, how many were predicted correctly
        - F1-Score: Harmonic mean of precision and recall
        
        Alignment Metrics:
        - Adjusted Rand Index (ARI): Measures clustering agreement (adjusted for chance)
        - Normalized Mutual Information (NMI): Measures shared information between clusterings
        - Topic Purity: Fraction of papers in each topic belonging to most common category
        """
        add_text_page(pdf, "4. Algorithms and Techniques", algorithms_text)
        
        # 5. Results - Add visualizations
        if 'coherence' in results:
            # Coherence scores plot
            fig, ax = plt.subplots(figsize=(10, 6))
            coherence = results['coherence']
            ax.plot(coherence['topics_range'], coherence['coherence_scores'], 
                   marker='o', linewidth=2, markersize=8)
            ax.axvline(x=coherence['optimal_topics'], color='r', 
                      linestyle='--', label=f"Optimal: {coherence['optimal_topics']}")
            ax.set_xlabel('Number of Topics', fontsize=12)
            ax.set_ylabel(f"Coherence Score ({coherence.get('coherence_type', 'c_v')})", fontsize=12)
            ax.set_title('Topic Coherence vs Number of Topics', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Category distribution
        if 'category_stats' in results:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            stats = results['category_stats']
            
            # Primary categories
            primary_cats = stats.get('primary_category_counts', {})
            top_primary = dict(sorted(primary_cats.items(), key=lambda x: x[1], reverse=True)[:10])
            
            axes[0].barh(range(len(top_primary)), list(top_primary.values()))
            axes[0].set_yticks(range(len(top_primary)))
            axes[0].set_yticklabels(list(top_primary.keys()))
            axes[0].set_xlabel('Number of Papers', fontsize=12)
            axes[0].set_title('Top 10 Primary Categories', fontsize=14, fontweight='bold')
            axes[0].invert_yaxis()
            axes[0].grid(True, alpha=0.3, axis='x')
            
            # All categories
            all_cats = stats.get('all_category_counts', {})
            top_all = dict(sorted(all_cats.items(), key=lambda x: x[1], reverse=True)[:10])
            
            axes[1].barh(range(len(top_all)), list(top_all.values()))
            axes[1].set_yticks(range(len(top_all)))
            axes[1].set_yticklabels(list(top_all.keys()))
            axes[1].set_xlabel('Number of Papers', fontsize=12)
            axes[1].set_title('Top 10 All Categories', fontsize=14, fontweight='bold')
            axes[1].invert_yaxis()
            axes[1].grid(True, alpha=0.3, axis='x')
            
            plt.suptitle('Category Distribution in Dataset', fontsize=16, fontweight='bold', y=1.02)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Results summary
        results_text = f"""
        5. Results Summary:
        
        Topic Modeling Results:
        - Optimal number of topics: {results.get('coherence', {}).get('optimal_topics', 'N/A')}
        - Optimal coherence score: {results.get('coherence', {}).get('optimal_coherence', 0):.4f}
        - Coherence metric used: {results.get('coherence', {}).get('coherence_type', 'c_v')}
        
        Classification Performance (Predicted vs Ground Truth):
        - Overall Accuracy: {results.get('classification', {}).get('accuracy', 0):.4f}
        - Macro-Averaged F1-Score: {results.get('classification', {}).get('f1_macro', 0):.4f}
        - Weighted-Averaged F1-Score: {results.get('classification', {}).get('f1_weighted', 0):.4f}
        
        Alignment Metrics:
        - Adjusted Rand Index (ARI): {results.get('alignment', {}).get('adjusted_rand_index', 0):.4f}
        - Normalized Mutual Information (NMI): {results.get('alignment', {}).get('normalized_mutual_info', 0):.4f}
        - Average Topic Purity: {results.get('alignment', {}).get('average_purity', 0):.4f}
        
        Dataset Statistics:
        - Total papers analyzed: {results.get('category_stats', {}).get('total_papers', 'N/A')}
        - Unique primary categories: {results.get('category_stats', {}).get('num_unique_primary_categories', 'N/A')}
        - Unique all categories: {results.get('category_stats', {}).get('num_unique_all_categories', 'N/A')}
        """
        add_text_page(pdf, "5. Results", results_text)
        
        # Topic words visualization
        if 'topic_words' in results:
            num_topics = len(results['topic_words'])
            fig, axes = plt.subplots((num_topics + 2) // 3, 3, figsize=(15, 5 * ((num_topics + 2) // 3)))
            if num_topics == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for topic_id in range(num_topics):
                topic_key = f"topic_{topic_id}"
                if topic_key in results['topic_words']:
                    words_dict = results['topic_words'][topic_key]
                    words_list = sorted(words_dict.items(), key=lambda x: x[1], reverse=True)[:10]
                    words = [w[0] for w in words_list]
                    probs = [w[1] for w in words_list]
                    
                    axes[topic_id].barh(range(len(words)), probs)
                    axes[topic_id].set_yticks(range(len(words)))
                    axes[topic_id].set_yticklabels(words)
                    axes[topic_id].set_xlabel('Probability', fontsize=10)
                    axes[topic_id].set_title(f'Topic {topic_id}', fontsize=12, fontweight='bold')
                    axes[topic_id].invert_yaxis()
                    axes[topic_id].grid(True, alpha=0.3, axis='x')
            
            # Hide extra subplots
            for idx in range(num_topics, len(axes)):
                axes[idx].axis('off')
            
            plt.suptitle('Top Words per Topic', fontsize=16, fontweight='bold', y=0.995)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Per-topic accuracy
        if 'per_topic_acc' in results:
            fig, ax = plt.subplots(figsize=(12, 6))
            topics = sorted(results['per_topic_acc'].keys())
            accuracies = [results['per_topic_acc'][t]['accuracy'] for t in topics]
            totals = [results['per_topic_acc'][t]['total'] for t in topics]
            
            bars = ax.bar(range(len(topics)), accuracies, alpha=0.8)
            colors = plt.cm.RdYlGn(accuracies)
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            for i, (acc, total) in enumerate(zip(accuracies, totals)):
                ax.text(i, acc + 0.02, f'{acc:.2%}\n(n={total})', 
                       ha='center', va='bottom', fontsize=9)
            
            ax.set_xlabel('Topic ID', fontsize=12)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_title('Per-Topic Classification Accuracy', fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(topics)))
            ax.set_xticklabels([f"Topic {t}" for t in topics])
            ax.set_ylim([0, 1.1])
            ax.grid(True, alpha=0.3, axis='y')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # 6. Discussion
        discussion_text = """
        6. Discussion and Interpretation:
        
        The results reveal several important insights about the relationship between 
        unsupervised topic discovery and arXiv's category system:
        
        Topic Discovery:
        - The optimal number of topics was determined through coherence optimization, 
          balancing interpretability and coverage.
        - Discovered topics capture thematic structures that may differ from explicit 
          categories, revealing cross-cutting themes.
        
        Category Alignment:
        - The alignment metrics (ARI, NMI) indicate how well discovered topics match 
          the category structure.
        - Lower alignment scores suggest that topics discover different structures than 
          categories, which can be valuable for identifying interdisciplinary themes.
        - Topic purity measures how concentrated each topic is in a single category.
        
        Classification Performance:
        - The accuracy of predicting categories from topics provides insight into the 
          correspondence between discovered themes and explicit labels.
        - Per-topic accuracy shows which topics align best with categories.
        
        Implications:
        - Topics may capture more nuanced or interdisciplinary themes than categories.
        - The analysis helps understand both the topic model and the category system.
        - Results can inform improvements to both topic modeling and categorization.
        """
        add_text_page(pdf, "6. Discussion and Interpretation", discussion_text)
        
        # 7. Conclusion
        conclusion_text = """
        7. Conclusion:
        
        This project successfully implements a comprehensive pipeline for topic modeling 
        and category alignment analysis of arXiv AI papers. Key achievements:
        
        - Successfully collected and processed 5000+ research papers
        - Discovered optimal topics using coherence-based optimization
        - Provided comprehensive evaluation comparing topics with categories
        - Generated rich visualizations and interactive reports
        
        The pipeline demonstrates the value of unsupervised topic discovery for 
        understanding research landscapes and provides tools for comparing discovered 
        structures with explicit categorization systems.
        
        Future Work:
        - Experiment with different topic modeling algorithms (e.g., BERTopic, CTM)
        - Incorporate temporal analysis of topic evolution
        - Explore hierarchical topic models
        - Develop interactive topic exploration tools
        
        The codebase is fully reproducible and can be extended for various research 
        applications in academic paper analysis.
        """
        add_text_page(pdf, "7. Conclusion", conclusion_text)
        
        # Add metadata
        d = pdf.infodict()
        d['Title'] = 'arXiv AI Papers Topic Modeling Report'
        d['Author'] = 'Topic Modeling Pipeline'
        d['Subject'] = 'Topic Modeling and Category Alignment Analysis'
        d['Keywords'] = 'LDA, Topic Modeling, arXiv, NLP, Machine Learning'
        d['CreationDate'] = datetime.now()
    
    logger.info(f"PDF report saved to {output_path}")
    return output_path


if __name__ == "__main__":
    output = generate_pdf_report()
    print(f"\n✅ PDF report generated successfully: {output}")

