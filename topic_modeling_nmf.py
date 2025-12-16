"""
NMF (Non-negative Matrix Factorization) topic modeling implementation.
"""

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from config import (
    NMF_TOPICS_RANGE, NMF_ALPHA, NMF_L1_RATIO, NMF_MAX_ITER, NMF_RANDOM_STATE,
    PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, TOP_WORDS_PER_TOPIC,
    MAX_FEATURES_TFIDF, MIN_DF, MAX_DF
)
from utils import (
    ensure_dir, save_pickle, load_pickle, save_json, logger
)

# Set style for plots
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")


def train_nmf_model(tfidf_matrix, num_topics, alpha=NMF_ALPHA, 
                    l1_ratio=NMF_L1_RATIO, max_iter=NMF_MAX_ITER, 
                    random_state=NMF_RANDOM_STATE):
    """
    Train an NMF model.
    
    Args:
        tfidf_matrix: TF-IDF matrix (scipy sparse matrix)
        num_topics: Number of topics
        alpha: Regularization parameter
        l1_ratio: L1/L2 regularization ratio
        max_iter: Maximum iterations
        random_state: Random seed
        
    Returns:
        Trained NMF model
    """
    logger.info(f"Training NMF model with {num_topics} topics...")
    
    # Convert sparse matrix to dense if needed for better initialization
    # Use random initialization which is more stable
    model = NMF(
        n_components=num_topics,
        alpha_W=alpha,  # Regularization for W matrix
        alpha_H=alpha,  # Regularization for H matrix
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        random_state=random_state,
        init='random',  # Use random initialization (more stable)
        solver='mu',  # Multiplicative update solver (works with sparse matrices)
        beta_loss='frobenius'  # Frobenius norm (standard)
    )
    
    # Fit the model
    with np.errstate(all='ignore'):  # Suppress warnings during fitting
        model.fit(tfidf_matrix)
    
    # Check if model trained successfully
    if np.all(model.components_ == 0):
        logger.warning("NMF components are all zeros, retrying with different parameters...")
        model = NMF(
            n_components=num_topics,
            alpha_W=0.0,  # No regularization
            alpha_H=0.0,
            l1_ratio=0.0,
            max_iter=max_iter * 2,  # More iterations
            random_state=random_state + 1,  # Different seed
            init='random',
            solver='mu',
            beta_loss='frobenius'
        )
        model.fit(tfidf_matrix)
    
    logger.info(f"NMF model trained with {num_topics} topics")
    return model


def find_optimal_nmf_topics(tfidf_matrix, topics_range=None, feature_names=None):
    """
    Find optimal number of topics for NMF using reconstruction error.
    
    Args:
        tfidf_matrix: TF-IDF matrix
        topics_range: List of topic numbers to try
        feature_names: Feature names from vectorizer
        
    Returns:
        Dictionary with results
    """
    if topics_range is None:
        topics_range = NMF_TOPICS_RANGE
    
    logger.info(f"Finding optimal NMF topics in range {topics_range}...")
    
    reconstruction_errors = []
    models = []
    
    for num_topics in tqdm(topics_range, desc="Evaluating NMF topics"):
        model = train_nmf_model(tfidf_matrix, num_topics)
        models.append(model)
        
        # Calculate reconstruction error (lower is better)
        W = model.transform(tfidf_matrix)
        H = model.components_
        reconstructed = np.dot(W, H)
        error = np.mean((tfidf_matrix.toarray() - reconstructed) ** 2)
        reconstruction_errors.append(error)
    
    # Find optimal (lowest reconstruction error)
    optimal_idx = np.argmin(reconstruction_errors)
    optimal_topics = topics_range[optimal_idx]
    optimal_model = models[optimal_idx]
    optimal_error = reconstruction_errors[optimal_idx]
    
    logger.info(f"Optimal NMF topics: {optimal_topics} (reconstruction error: {optimal_error:.4f})")
    
    return {
        'topics_range': topics_range,
        'reconstruction_errors': reconstruction_errors,
        'optimal_topics': optimal_topics,
        'optimal_model': optimal_model,
        'optimal_error': optimal_error,
        'models': models
    }


def get_nmf_topic_words(model, feature_names, num_words=TOP_WORDS_PER_TOPIC):
    """
    Get top words for each NMF topic.
    
    Args:
        model: Trained NMF model
        feature_names: Feature names from vectorizer
        num_words: Number of top words per topic
        
    Returns:
        List of topic words
    """
    topic_words = []
    
    for topic_idx in range(model.n_components):
        # Get top words for this topic (components_ contains topic-word weights)
        top_indices = model.components_[topic_idx].argsort()[-num_words:][::-1]
        top_words = [(feature_names[idx], float(model.components_[topic_idx][idx])) 
                     for idx in top_indices]
        topic_words.append(top_words)
    
    return topic_words


def get_nmf_document_topics(model, tfidf_matrix):
    """
    Get document-topic distributions from NMF model.
    
    Args:
        model: Trained NMF model
        tfidf_matrix: TF-IDF matrix
        
    Returns:
        Document-topic distribution matrix
    """
    # Transform documents to topic space
    # Use the same matrix that was used for training
    with np.errstate(all='ignore'):  # Suppress warnings
        doc_topics = model.transform(tfidf_matrix)
    
    # Ensure no negative values (shouldn't happen with NMF, but safety check)
    doc_topics = np.maximum(doc_topics, 0)
    
    # Normalize to get probabilities (sum to 1)
    row_sums = doc_topics.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    doc_topics = doc_topics / row_sums
    
    return doc_topics


def run_nmf_pipeline(tfidf_matrix=None, feature_names=None, texts=None, 
                    data_file=None, find_optimal=True):
    """
    Run complete NMF topic modeling pipeline.
    
    Args:
        tfidf_matrix: TF-IDF matrix (if None, will load)
        feature_names: Feature names (if None, will load)
        texts: Raw texts (for coherence if needed)
        data_file: Path to processed data CSV
        find_optimal: Whether to find optimal number of topics
        
    Returns:
        Dictionary with results
    """
    # Load TF-IDF matrix if not provided
    if tfidf_matrix is None:
        logger.info("Loading TF-IDF matrix...")
        from preprocessing import load_pickle
        tfidf_vectorizer = load_pickle(f"{PROCESSED_DATA_DIR}/processed_tfidf_vectorizer.pkl")
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        # Load processed texts
        if data_file is None:
            data_file = f"{PROCESSED_DATA_DIR}/processed_data.csv"
        df = pd.read_csv(data_file)
        texts = df['processed_text'].tolist()
        
        # Transform texts to TF-IDF
        tfidf_matrix = tfidf_vectorizer.transform(texts)
    
    ensure_dir(MODELS_DIR)
    ensure_dir(RESULTS_DIR)
    
    # Find optimal number of topics or use default
    if find_optimal:
        results = find_optimal_nmf_topics(tfidf_matrix, feature_names=feature_names)
        model = results['optimal_model']
        num_topics = results['optimal_topics']
        
        # Visualize reconstruction errors
        visualize_reconstruction_errors(
            results,
            f"{RESULTS_DIR}/nmf_reconstruction_errors.png"
        )
        
        # Save results
        save_json({
            'topics_range': results['topics_range'],
            'reconstruction_errors': [float(e) for e in results['reconstruction_errors']],
            'optimal_topics': results['optimal_topics'],
            'optimal_error': float(results['optimal_error'])
        }, f"{RESULTS_DIR}/nmf_results.json")
    else:
        num_topics = NMF_TOPICS_RANGE[len(NMF_TOPICS_RANGE) // 2]
        model = train_nmf_model(tfidf_matrix, num_topics)
    
    # Get topic words
    topic_words = get_nmf_topic_words(model, feature_names)
    
    # Get document topics
    doc_topics = get_nmf_document_topics(model, tfidf_matrix)
    
    # Save model
    model_path = f"{MODELS_DIR}/nmf_model_{num_topics}topics"
    save_pickle(model, f"{model_path}.pkl")
    logger.info(f"Saved NMF model to {model_path}.pkl")
    
    # Save topic words
    save_json(
        {f"topic_{i}": {word: float(prob) for word, prob in words} 
         for i, words in enumerate(topic_words)},
        f"{RESULTS_DIR}/nmf_topic_words.json"
    )
    
    # Save document topics
    save_pickle(doc_topics, f"{RESULTS_DIR}/nmf_document_topics.pkl")
    
    # Save document-topic distribution as CSV
    df_original = pd.read_csv(f"{PROCESSED_DATA_DIR}/processed_data.csv")
    save_document_topic_distribution_csv(
        doc_topics, df_original, f"{RESULTS_DIR}/nmf_document_topic_distribution.csv"
    )
    
    logger.info("NMF topic modeling pipeline complete!")
    
    return {
        'model': model,
        'num_topics': num_topics,
        'topic_words': topic_words,
        'document_topics': doc_topics,
        'tfidf_matrix': tfidf_matrix,
        'feature_names': feature_names,
        'results': results if find_optimal else None
    }


def visualize_reconstruction_errors(results, output_path=None):
    """Visualize reconstruction errors for different numbers of topics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results['topics_range'], results['reconstruction_errors'], 
           marker='o', linewidth=2, markersize=8)
    ax.axvline(x=results['optimal_topics'], color='r', 
               linestyle='--', label=f"Optimal: {results['optimal_topics']}")
    ax.set_xlabel('Number of Topics', fontsize=12)
    ax.set_ylabel('Reconstruction Error', fontsize=12)
    ax.set_title('NMF Reconstruction Error vs Number of Topics', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    
    if output_path:
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved reconstruction error plot to {output_path}")
    
    plt.close()


def save_document_topic_distribution_csv(document_topics, df_original, output_path):
    """Save document-topic distribution to a CSV file."""
    logger.info(f"Saving NMF document-topic distribution to {output_path}")
    num_topics = document_topics.shape[1]
    topic_cols = [f'topic_{i}' for i in range(num_topics)]
    
    doc_topic_df = pd.DataFrame(document_topics, columns=topic_cols)
    doc_topic_df['id'] = df_original['id'].reset_index(drop=True)
    doc_topic_df['dominant_topic'] = np.argmax(document_topics, axis=1)
    
    cols = ['id', 'dominant_topic'] + topic_cols
    doc_topic_df = doc_topic_df[cols]
    
    doc_topic_df.to_csv(output_path, index=False)
    logger.info(f"Created CSV with {len(doc_topic_df)} rows and {num_topics} topics")
    return doc_topic_df


if __name__ == "__main__":
    result = run_nmf_pipeline(find_optimal=True)
    print(f"\nNMF topic modeling complete!")
    print(f"Optimal topics: {result['num_topics']}")

