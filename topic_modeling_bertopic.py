"""
BERTopic topic modeling implementation.
"""

import os
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from config import (
    BERTOPIC_TOPICS_RANGE, BERTOPIC_MODEL, BERTOPIC_MIN_TOPIC_SIZE, BERTOPIC_RANDOM_STATE,
    PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, TOP_WORDS_PER_TOPIC
)
from utils import (
    ensure_dir, save_pickle, load_pickle, save_json, logger
)

try:
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from sentence_transformers import SentenceTransformer
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    logger.warning("BERTopic not available. Install with: pip install bertopic sentence-transformers")

# Set style for plots
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")


def train_bertopic_model(texts, num_topics=None, model_name=BERTOPIC_MODEL,
                        min_topic_size=BERTOPIC_MIN_TOPIC_SIZE, 
                        random_state=BERTOPIC_RANDOM_STATE):
    """
    Train a BERTopic model.
    
    Args:
        texts: List of text documents
        num_topics: Number of topics (if None, auto-detect)
        model_name: Sentence transformer model name
        min_topic_size: Minimum documents per topic
        random_state: Random seed
        
    Returns:
        Trained BERTopic model
    """
    if not BERTOPIC_AVAILABLE:
        raise ImportError("BERTopic is not installed. Install with: pip install bertopic sentence-transformers")
    
    logger.info(f"Training BERTopic model...")
    
    # Load sentence transformer model
    sentence_model = SentenceTransformer(model_name)
    
    # Create BERTopic model
    if num_topics is not None:
        # Use nr_topics parameter to control number of topics
        topic_model = BERTopic(
            embedding_model=sentence_model,
            nr_topics=num_topics,
            min_topic_size=min_topic_size,
            calculate_probabilities=True
        )
    else:
        # Auto-detect topics
        topic_model = BERTopic(
            embedding_model=sentence_model,
            min_topic_size=min_topic_size,
            calculate_probabilities=True
        )
    
    # Fit the model
    topics, probs = topic_model.fit_transform(texts)
    
    logger.info(f"BERTopic model trained with {len(set(topics))} topics")
    return topic_model, topics, probs


def find_optimal_bertopic_topics(texts, topics_range=None, model_name=BERTOPIC_MODEL):
    """
    Find optimal number of topics for BERTopic.
    Note: BERTopic uses HDBSCAN clustering, so exact topic count may vary.
    For simplicity, we'll train with a single target number and let it auto-detect.
    
    Args:
        texts: List of text documents
        topics_range: List of target topic numbers to try (not used, BERTopic auto-detects)
        model_name: Sentence transformer model name
        
    Returns:
        Dictionary with results
    """
    if topics_range is None:
        topics_range = BERTOPIC_TOPICS_RANGE
    
    logger.info(f"Training BERTopic (will auto-detect optimal topics)...")
    logger.info("Note: BERTopic uses HDBSCAN which auto-detects topics")
    
    # BERTopic auto-detects topics, so we train once
    # Use middle value from range as target, but it will auto-adjust
    target_topics = topics_range[len(topics_range) // 2]
    
    try:
        model, topics, probs = train_bertopic_model(
            texts, num_topics=None, model_name=model_name  # None = auto-detect
        )
        actual_topics = len(set(topics)) - (1 if -1 in topics else 0)  # Exclude outliers
        
        logger.info(f"BERTopic detected {actual_topics} topics")
        
        return {
            'topics_range': topics_range,
            'actual_topics': [actual_topics],
            'coherence_scores': [actual_topics],
            'optimal_topics': actual_topics,
            'optimal_model': model,
            'models': [model]
        }
    except Exception as e:
        logger.error(f"Failed to train BERTopic: {e}", exc_info=True)
        return {
            'topics_range': topics_range,
            'actual_topics': [0],
            'coherence_scores': [0],
            'optimal_topics': target_topics,
            'optimal_model': None,
            'models': [None]
        }


def get_bertopic_topic_words(model, num_words=TOP_WORDS_PER_TOPIC):
    """
    Get top words for each BERTopic topic.
    
    Args:
        model: Trained BERTopic model
        num_words: Number of top words per topic
        
    Returns:
        List of topic words
    """
    topic_words = []
    
    # Get topic info
    topic_info = model.get_topic_info()
    
    for topic_id in sorted([t for t in topic_info['Topic'].tolist() if t != -1]):
        words = model.get_topic(topic_id)
        if words:
            # words is list of (word, score) tuples
            top_words = words[:num_words]
            topic_words.append(top_words)
        else:
            topic_words.append([])
    
    return topic_words


def get_bertopic_document_topics(model, texts):
    """
    Get document-topic distributions from BERTopic model.
    
    Args:
        model: Trained BERTopic model
        texts: List of text documents
        
    Returns:
        Document-topic distribution matrix
    """
    # Transform documents
    topics, probs = model.transform(texts)
    
    # Convert probabilities to numpy array
    if probs is not None:
        doc_topics = np.array(probs)
    else:
        # If probabilities not available, create one-hot encoding
        num_topics = len(set(topics)) - (1 if -1 in topics else 0)
        doc_topics = np.zeros((len(topics), num_topics))
        for i, topic in enumerate(topics):
            if topic != -1:
                doc_topics[i, topic] = 1.0
    
    return doc_topics, topics


def run_bertopic_pipeline(texts=None, data_file=None, find_optimal=True):
    """
    Run complete BERTopic topic modeling pipeline.
    
    Args:
        texts: List of text documents (if None, will load)
        data_file: Path to processed data CSV
        find_optimal: Whether to find optimal number of topics
        
    Returns:
        Dictionary with results
    """
    # Load texts if not provided
    if texts is None:
        if data_file is None:
            data_file = f"{PROCESSED_DATA_DIR}/processed_data.csv"
        df = pd.read_csv(data_file)
        texts = df['processed_text'].tolist()
        # Use original abstracts for BERTopic (better semantic understanding)
        df_original = pd.read_csv(f"{PROCESSED_DATA_DIR}/../raw/arxiv_abstracts.csv")
        texts_original = df_original['abstract'].tolist()[:len(texts)]
    else:
        texts_original = texts
        if data_file:
            df_original = pd.read_csv(data_file)
        else:
            df_original = pd.read_csv(f"{PROCESSED_DATA_DIR}/processed_data.csv")
    
    ensure_dir(MODELS_DIR)
    ensure_dir(RESULTS_DIR)
    
    # Find optimal number of topics or use default
    if find_optimal:
        results = find_optimal_bertopic_topics(texts_original)
        model = results['optimal_model']
        num_topics = results['optimal_topics']
        
        # Get topics and probabilities for optimal model
        topics, probs = model.transform(texts_original)
        
        # Save results
        save_json({
            'topics_range': results['topics_range'],
            'actual_topics': results['actual_topics'],
            'coherence_scores': results['coherence_scores'],
            'optimal_topics': results['optimal_topics']
        }, f"{RESULTS_DIR}/bertopic_results.json")
    else:
        num_topics = BERTOPIC_TOPICS_RANGE[len(BERTOPIC_TOPICS_RANGE) // 2]
        model, topics, probs = train_bertopic_model(texts_original, num_topics=num_topics)
    
    # Get document topics
    doc_topics, assigned_topics = get_bertopic_document_topics(model, texts_original)
    
    # Get topic words
    topic_words = get_bertopic_topic_words(model)
    
    # Save model
    model_path = f"{MODELS_DIR}/bertopic_model_{num_topics}topics"
    model.save(model_path)
    logger.info(f"Saved BERTopic model to {model_path}")
    
    # Save topic words
    topic_words_dict = {}
    for i, words in enumerate(topic_words):
        topic_words_dict[f"topic_{i}"] = {word: float(score) for word, score in words}
    save_json(topic_words_dict, f"{RESULTS_DIR}/bertopic_topic_words.json")
    
    # Save document topics
    save_pickle(doc_topics, f"{RESULTS_DIR}/bertopic_document_topics.pkl")
    
    # Save document-topic distribution as CSV
    save_document_topic_distribution_csv(
        doc_topics, df_original, f"{RESULTS_DIR}/bertopic_document_topic_distribution.csv"
    )
    
    logger.info("BERTopic topic modeling pipeline complete!")
    
    return {
        'model': model,
        'num_topics': num_topics,
        'topic_words': topic_words,
        'document_topics': doc_topics,
        'assigned_topics': assigned_topics,
        'texts': texts_original,
        'results': results if find_optimal else None
    }


def save_document_topic_distribution_csv(document_topics, df_original, output_path):
    """Save document-topic distribution to a CSV file."""
    logger.info(f"Saving BERTopic document-topic distribution to {output_path}")
    num_topics = document_topics.shape[1]
    topic_cols = [f'topic_{i}' for i in range(num_topics)]
    
    doc_topic_df = pd.DataFrame(document_topics, columns=topic_cols)
    doc_topic_df['id'] = df_original['id'].reset_index(drop=True)[:len(doc_topic_df)]
    doc_topic_df['dominant_topic'] = np.argmax(document_topics, axis=1)
    
    cols = ['id', 'dominant_topic'] + topic_cols
    doc_topic_df = doc_topic_df[cols]
    
    doc_topic_df.to_csv(output_path, index=False)
    logger.info(f"Created CSV with {len(doc_topic_df)} rows and {num_topics} topics")
    return doc_topic_df


if __name__ == "__main__":
    result = run_bertopic_pipeline(find_optimal=True)
    print(f"\nBERTopic topic modeling complete!")
    print(f"Optimal topics: {result['num_topics']}")

