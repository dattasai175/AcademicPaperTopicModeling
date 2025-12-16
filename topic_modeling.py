"""
LDA-based topic modeling with coherence evaluation.
"""

import os
import numpy as np
import pandas as pd
from gensim.models import LdaModel, CoherenceModel
from gensim.corpora import Dictionary
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from config import (
    LDA_TOPICS_RANGE, LDA_ALPHA, LDA_BETA, LDA_PASSES, 
    LDA_ITERATIONS, LDA_RANDOM_STATE, COHERENCE_TYPE,
    PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, TOP_WORDS_PER_TOPIC
)
from utils import (
    ensure_dir, save_pickle, load_pickle, save_json, 
    print_topic_words, logger
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


def train_lda_model(corpus, dictionary, num_topics, alpha=LDA_ALPHA, 
                   beta=LDA_BETA, passes=LDA_PASSES, 
                   iterations=LDA_ITERATIONS, random_state=LDA_RANDOM_STATE):
    """
    Train an LDA model.
    
    Args:
        corpus: Gensim corpus
        dictionary: Gensim dictionary
        num_topics: Number of topics
        alpha: Document-topic prior
        beta: Topic-word prior
        passes: Number of passes through the corpus
        iterations: Number of iterations
        random_state: Random seed
        
    Returns:
        Trained LDA model
    """
    logger.info(f"Training LDA model with {num_topics} topics...")
    
    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        alpha=alpha,
        eta=beta,
        passes=passes,
        iterations=iterations,
        random_state=random_state,
        per_word_topics=True,
        chunksize=2000,  # Process documents in chunks for efficiency
        eval_every=10  # Evaluate model periodically
    )
    
    logger.info(f"LDA model trained with {num_topics} topics")
    return model


def evaluate_coherence(model, corpus, dictionary, texts, coherence_type=None):
    """
    Evaluate topic coherence.
    
    Args:
        model: Trained LDA model
        corpus: Gensim corpus
        dictionary: Gensim dictionary
        texts: Tokenized texts (list of lists)
        coherence_type: Type of coherence metric ('c_v', 'u_mass', 'c_uci', 'c_npmi')
        
    Returns:
        Coherence score
    """
    if coherence_type is None:
        coherence_type = COHERENCE_TYPE
    
    coherence_model = CoherenceModel(
        model=model,
        texts=texts,
        corpus=corpus,
        dictionary=dictionary,
        coherence=coherence_type
    )
    
    coherence_score = coherence_model.get_coherence()
    return coherence_score


def find_optimal_topics(corpus, dictionary, texts, topics_range=None, 
                       coherence_type=None):
    """
    Find optimal number of topics by evaluating coherence scores.
    
    Args:
        corpus: Gensim corpus
        dictionary: Gensim dictionary
        texts: Tokenized texts
        topics_range: List of topic numbers to try
        coherence_type: Type of coherence metric
        
    Returns:
        Dictionary with results and optimal model
    """
    if topics_range is None:
        topics_range = LDA_TOPICS_RANGE
    if coherence_type is None:
        coherence_type = COHERENCE_TYPE
    
    logger.info(f"Finding optimal number of topics from range: {topics_range}")
    logger.info(f"Using coherence metric: {coherence_type}")
    
    coherence_scores = []
    models = []
    
    for num_topics in tqdm(topics_range, desc="Evaluating topics"):
        # Train model
        model = train_lda_model(corpus, dictionary, num_topics)
        models.append(model)
        
        # Evaluate coherence
        coherence = evaluate_coherence(model, corpus, dictionary, texts, coherence_type)
        coherence_scores.append(coherence)
        
        logger.info(f"Topics: {num_topics}, Coherence ({coherence_type}): {coherence:.4f}")
    
    # Find optimal number of topics
    optimal_idx = np.argmax(coherence_scores)
    optimal_topics = topics_range[optimal_idx]
    optimal_model = models[optimal_idx]
    optimal_coherence = coherence_scores[optimal_idx]
    
    logger.info(f"\nOptimal number of topics: {optimal_topics}")
    logger.info(f"Optimal coherence score: {optimal_coherence:.4f}")
    
    results = {
        'topics_range': topics_range,
        'coherence_scores': coherence_scores,
        'optimal_topics': optimal_topics,
        'optimal_coherence': optimal_coherence,
        'optimal_model': optimal_model,
        'all_models': models,
        'coherence_type': coherence_type
    }
    
    return results


def get_topic_words(model, num_words=None):
    """
    Extract top words for each topic.
    
    Args:
        model: Trained LDA model
        num_words: Number of top words per topic
        
    Returns:
        List of tuples (word, probability) for each topic
    """
    if num_words is None:
        num_words = TOP_WORDS_PER_TOPIC
    
    topic_words = []
    for topic_id in range(model.num_topics):
        words = model.show_topic(topic_id, topn=num_words)
        topic_words.append(words)
    return topic_words


def get_document_topics(model, corpus):
    """
    Get topic distribution for each document.
    
    Args:
        model: Trained LDA model
        corpus: Gensim corpus
        
    Returns:
        Array of topic distributions (n_documents x n_topics)
    """
    doc_topics = []
    for doc in corpus:
        topic_dist = model.get_document_topics(doc, minimum_probability=0.0)
        doc_topics.append([prob for _, prob in topic_dist])
    
    return np.array(doc_topics)


def visualize_coherence_scores(results, output_path=None):
    """
    Visualize coherence scores across different numbers of topics.
    
    Args:
        results: Results dictionary from find_optimal_topics
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(results['topics_range'], results['coherence_scores'], 
             marker='o', linewidth=2, markersize=8)
    plt.axvline(x=results['optimal_topics'], color='r', 
                linestyle='--', label=f'Optimal: {results["optimal_topics"]}')
    plt.xlabel('Number of Topics', fontsize=12)
    plt.ylabel(f'Coherence Score ({results.get("coherence_type", "c_v")})', fontsize=12)
    plt.title('Topic Coherence vs Number of Topics', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if output_path:
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved coherence plot to {output_path}")
    
    plt.close()


def create_lda_visualization(model, corpus, dictionary, output_path=None):
    """
    Create interactive LDA visualization using pyLDAvis.
    
    Args:
        model: Trained LDA model
        corpus: Gensim corpus
        dictionary: Gensim dictionary
        output_path: Path to save HTML visualization
    """
    logger.info("Creating LDA visualization...")
    
    try:
        vis = gensimvis.prepare(model, corpus, dictionary, sort_topics=False)
        
        if output_path:
            ensure_dir(os.path.dirname(output_path))
            pyLDAvis.save_html(vis, output_path)
            logger.info(f"Saved LDA visualization to {output_path}")
        
        return vis
    except Exception as e:
        logger.warning(f"Could not create LDA visualization: {e}")
        return None


def run_topic_modeling_pipeline(corpus=None, dictionary=None, texts=None, 
                               data_file=None, find_optimal=True):
    """
    Run the complete topic modeling pipeline.
    
    Args:
        corpus: Gensim corpus (if None, will load from file)
        dictionary: Gensim dictionary (if None, will load from file)
        texts: Tokenized texts (if None, will load from file)
        data_file: Path to processed data CSV
        find_optimal: Whether to find optimal number of topics
        
    Returns:
        Dictionary with results
    """
    # Load data if not provided
    if corpus is None or dictionary is None:
        logger.info("Loading preprocessed data...")
        corpus = load_pickle(f"{PROCESSED_DATA_DIR}/processed_corpus.pkl")
        dictionary = load_pickle(f"{PROCESSED_DATA_DIR}/processed_dictionary.pkl")
    
    if texts is None:
        if data_file is None:
            data_file = f"{PROCESSED_DATA_DIR}/processed_data.csv"
        df = pd.read_csv(data_file)
        texts = [text.split() for text in df['processed_text'].tolist()]
    
    ensure_dir(MODELS_DIR)
    ensure_dir(RESULTS_DIR)
    
    # Find optimal number of topics or use default
    if find_optimal:
        results = find_optimal_topics(corpus, dictionary, texts)
        model = results['optimal_model']
        num_topics = results['optimal_topics']
        
        # Visualize coherence scores
        visualize_coherence_scores(
            results, 
            f"{RESULTS_DIR}/coherence_scores.png"
        )
        
        # Save coherence results
        save_json({
            'topics_range': results['topics_range'],
            'coherence_scores': results['coherence_scores'],
            'optimal_topics': results['optimal_topics'],
            'optimal_coherence': float(results['optimal_coherence']),
            'coherence_type': results['coherence_type']
        }, f"{RESULTS_DIR}/coherence_results.json")
    else:
        # Use default number of topics
        num_topics = LDA_TOPICS_RANGE[len(LDA_TOPICS_RANGE)//2]  # Default to middle value
        logger.info(f"Using default number of topics: {num_topics}")
        model = train_lda_model(corpus, dictionary, num_topics)
        results = {'optimal_model': model, 'optimal_topics': num_topics}
    
    # Get topic words
    topic_words = get_topic_words(model, num_words=TOP_WORDS_PER_TOPIC)
    
    # Print topic words
    logger.info("\n=== Top Words per Topic ===")
    print_topic_words(topic_words, top_n=10)
    
    # Get document topics
    doc_topics = get_document_topics(model, corpus)
    
    # Save model
    model_path = f"{MODELS_DIR}/lda_model_{num_topics}topics"
    model.save(model_path)
    logger.info(f"Saved LDA model to {model_path}")
    
    # Save dictionary
    dictionary_path = f"{MODELS_DIR}/lda_dictionary_{num_topics}topics"
    dictionary.save(dictionary_path)
    logger.info(f"Saved dictionary to {dictionary_path}")
    
    # Save topic words
    save_json(
        {f"topic_{i}": {word: float(prob) for word, prob in words} 
         for i, words in enumerate(topic_words)},
        f"{RESULTS_DIR}/topic_words.json"
    )
    
    # Save document topics
    save_pickle(doc_topics, f"{RESULTS_DIR}/document_topics.pkl")
    
    # Export document-topic distribution as CSV
    try:
        if data_file is None:
            data_file = f"{PROCESSED_DATA_DIR}/processed_data.csv"
        df_data = pd.read_csv(data_file)
        
        # Create DataFrame with document IDs and topic probabilities
        doc_topic_df = pd.DataFrame(
            doc_topics,
            columns=[f'topic_{i}' for i in range(num_topics)]
        )
        doc_topic_df.insert(0, 'id', df_data['id'].values[:len(doc_topics)])
        doc_topic_df.insert(1, 'dominant_topic', doc_topic_df.iloc[:, 1:].idxmax(axis=1).str.replace('topic_', '').astype(int))
        
        doc_topic_csv_path = f"{RESULTS_DIR}/document_topic_distribution.csv"
        doc_topic_df.to_csv(doc_topic_csv_path, index=False)
        logger.info(f"Saved document-topic distribution CSV to {doc_topic_csv_path}")
    except Exception as e:
        logger.warning(f"Could not export document-topic distribution CSV: {e}")
    
    # Export topic-word distribution as CSV
    try:
        topic_word_rows = []
        for topic_id, words in enumerate(topic_words):
            for word, prob in words:
                topic_word_rows.append({
                    'topic_id': topic_id,
                    'word': word,
                    'probability': float(prob)
                })
        
        topic_word_df = pd.DataFrame(topic_word_rows)
        topic_word_csv_path = f"{RESULTS_DIR}/topic_word_distribution.csv"
        topic_word_df.to_csv(topic_word_csv_path, index=False)
        logger.info(f"Saved topic-word distribution CSV to {topic_word_csv_path}")
    except Exception as e:
        logger.warning(f"Could not export topic-word distribution CSV: {e}")
    
    # Create visualization
    create_lda_visualization(
        model, corpus, dictionary,
        f"{RESULTS_DIR}/lda_visualization.html"
    )
    
    logger.info("Topic modeling pipeline complete!")
    
    return {
        'model': model,
        'dictionary': dictionary,
        'num_topics': num_topics,
        'topic_words': topic_words,
        'document_topics': doc_topics,
        'corpus': corpus,
        'results': results
    }


if __name__ == "__main__":
    result = run_topic_modeling_pipeline()
    print(f"\nTopic modeling complete!")
    print(f"Optimal number of topics: {result['num_topics']}")
    print(f"Document topics shape: {result['document_topics'].shape}")
