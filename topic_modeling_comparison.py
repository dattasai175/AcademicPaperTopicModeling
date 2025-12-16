"""
Unified topic modeling pipeline that runs LDA, NMF, and BERTopic and compares results.
"""

import os
import numpy as np
import pandas as pd
import logging
from config import (
    USE_LDA, USE_NMF, USE_BERTOPIC,
    PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR
)
from utils import logger, ensure_dir, load_pickle
from topic_modeling import run_topic_modeling_pipeline
from topic_modeling_nmf import run_nmf_pipeline

try:
    from topic_modeling_bertopic import run_bertopic_pipeline, BERTOPIC_AVAILABLE
except ImportError:
    BERTOPIC_AVAILABLE = False
    logger.warning("BERTopic module not available")


def run_all_topic_modeling_methods(find_optimal=True):
    """
    Run all enabled topic modeling methods (LDA, NMF, BERTopic).
    
    Args:
        find_optimal: Whether to find optimal number of topics for each method
        
    Returns:
        Dictionary with results from all methods
    """
    logger.info("=" * 70)
    logger.info("RUNNING ALL TOPIC MODELING METHODS")
    logger.info("=" * 70)
    
    results = {}
    
    # Load preprocessing results
    df_processed = pd.read_csv(f"{PROCESSED_DATA_DIR}/processed_data.csv")
    texts = [text.split() for text in df_processed['processed_text'].tolist()]
    
    # Run LDA
    if USE_LDA:
        logger.info("\n" + "=" * 70)
        logger.info("METHOD 1: LDA Topic Modeling")
        logger.info("=" * 70)
        try:
            dictionary = load_pickle(f"{PROCESSED_DATA_DIR}/processed_dictionary.pkl")
            corpus = load_pickle(f"{PROCESSED_DATA_DIR}/processed_corpus.pkl")
            
            lda_result = run_topic_modeling_pipeline(
                corpus=corpus,
                dictionary=dictionary,
                texts=texts,
                find_optimal=find_optimal
            )
            results['lda'] = lda_result
            logger.info(f"✅ LDA complete: {lda_result['num_topics']} topics")
        except Exception as e:
            logger.error(f"❌ LDA failed: {e}", exc_info=True)
            results['lda'] = None
    
    # Run NMF
    if USE_NMF:
        logger.info("\n" + "=" * 70)
        logger.info("METHOD 2: NMF Topic Modeling")
        logger.info("=" * 70)
        try:
            # Check if TF-IDF vectorizer exists
            tfidf_vectorizer_path = f"{PROCESSED_DATA_DIR}/processed_tfidf_vectorizer.pkl"
            if not os.path.exists(tfidf_vectorizer_path):
                logger.warning(f"TF-IDF vectorizer not found at {tfidf_vectorizer_path}")
                logger.warning("NMF requires TF-IDF. Make sure USE_TFIDF_FOR_LDA=True in config.py")
                logger.warning("Re-run preprocessing to generate TF-IDF vectorizer.")
                results['nmf'] = None
            else:
                # Load TF-IDF vectorizer
                tfidf_vectorizer = load_pickle(tfidf_vectorizer_path)
                feature_names = tfidf_vectorizer.get_feature_names_out()
                texts_list = df_processed['processed_text'].tolist()
                tfidf_matrix = tfidf_vectorizer.transform(texts_list)
                
                nmf_result = run_nmf_pipeline(
                    tfidf_matrix=tfidf_matrix,
                    feature_names=feature_names,
                    texts=texts_list,
                    find_optimal=find_optimal
                )
                results['nmf'] = nmf_result
                logger.info(f"✅ NMF complete: {nmf_result['num_topics']} topics")
        except Exception as e:
            logger.error(f"❌ NMF failed: {e}", exc_info=True)
            results['nmf'] = None
    
    # Run BERTopic
    if USE_BERTOPIC:
        if not BERTOPIC_AVAILABLE:
            logger.warning("BERTopic requested but not available. Skipping.")
            logger.warning("Install with: pip install bertopic sentence-transformers")
            results['bertopic'] = None
        else:
            logger.info("\n" + "=" * 70)
            logger.info("METHOD 3: BERTopic Topic Modeling")
            logger.info("=" * 70)
            try:
                # Use original abstracts for BERTopic (better semantic understanding)
                df_raw = pd.read_csv(f"{PROCESSED_DATA_DIR}/../raw/arxiv_abstracts.csv")
                texts_original = df_raw['abstract'].tolist()[:len(df_processed)]
                
                bertopic_result = run_bertopic_pipeline(
                    texts=texts_original,
                    find_optimal=find_optimal
                )
                results['bertopic'] = bertopic_result
                logger.info(f"✅ BERTopic complete: {bertopic_result['num_topics']} topics")
            except Exception as e:
                logger.error(f"❌ BERTopic failed: {e}", exc_info=True)
                results['bertopic'] = None
    else:
        results['bertopic'] = None
    
    logger.info("\n" + "=" * 70)
    logger.info("ALL TOPIC MODELING METHODS COMPLETE")
    logger.info("=" * 70)
    
    return results


if __name__ == "__main__":
    results = run_all_topic_modeling_methods(find_optimal=True)
    print("\n✅ All topic modeling methods completed!")
    for method, result in results.items():
        if result:
            print(f"  {method.upper()}: {result['num_topics']} topics")

