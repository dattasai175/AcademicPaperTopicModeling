"""
Script to run NMF and BERTopic topic modeling methods individually.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from utils import logger, load_pickle
from config import PROCESSED_DATA_DIR, RESULTS_DIR
from topic_modeling_nmf import run_nmf_pipeline

try:
    from topic_modeling_bertopic import run_bertopic_pipeline, BERTOPIC_AVAILABLE
except ImportError:
    BERTOPIC_AVAILABLE = False

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("RUNNING NMF AND BERTOPIC")
    logger.info("=" * 70)
    
    df_processed = pd.read_csv(f"{PROCESSED_DATA_DIR}/processed_data.csv")
    
    # Run NMF
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING NMF")
    logger.info("=" * 70)
    try:
        tfidf_vectorizer = load_pickle(f"{PROCESSED_DATA_DIR}/processed_tfidf_vectorizer.pkl")
        feature_names = tfidf_vectorizer.get_feature_names_out()
        texts_list = df_processed['processed_text'].tolist()
        tfidf_matrix = tfidf_vectorizer.transform(texts_list)
        
        nmf_result = run_nmf_pipeline(
            tfidf_matrix=tfidf_matrix,
            feature_names=feature_names,
            texts=texts_list,
            find_optimal=True
        )
        logger.info(f"✅ NMF complete: {nmf_result['num_topics']} topics")
    except Exception as e:
        logger.error(f"❌ NMF failed: {e}", exc_info=True)
    
    # Run BERTopic
    if BERTOPIC_AVAILABLE:
        logger.info("\n" + "=" * 70)
        logger.info("RUNNING BERTOPIC")
        logger.info("=" * 70)
        try:
            df_raw = pd.read_csv(f"{PROCESSED_DATA_DIR}/../raw/arxiv_abstracts.csv")
            texts_original = df_raw['abstract'].tolist()[:len(df_processed)]
            
            bertopic_result = run_bertopic_pipeline(
                texts=texts_original,
                find_optimal=True
            )
            logger.info(f"✅ BERTopic complete: {bertopic_result['num_topics']} topics")
        except Exception as e:
            logger.error(f"❌ BERTopic failed: {e}", exc_info=True)
    else:
        logger.warning("BERTopic not available")
    
    logger.info("\n✅ Done!")

