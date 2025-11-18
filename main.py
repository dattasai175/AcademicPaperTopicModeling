"""
Main pipeline orchestrator for academic paper topic modeling and classification.
"""

import logging
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import *
from utils import logger, ensure_dir
from data_collection import collect_arxiv_abstracts
from preprocessing import run_preprocessing_pipeline
from topic_modeling import run_topic_modeling_pipeline
from classification import run_classification_pipeline
from evaluation import generate_evaluation_report

def main():
    """Run the complete pipeline."""
    
    logger.info("=" * 60)
    logger.info("Academic Paper Topic Modeling and Classification Pipeline")
    logger.info("=" * 60)
    
    # Ensure directories exist
    ensure_dir(DATA_DIR)
    ensure_dir(RAW_DATA_DIR)
    ensure_dir(PROCESSED_DATA_DIR)
    ensure_dir(MODELS_DIR)
    ensure_dir(RESULTS_DIR)
    
    try:
        # Step 1: Data Collection
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: Data Collection")
        logger.info("=" * 60)
        
        # Check if data already exists
        data_file = f"{RAW_DATA_DIR}/arxiv_abstracts.csv"
        if Path(data_file).exists():
            logger.info(f"Data file already exists: {data_file}")
            logger.info("Skipping data collection. Delete the file to re-collect.")
        else:
            df = collect_arxiv_abstracts()
            logger.info(f"Collected {len(df)} abstracts")
        
        # Step 2: Preprocessing
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Preprocessing")
        logger.info("=" * 60)
        
        preprocessing_result = run_preprocessing_pipeline()
        logger.info(f"Preprocessed {len(preprocessing_result['df'])} abstracts")
        
        # Step 3: Topic Modeling
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Topic Modeling")
        logger.info("=" * 60)
        
        topic_modeling_result = run_topic_modeling_pipeline(
            corpus=preprocessing_result['corpus'],
            dictionary=preprocessing_result['dictionary'],
            find_optimal=True
        )
        logger.info(f"Optimal number of topics: {topic_modeling_result['num_topics']}")
        
        # Step 4: Classification
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: Classification")
        logger.info("=" * 60)
        
        classification_result = run_classification_pipeline(
            tfidf_matrix=preprocessing_result['tfidf_matrix'],
            topic_words=topic_modeling_result['topic_words'],
            document_topics=topic_modeling_result['document_topics']
        )
        logger.info(f"Classification accuracy: {classification_result['metrics']['accuracy']:.4f}")
        
        # Step 5: Evaluation
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: Evaluation")
        logger.info("=" * 60)
        
        evaluation_result = generate_evaluation_report()
        logger.info("Evaluation complete!")
        
        # Final Summary
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Number of abstracts processed: {len(preprocessing_result['df'])}")
        logger.info(f"Optimal number of topics: {topic_modeling_result['num_topics']}")
        logger.info(f"Classification accuracy: {classification_result['metrics']['accuracy']:.4f}")
        logger.info(f"Classification F1-score: {classification_result['metrics']['f1_weighted']:.4f}")
        logger.info(f"Topic-Classification agreement: {evaluation_result['comparison_results']['overall_agreement']:.4f}")
        logger.info("\nResults saved in:")
        logger.info(f"  - Models: {MODELS_DIR}/")
        logger.info(f"  - Results: {RESULTS_DIR}/")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

