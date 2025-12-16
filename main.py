"""
Main pipeline orchestrator for arXiv AI Papers Topic Modeling & Category Alignment Analysis.
"""

import logging
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import *
from config import USE_TFIDF_FOR_LDA, USE_LDA, USE_NMF, USE_BERTOPIC
from utils import logger, ensure_dir
from data_collection import collect_arxiv_abstracts
from preprocessing import run_preprocessing_pipeline
from topic_modeling import run_topic_modeling_pipeline
from topic_modeling_comparison import run_all_topic_modeling_methods
from evaluation import generate_analysis_report
from evaluation_comparison import compare_all_methods


def main():
    """Run the complete pipeline."""
    
    logger.info("=" * 60)
    logger.info("arXiv AI Papers Topic Modeling & Category Alignment Analysis")
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
        logger.info(f"Dictionary size: {len(preprocessing_result['dictionary'])}")
        if USE_TFIDF_FOR_LDA:
            logger.info("✅ Using TF-IDF weighted corpus for LDA (better topic quality)")
        else:
            logger.info("Using BOW corpus for LDA")
        
        # Step 3: Topic Modeling (All Methods)
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Topic Modeling (LDA, NMF, BERTopic)")
        logger.info("=" * 60)
        
        # Run all topic modeling methods
        topic_modeling_results = run_all_topic_modeling_methods(find_optimal=True)
        
        # Log results for each method
        for method_name, result in topic_modeling_results.items():
            if result:
                logger.info(f"{method_name.upper()}: {result['num_topics']} topics")
        
        # Step 4: Analysis & Evaluation (Comparison)
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: Analysis & Evaluation (Compare All Methods)")
        logger.info("=" * 60)
        
        # Check which methods completed successfully
        completed_methods = [name for name, result in topic_modeling_results.items() if result is not None]
        logger.info(f"Methods completed: {', '.join([m.upper() for m in completed_methods])}")
        
        if len(completed_methods) == 0:
            logger.warning("No topic modeling methods completed successfully!")
        else:
            # Compare all methods that completed
            comparison_results = compare_all_methods()
            
            # Also run individual LDA analysis for backward compatibility
            if USE_LDA and topic_modeling_results.get('lda'):
                logger.info("\nGenerating detailed LDA analysis report...")
                analysis_result = generate_analysis_report()
                logger.info("LDA analysis complete!")
        
        logger.info("\n" + "=" * 60)
        logger.info("METHOD COMPARISON SUMMARY")
        logger.info("=" * 60)
        
        from utils import load_json
        import os
        
        summary_file = f"{RESULTS_DIR}/method_comparison_summary.json"
        if os.path.exists(summary_file):
            summary = load_json(summary_file)
            
            if len(summary.get('methods', {})) > 0:
                logger.info("\nClassification Accuracy (All Categories):")
                for method, metrics in summary['methods'].items():
                    logger.info(f"  {method.upper()}: {metrics['accuracy']:.4f} "
                               f"(F1-weighted: {metrics['f1_weighted']:.4f})")
                
                if summary.get('best_method'):
                    logger.info(f"\n🏆 Best Method: {summary['best_method'].upper()} "
                               f"(Accuracy: {summary['best_accuracy']:.4f})")
            else:
                logger.warning("No methods available for comparison.")
        else:
            logger.warning("Comparison summary not found. Some methods may not have completed.")
        
        # Final Summary
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Number of abstracts processed: {len(preprocessing_result['df'])}")
        
        # Summary from comparison
        if comparison_results:
            logger.info("\nFinal Method Comparison:")
            for method_name, result in comparison_results.items():
                if result:
                    metrics = result['classification_metrics']
                    logger.info(f"  {method_name.upper()}: "
                              f"Accuracy={metrics['accuracy']:.4f}, "
                              f"F1-weighted={metrics['f1_weighted']:.4f}")
        
        logger.info("\nResults saved in:")
        logger.info(f"  - Models: {MODELS_DIR}/")
        logger.info(f"  - Results: {RESULTS_DIR}/")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
