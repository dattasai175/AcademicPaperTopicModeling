"""
TF-IDF + Linear SVM classification for AI subfield assignment.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import pickle
import logging
from config import (
    TEST_SIZE, RANDOM_STATE, CV_FOLDS, SVM_C, SVM_MAX_ITER,
    PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, AI_SUBFIELDS
)
from utils import (
    ensure_dir, save_pickle, load_pickle, save_json, 
    load_json, logger, get_dominant_topic
)

# Set random seed for reproducibility
np.random.seed(RANDOM_STATE)


def create_subfield_labels_from_topics(df, document_topics, topic_to_subfield_mapping):
    """
    Create subfield labels from topic distributions using a mapping.
    
    Args:
        df: DataFrame with abstracts
        document_topics: Array of topic distributions (n_documents x n_topics)
        topic_to_subfield_mapping: Dictionary mapping topic_id to subfield
        
    Returns:
        Array of subfield labels
    """
    labels = []
    for topic_dist in document_topics:
        dominant_topic = get_dominant_topic(topic_dist)
        subfield = topic_to_subfield_mapping.get(dominant_topic, 'Other')
        labels.append(subfield)
    return np.array(labels)


def create_manual_subfield_mapping(topic_words, num_topics):
    """
    Create a manual mapping from topics to subfields based on top words.
    This is a helper function - in practice, this would be done manually
    by inspecting the topics.
    
    Args:
        topic_words: List of topic words (from get_topic_words)
        num_topics: Number of topics
        
    Returns:
        Dictionary mapping topic_id to subfield
    """
    # This is a heuristic-based mapping - should be refined manually
    mapping = {}
    
    # Keywords for each subfield
    subfield_keywords = {
        'Natural Language Processing': ['language', 'text', 'nlp', 'translation', 
                                       'semantic', 'sentence', 'word', 'bert', 
                                       'transformer', 'linguistic'],
        'Computer Vision': ['image', 'visual', 'vision', 'pixel', 'convolutional',
                          'detection', 'recognition', 'camera', 'video', 'object'],
        'Machine Learning Theory': ['optimization', 'convergence', 'theoretical',
                                   'bound', 'algorithm', 'complexity', 'regret',
                                   'generalization', 'sample'],
        'Reinforcement Learning': ['reinforcement', 'reward', 'policy', 'agent',
                                  'q-learning', 'actor', 'critic', 'environment',
                                  'episode', 'action'],
        'Neural Networks': ['neural', 'network', 'deep', 'layer', 'activation',
                          'gradient', 'backpropagation', 'cnn', 'rnn', 'lstm'],
        'Knowledge Representation': ['knowledge', 'reasoning', 'logic', 'ontology',
                                    'inference', 'semantic', 'graph', 'rule'],
        'Robotics': ['robot', 'robotic', 'manipulation', 'motion', 'control',
                    'sensor', 'actuator', 'navigation', 'grasp']
    }
    
    for topic_id, words in enumerate(topic_words):
        # Get top words for this topic
        top_words = [word.lower() for word, _ in words[:20]]
        
        # Count matches for each subfield
        subfield_scores = {}
        for subfield, keywords in subfield_keywords.items():
            score = sum(1 for keyword in keywords if keyword in ' '.join(top_words))
            subfield_scores[subfield] = score
        
        # Assign to subfield with highest score, or 'Other' if no match
        if max(subfield_scores.values()) > 0:
            mapping[topic_id] = max(subfield_scores, key=subfield_scores.get)
        else:
            mapping[topic_id] = 'Other'
    
    return mapping


def train_svm_classifier(X_train, y_train, C=SVM_C, max_iter=SVM_MAX_ITER):
    """
    Train a Linear SVM classifier.
    
    Args:
        X_train: Training features (TF-IDF matrix)
        y_train: Training labels
        C: Regularization parameter
        max_iter: Maximum iterations
        
    Returns:
        Trained classifier
    """
    logger.info("Training Linear SVM classifier...")
    logger.info(f"Training set size: {X_train.shape[0]}")
    logger.info(f"Feature dimensions: {X_train.shape[1]}")
    logger.info(f"Number of classes: {len(np.unique(y_train))}")
    
    clf = LinearSVC(C=C, max_iter=max_iter, random_state=RANDOM_STATE, dual=False)
    clf.fit(X_train, y_train)
    
    logger.info("SVM classifier trained successfully")
    return clf


def evaluate_classifier(clf, X_test, y_test, class_names=None):
    """
    Evaluate classifier performance.
    
    Args:
        clf: Trained classifier
        X_test: Test features
        y_test: Test labels
        class_names: List of class names
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Predictions
    y_pred = clf.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision_weighted': float(precision),
        'recall_weighted': float(recall),
        'f1_weighted': float(f1),
        'precision_per_class': precision_per_class.tolist() if isinstance(precision_per_class, np.ndarray) else precision_per_class,
        'recall_per_class': recall_per_class.tolist() if isinstance(recall_per_class, np.ndarray) else recall_per_class,
        'f1_per_class': f1_per_class.tolist() if isinstance(f1_per_class, np.ndarray) else f1_per_class,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names if class_names else list(np.unique(y_test))
    }
    
    # Print classification report
    logger.info("\n=== Classification Report ===")
    logger.info(f"\nAccuracy: {accuracy:.4f}")
    logger.info(f"Precision (weighted): {precision:.4f}")
    logger.info(f"Recall (weighted): {recall:.4f}")
    logger.info(f"F1-score (weighted): {f1:.4f}")
    
    if class_names:
        logger.info("\n=== Per-Class Metrics ===")
        for i, class_name in enumerate(class_names):
            logger.info(f"{class_name}:")
            logger.info(f"  Precision: {precision_per_class[i]:.4f}")
            logger.info(f"  Recall: {recall_per_class[i]:.4f}")
            logger.info(f"  F1-score: {f1_per_class[i]:.4f}")
    
    return metrics, y_pred


def cross_validate_classifier(X, y, C=SVM_C, max_iter=SVM_MAX_ITER, cv_folds=CV_FOLDS):
    """
    Perform cross-validation on the classifier.
    
    Args:
        X: Features
        y: Labels
        C: Regularization parameter
        max_iter: Maximum iterations
        cv_folds: Number of CV folds
        
    Returns:
        Dictionary with CV results
    """
    logger.info(f"Performing {cv_folds}-fold cross-validation...")
    
    clf = LinearSVC(C=C, max_iter=max_iter, random_state=RANDOM_STATE, dual=False)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    
    # Cross-validation scores
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='f1_weighted')
    
    results = {
        'cv_scores': cv_scores.tolist(),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'cv_folds': cv_folds
    }
    
    logger.info(f"CV F1-score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return results


def run_classification_pipeline(tfidf_matrix=None, labels=None, 
                               topic_words=None, document_topics=None,
                               use_topic_features=False):
    """
    Run the complete classification pipeline.
    
    Args:
        tfidf_matrix: TF-IDF feature matrix
        labels: Ground truth labels (if available)
        topic_words: Topic words from LDA (for creating labels)
        document_topics: Document topic distributions (for creating labels)
        use_topic_features: Whether to use topic distributions as features
        
    Returns:
        Dictionary with results
    """
    ensure_dir(MODELS_DIR)
    ensure_dir(RESULTS_DIR)
    
    # Load data if not provided
    if tfidf_matrix is None:
        logger.info("Loading TF-IDF features...")
        tfidf_matrix = load_pickle(f"{PROCESSED_DATA_DIR}/processed_tfidf_matrix.pkl")
    
    # Create labels from topics if not provided
    if labels is None:
        if document_topics is None:
            logger.info("Loading document topics...")
            document_topics = load_pickle(f"{RESULTS_DIR}/document_topics.pkl")
        
        if topic_words is None:
            logger.info("Loading topic words...")
            topic_words_json = load_json(f"{RESULTS_DIR}/topic_words.json")
            topic_words = [
                [(word, prob) for word, prob in words.items()]
                for words in topic_words_json.values()
            ]
        
        # Create topic-to-subfield mapping
        num_topics = len(topic_words)
        topic_to_subfield = create_manual_subfield_mapping(topic_words, num_topics)
        
        logger.info("Topic-to-subfield mapping:")
        for topic_id, subfield in topic_to_subfield.items():
            logger.info(f"  Topic {topic_id} -> {subfield}")
        
        # Create labels
        labels = create_subfield_labels_from_topics(
            None, document_topics, topic_to_subfield
        )
        
        # Save mapping
        save_json(topic_to_subfield, f"{RESULTS_DIR}/topic_to_subfield_mapping.json")
    
    # Prepare features
    if use_topic_features:
        logger.info("Using topic distributions as features...")
        X = document_topics
    else:
        logger.info("Using TF-IDF features...")
        X = tfidf_matrix
    
    # Get unique labels
    unique_labels = np.unique(labels)
    logger.info(f"Number of classes: {len(unique_labels)}")
    logger.info(f"Classes: {unique_labels}")
    
    # Get indices for train-test split
    indices = np.arange(len(labels))
    train_indices, test_indices = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, 
        stratify=labels
    )
    
    # Split data using indices
    if hasattr(X, 'toarray'):  # Sparse matrix
        X_train = X[train_indices]
        X_test = X[test_indices]
    else:
        X_train = X[train_indices]
        X_test = X[test_indices]
    
    y_train = labels[train_indices]
    y_test = labels[test_indices]
    
    # Cross-validation
    cv_results = cross_validate_classifier(X_train, y_train)
    save_json(cv_results, f"{RESULTS_DIR}/cv_results.json")
    
    # Train classifier
    clf = train_svm_classifier(X_train, y_train)
    
    # Evaluate
    metrics, y_pred = evaluate_classifier(clf, X_test, y_test, 
                                         class_names=unique_labels.tolist())
    
    # Save model
    model_path = f"{MODELS_DIR}/svm_classifier.pkl"
    save_pickle(clf, model_path)
    
    # Save metrics
    save_json(metrics, f"{RESULTS_DIR}/classification_metrics.json")
    
    # Save predictions with test indices
    results_df = pd.DataFrame({
        'test_index': test_indices,
        'true_label': y_test,
        'predicted_label': y_pred
    })
    results_df.to_csv(f"{RESULTS_DIR}/predictions.csv", index=False)
    
    # Save test indices for evaluation
    save_pickle(test_indices, f"{RESULTS_DIR}/test_indices.pkl")
    
    logger.info("Classification pipeline complete!")
    
    return {
        'classifier': clf,
        'metrics': metrics,
        'predictions': y_pred,
        'cv_results': cv_results,
        'topic_to_subfield': topic_to_subfield if labels is None else None
    }


if __name__ == "__main__":
    result = run_classification_pipeline()
    print(f"\nClassification complete!")
    print(f"Accuracy: {result['metrics']['accuracy']:.4f}")
    print(f"F1-score: {result['metrics']['f1_weighted']:.4f}")

