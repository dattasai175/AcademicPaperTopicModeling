"""
Configuration parameters for the topic modeling and classification pipeline.
"""

# Data Collection
ARXIV_CATEGORIES = ['cs.AI', 'cs.CL', 'cs.CV', 'cs.LG', 'cs.NE', 'stat.ML']
MIN_YEAR = 2018  # Last 5-7 years
MAX_ABSTRACTS = 1500
MIN_ABSTRACTS = 500

# Preprocessing
MIN_WORD_LENGTH = 2
MAX_FEATURES_TFIDF = 10000
NGRAM_RANGE = (1, 2)  # Unigrams and bigrams
MIN_DF = 2  # Minimum document frequency
MAX_DF = 0.95  # Maximum document frequency

# Topic Modeling (LDA)
LDA_TOPICS_RANGE = [5, 8, 10, 15, 20]  # Number of topics to explore
LDA_ALPHA = 'auto'  # Document-topic prior
LDA_BETA = 'auto'  # Topic-word prior
LDA_PASSES = 10
LDA_ITERATIONS = 50
LDA_RANDOM_STATE = 42

# Classification
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
SVM_C = 1.0
SVM_MAX_ITER = 1000

# Paths
DATA_DIR = 'data'
RAW_DATA_DIR = f'{DATA_DIR}/raw'
PROCESSED_DATA_DIR = f'{DATA_DIR}/processed'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'

# AI Subfields for classification
AI_SUBFIELDS = [
    'Natural Language Processing',
    'Computer Vision',
    'Machine Learning Theory',
    'Reinforcement Learning',
    'Neural Networks',
    'Knowledge Representation',
    'Robotics',
    'Other'
]

