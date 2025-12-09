"""
Configuration parameters for the arXiv AI Papers Topic Modeling pipeline.
"""

# Data Collection
ARXIV_CATEGORIES = ['cs.AI', 'cs.LG', 'cs.CV', 'cs.CL', 'cs.NE', 'cs.RO', 'stat.ML']
MIN_YEAR = 2018  # Minimum publication year
MAX_ABSTRACTS = 5000  # Target number of abstracts to collect
ARXIV_MAX_RESULTS_PER_QUERY = 5000  # arXiv API limit per query

# Preprocessing
MIN_WORD_LENGTH = 2
MAX_FEATURES_TFIDF = 10000  # For potential future use
NGRAM_RANGE = (1, 2)  # Unigrams and bigrams
MIN_DF = 2  # Minimum document frequency
MAX_DF = 0.95  # Maximum document frequency

# Topic Modeling (LDA)
LDA_TOPICS_RANGE = [3, 5, 8, 10, 12, 15,17, 20,25,30,35,40,45,50]  # Number of topics to explore
LDA_ALPHA = 'auto'  # Document-topic prior
LDA_BETA = 'auto'  # Topic-word prior
LDA_PASSES = 10
LDA_ITERATIONS = 50
LDA_RANDOM_STATE = 42
COHERENCE_TYPE = 'c_v'  # Coherence metric type

# Paths
DATA_DIR = 'data'
RAW_DATA_DIR = f'{DATA_DIR}/raw'
PROCESSED_DATA_DIR = f'{DATA_DIR}/processed'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'

# Analysis/Evaluation
TOP_WORDS_PER_TOPIC = 15  # Number of top words to display per topic
TOP_CATEGORIES_PER_TOPIC = 10  # Number of top categories to show per topic
