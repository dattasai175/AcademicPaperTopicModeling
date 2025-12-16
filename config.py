"""
Configuration parameters for the arXiv AI Papers Topic Modeling pipeline.
"""

# Data Collection
ARXIV_CATEGORIES = ['cs.AI', 'cs.LG', 'cs.CV', 'cs.CL', 'cs.NE', 'cs.RO', 'stat.ML']
MIN_YEAR = 2018  # Minimum publication year
MAX_ABSTRACTS = 5000  # Target number of abstracts to collect
ARXIV_MAX_RESULTS_PER_QUERY = 2000  # arXiv API limit per query (actual limit)

# Preprocessing
MIN_WORD_LENGTH = 2
MAX_FEATURES_TFIDF = 10000  # Maximum features for TF-IDF
NGRAM_RANGE = (1, 1)  # Use unigrams for LDA (bigrams can be too sparse)
MIN_DF = 2  # Minimum document frequency
MAX_DF = 0.95  # Maximum document frequency
USE_TFIDF_FOR_LDA = True  # Use TF-IDF instead of BOW for LDA (better results)

# Topic Modeling - Multiple Methods
USE_LDA = True  # Use LDA topic modeling
USE_NMF = True  # Use NMF topic modeling
USE_BERTOPIC = True  # Use BERTopic topic modeling

# LDA Parameters
LDA_TOPICS_RANGE = [3, 5, 7, 10]  # Number of topics to explore
LDA_ALPHA = 'auto'  # Document-topic prior (auto = 1/num_topics)
LDA_BETA = 0.01  # Topic-word prior (lower = more focused topics)
LDA_PASSES = 20  # More passes for better convergence
LDA_ITERATIONS = 100  # More iterations for better convergence
LDA_RANDOM_STATE = 42
COHERENCE_TYPE = 'c_v'  # Coherence metric type

# NMF Parameters
NMF_TOPICS_RANGE = [5, 8, 10, 12, 15, 18, 20]  # Number of topics to explore (same as LDA for comparison)
NMF_ALPHA = 0.1  # Regularization parameter (alpha_W and alpha_H)
NMF_L1_RATIO = 0.5  # L1/L2 regularization ratio
NMF_MAX_ITER = 200  # Maximum iterations
NMF_RANDOM_STATE = 42

# BERTopic Parameters
BERTOPIC_TOPICS_RANGE = [3, 5, 7, 10]  # Number of topics to explore
BERTOPIC_MODEL = 'all-MiniLM-L6-v2'  # Sentence transformer model
BERTOPIC_MIN_TOPIC_SIZE = 10  # Minimum documents per topic
BERTOPIC_RANDOM_STATE = 42

# Evaluation - Focus on major categories
MIN_CATEGORY_COUNT = 50  # Minimum papers per category for evaluation
TOP_CATEGORIES_FOR_EVAL = 10  # Top N categories to focus on

# Paths
DATA_DIR = 'data'
RAW_DATA_DIR = f'{DATA_DIR}/raw'
PROCESSED_DATA_DIR = f'{DATA_DIR}/processed'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'

# Analysis/Evaluation
TOP_WORDS_PER_TOPIC = 15  # Number of top words to display per topic
TOP_CATEGORIES_PER_TOPIC = 10  # Number of top categories to show per topic
