"""
Text preprocessing and feature extraction pipeline for topic modeling.
"""

import re
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import logging
from tqdm import tqdm
from config import (
    MIN_WORD_LENGTH, MAX_FEATURES_TFIDF, NGRAM_RANGE, 
    MIN_DF, MAX_DF, RAW_DATA_DIR, PROCESSED_DATA_DIR
)
from utils import (
    ensure_dir, save_pickle, load_pickle, save_dataframe, 
    load_dataframe, logger
)

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)


class TextPreprocessor:
    """Text preprocessing class for cleaning and normalizing abstracts."""
    
    def __init__(self, min_word_length=MIN_WORD_LENGTH):
        self.min_word_length = min_word_length
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Add domain-specific stopwords
        self.stop_words.update(['abstract', 'paper', 'propose', 'proposed', 
                               'method', 'approach', 'result', 'results',
                               'show', 'demonstrate', 'present', 'study'])
    
    def clean_text(self, text):
        """Clean and normalize text."""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """Tokenize text into words."""
        tokens = word_tokenize(text)
        return tokens
    
    def lemmatize(self, tokens):
        """Lemmatize tokens."""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def filter_tokens(self, tokens):
        """Filter tokens by length and stopwords."""
        filtered = [
            token for token in tokens
            if len(token) >= self.min_word_length
            and token not in self.stop_words
            and token.isalpha()  # Keep only alphabetic tokens
        ]
        return filtered
    
    def preprocess(self, text):
        """Complete preprocessing pipeline."""
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.lemmatize(tokens)
        tokens = self.filter_tokens(tokens)
        return ' '.join(tokens)


def preprocess_abstracts(df, text_column='abstract'):
    """
    Preprocess all abstracts in the DataFrame.
    
    Args:
        df: DataFrame with abstracts
        text_column: Name of the column containing text
        
    Returns:
        DataFrame with preprocessed text
    """
    logger.info("Starting text preprocessing...")
    
    preprocessor = TextPreprocessor()
    
    # Preprocess abstracts
    processed_texts = []
    for idx, text in enumerate(tqdm(df[text_column], desc="Preprocessing")):
        processed = preprocessor.preprocess(text)
        processed_texts.append(processed)
    
    df_processed = df.copy()
    df_processed['processed_text'] = processed_texts
    
    # Remove empty processed texts
    initial_count = len(df_processed)
    df_processed = df_processed[df_processed['processed_text'].str.len() > 0]
    logger.info(f"Removed {initial_count - len(df_processed)} empty processed texts")
    
    logger.info(f"Preprocessing complete. Processed {len(df_processed)} abstracts")
    
    return df_processed


def create_bow_features(texts, min_df=MIN_DF, max_df=MAX_DF):
    """
    Create bag-of-words features for LDA.
    
    Args:
        texts: List of preprocessed text strings
        min_df: Minimum document frequency
        max_df: Maximum document frequency
        
    Returns:
        BOW matrix, vectorizer, dictionary (for Gensim), and corpus
    """
    logger.info("Creating bag-of-words features for LDA...")
    
    # Create Gensim dictionary
    from gensim.corpora import Dictionary
    
    # Convert texts to list of token lists
    tokenized_texts = [text.split() for text in texts]
    dictionary = Dictionary(tokenized_texts)
    
    # Filter extremes
    dictionary.filter_extremes(no_below=min_df, no_above=max_df)
    
    # Create corpus
    corpus = [dictionary.doc2bow(text.split()) for text in texts]
    
    logger.info(f"Dictionary size: {len(dictionary)}")
    logger.info(f"Corpus size: {len(corpus)} documents")
    
    return dictionary, corpus


def run_preprocessing_pipeline(input_file=None, output_prefix='processed'):
    """
    Run the complete preprocessing pipeline.
    
    Args:
        input_file: Path to input CSV file (default: raw data)
        output_prefix: Prefix for output files
        
    Returns:
        Dictionary with processed data and features
    """
    # Load data
    if input_file is None:
        input_file = f"{RAW_DATA_DIR}/arxiv_abstracts.csv"
    
    logger.info(f"Loading data from {input_file}")
    df = load_dataframe(input_file)
    
    # Preprocess texts
    df_processed = preprocess_abstracts(df)
    
    # Save processed DataFrame
    output_file = f"{PROCESSED_DATA_DIR}/{output_prefix}_data.csv"
    ensure_dir(PROCESSED_DATA_DIR)
    save_dataframe(df_processed, output_file)
    
    # Create BOW features for LDA
    dictionary, corpus = create_bow_features(
        df_processed['processed_text'].tolist()
    )
    
    # Save features
    save_pickle(dictionary, f"{PROCESSED_DATA_DIR}/{output_prefix}_dictionary.pkl")
    save_pickle(corpus, f"{PROCESSED_DATA_DIR}/{output_prefix}_corpus.pkl")
    
    logger.info("Preprocessing pipeline complete!")
    
    return {
        'df': df_processed,
        'dictionary': dictionary,
        'corpus': corpus
    }


if __name__ == "__main__":
    result = run_preprocessing_pipeline()
    print(f"\nPreprocessing complete!")
    print(f"Processed {len(result['df'])} abstracts")
    print(f"Dictionary size: {len(result['dictionary'])}")
