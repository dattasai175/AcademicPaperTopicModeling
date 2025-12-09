"""
Utility functions for the topic modeling and category alignment analysis pipeline.
"""

import os
import pickle
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def save_pickle(obj, filepath):
    """Save object as pickle file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    logger.info(f"Saved pickle to {filepath}")


def load_pickle(filepath):
    """Load object from pickle file."""
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    logger.info(f"Loaded pickle from {filepath}")
    return obj


def save_json(obj, filepath, indent=2):
    """Save object as JSON file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
    logger.info(f"Saved JSON to {filepath}")


def load_json(filepath):
    """Load object from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    logger.info(f"Loaded JSON from {filepath}")
    return obj


def save_dataframe(df, filepath, index=False):
    """Save DataFrame to CSV."""
    ensure_dir(os.path.dirname(filepath))
    df.to_csv(filepath, index=index, encoding='utf-8')
    logger.info(f"Saved DataFrame to {filepath}")


def load_dataframe(filepath):
    """Load DataFrame from CSV."""
    df = pd.read_csv(filepath, encoding='utf-8')
    logger.info(f"Loaded DataFrame from {filepath}")
    return df


def print_topic_words(topic_words, top_n=10):
    """Print top words for each topic."""
    for topic_id, words in enumerate(topic_words):
        print(f"\nTopic {topic_id}:")
        print(", ".join([f"{word} ({prob:.3f})" for word, prob in words[:top_n]]))


def get_dominant_topic(topic_distribution):
    """Get the dominant topic index from a topic distribution."""
    return np.argmax(topic_distribution)


def calculate_topic_diversity(topic_words, top_n=10):
    """Calculate topic diversity as the average unique words across topics."""
    all_words = set()
    for words in topic_words:
        all_words.update([w[0] for w in words[:top_n]])
    return len(all_words) / (len(topic_words) * top_n)

