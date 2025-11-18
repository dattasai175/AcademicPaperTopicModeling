"""
Setup script to download required NLTK data.
"""

import nltk
import sys

def download_nltk_data():
    """Download required NLTK data."""
    print("Downloading NLTK data...")
    
    try:
        nltk.download('punkt', quiet=True)
        print("✓ Downloaded punkt")
    except Exception as e:
        print(f"✗ Error downloading punkt: {e}")
    
    try:
        nltk.download('stopwords', quiet=True)
        print("✓ Downloaded stopwords")
    except Exception as e:
        print(f"✗ Error downloading stopwords: {e}")
    
    try:
        nltk.download('wordnet', quiet=True)
        print("✓ Downloaded wordnet")
    except Exception as e:
        print(f"✗ Error downloading wordnet: {e}")
    
    try:
        nltk.download('omw-1.4', quiet=True)
        print("✓ Downloaded omw-1.4")
    except Exception as e:
        print(f"✗ Error downloading omw-1.4: {e}")
    
    print("\nNLTK data download complete!")

if __name__ == "__main__":
    download_nltk_data()

