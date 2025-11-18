"""
Quick test script to verify all imports work correctly.
"""

import sys

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import numpy
        print("✓ numpy")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False
    
    try:
        import pandas
        print("✓ pandas")
    except ImportError as e:
        print(f"✗ pandas: {e}")
        return False
    
    try:
        import sklearn
        print("✓ scikit-learn")
    except ImportError as e:
        print(f"✗ scikit-learn: {e}")
        return False
    
    try:
        import gensim
        print("✓ gensim")
    except ImportError as e:
        print(f"✗ gensim: {e}")
        return False
    
    try:
        import nltk
        print("✓ nltk")
    except ImportError as e:
        print(f"✗ nltk: {e}")
        return False
    
    try:
        import matplotlib
        print("✓ matplotlib")
    except ImportError as e:
        print(f"✗ matplotlib: {e}")
        return False
    
    try:
        import seaborn
        print("✓ seaborn")
    except ImportError as e:
        print(f"✗ seaborn: {e}")
        return False
    
    try:
        import pyLDAvis
        print("✓ pyLDAvis")
    except ImportError as e:
        print(f"✗ pyLDAvis: {e}")
        return False
    
    try:
        import arxiv
        print("✓ arxiv")
    except ImportError as e:
        print(f"✗ arxiv: {e}")
        return False
    
    try:
        import tqdm
        print("✓ tqdm")
    except ImportError as e:
        print(f"✗ tqdm: {e}")
        return False
    
    # Test project modules
    try:
        import config
        print("✓ config")
    except ImportError as e:
        print(f"✗ config: {e}")
        return False
    
    try:
        import utils
        print("✓ utils")
    except ImportError as e:
        print(f"✗ utils: {e}")
        return False
    
    print("\nAll imports successful!")
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)

