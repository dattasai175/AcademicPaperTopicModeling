# Topic Modeling Methods Comparison

## Overview

The project now supports **three topic modeling methods** and automatically compares their classification accuracy:

1. **LDA (Latent Dirichlet Allocation)** - Probabilistic topic model
2. **NMF (Non-negative Matrix Factorization)** - Matrix factorization approach
3. **BERTopic** - Neural topic modeling using sentence transformers

## New Files Added

1. **`topic_modeling_nmf.py`** - NMF topic modeling implementation
2. **`topic_modeling_bertopic.py`** - BERTopic implementation
3. **`topic_modeling_comparison.py`** - Unified pipeline to run all methods
4. **`evaluation_comparison.py`** - Compare classification accuracy across methods

## Configuration

In `config.py`, you can enable/disable methods:

```python
USE_LDA = True      # Use LDA topic modeling
USE_NMF = True      # Use NMF topic modeling
USE_BERTOPIC = True # Use BERTopic topic modeling
```

## How It Works

### 1. Topic Modeling Phase
- Runs all enabled methods (LDA, NMF, BERTopic)
- Each method finds optimal number of topics
- Saves models and results separately

### 2. Evaluation Phase
- Evaluates each method's classification accuracy
- Compares predicted categories vs ground truth
- Generates comparison visualizations

### 3. Comparison Results
- **Accuracy comparison** (all categories and top categories)
- **F1-score comparison** (weighted and macro)
- **Best method identification**

## Usage

### Run Full Pipeline
```bash
python main.py
```

This will:
1. Collect/preprocess data (if needed)
2. Run all three topic modeling methods
3. Compare classification accuracy
4. Generate comparison visualizations

### Run Individual Methods
```bash
# LDA only
python topic_modeling.py

# NMF only
python topic_modeling_nmf.py

# BERTopic only
python topic_modeling_bertopic.py
```

### Compare Methods (after running all)
```bash
python evaluation_comparison.py
```

## Output Files

### Method-Specific Results
- `results/lda_*` - LDA results
- `results/nmf_*` - NMF results
- `results/bertopic_*` - BERTopic results

### Comparison Results
- `results/method_comparison_summary.json` - Summary metrics
- `results/method_comparison.png` - Comparison visualization

## Expected Improvements

**BERTopic** typically performs best because:
- Uses semantic embeddings (sentence transformers)
- Better understanding of document meaning
- Handles synonyms and related concepts better

**NMF** often performs better than LDA because:
- Non-negative constraints lead to more interpretable topics
- Better for sparse data
- More stable factorization

**LDA** is good baseline:
- Well-established method
- Fast and interpretable
- Good for comparison

## Installation

Install new dependencies:
```bash
pip install bertopic sentence-transformers umap-learn hdbscan
```

Or update requirements:
```bash
pip install -r requirements.txt
```

## Notes

- **BERTopic** uses original abstracts (not preprocessed) for better semantic understanding
- **NMF** uses TF-IDF matrix (same as LDA)
- **LDA** uses TF-IDF corpus (when `USE_TFIDF_FOR_LDA=True`)
- All methods are evaluated on the same dataset for fair comparison

