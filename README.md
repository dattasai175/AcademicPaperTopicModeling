# arXiv AI Papers Topic Modeling & Category Alignment Analysis

A comprehensive pipeline for collecting AI research papers from arXiv, discovering latent topics using LDA, and analyzing how discovered topics align with ground-truth arXiv categories.

## Project Overview

This project implements an unsupervised topic discovery pipeline that:

- **Data Collection**: Automated collection of up to 5000 recent AI research abstracts from arXiv API
- **Topic Modeling**: LDA-based topic discovery with coherence score optimization to find the optimal number of topics
- **Category Alignment Analysis**: Comprehensive comparison between discovered LDA topics and actual arXiv categories (primary and secondary)
- **Visualization**: Rich visualizations showing topic-category relationships, topic distributions, and temporal trends

## Key Features

- **Large-Scale Data Collection**: Collects up to 5000 papers with proper pagination handling
- **Category Preservation**: Maintains both primary and all secondary categories from arXiv
- **Optimal Topic Discovery**: Automatically finds the best number of topics using coherence scores (C_v metric)
- **Alignment Metrics**: Computes Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), and topic purity
- **Rich Visualizations**: Heatmaps, stacked bar charts, topic-word visualizations, and temporal analysis

## Installation

### Step 0: Create a Virtual Environment (Recommended)

It's recommended to use a virtual environment to isolate project dependencies.

**On Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate.bat
```

**On macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

After activation, you should see `(venv)` in your terminal prompt.

**To deactivate the virtual environment later:**
```bash
deactivate
```

### Step 1: Install Dependencies

Once your virtual environment is activated, install the required packages:

```bash
pip install -r requirements.txt
```

**Note**: This project requires Python 3.9-3.12. Python 3.14+ is not yet compatible with gensim.

### Step 2: Download NLTK Data

Download required NLTK data:
```bash
python setup.py
```

Or manually:
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt_tab')"
```

## Usage

### Quick Start

Run the complete pipeline:
```bash
python main.py
```

The pipeline will:
1. Collect abstracts from arXiv (if not already collected)
2. Preprocess and clean the text
3. Discover optimal topics using LDA
4. Analyze topic-category alignment
5. Generate visualizations and reports

### Step-by-Step Execution

1. **Collect data from arXiv**:
```bash
python data_collection.py
```

2. **Preprocess and extract features**:
```bash
python preprocessing.py
```

3. **Run topic modeling**:
```bash
python topic_modeling.py
```

4. **Generate analysis and evaluation**:
```bash
python evaluation.py
```

## Project Structure

```
.
├── main.py                 # Main pipeline orchestrator
├── data_collection.py      # arXiv API data collection with pagination
├── preprocessing.py        # Text cleaning and feature extraction
├── topic_modeling.py       # LDA topic modeling with coherence optimization
├── evaluation.py           # Topic-category alignment analysis
├── utils.py                # Utility functions
├── config.py               # Configuration parameters
├── setup.py                # NLTK data download script
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Data

- **Raw Data**: Collected abstracts are stored in `data/raw/arxiv_abstracts.csv`
  - Includes: title, abstract, authors, published date, categories, primary_category
- **Processed Data**: Cleaned and preprocessed data in `data/processed/`
- **Models**: Trained LDA models and dictionaries saved in `models/`
- **Results**: Analysis results, visualizations, and metrics in `results/`

## Configuration

Edit `config.py` to adjust:

- **Data Collection**:
  - `MAX_ABSTRACTS`: Number of papers to collect (default: 5000)
  - `ARXIV_CATEGORIES`: List of arXiv categories to search
  - `MIN_YEAR`: Minimum publication year

- **Topic Modeling**:
  - `LDA_TOPICS_RANGE`: List of topic numbers to evaluate (default: [5, 8, 10, 15, 20, 25])
  - `LDA_PASSES`: Number of passes through corpus
  - `LDA_ITERATIONS`: Number of iterations per pass
  - `COHERENCE_TYPE`: Coherence metric type ('c_v', 'u_mass', 'c_uci', 'c_npmi')

- **Analysis**:
  - `TOP_WORDS_PER_TOPIC`: Number of top words to display per topic
  - `TOP_CATEGORIES_PER_TOPIC`: Number of top categories to show per topic

## Outputs

The pipeline generates comprehensive outputs:

### Metrics & Statistics
- **Coherence Scores**: Evaluation of different topic numbers to find optimal
- **Alignment Metrics**: Adjusted Rand Index (ARI), Normalized Mutual Information (NMI)
- **Topic Purity**: Proportion of papers in each topic that belong to the most common category
- **Category Distributions**: Statistics on arXiv category usage

### Visualizations
- **Coherence Scores Plot**: Shows coherence vs. number of topics
- **Category Distribution**: Bar charts of primary and all categories
- **Topic-Category Heatmap**: Shows which categories are most associated with each topic
- **Stacked Bar Chart**: Category proportions per topic
- **Topic Words Visualization**: Top words for each discovered topic
- **Topic Distribution Over Time**: How topics evolve across years
- **Interactive LDA Visualization**: HTML file with pyLDAvis interactive exploration

### Data Files
- `topic_words.json`: Top words and probabilities for each topic
- `document_topics.pkl`: Topic distribution for each document
- `category_statistics.json`: Category distribution statistics
- `topic_category_alignment.json`: Detailed alignment analysis
- `analysis_data.csv`: Combined data with dominant topics

## Understanding the Results

### Topic-Category Alignment

The analysis compares discovered LDA topics with ground-truth arXiv categories:

- **High ARI/NMI**: Topics align well with arXiv categories (topics capture category structure)
- **Low ARI/NMI**: Topics discover different structure than categories (topics may capture cross-cutting themes)
- **Topic Purity**: For each topic, what fraction belongs to the most common category
  - High purity: Topic is dominated by one category
  - Low purity: Topic spans multiple categories

### Interpreting Topics

Each topic is represented by:
- **Top Words**: Most probable words in the topic
- **Category Distribution**: Which arXiv categories are most common in papers assigned to this topic
- **Temporal Trends**: How the topic's prevalence changes over time

## Reproducibility

All random seeds are set for reproducibility. The pipeline includes:
- Version tracking of dependencies
- Configurable parameters
- Detailed logging
- Saved intermediate results

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **NLTK Data Missing**: Run the setup script:
   ```bash
   python setup.py
   ```

3. **Python Version**: This project requires Python 3.9-3.12. If you have Python 3.14+, create a virtual environment with an older Python version:
   ```bash
   /usr/bin/python3 -m venv venv  # Use system Python 3.9
   source venv/bin/activate
   ```

4. **Data Collection Timeout**: If arXiv API requests timeout, the script includes retry logic. You can also reduce `MAX_ABSTRACTS` in `config.py` for faster testing.

5. **Memory Issues**: If you run out of memory during topic modeling:
   - Reduce `MAX_ABSTRACTS` in `config.py`
   - Reduce the number of topics in `LDA_TOPICS_RANGE`
   - Reduce `LDA_PASSES` and `LDA_ITERATIONS`

6. **arXiv API Rate Limiting**: The script includes rate limiting (0.1s delay between requests). If you encounter rate limits, increase the delay in `data_collection.py`.

### Verifying Installation

Test that all imports work:
```bash
python test_imports.py
```

### Resuming from Errors

If the pipeline fails at a specific step:
- Delete the output files from the failed step
- Re-run `main.py` (it will skip completed steps)
- Or run individual scripts starting from the failed step

## Example Workflow

```bash
# 1. Set up environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python setup.py

# 2. Run complete pipeline
python main.py

# 3. Explore results
# - Open results/lda_visualization.html in browser
# - Check results/topic_category_alignment.json for metrics
# - View visualizations in results/ directory
```

## Citation

If you use this code in your research, please cite:

```
arXiv AI Papers Topic Modeling & Category Alignment Analysis
A pipeline for unsupervised topic discovery and category alignment analysis
```

## License

This project is provided as-is for educational and research purposes.
