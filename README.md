# Academic Paper Topic Modeling and Classification

A reproducible pipeline for collecting AI research abstracts, discovering topics using LDA, and classifying them into AI subfields using TF-IDF and SVM.

## Project Overview

This project implements:
- **Data Collection**: Automated collection of abstracts from arXiv API
- **Topic Modeling**: LDA-based topic discovery with coherence evaluation
- **Subfield Classification**: TF-IDF + Linear SVM classifier for AI subfield assignment
- **Evaluation**: Comprehensive metrics and comparison between unsupervised and supervised approaches

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

4. **Train classifier**:
```bash
python classification.py
```

5. **Evaluate and compare**:
```bash
python evaluation.py
```

## Project Structure

```
.
├── main.py                 # Main pipeline orchestrator
├── data_collection.py      # arXiv API data collection
├── preprocessing.py        # Text cleaning and feature extraction
├── topic_modeling.py      # LDA topic modeling
├── classification.py      # TF-IDF + SVM classification
├── evaluation.py          # Evaluation and comparison
├── utils.py               # Utility functions
├── config.py              # Configuration parameters
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Data

- Collected abstracts are stored in `data/raw/arxiv_abstracts.csv`
- Processed data is stored in `data/processed/`
- Models and results are saved in `models/` and `results/`

## Configuration

Edit `config.py` to adjust:
- Number of abstracts to collect
- Date range filters
- LDA hyperparameters (number of topics, alpha, beta)
- Classification parameters
- Evaluation metrics

## Outputs

- Topic coherence scores and visualizations
- Classification metrics (accuracy, precision, recall, F1)
- Topic-word distributions
- Per-document topic assignments
- Comparison between topic clusters and supervised classifications

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

3. **PowerShell Execution Policy**: If you get an execution policy error when activating the virtual environment:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

4. **Data Collection Timeout**: If arXiv API requests timeout, the script will retry. You can also reduce `MAX_ABSTRACTS` in `config.py` for faster testing.

5. **Memory Issues**: If you run out of memory during topic modeling, reduce the number of topics being evaluated in `LDA_TOPICS_RANGE` in `config.py`.

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

