"""
Data collection script for arXiv abstracts.
"""

import arxiv
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm
import logging
from config import ARXIV_CATEGORIES, MIN_YEAR, MAX_ABSTRACTS, RAW_DATA_DIR
from utils import ensure_dir, save_dataframe, logger

def collect_arxiv_abstracts(categories=None, max_results=None, min_year=None):
    """
    Collect abstracts from arXiv API.
    
    Args:
        categories: List of arXiv categories to search
        max_results: Maximum number of abstracts to collect
        min_year: Minimum publication year
        
    Returns:
        DataFrame with collected abstracts
    """
    if categories is None:
        categories = ARXIV_CATEGORIES
    if max_results is None:
        max_results = MAX_ABSTRACTS
    if min_year is None:
        min_year = MIN_YEAR
    
    logger.info(f"Collecting abstracts from arXiv categories: {categories}")
    logger.info(f"Target: {max_results} abstracts, minimum year: {min_year}")
    
    all_papers = []
    collected_ids = set()
    
    # Search each category
    for category in categories:
        if len(all_papers) >= max_results:
            break
            
        logger.info(f"Searching category: {category}")
        query = f"cat:{category}"
        
        # Search with date filter
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        category_count = 0
        for paper in tqdm(search.results(), desc=f"Collecting {category}"):
            # Check if we've already collected this paper
            if paper.entry_id in collected_ids:
                continue
            
            # Filter by year
            published_year = paper.published.year
            if published_year < min_year:
                continue
            
            # Extract relevant information
            paper_data = {
                'id': paper.entry_id,
                'title': paper.title,
                'abstract': paper.summary,
                'authors': ', '.join([author.name for author in paper.authors]),
                'published': paper.published,
                'year': published_year,
                'categories': ', '.join(paper.categories),
                'primary_category': paper.primary_category,
                'arxiv_id': paper.entry_id.split('/')[-1]
            }
            
            all_papers.append(paper_data)
            collected_ids.add(paper.entry_id)
            category_count += 1
            
            if len(all_papers) >= max_results:
                break
            
            # Rate limiting
            time.sleep(0.1)
        
        logger.info(f"Collected {category_count} papers from {category}")
    
    # Create DataFrame
    df = pd.DataFrame(all_papers)
    logger.info(f"Total collected: {len(df)} abstracts")
    
    # Remove duplicates based on title similarity or ID
    initial_count = len(df)
    df = df.drop_duplicates(subset=['id'], keep='first')
    logger.info(f"Removed {initial_count - len(df)} duplicates")
    
    # Save to CSV
    output_path = f"{RAW_DATA_DIR}/arxiv_abstracts.csv"
    ensure_dir(RAW_DATA_DIR)
    save_dataframe(df, output_path)
    
    # Print summary statistics
    logger.info("\n=== Collection Summary ===")
    logger.info(f"Total abstracts: {len(df)}")
    logger.info(f"Year range: {df['year'].min()} - {df['year'].max()}")
    logger.info(f"Categories: {df['primary_category'].value_counts().to_dict()}")
    
    return df


if __name__ == "__main__":
    df = collect_arxiv_abstracts()
    print(f"\nCollected {len(df)} abstracts")
    print(f"\nFirst few entries:")
    print(df[['title', 'year', 'primary_category']].head())

