"""
Data collection script for arXiv abstracts with support for large-scale collection.
"""

import arxiv
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm
import logging
from config import (
    ARXIV_CATEGORIES, MIN_YEAR, MAX_ABSTRACTS, 
    ARXIV_MAX_RESULTS_PER_QUERY, RAW_DATA_DIR
)
from utils import ensure_dir, save_dataframe, logger


def collect_arxiv_abstracts(categories=None, max_results=None, min_year=None):
    """
    Collect abstracts from arXiv API with pagination support.
    
    Args:
        categories: List of arXiv categories to search
        max_results: Maximum number of abstracts to collect
        min_year: Minimum publication year
        
    Returns:
        DataFrame with collected abstracts including categories and primary_category
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
        
        # Calculate how many results we need from this category
        remaining = max_results - len(all_papers)
        results_per_category = min(remaining, ARXIV_MAX_RESULTS_PER_QUERY)
        
        # Use pagination if we need more than ARXIV_MAX_RESULTS_PER_QUERY
        offset = 0
        category_count = 0
        
        while len(all_papers) < max_results and offset < max_results:
            # Determine how many to fetch in this batch
            batch_size = min(
                ARXIV_MAX_RESULTS_PER_QUERY,
                max_results - len(all_papers),
                max_results - offset
            )
            
            if batch_size <= 0:
                break
            
            try:
                # Search with date filter and pagination
                search = arxiv.Search(
                    query=query,
                    max_results=batch_size,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                )
                
                # Collect papers from this batch
                batch_papers = []
                for paper in tqdm(search.results(), 
                                 desc=f"Collecting {category} (batch {offset//ARXIV_MAX_RESULTS_PER_QUERY + 1})",
                                 total=batch_size):
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
                        'categories': ', '.join(paper.categories),  # All categories
                        'primary_category': paper.primary_category,  # Primary category
                        'arxiv_id': paper.entry_id.split('/')[-1]
                    }
                    
                    batch_papers.append(paper_data)
                    collected_ids.add(paper.entry_id)
                    category_count += 1
                    
                    if len(all_papers) + len(batch_papers) >= max_results:
                        break
                
                all_papers.extend(batch_papers)
                offset += batch_size
                
                # Rate limiting
                time.sleep(0.1)
                
                # If we got fewer results than requested, we've reached the end
                if len(batch_papers) < batch_size:
                    break
                    
            except Exception as e:
                logger.warning(f"Error collecting batch from {category} at offset {offset}: {e}")
                break
        
        logger.info(f"Collected {category_count} papers from {category}")
    
    # Create DataFrame
    df = pd.DataFrame(all_papers)
    logger.info(f"Total collected: {len(df)} abstracts")
    
    # Remove duplicates based on ID
    initial_count = len(df)
    df = df.drop_duplicates(subset=['id'], keep='first')
    logger.info(f"Removed {initial_count - len(df)} duplicates")
    
    # Ensure categories columns are present
    if 'categories' not in df.columns:
        logger.warning("Categories column missing, adding empty column")
        df['categories'] = ''
    if 'primary_category' not in df.columns:
        logger.warning("Primary category column missing, adding empty column")
        df['primary_category'] = ''
    
    # Save to CSV
    output_path = f"{RAW_DATA_DIR}/arxiv_abstracts.csv"
    ensure_dir(RAW_DATA_DIR)
    save_dataframe(df, output_path)
    
    # Print summary statistics
    logger.info("\n=== Collection Summary ===")
    logger.info(f"Total abstracts: {len(df)}")
    logger.info(f"Year range: {df['year'].min()} - {df['year'].max()}")
    
    if 'primary_category' in df.columns:
        category_counts = df['primary_category'].value_counts()
        logger.info(f"Primary categories distribution:")
        for cat, count in category_counts.head(10).items():
            logger.info(f"  {cat}: {count}")
    
    return df


if __name__ == "__main__":
    df = collect_arxiv_abstracts()
    print(f"\nCollected {len(df)} abstracts")
    print(f"\nFirst few entries:")
    print(df[['title', 'year', 'primary_category', 'categories']].head())
    print(f"\nColumn names: {df.columns.tolist()}")
