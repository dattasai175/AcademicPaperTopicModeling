"""
Interactive HTML visualizations for topic modeling results.
"""

import os
import pandas as pd
import numpy as np
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from config import RESULTS_DIR
from utils import ensure_dir, logger

# Set style
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for embedding in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str


def create_topic_word_interactive_html(topic_word_df, output_path=None, top_n=15):
    """
    Create interactive HTML visualization for topic-word distributions.
    
    Args:
        topic_word_df: DataFrame with columns topic_id, word, probability
        output_path: Path to save HTML file
        top_n: Number of top words to show per topic
    """
    if output_path is None:
        output_path = f"{RESULTS_DIR}/topic_word_distribution_interactive.html"
    
    num_topics = topic_word_df['topic_id'].nunique()
    
    # Create visualizations for each topic
    topic_images = {}
    for topic_id in range(num_topics):
        topic_data = topic_word_df[topic_word_df['topic_id'] == topic_id].head(top_n)
        topic_data = topic_data.sort_values('probability', ascending=True)
        
        fig, ax = plt.subplots(figsize=(8, max(6, len(topic_data) * 0.4)))
        ax.barh(range(len(topic_data)), topic_data['probability'], 
                color=plt.cm.Set3(topic_id % 12))
        ax.set_yticks(range(len(topic_data)))
        ax.set_yticklabels(topic_data['word'])
        ax.set_xlabel('Probability', fontsize=12)
        ax.set_title(f'Topic {topic_id} - Top {top_n} Words', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, prob in enumerate(topic_data['probability']):
            ax.text(prob + 0.001, i, f'{prob:.3f}', va='center', fontsize=9)
        
        topic_images[topic_id] = fig_to_base64(fig)
    
    # Create HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Topic-Word Distribution Visualization</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }}
            h1 {{
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }}
            .topic-section {{
                margin: 30px 0;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
                border-left: 5px solid #667eea;
            }}
            .topic-section h2 {{
                color: #667eea;
                margin-top: 0;
            }}
            .topic-image {{
                text-align: center;
                margin: 20px 0;
            }}
            .topic-image img {{
                max-width: 100%;
                height: auto;
                border-radius: 5px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            .controls {{
                text-align: center;
                margin: 20px 0;
                padding: 15px;
                background: #f0f0f0;
                border-radius: 5px;
            }}
            .controls button {{
                padding: 10px 20px;
                margin: 5px;
                background: #667eea;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 14px;
            }}
            .controls button:hover {{
                background: #5568d3;
            }}
            .hidden {{
                display: none;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Topic-Word Distribution Visualization</h1>
            <p style="text-align: center; color: #666;">Total Topics: {num_topics} | Words per Topic: {top_n}</p>
            
            <div class="controls">
                <button onclick="showAll()">Show All Topics</button>
                <button onclick="showTopic(0)">Topic 0</button>
    """
    
    for i in range(1, num_topics):
        html_content += f'<button onclick="showTopic({i})">Topic {i}</button>\n'
    
    html_content += """
            </div>
    """
    
    # Add topic sections
    for topic_id in range(num_topics):
        html_content += f"""
            <div class="topic-section" id="topic-{topic_id}">
                <h2>Topic {topic_id}</h2>
                <div class="topic-image">
                    <img src="data:image/png;base64,{topic_images[topic_id]}" alt="Topic {topic_id}">
                </div>
            </div>
        """
    
    html_content += """
        </div>
        
        <script>
            function showAll() {
                for (let i = 0; i < """ + str(num_topics) + """; i++) {
                    document.getElementById('topic-' + i).style.display = 'block';
                }
            }
            
            function showTopic(topicId) {
                for (let i = 0; i < """ + str(num_topics) + """; i++) {
                    const element = document.getElementById('topic-' + i);
                    if (i === topicId) {
                        element.style.display = 'block';
                        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    } else {
                        element.style.display = 'none';
                    }
                }
            }
            
            // Show all by default
            showAll();
        </script>
    </body>
    </html>
    """
    
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Saved interactive topic-word visualization to {output_path}")


def create_document_topic_interactive_html(doc_topic_df, output_path=None, sample_size=500):
    """
    Create interactive HTML visualization for document-topic distributions.
    
    Args:
        doc_topic_df: DataFrame with id, dominant_topic, and topic_* columns
        output_path: Path to save HTML file
        sample_size: Number of documents to show
    """
    if output_path is None:
        output_path = f"{RESULTS_DIR}/document_topic_distribution_interactive.html"
    
    # Sample data if too large
    if len(doc_topic_df) > sample_size:
        doc_topic_df_sample = doc_topic_df.sample(n=sample_size, random_state=42).sort_index()
        logger.info(f"Sampling {sample_size} documents for visualization")
    else:
        doc_topic_df_sample = doc_topic_df.copy()
    
    # Get topic columns
    topic_cols = [col for col in doc_topic_df_sample.columns if col.startswith('topic_')]
    num_topics = len(topic_cols)
    
    # Create heatmap visualization
    fig, ax = plt.subplots(figsize=(max(12, num_topics * 1.2), max(8, len(doc_topic_df_sample) * 0.15)))
    heatmap_data = doc_topic_df_sample[topic_cols].values.T
    sns.heatmap(heatmap_data, 
                yticklabels=topic_cols,
                xticklabels=False,
                cmap='YlOrRd',
                cbar_kws={'label': 'Probability'},
                ax=ax)
    ax.set_title(f'Document-Topic Probability Heatmap ({len(doc_topic_df_sample)} Documents)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Documents', fontsize=12)
    ax.set_ylabel('Topics', fontsize=12)
    heatmap_img = fig_to_base64(fig)
    
    # Create distribution chart
    fig, ax = plt.subplots(figsize=(max(10, num_topics * 0.8), 6))
    topic_counts = doc_topic_df_sample['dominant_topic'].value_counts().sort_index()
    ax.bar(range(len(topic_counts)), topic_counts.values, color=plt.cm.Set3(range(len(topic_counts))))
    ax.set_xticks(range(len(topic_counts)))
    ax.set_xticklabels([f'Topic {i}' for i in topic_counts.index])
    ax.set_xlabel('Topic', fontsize=12)
    ax.set_ylabel('Number of Documents', fontsize=12)
    ax.set_title('Distribution of Dominant Topics', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(topic_counts.values):
        ax.text(i, v + max(topic_counts.values) * 0.01, str(v), ha='center', va='bottom')
    dist_img = fig_to_base64(fig)
    
    # Create HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Document-Topic Distribution Visualization</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }}
            .container {{
                max-width: 1600px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }}
            h1 {{
                color: #333;
                text-align: center;
                margin-bottom: 10px;
            }}
            .section {{
                margin: 30px 0;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
            }}
            .section h2 {{
                color: #667eea;
                margin-top: 0;
            }}
            .chart-container {{
                text-align: center;
                margin: 20px 0;
            }}
            .chart-container img {{
                max-width: 100%;
                height: auto;
                border-radius: 5px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            .controls {{
                margin: 20px 0;
                padding: 15px;
                background: #f0f0f0;
                border-radius: 5px;
            }}
            .controls input, .controls select {{
                padding: 8px;
                margin: 5px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            .controls button {{
                padding: 8px 16px;
                background: #667eea;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }}
            .controls button:hover {{
                background: #5568d3;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
                font-size: 12px;
            }}
            th, td {{
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #667eea;
                color: white;
                cursor: pointer;
                position: sticky;
                top: 0;
            }}
            th:hover {{
                background-color: #5568d3;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .prob-cell {{
                text-align: right;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Document-Topic Distribution Visualization</h1>
            <p style="text-align: center; color: #666;">
                Total documents: {len(doc_topic_df_sample):,} | Topics: {num_topics} | 
                Showing sample of {len(doc_topic_df_sample):,} documents
            </p>
            
            <div class="section">
                <h2>Topic Distribution Chart</h2>
                <div class="chart-container">
                    <img src="data:image/png;base64,{dist_img}" alt="Topic Distribution">
                </div>
            </div>
            
            <div class="section">
                <h2>Document-Topic Heatmap</h2>
                <div class="chart-container">
                    <img src="data:image/png;base64,{heatmap_img}" alt="Heatmap">
                </div>
            </div>
            
            <div class="section">
                <h2>Document-Topic Assignments Table</h2>
                <div class="controls">
                    <input type="text" id="searchInput" placeholder="Search by ID..." 
                           onkeyup="filterTable()" style="width: 300px;">
                    <select id="topicFilter" onchange="filterTable()">
                        <option value="">All Topics</option>
    """
    
    for i in range(num_topics):
        html_content += f'<option value="{i}">Topic {i}</option>\n'
    
    html_content += """
                    </select>
                    <button onclick="resetFilters()">Reset</button>
                    <button onclick="exportTable()">Export Filtered CSV</button>
                </div>
                
                <div style="overflow-x: auto; max-height: 600px; overflow-y: auto;">
                    <table id="dataTable">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Dominant Topic</th>
    """
    
    for col in topic_cols:
        html_content += f'<th onclick="sortTable({topic_cols.index(col) + 2})">{col}</th>'
    
    html_content += """
                            </tr>
                        </thead>
                        <tbody>
    """
    
    # Add table rows
    for idx, row in doc_topic_df_sample.iterrows():
        html_content += '<tr>'
        html_content += f'<td style="max-width: 200px; overflow: hidden; text-overflow: ellipsis;">{row["id"]}</td>'
        html_content += f'<td><strong style="color: #667eea;">Topic {row["dominant_topic"]}</strong></td>'
        for col in topic_cols:
            prob = row[col]
            intensity = prob * 0.7 + 0.3
            html_content += f'<td class="prob-cell" style="background-color: rgba(102, 126, 234, {intensity});">{prob:.4f}</td>'
        html_content += '</tr>'
    
    html_content += """
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <script>
            let allRows = Array.from(document.querySelectorAll('#dataTable tbody tr'));
            let sortDirection = {{}};
            
            function filterTable() {{
                const searchInput = document.getElementById('searchInput').value.toLowerCase();
                const topicFilter = document.getElementById('topicFilter').value;
                const rows = document.querySelectorAll('#dataTable tbody tr');
                
                rows.forEach(row => {{
                    const cells = row.querySelectorAll('td');
                    const id = cells[0].textContent.toLowerCase();
                    const dominantTopic = cells[1].textContent.match(/\\d+/)[0];
                    
                    const matchesSearch = id.includes(searchInput);
                    const matchesTopic = !topicFilter || dominantTopic === topicFilter;
                    
                    row.style.display = (matchesSearch && matchesTopic) ? '' : 'none';
                }});
            }}
            
            function resetFilters() {{
                document.getElementById('searchInput').value = '';
                document.getElementById('topicFilter').value = '';
                filterTable();
            }}
            
            function sortTable(col) {{
                const table = document.getElementById('dataTable');
                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));
                
                const dir = sortDirection[col] || 'asc';
                sortDirection[col] = dir === 'asc' ? 'desc' : 'asc';
                
                rows.sort((a, b) => {{
                    const aVal = parseFloat(a.querySelectorAll('td')[col].textContent) || 0;
                    const bVal = parseFloat(b.querySelectorAll('td')[col].textContent) || 0;
                    return dir === 'asc' ? aVal - bVal : bVal - aVal;
                }});
                
                rows.forEach(row => tbody.appendChild(row));
            }}
            
            function exportTable() {{
                const visibleRows = Array.from(document.querySelectorAll('#dataTable tbody tr'))
                    .filter(row => row.style.display !== 'none');
                
                let csv = 'id,dominant_topic,' + """ + str(','.join(topic_cols)) + """ + '\\n';
                
                visibleRows.forEach(row => {{
                    const cells = row.querySelectorAll('td');
                    const rowData = Array.from(cells).map(cell => cell.textContent.trim());
                    csv += rowData.join(',') + '\\n';
                }});
                
                const blob = new Blob([csv], {{ type: 'text/csv' }});
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'filtered_document_topic_distribution.csv';
                a.click();
            }}
        </script>
    </body>
    </html>
    """
    
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Saved interactive document-topic visualization to {output_path}")


def create_topic_dashboard_html(topic_word_df, doc_topic_df, output_path=None):
    """
    Create a comprehensive dashboard HTML with both visualizations.
    
    Args:
        topic_word_df: DataFrame with topic-word distributions
        doc_topic_df: DataFrame with document-topic distributions
        output_path: Path to save HTML file
    """
    if output_path is None:
        output_path = f"{RESULTS_DIR}/topic_modeling_dashboard.html"
    
    num_topics = topic_word_df['topic_id'].nunique()
    topic_cols = [col for col in doc_topic_df.columns if col.startswith('topic_')]
    
    # Create topic-word images
    topic_images = {}
    for topic_id in range(num_topics):
        topic_data = topic_word_df[topic_word_df['topic_id'] == topic_id].head(15)
        topic_data = topic_data.sort_values('probability', ascending=True)
        
        fig, ax = plt.subplots(figsize=(8, max(6, len(topic_data) * 0.4)))
        ax.barh(range(len(topic_data)), topic_data['probability'], 
                color=plt.cm.Set3(topic_id % 12))
        ax.set_yticks(range(len(topic_data)))
        ax.set_yticklabels(topic_data['word'])
        ax.set_xlabel('Probability', fontsize=10)
        ax.set_title(f'Topic {topic_id}', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        topic_images[topic_id] = fig_to_base64(fig)
    
    # Create distribution chart
    fig, ax = plt.subplots(figsize=(max(10, num_topics * 0.8), 6))
    topic_counts = doc_topic_df['dominant_topic'].value_counts().sort_index()
    ax.bar(range(len(topic_counts)), topic_counts.values, color=plt.cm.Set3(range(len(topic_counts))))
    ax.set_xticks(range(len(topic_counts)))
    ax.set_xticklabels([f'Topic {i}' for i in topic_counts.index])
    ax.set_xlabel('Topic', fontsize=12)
    ax.set_ylabel('Number of Documents', fontsize=12)
    ax.set_title('Distribution of Dominant Topics', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    dist_img = fig_to_base64(fig)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Topic Modeling Dashboard</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }}
            .dashboard {{
                max-width: 1600px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 2.5em;
            }}
            .content {{
                padding: 30px;
            }}
            .stats {{
                display: flex;
                justify-content: space-around;
                margin: 20px 0;
            }}
            .stat-box {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                flex: 1;
                margin: 0 10px;
            }}
            .stat-box h3 {{
                margin: 0;
                font-size: 2em;
            }}
            .section {{
                margin: 40px 0;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
            }}
            .section h2 {{
                color: #667eea;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
            }}
            .topic-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .topic-card {{
                background: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .topic-card img {{
                width: 100%;
                height: auto;
                border-radius: 5px;
            }}
            .chart-container {{
                text-align: center;
                margin: 20px 0;
            }}
            .chart-container img {{
                max-width: 100%;
                height: auto;
                border-radius: 5px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
        </style>
    </head>
    <body>
        <div class="dashboard">
            <div class="header">
                <h1>📊 Topic Modeling Dashboard</h1>
                <p>Comprehensive Visualization of Document-Topic and Topic-Word Distributions</p>
            </div>
            <div class="content">
                <div class="stats">
                    <div class="stat-box">
                        <h3>{len(doc_topic_df):,}</h3>
                        <p>Documents</p>
                    </div>
                    <div class="stat-box">
                        <h3>{num_topics}</h3>
                        <p>Topics</p>
                    </div>
                    <div class="stat-box">
                        <h3>{len(topic_word_df)}</h3>
                        <p>Topic-Word Pairs</p>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Topic Distribution</h2>
                    <div class="chart-container">
                        <img src="data:image/png;base64,{dist_img}" alt="Topic Distribution">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Topic-Word Distributions</h2>
                    <div class="topic-grid">
    """
    
    for topic_id in range(num_topics):
        html_content += f"""
                        <div class="topic-card">
                            <h3 style="color: #667eea; margin-top: 0;">Topic {topic_id}</h3>
                            <img src="data:image/png;base64,{topic_images[topic_id]}" alt="Topic {topic_id}">
                        </div>
        """
    
    html_content += """
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Saved topic modeling dashboard to {output_path}")


if __name__ == "__main__":
    # Load data
    topic_word_df = pd.read_csv(f"{RESULTS_DIR}/topic_word_distribution.csv")
    doc_topic_df = pd.read_csv(f"{RESULTS_DIR}/document_topic_distribution.csv")
    
    # Create visualizations
    create_topic_word_interactive_html(topic_word_df)
    create_document_topic_interactive_html(doc_topic_df)
    create_topic_dashboard_html(topic_word_df, doc_topic_df)
    print("\n✅ All interactive HTML visualizations created successfully!")
