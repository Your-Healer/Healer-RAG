import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import logging

def parse_retrieval_log(log_file='retrieval.log'):
    """
    Parse the retrieval log file to extract query and document information.
    
    Args:
        log_file (str): Path to the log file
        
    Returns:
        pd.DataFrame: DataFrame with retrieval information
    """
    queries = []
    retrieved_docs = []
    current_query = None
    current_docs = []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Extract query information
                query_match = re.search(r'RETRIEVAL QUERY: (.*) \(symptom-based: (.*), language: (.*)\)', line)
                if query_match:
                    # If we were processing a query, save it before starting a new one
                    if current_query and current_docs:
                        queries.append(current_query)
                        retrieved_docs.append(current_docs.copy())
                    
                    current_query = {
                        'query': query_match.group(1),
                        'symptom_based': query_match.group(2) == 'True',
                        'language': query_match.group(3)
                    }
                    current_docs = []
                    continue
                
                # Extract document information
                doc_match = re.search(r'Doc (\d+): (.*) - (.*)', line)
                if doc_match and current_query:
                    current_docs.append({
                        'position': int(doc_match.group(1)),
                        'disease': doc_match.group(2),
                        'section': doc_match.group(3)
                    })
                
                # Check for query completion
                if 'QUERY COMPLETED' in line and current_query and current_docs:
                    queries.append(current_query)
                    retrieved_docs.append(current_docs.copy())
                    current_query = None
                    current_docs = []
        
        # Create a DataFrame from the parsed data
        data = []
        for i in range(len(queries)):
            query = queries[i]
            docs = retrieved_docs[i]
            
            for doc in docs:
                data.append({
                    'query': query['query'],
                    'symptom_based': query['symptom_based'],
                    'language': query['language'],
                    'doc_position': doc['position'],
                    'disease': doc['disease'],
                    'section': doc['section']
                })
        
        return pd.DataFrame(data)
    
    except Exception as e:
        logging.error(f"Error parsing retrieval log: {str(e)}")
        return pd.DataFrame()

def analyze_retrieval_data(df):
    """
    Analyze retrieval data and generate insights.
    
    Args:
        df (pd.DataFrame): DataFrame with retrieval information
        
    Returns:
        dict: Dictionary with analysis results
    """
    if df.empty:
        return {"error": "No data to analyze"}
    
    results = {}
    
    # Count symptom-based vs. disease-based queries
    query_types = df[['query', 'symptom_based']].drop_duplicates()
    results['query_type_counts'] = query_types['symptom_based'].value_counts().to_dict()
    
    # Most frequently retrieved diseases
    results['top_diseases'] = df['disease'].value_counts().head(10).to_dict()
    
    # Most frequently retrieved sections
    results['top_sections'] = df['section'].value_counts().head(10).to_dict()
    
    # Retrieval position analysis
    results['position_distribution'] = df['doc_position'].value_counts().sort_index().to_dict()
    
    # Language distribution
    results['language_distribution'] = df[['query', 'language']].drop_duplicates()['language'].value_counts().to_dict()
    
    return results

def visualize_retrieval_analysis(analysis_results, output_dir='.'):
    """
    Visualize retrieval analysis results.
    
    Args:
        analysis_results (dict): Results from analyze_retrieval_data
        output_dir (str): Directory to save visualizations
    """
    if 'error' in analysis_results:
        print(f"Error: {analysis_results['error']}")
        return
    
    # Query Type Distribution
    plt.figure(figsize=(10, 6))
    labels = ['Disease-based', 'Symptom-based']
    values = [
        analysis_results['query_type_counts'].get(False, 0),
        analysis_results['query_type_counts'].get(True, 0)
    ]
    plt.bar(labels, values, color=['#3498db', '#e74c3c'])
    plt.title('Query Type Distribution')
    plt.ylabel('Count')
    plt.savefig(f"{output_dir}/query_types.png")
    
    # Top Diseases
    plt.figure(figsize=(12, 6))
    diseases = list(analysis_results['top_diseases'].keys())
    counts = list(analysis_results['top_diseases'].values())
    plt.barh(diseases, counts, color='#2ecc71')
    plt.title('Top Retrieved Diseases')
    plt.xlabel('Count')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_diseases.png")
    
    # Top Sections
    plt.figure(figsize=(12, 6))
    sections = list(analysis_results['top_sections'].keys())
    counts = list(analysis_results['top_sections'].values())
    plt.barh(sections, counts, color='#9b59b6')
    plt.title('Top Retrieved Sections')
    plt.xlabel('Count')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_sections.png")
    
    # Document Position Distribution
    plt.figure(figsize=(10, 6))
    positions = list(analysis_results['position_distribution'].keys())
    counts = list(analysis_results['position_distribution'].values())
    plt.bar(positions, counts, color='#f39c12')
    plt.title('Document Position Distribution')
    plt.xlabel('Position')
    plt.ylabel('Count')
    plt.savefig(f"{output_dir}/position_distribution.png")
    
    print(f"Visualizations saved to {output_dir}")

def main():
    """Run the retrieval analysis pipeline."""
    print("Analyzing retrieval logs...")
    df = parse_retrieval_log()
    
    if df.empty:
        print("No retrieval data found. Make sure the retrieval.log file exists.")
        return
    
    print(f"Found {len(df)} retrieval records")
    
    analysis_results = analyze_retrieval_data(df)
    visualize_retrieval_analysis(analysis_results)
    
    print("Analysis complete.")
    print("\nSummary:")
    print(f"- Total unique queries: {len(df['query'].unique())}")
    print(f"- Symptom-based queries: {analysis_results['query_type_counts'].get(True, 0)}")
    print(f"- Disease-based queries: {analysis_results['query_type_counts'].get(False, 0)}")
    print(f"- Most common disease: {max(analysis_results['top_diseases'], key=analysis_results['top_diseases'].get)}")
    print(f"- Most common section: {max(analysis_results['top_sections'], key=analysis_results['top_sections'].get)}")

if __name__ == "__main__":
    main()
