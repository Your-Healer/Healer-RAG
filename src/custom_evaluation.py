import pandas as pd
import json
from symptom_evaluation import SymptomEvaluator
from main import query_rag_system, setup_rag_system
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def evaluate_custom_queries(custom_queries, language="vietnamese", output_file="custom_evaluation_results.json"):
    """
    Evaluate a list of custom symptom queries and analyze the results.
    
    Args:
        custom_queries: List of symptom queries to evaluate
        language: Language for evaluation
        output_file: File to save results
        
    Returns:
        dict: Evaluation results
    """
    # Initialize RAG system
    print("Initializing RAG system...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )
    qa_chain, enhanced_retriever = setup_rag_system(vector_store, language)
    
    # Initialize evaluator
    evaluator = SymptomEvaluator(qa_chain, enhanced_retriever, language=language)
    
    # Process each query
    results = []
    for i, query in enumerate(custom_queries):
        print(f"Processing query {i+1}/{len(custom_queries)}: {query}")
        
        # Query the RAG system
        response = query_rag_system(qa_chain, query, enhanced_retriever, language)
        
        # Extract disease IDs from response
        predicted_disease_ids = evaluator.extract_disease_ids_from_response(response["result"])
        
        # Get retrieved disease IDs from metadata
        retrieved_disease_ids = []
        for doc in response.get("retrieved_documents", []):
            if hasattr(doc, "metadata") and "disease_id" in doc.metadata:
                retrieved_disease_ids.append(doc.metadata["disease_id"])
        
        # Add to results
        results.append({
            "query": query,
            "predicted_disease_ids": predicted_disease_ids,
            "predicted_diseases": [evaluator.disease_id_to_name.get(did, "Unknown") for did in predicted_disease_ids],
            "retrieved_disease_ids": retrieved_disease_ids,
            "retrieved_diseases": [evaluator.disease_id_to_name.get(did, "Unknown") for did in retrieved_disease_ids],
            "response": response["result"]
        })
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Evaluation results saved to {output_file}")
    return results

def create_evaluation_dataset(queries_with_diseases, output_file="evaluation_dataset.json"):
    """
    Create an evaluation dataset from a list of queries and expected diseases.
    
    Args:
        queries_with_diseases: List of tuples (query, [disease_ids])
        output_file: File to save the dataset
        
    Returns:
        list: Evaluation dataset
    """
    dataset = []
    
    for query, disease_ids in queries_with_diseases:
        dataset.append({
            "query": query,
            "expected_disease_ids": disease_ids
        })
    
    # Save dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"Evaluation dataset saved to {output_file}")
    return dataset

# Example usage
if __name__ == "__main__":
    # Example custom queries
    custom_queries = [
        "Tôi bị đau đầu, sốt và đau cơ đã 3 ngày nay, tôi có thể bị bệnh gì?",
        "Tôi bị ho nhiều, đau họng và sổ mũi, tôi có thể mắc bệnh gì?",
        "Tôi bị phát ban trên da và sốt cao, tôi có bị bệnh gì không?",
        "Tôi có đau bụng, tiêu chảy và buồn nôn, tôi bị làm sao vậy?",
        "Tôi bị đau khớp, mệt mỏi và sốt nhẹ, đây có thể là bệnh gì?"
    ]
    
    # Evaluate custom queries
    evaluate_custom_queries(custom_queries)
    
    # Example queries with expected diseases for creating an evaluation dataset
    queries_with_diseases = [
        ("Tôi bị đau đầu, sốt và đau cơ", ["INFL-001", "DENG-001"]),
        ("Tôi bị ho nhiều, đau họng và sổ mũi", ["COMM-001", "FLU-001"]),
        ("Tôi bị phát ban trên da và sốt cao", ["MEAS-001", "CHIC-001"])
    ]
    
    # Create evaluation dataset
    create_evaluation_dataset(queries_with_diseases)
