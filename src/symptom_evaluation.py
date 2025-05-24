import json
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from dotenv import load_dotenv
import numpy as np
from collections import defaultdict
from main import query_rag_system, setup_rag_system, is_symptom_query
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

class SymptomEvaluator:
    """Evaluator for symptom-based disease prediction."""
    
    def __init__(self, qa_chain=None, enhanced_retriever=None, db_path="./chroma_langchain_db", language="vietnamese"):
        """
        Initialize the evaluator.
        
        Args:
            qa_chain: The RAG chain (if None, will initialize)
            enhanced_retriever: The enhanced retriever (if None, will initialize)
            db_path: Path to the vector database
            language: Language for evaluation
        """
        self.language = language
        
        # Initialize RAG system if not provided
        if qa_chain is None or enhanced_retriever is None:
            print("Initializing RAG system...")
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            vector_store = Chroma(
                collection_name="example_collection",
                embedding_function=embeddings,
                persist_directory=db_path,
            )
            self.qa_chain, self.enhanced_retriever = setup_rag_system(vector_store, language)
        else:
            self.qa_chain = qa_chain
            self.enhanced_retriever = enhanced_retriever
        
        # Load disease data for ground truth
        self.disease_data = self._load_disease_data()
        self.disease_id_to_name = {data["disease_id"]: data["disease_name"] for data in self.disease_data}
        
        # Create symptom to disease_id mapping for ground truth
        self.symptom_to_disease_ids = self._create_symptom_mapping()
    
    def _load_disease_data(self):
        """Load all disease data from JSON files."""
        json_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../json_data")
        
        if not os.path.exists(json_folder):
            raise Exception(f"Directory not found: {json_folder}")
        
        json_files = [f for f in os.listdir(json_folder) if f.endswith('.json') and f != "template.json"]
        
        json_data_list = []
        for json_file in json_files:
            file_path = os.path.join(json_folder, json_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    json_data_list.append(data)
            except Exception as e:
                print(f"Error loading {json_file}: {str(e)}")
        
        return json_data_list
    
    def _create_symptom_mapping(self):
        """Create mapping from symptoms to disease IDs."""
        symptom_to_disease_ids = defaultdict(list)
        
        for disease in self.disease_data:
            disease_id = disease.get("disease_id", "")
            
            # Skip entries without disease_id
            if not disease_id:
                continue
            
            # Process symptoms
            symptoms = disease.get("symptoms", [])
            for symptom in symptoms:
                symptom_name = symptom.get("name", "").lower()
                if symptom_name:
                    symptom_to_disease_ids[symptom_name].append(disease_id)
        
        return symptom_to_disease_ids
    
    def extract_disease_ids_from_response(self, response_text):
        """Extract disease IDs from the response text."""
        extracted_ids = []
        
        # Try to match direct mentions of disease IDs
        id_pattern = r'(?:disease_id|mã bệnh|ID):\s*([A-Z0-9-]+)'
        direct_matches = re.findall(id_pattern, response_text, re.IGNORECASE)
        extracted_ids.extend(direct_matches)
        
        # Match disease names and map to IDs
        for disease_id, disease_name in self.disease_id_to_name.items():
            if disease_name.lower() in response_text.lower():
                extracted_ids.append(disease_id)
        
        # Return unique IDs
        return list(set(extracted_ids))
    
    def generate_evaluation_queries(self, num_queries=20):
        """
        Generate symptom-based queries for evaluation.
        
        Args:
            num_queries: Number of queries to generate
            
        Returns:
            list: List of dicts with query, symptoms, and expected disease IDs
        """
        evaluation_queries = []
        
        # Get symptoms that have associated diseases
        symptoms_with_diseases = [(symptom, disease_ids) 
                                  for symptom, disease_ids in self.symptom_to_disease_ids.items()
                                  if disease_ids]
        
        # If we don't have enough symptoms, use all we have
        num_queries = min(num_queries, len(symptoms_with_diseases))
        
        # Select random symptoms
        import random
        selected_symptoms = random.sample(symptoms_with_diseases, num_queries)
        
        # Generate queries
        for symptom, disease_ids in selected_symptoms:
            if self.language == "vietnamese":
                query = f"Tôi bị {symptom}, tôi có thể mắc bệnh gì?"
            else:
                query = f"I have {symptom}, what disease might I have?"
            
            evaluation_queries.append({
                "query": query,
                "symptoms": [symptom],
                "expected_disease_ids": disease_ids
            })
        
        return evaluation_queries
    
    def evaluate_queries(self, evaluation_queries):
        """
        Run evaluation on the given queries.
        
        Args:
            evaluation_queries: List of evaluation query dicts
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        
        print(f"Evaluating {len(evaluation_queries)} queries...")
        
        for i, eval_query in enumerate(evaluation_queries):
            query = eval_query["query"]
            expected_disease_ids = eval_query["expected_disease_ids"]
            
            print(f"Processing query {i+1}/{len(evaluation_queries)}: {query}")
            
            # Query the RAG system
            response = query_rag_system(
                self.qa_chain, 
                query, 
                self.enhanced_retriever, 
                self.language
            )
            
            # Extract disease IDs from response
            predicted_disease_ids = self.extract_disease_ids_from_response(response["result"])
            
            # Get retrieved disease IDs from metadata
            retrieved_disease_ids = []
            for doc in response.get("retrieved_documents", []):
                if hasattr(doc, "metadata") and "disease_id" in doc.metadata:
                    retrieved_disease_ids.append(doc.metadata["disease_id"])
            
            # Compute metrics
            correct_predictions = [did for did in predicted_disease_ids if did in expected_disease_ids]
            precision = len(correct_predictions) / len(predicted_disease_ids) if predicted_disease_ids else 0
            recall = len(correct_predictions) / len(expected_disease_ids) if expected_disease_ids else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Add to results
            results.append({
                "query": query,
                "symptoms": eval_query["symptoms"],
                "expected_disease_ids": expected_disease_ids,
                "predicted_disease_ids": predicted_disease_ids,
                "retrieved_disease_ids": retrieved_disease_ids,
                "correct_predictions": correct_predictions,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "response": response["result"]
            })
        
        return pd.DataFrame(results)
    
    def analyze_results(self, df):
        """
        Analyze evaluation results.
        
        Args:
            df: DataFrame with evaluation results
            
        Returns:
            dict with analysis metrics
        """
        analysis = {}
        
        # Overall metrics
        analysis["overall"] = {
            "precision": df["precision"].mean(),
            "recall": df["recall"].mean(),
            "f1_score": df["f1_score"].mean(),
            "total_queries": len(df),
            "queries_with_predictions": len(df[df["predicted_disease_ids"].str.len() > 0]),
            "perfect_matches": len(df[df["precision"] == 1.0]),
            "no_matches": len(df[df["recall"] == 0.0])
        }
        
        # Most common correct predictions
        all_correct = [disease_id for sublist in df["correct_predictions"].tolist() for disease_id in sublist]
        analysis["top_correct_diseases"] = pd.Series(all_correct).value_counts().head(10).to_dict()
        
        # Most common incorrect predictions
        all_incorrect = []
        for _, row in df.iterrows():
            incorrect = [did for did in row["predicted_disease_ids"] if did not in row["expected_disease_ids"]]
            all_incorrect.extend(incorrect)
        
        analysis["top_incorrect_diseases"] = pd.Series(all_incorrect).value_counts().head(10).to_dict()
        
        # Top missed diseases (false negatives)
        all_missed = []
        for _, row in df.iterrows():
            missed = [did for did in row["expected_disease_ids"] if did not in row["predicted_disease_ids"]]
            all_missed.extend(missed)
        
        analysis["top_missed_diseases"] = pd.Series(all_missed).value_counts().head(10).to_dict()
        
        return analysis
    
    def visualize_results(self, df, analysis, output_dir="./evaluation_results"):
        """
        Visualize evaluation results.
        
        Args:
            df: DataFrame with evaluation results
            analysis: Dictionary with analysis metrics
            output_dir: Directory to save visualizations
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot overall metrics
        metrics = ["precision", "recall", "f1_score"]
        plt.figure(figsize=(10, 6))
        plt.bar(metrics, [analysis["overall"][m] for m in metrics], color=["blue", "orange", "green"])
        plt.ylim(0, 1.0)
        plt.title("Overall Performance Metrics")
        plt.savefig(f"{output_dir}/overall_metrics.png")
        
        # Plot distribution of metrics
        plt.figure(figsize=(12, 6))
        for i, metric in enumerate(metrics):
            plt.subplot(1, 3, i+1)
            sns.histplot(df[metric], bins=10, kde=True)
            plt.title(f"Distribution of {metric}")
            plt.xlim(0, 1.0)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/metrics_distribution.png")
        
        # Plot top correct diseases
        plt.figure(figsize=(12, 8))
        top_correct = pd.Series(analysis["top_correct_diseases"])
        labels = [f"{self.disease_id_to_name.get(did, did)} ({did})" for did in top_correct.index]
        plt.barh(labels, top_correct.values)
        plt.title("Top Correctly Predicted Diseases")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/top_correct_diseases.png")
        
        # Plot top incorrect diseases
        if analysis["top_incorrect_diseases"]:
            plt.figure(figsize=(12, 8))
            top_incorrect = pd.Series(analysis["top_incorrect_diseases"])
            labels = [f"{self.disease_id_to_name.get(did, did)} ({did})" for did in top_incorrect.index]
            plt.barh(labels, top_incorrect.values)
            plt.title("Top Incorrectly Predicted Diseases")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/top_incorrect_diseases.png")
        
        # Plot top missed diseases
        if analysis["top_missed_diseases"]:
            plt.figure(figsize=(12, 8))
            top_missed = pd.Series(analysis["top_missed_diseases"])
            labels = [f"{self.disease_id_to_name.get(did, did)} ({did})" for did in top_missed.index]
            plt.barh(labels, top_missed.values)
            plt.title("Top Missed Diseases")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/top_missed_diseases.png")
        
        # Save detailed results to CSV
        df.to_csv(f"{output_dir}/detailed_results.csv", index=False)
        
        # Save analysis summary
        with open(f"{output_dir}/analysis_summary.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Visualizations and results saved to {output_dir}")
    
    def run_evaluation(self, num_queries=20, output_dir="./evaluation_results"):
        """
        Run the complete evaluation pipeline.
        
        Args:
            num_queries: Number of queries to evaluate
            output_dir: Directory to save results
        """
        print(f"Starting symptom-based disease prediction evaluation...")
        
        # Generate evaluation queries
        evaluation_queries = self.generate_evaluation_queries(num_queries)
        print(f"Generated {len(evaluation_queries)} evaluation queries")
        
        # Run evaluation
        results_df = self.evaluate_queries(evaluation_queries)
        
        # Analyze results
        analysis = self.analyze_results(results_df)
        
        # Visualize results
        self.visualize_results(results_df, analysis, output_dir)
        
        # Print summary
        print("\nEvaluation Summary:")
        print(f"Total queries: {analysis['overall']['total_queries']}")
        print(f"Average precision: {analysis['overall']['precision']:.4f}")
        print(f"Average recall: {analysis['overall']['recall']:.4f}")
        print(f"Average F1 score: {analysis['overall']['f1_score']:.4f}")
        print(f"Perfect matches: {analysis['overall']['perfect_matches']} ({analysis['overall']['perfect_matches']/analysis['overall']['total_queries']*100:.1f}%)")
        print(f"No matches: {analysis['overall']['no_matches']} ({analysis['overall']['no_matches']/analysis['overall']['total_queries']*100:.1f}%)")
        
        return results_df, analysis

if __name__ == "__main__":
    evaluator = SymptomEvaluator()
    evaluator.run_evaluation(num_queries=20)
