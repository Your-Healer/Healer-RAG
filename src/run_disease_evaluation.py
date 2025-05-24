import argparse
from symptom_evaluation import SymptomEvaluator
from custom_evaluation import evaluate_custom_queries, create_evaluation_dataset
import json
import os

def main():
    parser = argparse.ArgumentParser(description="Evaluate disease prediction from symptoms")
    parser.add_argument("--mode", choices=["auto", "custom", "dataset"], default="auto",
                      help="Evaluation mode: auto (generate queries), custom (use provided queries), dataset (use evaluation dataset)")
    parser.add_argument("--num_queries", type=int, default=20,
                      help="Number of queries to generate in auto mode")
    parser.add_argument("--language", choices=["vietnamese", "english"], default="vietnamese",
                      help="Language for evaluation")
    parser.add_argument("--input_file", type=str, default=None,
                      help="Input file for custom queries or evaluation dataset")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                      help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == "auto":
        print(f"Running automatic evaluation with {args.num_queries} generated queries...")
        evaluator = SymptomEvaluator(language=args.language)
        evaluator.run_evaluation(num_queries=args.num_queries, output_dir=args.output_dir)
    
    elif args.mode == "custom":
        if not args.input_file:
            # Use default questions if no input file provided
            custom_queries = [
                "Tôi bị đau đầu, sốt và đau cơ đã 3 ngày nay, tôi có thể bị bệnh gì?",
                "Tôi bị ho nhiều, đau họng và sổ mũi, tôi có thể mắc bệnh gì?",
                "Tôi bị phát ban trên da và sốt cao, tôi có bị bệnh gì không?",
                "Tôi có đau bụng, tiêu chảy và buồn nôn, tôi bị làm sao vậy?",
                "Tôi bị đau khớp, mệt mỏi và sốt nhẹ, đây có thể là bệnh gì?"
            ]
        else:
            # Load queries from input file
            with open(args.input_file, 'r', encoding='utf-8') as f:
                custom_queries = json.load(f)
        
        print(f"Running evaluation with {len(custom_queries)} custom queries...")
        output_file = os.path.join(args.output_dir, "custom_evaluation_results.json")
        evaluate_custom_queries(custom_queries, language=args.language, output_file=output_file)
    
    elif args.mode == "dataset":
        if not args.input_file:
            print("Error: Input file required for dataset evaluation mode")
            return
        
        # Load evaluation dataset
        with open(args.input_file, 'r', encoding='utf-8') as f:
            evaluation_dataset = json.load(f)
        
        print(f"Running evaluation with dataset containing {len(evaluation_dataset)} queries...")
        evaluator = SymptomEvaluator(language=args.language)
        results_df, analysis = evaluator.evaluate_queries(evaluation_dataset)
        evaluator.visualize_results(results_df, analysis, output_dir=args.output_dir)
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main()
