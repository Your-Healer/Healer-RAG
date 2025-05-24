from langchain.evaluation import QAEvalChain
from langchain_openai import ChatOpenAI

def evaluate_rag_system(qa_chain, eval_data):
    """
    Evaluate the RAG system using a set of evaluation data.
    
    Args:
        qa_chain: The RAG system chain
        eval_data (list): List of dictionaries with 'question' and 'answer' keys
        
    Returns:
        dict: Evaluation results
    """
    llm = ChatOpenAI(temperature=0)
    eval_chain = QAEvalChain.from_llm(llm)
    
    # Generate predictions
    predictions = []
    references = []
    
    for example in eval_data:
        result = qa_chain({"query": example["question"]})
        predictions.append({"text": result["result"]})
        references.append({"text": example["answer"]})
    
    # Evaluate
    graded_outputs = eval_chain.evaluate(
        predictions=predictions,
        references=references
    )
    
    return graded_outputs
