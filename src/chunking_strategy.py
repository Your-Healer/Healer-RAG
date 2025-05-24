import json
from langchain_text_splitters import RecursiveJsonSplitter

def create_optimized_chunks(json_data, max_chunk_size=300):
    """
    Create optimized chunks from JSON data with a comprehensive strategy.
    
    Ensures all fields from the medical template are properly chunked while
    preserving semantic relationships and field structures.
    
    Args:
        json_data: The JSON data to chunk
        max_chunk_size: Maximum chunk size in tokens
        
    Returns:
        list: List of optimized chunks
    """
    chunks = []
    
    # Always add a base chunk with core disease information
    base_chunk = {
        "disease_id": json_data.get("disease_id", ""),
        "disease_name": json_data.get("disease_name", ""),
        "disease_alternative_names": json_data.get("disease_alternative_names", []),
        "disease_definition": json_data.get("disease_definition", ""),
        "section": "core_information"
    }
    chunks.append(base_chunk)
    
    # Create a chunk for symptoms
    if "symptoms" in json_data and json_data["symptoms"]:
        symptoms_chunk = {
            "disease_id": json_data.get("disease_id", ""),
            "disease_name": json_data.get("disease_name", ""),
            "symptoms": json_data["symptoms"],
            "section": "symptoms"
        }
        chunks.append(symptoms_chunk)
    
    # Create a chunk for causes
    if "causes" in json_data and json_data["causes"]:
        causes_chunk = {
            "disease_id": json_data.get("disease_id", ""),
            "disease_name": json_data.get("disease_name", ""),
            "causes": json_data["causes"],
            "section": "causes"
        }
        chunks.append(causes_chunk)
    
    # Create a chunk for risk factors
    if "risk_factors" in json_data and json_data["risk_factors"]:
        risk_factors_chunk = {
            "disease_id": json_data.get("disease_id", ""),
            "disease_name": json_data.get("disease_name", ""),
            "risk_factors": json_data["risk_factors"],
            "section": "risk_factors"
        }
        chunks.append(risk_factors_chunk)
    
    # Create a chunk for complications
    if "complications" in json_data and json_data["complications"]:
        complications_chunk = {
            "disease_id": json_data.get("disease_id", ""),
            "disease_name": json_data.get("disease_name", ""),
            "complications": json_data["complications"],
            "section": "complications"
        }
        chunks.append(complications_chunk)
    
    # Create a chunk for diagnosis
    if "diagnosis" in json_data and json_data["diagnosis"]:
        diagnosis_chunk = {
            "disease_id": json_data.get("disease_id", ""),
            "disease_name": json_data.get("disease_name", ""),
            "diagnosis": json_data["diagnosis"],
            "section": "diagnosis"
        }
        chunks.append(diagnosis_chunk)
    
    # Create a chunk for treatment
    if "treatment" in json_data and json_data["treatment"]:
        treatment_chunk = {
            "disease_id": json_data.get("disease_id", ""),
            "disease_name": json_data.get("disease_name", ""),
            "treatment": json_data["treatment"],
            "section": "treatment"
        }
        chunks.append(treatment_chunk)
    
    # Create a chunk for prevention
    if "prevention" in json_data and json_data["prevention"]:
        prevention_chunk = {
            "disease_id": json_data.get("disease_id", ""),
            "disease_name": json_data.get("disease_name", ""),
            "prevention": json_data["prevention"],
            "section": "prevention"
        }
        chunks.append(prevention_chunk)
    
    # Create chunks for FAQs in smaller groups
    if "faqs" in json_data and json_data["faqs"]:
        # Process FAQs in smaller groups to avoid token limits
        faq_batch_size = 3  # Process 3 FAQs at a time
        for i in range(0, len(json_data["faqs"]), faq_batch_size):
            faq_chunk = {
                "disease_id": json_data.get("disease_id", ""),
                "disease_name": json_data.get("disease_name", ""),
                "faqs": json_data["faqs"][i:i+faq_batch_size],
                "section": f"faqs_{i//faq_batch_size + 1}"
            }
            chunks.append(faq_chunk)
    
    # Create chunks for common questions
    if "common_questions" in json_data and json_data["common_questions"]:
        # Process common questions in smaller groups
        question_batch_size = 3
        for i in range(0, len(json_data["common_questions"]), question_batch_size):
            question_chunk = {
                "disease_id": json_data.get("disease_id", ""),
                "disease_name": json_data.get("disease_name", ""),
                "common_questions": json_data["common_questions"][i:i+question_batch_size],
                "section": f"common_questions_{i//question_batch_size + 1}"
            }
            chunks.append(question_chunk)
    
    # Apply recursive splitting only if necessary (if chunks might be too large)
    splitter = RecursiveJsonSplitter(max_chunk_size=max_chunk_size)
    final_chunks = []
    
    for chunk in chunks:
        # Check if this chunk might be too large
        chunk_str = json.dumps(chunk, ensure_ascii=False)
        if len(chunk_str) > max_chunk_size * 4:  # Rough estimate: 4 chars per token
            # Apply recursive splitting
            split_chunks = splitter.split_json(json_data=chunk)
            final_chunks.extend(split_chunks)
        else:
            # Keep as is
            final_chunks.append(chunk)
    
    return final_chunks

def chunking_json_data(json_data, max_chunk_size=300):
    """
    Split JSON data using optimized chunking strategy.
    
    Args:
        json_data (dict): JSON data to be chunked
        max_chunk_size (int): Maximum chunk size in tokens
        
    Returns:
        list: List of JSON chunks
    """
    try:
        return create_optimized_chunks(json_data, max_chunk_size)
    except Exception as e:
        print(f"Error in optimized chunking: {str(e)}")
        # Fallback to standard chunking
        splitter = RecursiveJsonSplitter(max_chunk_size=max_chunk_size)
        return splitter.split_json(json_data=json_data)
