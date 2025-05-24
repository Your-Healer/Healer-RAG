import json
import os
import getpass
import logging
import argparse
from dotenv import load_dotenv
from typing import List
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from preprocess import preprocess_json_data
# Import the enhanced retrieval system
from enhanced_retrieval import EnhancedRetrieval, setup_enhanced_retrieval
# Import the optimized chunking strategy
from chunking_strategy import chunking_json_data
# Import vector store factory
from vector_store_factory import create_vector_store, check_vector_store_exists

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='retrieval.log'
)
logger = logging.getLogger('retrieval')

def load_all_json_data():
    """
    Load all JSON data from json_data directory.
    
    Returns:
        dict: A dictionary with disease_id as keys and the corresponding JSON data as values.
    """
    json_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../json_data")
    
    if not os.path.exists(json_folder):
        raise Exception(f"Directory not found: {json_folder}")
    
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    
    json_data_list = []
    if not json_files:
        print("No JSON files found in the directory.")
        return json_data_list
    
    for json_file in json_files:
        file_path = os.path.join(json_folder, json_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data = preprocess_json_data(data)  # Preprocess the data
                json_data_list.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {str(e)}")
    
    return json_data_list

# Modify the embedding function to handle batching
def embedding_json_data(vector_store, json_chunks):
    """
    Embed JSON chunks and store them in the vector database.
    
    Args:
        vector_store: The vector store for embedding
        json_chunks (list): List of JSON chunks to be embedded
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Prepare all texts and metadata
        all_texts = []
        all_metadatas = []
        
        for chunk in json_chunks:
            # Convert chunk to string for embedding
            chunk_text = json.dumps(chunk, ensure_ascii=False)
            
            # Extract metadata for retrieval
            metadata = {
                "disease_id": chunk.get("disease_id", ""),
                "disease_name": chunk.get("disease_name", "")
            }
            
            # If the chunk is a section, add section information
            if "section" in chunk:
                metadata["section"] = chunk["section"]
            
            all_texts.append(chunk_text)
            all_metadatas.append(metadata)
        
        # Process in batches to stay within token limits
        batch_size = 100  # Adjust this based on your average token count per document
        total_added = 0
        
        print(f"Processing {len(all_texts)} chunks in batches of {batch_size}...")
        
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i+batch_size]
            batch_metadatas = all_metadatas[i:i+batch_size]
            
            try:
                # Add batch to vector store
                vector_store.add_texts(texts=batch_texts, metadatas=batch_metadatas)
                total_added += len(batch_texts)
                print(f"Processed batch {i//batch_size + 1}/{len(all_texts)//batch_size + 1} - {total_added}/{len(all_texts)} documents")
            except Exception as batch_error:
                print(f"Error embedding batch {i//batch_size + 1}: {str(batch_error)}")
                
                # If the batch is too big, try with smaller batches
                if "max_tokens" in str(batch_error).lower() and batch_size > 10:
                    smaller_batch_size = batch_size // 2
                    print(f"Trying with smaller batch size of {smaller_batch_size}...")
                    
                    for j in range(i, min(i+batch_size, len(all_texts)), smaller_batch_size):
                        try:
                            sub_batch_texts = all_texts[j:j+smaller_batch_size]
                            sub_batch_metadatas = all_metadatas[j:j+smaller_batch_size]
                            vector_store.add_texts(texts=sub_batch_texts, metadatas=sub_batch_metadatas)
                            total_added += len(sub_batch_texts)
                            print(f"Processed sub-batch - {total_added}/{len(all_texts)} documents")
                        except Exception as sub_batch_error:
                            print(f"Error embedding sub-batch: {str(sub_batch_error)}")
        
        # Check if persist method exists before calling it
        # Different Chroma versions handle persistence differently
        try:
            if hasattr(vector_store, 'persist'):
                vector_store.persist()
            elif hasattr(vector_store, '_persist'):
                vector_store._persist()
            elif hasattr(vector_store, '_collection') and hasattr(vector_store._collection, 'persist'):
                vector_store._collection.persist()
            else:
                # New versions of Chroma might persist automatically
                print("No persist method found, Chroma may persist data automatically")
        except Exception as e:
            print(f"Warning: Could not persist vector store: {str(e)}")
        
        print(f"Successfully embedded {total_added} out of {len(all_texts)} documents")
        return total_added > 0
    except Exception as e:
        print(f"Error embedding JSON data: {str(e)}")
        return False

def setup_rag_system(vector_store, language="vietnamese"):
    """
    Set up the RAG system for medical question answering.
    
    Args:
        vector_store: The vector store to use for retrieval
        language (str): The language to use for the response
        
    Returns:
        tuple: (qa_chain, enhanced_retriever) - The QA chain and the enhanced retriever
    """
    # Initialize language model with more specific parameters for compatibility
    llm = ChatOpenAI(
        model_name="gpt-4.1-nano",  # Try with a more current model
        temperature=0.0,
        verbose=True  # Enable verbose mode to see what's happening
    )
    
    # Modify the prompt template to use "input_language" consistently throughout
    # This helps avoid confusion with the variable name "language"
    prompt_template = """
    You are an advanced medical assistant specializing in Vietnamese diseases with access to a comprehensive medical database. Your purpose is to provide accurate, helpful, and contextually relevant information.

    You have two primary functions:
    1. DISEASE INFORMATION: When users ask about specific diseases, provide detailed, organized information
    2. DISEASE PREDICTION: When users describe symptoms, analyze them and suggest potential diseases that match

    CONTEXT INFORMATION:
    {context}

    USER QUERY:
    {question}

    RESPONSE GUIDELINES:
    - Respond ONLY in {input_language} language
    - Be compassionate but professional in your tone
    - For serious conditions, encourage seeking professional medical advice
    - NEVER invent or hallucinate information not present in the context provided
    - If information is incomplete or unavailable, acknowledge limitations clearly
    - Avoid medical jargon when possible; explain specialized terms if necessary
    - Maintain cultural sensitivity relevant to Vietnamese medical practices

    FOR DISEASE INFORMATION QUERIES:
    - Provide a concise definition of the disease first
    - Organize information into clear sections: Symptoms, Causes, Diagnosis, Treatment, Prevention
    - Include common local names or alternative names used in Vietnam if available
    - Highlight important warning signs that require urgent medical attention
    - Mention relevant statistics or epidemiology specific to Vietnam when available

    FOR SYMPTOM-BASED QUERIES:
    - Summarize and confirm the symptoms described by the user
    - List potential diseases in order of likelihood based on the symptoms
    - For each disease suggestion, clearly explain WHY it matches the symptoms described
    - Include severity assessment and urgency indicators when appropriate
    - Suggest what additional symptoms might help narrow down the diagnosis
    - Recommend appropriate types of medical specialists to consult if needed
    - Provide practical next steps based on symptom severity
    
    YOUR RESPONSE ({input_language}):
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "input_language"],
        partial_variables={"input_language": language}
    )
    
    # Create a standard retriever with compatible configuration
    standard_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # Create the RAG chain with the standard retriever
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=standard_retriever,
        return_source_documents=True,
        verbose=True,
        chain_type_kwargs={
            "prompt": PROMPT,
            "document_separator": "\n\n---\n\n",
            "verbose": True
        }
    )
    
    # Create an enhanced retriever with the vector store
    enhanced_retriever = setup_enhanced_retrieval(vector_store)
    
    return qa_chain, enhanced_retriever

def retrieve_relevant_documents(query: str, retriever=None, enhanced_retriever=None, language: str = "vietnamese", k: int = 5) -> List[Document]:
    """
    Retrieve relevant documents based on the query.
    
    Args:
        query (str): The user query
        retriever: The standard retriever object
        enhanced_retriever: The enhanced retriever object (optional)
        language (str): The language of the query
        k (int): Number of documents to retrieve
        
    Returns:
        List[Document]: List of relevant documents
    """
    try:
        # Process query based on query type and language
        symptom_based = is_symptom_query(query, language)
        
        processed_query = query
        if symptom_based:
            if language == "vietnamese":
                processed_query = f"Triệu chứng: {query}"
            else:
                processed_query = f"Symptoms: {query}"
                
            # For symptom queries, we might want more documents
            k = max(k, 7)
        
        # Log the retrieval query
        logger.info(f"RETRIEVAL QUERY: {processed_query} (symptom-based: {symptom_based}, language: {language})")
        
        # Use enhanced retrieval if available
        if enhanced_retriever is not None:
            try:
                # Use the enhanced retriever directly
                relevant_docs = enhanced_retriever.retrieve(
                    processed_query, 
                    language=language, 
                    k=k
                )
                logger.info("Using enhanced retrieval")
            except Exception as e:
                logger.error(f"Enhanced retrieval failed: {str(e)}, falling back to standard retrieval")
                relevant_docs = retriever.get_relevant_documents(processed_query, k=k)
        else:
            # Fallback to standard retriever
            relevant_docs = retriever.get_relevant_documents(processed_query, k=k)
        
        # Log retrieved document count and metadata
        logger.info(f"RETRIEVED {len(relevant_docs)} DOCUMENTS")
        for i, doc in enumerate(relevant_docs):
            meta = doc.metadata if hasattr(doc, "metadata") else {}
            logger.info(f"  Doc {i+1}: {meta.get('disease_name', 'Unknown')} - {meta.get('section', 'General')}")
        
        return relevant_docs
    except Exception as e:
        logger.error(f"Error in retrieval: {str(e)}")
        return []

def format_retrieved_documents(docs: List[Document], language: str = "vietnamese") -> str:
    """
    Format retrieved documents for display.
    
    Args:
        docs (List[Document]): The retrieved documents
        language (str): The language for formatting
        
    Returns:
        str: Formatted document summary
    """
    if not docs:
        return "No relevant documents found."
    
    # Format based on language
    header = "Tài liệu tham khảo:" if language == "vietnamese" else "Reference Documents:"
    
    formatted_text = [header]
    
    for i, doc in enumerate(docs):
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        disease_name = meta.get("disease_name", "Unknown disease")
        section = meta.get("section", "General information")
        
        # Format based on language
        if language == "vietnamese":
            formatted_text.append(f"{i+1}. Bệnh: {disease_name}, Phần: {section}")
        else:
            formatted_text.append(f"{i+1}. Disease: {disease_name}, Section: {section}")
    
    return "\n".join(formatted_text)

def is_symptom_query(query, language="vietnamese"):
    """
    Determine if a query is asking about symptoms to predict a disease.
    
    Args:
        query (str): The user query
        language (str): The language of the query
        
    Returns:
        bool: True if the query is symptom-based, False otherwise
    """
    # Common phrases that indicate symptom queries in Vietnamese
    vn_symptom_phrases = [
        "triệu chứng", "dấu hiệu", "bị đau", "cảm thấy", "có những", 
        "tôi bị", "tôi có", "tôi đang", "tôi thấy", "bệnh gì",
        "mắc bệnh", "mắc phải", "tôi bị làm sao", "tôi bị sao",
        "chẩn đoán", "có phải tôi bị", "nghi ngờ bị", "đau", "nhức",
        "sốt", "nóng", "lạnh", "phát ban", "nổi mẩn", "buồn nôn",
        "nôn", "mệt mỏi", "mệt", "khó thở", "ho", "đau đầu",
        "chóng mặt", "ăn không ngon"
    ]
    
    # Common phrases that indicate symptom queries in English
    en_symptom_phrases = [
        "symptom", "sign", "feel", "having", "suffering from",
        "i have", "i am", "i feel", "what disease", "diagnose",
        "do i have", "might i have", "could i have", "is it possible",
        "pain", "ache", "fever", "hot", "cold", "rash", "nausea",
        "vomiting", "tired", "fatigue", "breathing", "cough", "headache",
        "dizzy", "no appetite", "hurts", "sore", "swollen", "swelling"
    ]
    
    symptom_phrases = vn_symptom_phrases if language == "vietnamese" else en_symptom_phrases
    
    # Check if any of the phrases are in the query
    lower_query = query.lower()
    return any(phrase in lower_query for phrase in symptom_phrases)

def query_rag_system(qa_chain, query, enhanced_retriever=None, language="vietnamese", show_references=False):
    """
    Query the RAG system with enhanced handling for symptom-based queries.
    
    Args:
        qa_chain: The RAG system chain
        query (str): User query
        enhanced_retriever: The enhanced retriever (optional)
        language (str): The language to use for the response
        show_references (bool): Whether to show reference documents
        
    Returns:
        dict: Response from the RAG system with additional retrieval information
    """
    try:
        # Get the standard retriever from the chain
        retriever = qa_chain.retriever
        
        # First, explicitly retrieve relevant documents
        relevant_docs = retrieve_relevant_documents(
            query=query,
            retriever=retriever,
            enhanced_retriever=enhanced_retriever,
            language=language,
            k=5 if not is_symptom_query(query, language) else 7
        )
        
        # Check if this is a symptom-based query
        symptom_based = is_symptom_query(query, language)
        
        # For symptom-based queries, enhance the query to focus on symptoms
        if symptom_based:
            # Determine appropriate phrasing based on language
            if language == "vietnamese":
                enhanced_query = f"Phân tích các triệu chứng sau và dự đoán bệnh có thể mắc phải: {query}"
            else:
                enhanced_query = f"Analyze the following symptoms and predict possible diseases: {query}"
                
            # Use invoke method instead of direct calling, with input_language parameter
            try:
                result = qa_chain.invoke({
                    "query": enhanced_query,
                    "input_language": language,  # Changed from "language" to "input_language"
                })
            except AttributeError:
                # Fallback for older versions of LangChain
                result = qa_chain({"query": enhanced_query, "input_language": language})
        else:
            # Regular disease information query
            try:
                result = qa_chain.invoke({
                    "query": query, 
                    "input_language": language,  # Changed from "language" to "input_language"
                })
            except AttributeError:
                # Fallback for older versions of LangChain
                result = qa_chain({"query": query, "input_language": language})
        
        # Add retrieval information to the result
        result["retrieved_documents"] = relevant_docs
        result["formatted_references"] = format_retrieved_documents(relevant_docs, language)
        
        # Log the completion
        logger.info(f"QUERY COMPLETED: {query}")
        
        return result
    except Exception as e:
        error_msg = f"Error querying RAG system: {str(e)}"
        logger.error(error_msg)
        # Print the full error stack trace for debugging
        import traceback
        logger.error(traceback.format_exc())
        return {
            "result": f"Error processing your query. ({language}): {str(e)}",
            "retrieved_documents": [],
            "formatted_references": "",
            "error": error_msg
        }

def main():
    parser = argparse.ArgumentParser(description="Medical RAG System")
    parser.add_argument("--vector-store", choices=["chroma", "pgvector"], default="chroma",
                        help="Vector store backend to use (default: chroma)")
    parser.add_argument("--db-path", default="./chroma_langchain_db",
                        help="Path to the database (for Chroma)")
    parser.add_argument("--collection", default="example_collection",
                        help="Collection name")
    parser.add_argument("--conn-string", default=None,
                        help="Database connection string (for PGVector)")
    parser.add_argument("--language", choices=["vietnamese", "english"], default="vietnamese",
                        help="Response language")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--query", default=None,
                        help="Single query to run (non-interactive mode)")
    
    args = parser.parse_args()
    
    # Check if database exists
    database_exists = check_vector_store_exists(
        store_type=args.vector_store,
        persist_directory=args.db_path if args.vector_store == "chroma" else None,
        collection_name=args.collection,
        connection_string=args.conn_string if args.vector_store == "pgvector" else None
    )
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Create the vector store
    print(f"Using {args.vector_store} as vector store backend")
    vector_store = create_vector_store(
        store_type=args.vector_store,
        embedding_function=embeddings,
        persist_directory=args.db_path if args.vector_store == "chroma" else None,
        collection_name=args.collection,
        connection_string=args.conn_string if args.vector_store == "pgvector" else None
    )
    
    # Only process and embed data if the database doesn't exist
    if not database_exists:
        print(f"No existing {args.vector_store} database found. Creating new vector database...")
        
        print("Loading JSON data...")
        json_data_list = load_all_json_data()
        
        # Process and embed all JSON data
        print("Processing and embedding data...")
        all_chunks = []
        for json_data in json_data_list:
            # Use the imported chunking_json_data function directly
            chunks = chunking_json_data(json_data)
            all_chunks.extend(chunks)
        
        if all_chunks:
            print(f"Generated {len(all_chunks)} chunks from {len(json_data_list)} documents")
            embedding_json_data(vector_store, all_chunks)
            print("Data embedded and stored successfully!")
    else:
        print(f"Using existing {args.vector_store} database")
    
    # Set up RAG system
    print("Setting up RAG system...")
    qa_chain, enhanced_retriever = setup_rag_system(vector_store, args.language)
    
    # Language selection (if in interactive mode or not specified)
    language = args.language
    if args.interactive and not args.query:
        print("\nSelect response language:")
        print("1. Vietnamese")
        print("2. English")
        language_choice = input("Enter your choice (1/2, default is 1): ").strip()
        language = "vietnamese" if language_choice != "2" else "english"
        print(f"\nLanguage set to: {language.capitalize()}")
    
    # Run in interactive mode or process a single query
    if args.interactive and not args.query:
        # Interactive query mode
        print("\nMedical Assistant Ready! Type 'exit' to quit, 'language' to change language, or 'refs' to toggle references.")
        show_references = False
        
        while True:
            # Get user input
            query = input("\nYour question: ")
            if query.lower() in ["exit", "quit"]:
                break
            elif query.lower() == "language":
                print("\nSelect response language:")
                print("1. Vietnamese")
                print("2. English")
                language_choice = input("Enter your choice (1/2): ").strip()
                language = "vietnamese" if language_choice != "2" else "english"
                print(f"\nLanguage set to: {language.capitalize()}")
                continue
            elif query.lower() in ["refs", "references", "sources"]:
                show_references = not show_references
                status = "enabled" if show_references else "disabled"
                print(f"\nReference display {status}")
                continue
            
            # Add a try/except block for extra robustness
            try:
                # Query the system with the enhanced retriever passed as a separate parameter
                print("Sending query to RAG system...")
                response = query_rag_system(qa_chain, query, enhanced_retriever, language, show_references)
                
                # Check if we got a valid result or an error
                if "error" in response and response["error"]:
                    print(f"\nError details: {response['error']}")
                    
                # Print the assistant's response
                print(f"\nAssistant ({language.capitalize()}):", response["result"])
                
                # Show references if enabled
                if show_references and "formatted_references" in response:
                    print("\n" + response["formatted_references"])
            except Exception as e:
                print(f"\nUnhandled error: {str(e)}")
                import traceback
                print(traceback.format_exc())
                print("Please try again with a different query.")
        
        print("\nThank you for using the Medical Assistant. Goodbye!")
    else:
        # Process a single query
        query = args.query or "Tôi bị đau đầu, sốt và đau cơ, tôi có thể bị bệnh gì?"
        print(f"\nProcessing query: {query}")
        
        try:
            # Query the system
            response = query_rag_system(qa_chain, query, enhanced_retriever, language, show_references=True)
            
            # Print the assistant's response
            print(f"\nAssistant ({language.capitalize()}):", response["result"])
            
            # Show references
            if "formatted_references" in response:
                print("\n" + response["formatted_references"])
        except Exception as e:
            print(f"\nError: {str(e)}")
    
    print("\nThank you for using the Medical Assistant. Goodbye!")

if __name__ == "__main__":
    main()