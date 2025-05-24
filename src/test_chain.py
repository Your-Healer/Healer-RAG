import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma

load_dotenv()

def test_simple_chain():
    """
    Test a simple chain to verify that LangChain is working correctly.
    This helps isolate issues with our RAG implementation.
    """
    # Initialize basic components
    print("Initializing test chain...")
    
    # Use a small model for testing
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    # Create a simple prompt
    prompt = PromptTemplate(
        input_variables=["question"],
        template="Answer this question: {question}"
    )
    
    # Try a simple query
    print("Testing LLM directly...")
    try:
        response = llm.invoke("What is the capital of France?")
        print(f"LLM Response: {response.content}")
        print("✅ Direct LLM call successful")
    except Exception as e:
        print(f"❌ Error calling LLM directly: {str(e)}")
        return
    
    # Try connecting to the vector store
    print("\nTesting vector store connection...")
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=embeddings,
            persist_directory="./chroma_langchain_db"
        )
        print("✅ Vector store connection successful")
    except Exception as e:
        print(f"❌ Error connecting to vector store: {str(e)}")
        return
    
    # Try creating a retriever
    print("\nTesting retriever...")
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 1})
        docs = retriever.get_relevant_documents("test query")
        print(f"Retrieved {len(docs)} documents")
        print("✅ Retriever test successful")
    except Exception as e:
        print(f"❌ Error using retriever: {str(e)}")
        return
    
    # Try creating a full chain
    print("\nTesting full retrieval chain...")
    try:
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            verbose=True
        )
        print("✅ Chain creation successful")
    except Exception as e:
        print(f"❌ Error creating chain: {str(e)}")
        return
    
    # Try running the chain
    print("\nTesting chain execution...")
    try:
        # Try with invoke method
        result = chain.invoke({"query": "What is a common disease?"})
        print(f"Chain response: {result}")
        print("✅ Chain execution successful")
    except Exception as e:
        print(f"❌ Error executing chain: {str(e)}")
        try:
            # Try with call method as fallback
            result = chain({"query": "What is a common disease?"})
            print(f"Chain response (using call method): {result}")
            print("✅ Chain execution successful with call method")
        except Exception as e2:
            print(f"❌ Error executing chain with call method: {str(e2)}")
            return
            
    print("\nAll tests completed. If you see failures, check your environment, dependencies, or API keys.")

if __name__ == "__main__":
    test_simple_chain()
