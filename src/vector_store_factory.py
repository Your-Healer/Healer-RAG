from langchain_chroma import Chroma
from typing import Literal, Optional, Dict, Any
import os
import logging
from pgvector_store import create_pg_vector_store, get_pg_connection_string, check_pgvector_connection

logger = logging.getLogger('retrieval')

def create_vector_store(
    store_type: Literal["chroma", "pgvector"] = "chroma",
    embedding_function = None,
    persist_directory: Optional[str] = None,
    collection_name: str = "example_collection",
    connection_string: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Factory function to create a vector store of the specified type.
    
    Args:
        store_type: Type of vector store to create (chroma or pgvector)
        embedding_function: Embedding function to use
        persist_directory: Directory for persistent storage (for Chroma)
        collection_name: Name of the collection
        connection_string: PostgreSQL connection string (for PGVector)
        **kwargs: Additional arguments for the vector store
        
    Returns:
        The created vector store
    """
    if store_type == "chroma":
        logger.info(f"Creating Chroma vector store with collection: {collection_name}")
        return Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory,
            **kwargs
        )
    elif store_type == "pgvector":
        logger.info(f"Creating PGVector store with collection: {collection_name}")
        # Use provided connection string or generate one from environment variables
        conn_str = connection_string or get_pg_connection_string()
        return create_pg_vector_store(
            connection_string=conn_str,
            embedding_function=embedding_function,
            collection_name=collection_name
        )
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")

def check_vector_store_exists(
    store_type: Literal["chroma", "pgvector"] = "chroma",
    persist_directory: Optional[str] = None,
    collection_name: str = "example_collection",
    connection_string: Optional[str] = None
) -> bool:
    """
    Check if a vector store exists.
    
    Args:
        store_type: Type of vector store to check
        persist_directory: Directory for persistent storage (for Chroma)
        collection_name: Name of the collection
        connection_string: PostgreSQL connection string (for PGVector)
        
    Returns:
        bool: Whether the vector store exists
    """
    if store_type == "chroma":
        # Check if directory exists and is not empty
        if not persist_directory:
            return False
        
        return os.path.exists(persist_directory) and os.path.isdir(persist_directory) and len(os.listdir(persist_directory)) > 0
    
    elif store_type == "pgvector":
        # For PGVector, check connection health
        is_healthy, message = check_pgvector_connection(connection_string)
        logger.info(f"PGVector health check: {message}")
        
        # This only checks if we can connect, not if the collection exists
        # A full check would require initializing the store and checking collections
        return is_healthy
    
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")
