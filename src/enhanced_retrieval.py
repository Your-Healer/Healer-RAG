import logging
from typing import List, Dict, Any, Tuple
from langchain.schema import Document
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

logger = logging.getLogger('retrieval')

class EnhancedRetrieval:
    """Enhanced retrieval system with multi-stage search and re-ranking."""
    
    def __init__(self, vector_store):
        """
        Initialize the enhanced retrieval system.
        
        Args:
            vector_store: The vector store to use for retrieval
        """
        self.vector_store = vector_store
        self.query_cache = {}  # Simple cache for frequent queries
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True, 
            stop_words='english',
            ngram_range=(1, 2)
        )
        # Initialize TF-IDF from existing documents if available
        self._initialize_tfidf()
    
    def _initialize_tfidf(self):
        """Initialize TF-IDF vectorizer with existing documents."""
        try:
            # Try different methods to access documents based on vector store type
            docs = []
            
            # For Chroma
            if hasattr(self.vector_store, '_collection'):
                try:
                    collection = self.vector_store._collection
                    if hasattr(collection, 'get'):
                        results = collection.get(limit=1000)
                        docs = results.get('documents', [])
                except Exception as e:
                    logger.warning(f"Could not get documents from Chroma: {str(e)}")
            
            # For PGVector
            elif hasattr(self.vector_store, 'store'):
                try:
                    # For our custom PGVector wrapper
                    if hasattr(self.vector_store.store, 'collection_name'):
                        # We need a simple query to get some documents
                        # This is a workaround as PGVector doesn't have a direct "get all" method
                        sample_docs = self.vector_store.similarity_search("medicine", k=100)
                        docs = [doc.page_content for doc in sample_docs]
                except Exception as e:
                    logger.warning(f"Could not get documents from PGVector: {str(e)}")
            
            # Generic approach for any vector store that has get_relevant_documents
            if not docs and hasattr(self.vector_store, 'similarity_search'):
                try:
                    # Try a generic search query to get some documents
                    sample_docs = self.vector_store.similarity_search("medicine health disease", k=100)
                    docs = [doc.page_content for doc in sample_docs]
                except Exception as e:
                    logger.warning(f"Could not get documents from vector store: {str(e)}")
            
            if docs:
                self.tfidf_vectorizer.fit(docs)
                logger.info(f"TF-IDF vectorizer initialized with {len(docs)} documents")
            else:
                logger.warning("No documents found to initialize TF-IDF vectorizer")
                
        except Exception as e:
            logger.warning(f"Could not initialize TF-IDF vectorizer: {str(e)}")
    
    def _expand_query(self, query: str) -> List[str]:
        """
        Expand the original query with related terms.
        
        Args:
            query: The original query
            
        Returns:
            List of expanded queries
        """
        # Simple expansion for now - could be enhanced with medical ontologies
        expanded_queries = [query]
        
        # Add a more general version
        expanded_queries.append(f"general information about {query}")
        
        # For symptom queries, add variations
        if "symptoms" in query.lower() or "signs" in query.lower():
            expanded_queries.append(f"clinical manifestations of {query}")
        
        return expanded_queries
    
    def _hybrid_search(self, query: str, k: int = 10) -> List[Document]:
        """
        Perform hybrid search using both vector and keyword matching.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        # Vector search with direct similarity search
        # Use simple k parameter without fetch_k
        vector_docs = self.vector_store.similarity_search(query, k=k)
        
        # Extract keywords from query for filtering
        keywords = [w for w in query.lower().split() if len(w) > 3]
        
        # Score documents based on keyword presence
        scored_docs = []
        for doc in vector_docs:
            # Get document text
            doc_text = doc.page_content.lower()
            
            # Count keyword occurrences
            keyword_score = sum(doc_text.count(kw) for kw in keywords)
            
            # Calculate section relevance (custom scoring based on document sections)
            section_score = 0
            if hasattr(doc, 'metadata'):
                section = doc.metadata.get('section', '').lower()
                # Prioritize key sections
                if 'symptom' in section:
                    section_score += 3
                elif 'treatment' in section:
                    section_score += 2
                elif 'cause' in section or 'diagnosis' in section:
                    section_score += 1
            
            # Combined score
            total_score = keyword_score + section_score
            
            scored_docs.append((doc, total_score))
        
        # Sort by score and return documents
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:k]]
    
    def _rerank_documents(self, docs: List[Document], query: str) -> List[Document]:
        """
        Rerank documents based on multiple signals.
        
        Args:
            docs: List of retrieved documents
            query: The original query
            
        Returns:
            Reranked list of documents
        """
        if not docs:
            return []
            
        # Extract features for reranking
        rerank_features = []
        
        for doc in docs:
            features = {}
            
            # 1. Calculate semantic similarity (already done in vector search)
            features['semantic_similarity'] = 1.0  # Placeholder, already ranked by this
            
            # 2. Keyword matching
            query_terms = set(query.lower().split())
            content = doc.page_content.lower()
            term_matches = sum(1 for term in query_terms if term in content)
            features['keyword_match'] = term_matches / len(query_terms) if query_terms else 0
            
            # 3. Document section relevance
            section_relevance = 0.5  # Default
            if hasattr(doc, 'metadata') and 'section' in doc.metadata:
                section = doc.metadata['section'].lower()
                # Adjust based on query intent (simplified)
                if 'symptom' in query.lower() and 'symptom' in section:
                    section_relevance = 1.0
                elif 'treatment' in query.lower() and 'treatment' in section:
                    section_relevance = 1.0
                elif 'cause' in query.lower() and 'cause' in section:
                    section_relevance = 1.0
            features['section_relevance'] = section_relevance
            
            # 4. Document completeness (length as a proxy)
            features['completeness'] = min(1.0, len(content) / 1000)  # Cap at 1.0
            
            # Calculate final score (weighted average)
            final_score = (
                0.4 * features['semantic_similarity'] +
                0.3 * features['keyword_match'] +
                0.2 * features['section_relevance'] +
                0.1 * features['completeness']
            )
            
            rerank_features.append((doc, final_score))
        
        # Sort by final score
        rerank_features.sort(key=lambda x: x[1], reverse=True)
        
        # Log reranking results
        logger.info(f"Reranked {len(docs)} documents")
        
        # Return reranked documents
        return [doc for doc, _ in rerank_features]
    
    def retrieve(self, query: str, language: str = "vietnamese", k: int = 5) -> List[Document]:
        """
        Main retrieval method with enhanced capabilities.
        
        Args:
            query: The user query
            language: Query language
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        # Safety check for empty query
        if not query or query.strip() == "":
            logger.warning("Empty query provided to retrieval")
            return []
            
        # Check cache first
        cache_key = f"{query}_{language}_{k}"
        if cache_key in self.query_cache:
            logger.info(f"Cache hit for query: {query}")
            return self.query_cache[cache_key]
        
        try:
            # Preprocess query based on language
            if language == "vietnamese":
                # Ensure Unicode normalization for Vietnamese
                import unicodedata
                query = unicodedata.normalize('NFC', query)
            
            # Expand query
            expanded_queries = self._expand_query(query)
            logger.info(f"Expanded query '{query}' to {len(expanded_queries)} variations")
            
            # Multi-stage retrieval
            # Stage 1: Broad retrieval with expanded queries
            all_docs = []
            for expanded_query in expanded_queries:
                # Use hybrid search for each expanded query
                try:
                    query_docs = self._hybrid_search(expanded_query, k=k)
                    all_docs.extend(query_docs)
                except Exception as e:
                    logger.error(f"Error in hybrid search for '{expanded_query}': {str(e)}")
                    # Try direct search as fallback
                    try:
                        query_docs = self.vector_store.similarity_search(expanded_query, k=k)
                        all_docs.extend(query_docs)
                    except Exception as inner_e:
                        logger.error(f"Fallback search also failed: {str(inner_e)}")
            
            # If no documents were retrieved, return empty list
            if not all_docs:
                logger.warning(f"No documents retrieved for query: {query}")
                return []
                
            # Remove duplicates
            unique_docs = []
            seen_contents = set()
            for doc in all_docs:
                if doc.page_content not in seen_contents:
                    unique_docs.append(doc)
                    seen_contents.add(doc.page_content)
            
            # Stage 2: Rerank documents
            try:
                reranked_docs = self._rerank_documents(unique_docs, query)
            except Exception as e:
                logger.error(f"Error in reranking: {str(e)}")
                reranked_docs = unique_docs  # Use unranked docs as fallback
            
            # Take top k
            final_docs = reranked_docs[:k]
            
            # Update cache
            self.query_cache[cache_key] = final_docs
            
            # Log retrieval stats
            logger.info(f"Enhanced retrieval: {len(final_docs)} documents for query '{query}'")
            
            return final_docs
            
        except Exception as e:
            logger.error(f"Error in enhanced retrieval: {str(e)}")
            # Fallback to basic retrieval
            try:
                return self.vector_store.similarity_search(query, k=k)
            except Exception as fallback_e:
                logger.error(f"Fallback retrieval also failed: {str(fallback_e)}")
                return []  # Return empty list as last resort
    
    def retrieve_with_metadata_filter(self, query: str, filters: Dict[str, Any], k: int = 5) -> List[Document]:
        """
        Retrieve documents with metadata filtering.
        
        Args:
            query: The search query
            filters: Metadata filters to apply
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        try:
            docs = self.vector_store.similarity_search_with_filter(query, filter=filters, k=k)
            return docs
        except Exception as e:
            logger.error(f"Error in metadata-filtered retrieval: {str(e)}")
            return self.retrieve(query, k=k)

def setup_enhanced_retrieval(vector_store):
    """Create and return an enhanced retrieval system."""
    return EnhancedRetrieval(vector_store)
