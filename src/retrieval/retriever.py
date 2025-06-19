# src/retrieval/optimized_retriever.py
import asyncio
import time
from functools import lru_cache
from typing import List, Dict, Optional, Any
import hashlib
import pickle
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from src.data_ingestion.vector_store_manager import VectorStoreManager

class HybridRetriever:
    """
    A high-performance retriever with in-memory caching and async operations.
    Features:
    - Embedding caching for faster repeated queries
    - Async operations for better concurrency
    - Hybrid search capabilities
    - Performance tracking and statistics
    """
    
    def __init__(self, weaviate_client, cache_size: int = 1000):
        self.weaviate_client = weaviate_client
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_cache = {}
        self.cache_size = cache_size
        self._cache_lock = asyncio.Lock()
        print("HybridRetriever initialized with embedding cache")
        
        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize search results cache
        self.search_results_cache = {}
    
    def _get_cache_key(self, query_text: str, search_params: dict = None) -> str:
        """Generate cache key for query + parameters"""
        params_str = str(sorted(search_params.items())) if search_params else ""
        content = f"{query_text}:{params_str}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _manage_cache_size(self, cache_dict: dict):
        """Simple LRU cache management - remove oldest entries"""
        if len(cache_dict) > self.cache_size:
            # Remove oldest 20% of entries
            items_to_remove = len(cache_dict) - int(self.cache_size * 0.8)
            keys_to_remove = list(cache_dict.keys())[:items_to_remove]
            for key in keys_to_remove:
                del cache_dict[key]
    
    async def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache with thread safety"""
        async with self._cache_lock:
            return self.embedding_cache.get(text)

    async def _cache_embedding(self, text: str, embedding: np.ndarray):
        """Cache embedding with thread safety and size limit"""
        async with self._cache_lock:
            # If cache is full, remove oldest item
            if len(self.embedding_cache) >= self.cache_size:
                oldest_key = next(iter(self.embedding_cache))
                del self.embedding_cache[oldest_key]
            self.embedding_cache[text] = embedding

    async def _embed_query(self, query_text: str) -> np.ndarray:
        """Get embedding for query text, using cache if available"""
        # Check cache first
        cached_embedding = await self._get_cached_embedding(query_text)
        if cached_embedding is not None:
            print("Cache hit, using cached embedding")
            return cached_embedding

        # Generate new embedding
        print("Cache miss, generating embedding...")
        try:
            # Run the CPU-bound embedding generation in a thread pool
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: self.embedding_model.encode(query_text, convert_to_tensor=False)
            )
            # Cache the new embedding
            await self._cache_embedding(query_text, embedding)
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise
    
    async def get_all_documents(self, limit: int = 1000) -> List[Dict]:
        """Get all documents without semantic search"""
        try:
            print("Fetching all documents from Weaviate...")
            collection = self.weaviate_client.collections.get("MathConcept")
            response = collection.query.fetch_objects(limit=limit)
            
            print(f"Raw Weaviate response (first 500 chars): {str(response)[:500]}")
            
            if not response or not hasattr(response, 'objects'):
                print("Invalid response from Weaviate")
                return []
                
            documents = []
            for obj in response.objects:
                doc = {
                    'chunk_id': obj.properties.get('chunk_id'),
                    'doc_id': obj.properties.get('doc_id'),
                    'filename': obj.properties.get('filename'),
                    'concept_name': obj.properties.get('concept_name'),
                    'concept_type': obj.properties.get('concept_type'),
                    'source_path': obj.properties.get('source_path'),
                    'original_doc_type': obj.properties.get('original_doc_type'),
                    'parent_block_id': obj.properties.get('parent_block_id'),
                    'sequence_in_block': obj.properties.get('sequence_in_block'),
                    'id': str(obj.uuid)
                }
                documents.append(doc)
            
            print(f"Found {len(documents)} documents")
            return documents
            
        except Exception as e:
            print(f"Error getting all documents: {e}")
            return []

    async def fast_semantic_search(self, query_text: str, limit: int = 5) -> List[Dict]:
        """Perform semantic search with caching"""
        try:
            print(f"Starting fast_semantic_search for query: {query_text}")
            
            # Get embedding
            embedding = await self._embed_query(query_text)
            
            # Perform search
            results = await self._execute_weaviate_search(embedding, limit)
            
            print(f"Found {len(results)} results for query: {query_text}")
            return results
            
        except Exception as e:
            print(f"Error in fast_semantic_search: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return []
    
    async def _execute_weaviate_search(self, embedding: np.ndarray, limit: int) -> List[Dict]:
        """Execute Weaviate search with error handling"""
        try:
            print("Executing Weaviate search...")
            collection = self.weaviate_client.collections.get("MathConcept")
            response = collection.query.near_vector(
                near_vector=embedding.tolist(),
                limit=limit
            )
            
            print(f"Raw Weaviate search response (first 500 chars): {str(response)[:500]}")
            
            if not response or not hasattr(response, 'objects'):
                print("Invalid response from Weaviate")
                return []
                
            results = []
            for obj in response.objects:
                result = {
                    'chunk_id': obj.properties.get('chunk_id'),
                    'doc_id': obj.properties.get('doc_id'),
                    'filename': obj.properties.get('filename'),
                    'concept_name': obj.properties.get('concept_name'),
                    'concept_type': obj.properties.get('concept_type'),
                    'source_path': obj.properties.get('source_path'),
                    'original_doc_type': obj.properties.get('original_doc_type'),
                    'parent_block_id': obj.properties.get('parent_block_id'),
                    'sequence_in_block': obj.properties.get('sequence_in_block'),
                    'id': str(obj.uuid)
                }
                results.append(result)
            
            print(f"Found {len(results)} results")
            return results
            
        except Exception as e:
            print(f"Error executing Weaviate search: {e}")
            return []
    
    async def fast_hybrid_search(self, 
                                query_text: str,
                                alpha: Optional[float] = None,
                                limit: Optional[int] = None,
                                **kwargs) -> List[Dict]:
        """Async hybrid search with caching"""
        search_params = {
            'alpha': alpha or self.default_hybrid_alpha,
            'limit': limit or self.default_limit,
            'search_type': 'hybrid'
        }
        cache_key = self._get_cache_key(query_text, search_params)
        
        if cache_key in self.search_results_cache:
            return self.search_results_cache[cache_key]
        
        # Run in parallel: embedding + keyword prep
        loop = asyncio.get_event_loop()
        embedding_task = loop.run_in_executor(None, self._embed_query, query_text)
        
        embedding = await embedding_task
        if not embedding.any():
            return []
        
        # Execute hybrid search
        results = await loop.run_in_executor(
            None,
            self._execute_hybrid_search,
            query_text,
            embedding,
            search_params
        )
        
        # Cache results
        self.search_results_cache[cache_key] = results
        self._manage_cache_size(self.search_results_cache)
        
        return results
    
    def _execute_hybrid_search(self, query_text: str, embedding: np.ndarray, params: dict) -> List[Dict]:
        """Execute hybrid search synchronously"""
        hybrid_params = {
            "query": query_text,
            "alpha": params['alpha'],
            "properties": self.default_hybrid_bm25_properties,
            "vector": embedding.tolist()
        }
        
        try:
            query_chain = self.client.query.get(self.weaviate_class_name, self.DEFAULT_RETURN_PROPERTIES)
            query_chain = query_chain.with_hybrid(**hybrid_params)
            query_chain = query_chain.with_limit(params['limit'])
            query_chain = query_chain.with_additional(self.DEFAULT_ADDITIONAL_PROPERTIES)
            
            response = query_chain.do()
            return self._format_results(response)
        except Exception as e:
            print(f"Hybrid search error: {e}")
            return []
    
    def get_cache_stats(self) -> dict:
        """Get cache performance statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "embedding_cache_size": len(self.embedding_cache),
            "results_cache_size": len(self.search_results_cache),
            "max_cache_size": self.cache_size
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self.embedding_cache.clear()
        self.search_results_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        print("All caches cleared")

    def _format_results(self, weaviate_response: Dict, requested_properties: Optional[List[str]] = None) -> List[Dict]:
        """
        Formats the raw response from Weaviate into a list of result dictionaries.

        Args:
            weaviate_response: The raw JSON response from Weaviate's .do() call.
            requested_properties: The list of properties that were asked for in the query.
                                  If None, uses self.DEFAULT_RETURN_PROPERTIES.
        """
        results = []
        props_to_extract = requested_properties if requested_properties else self.DEFAULT_RETURN_PROPERTIES

        if not weaviate_response or "data" not in weaviate_response or \
           "Get" not in weaviate_response["data"] or not weaviate_response["data"]["Get"].get(self.weaviate_class_name):
            if weaviate_response and "errors" in weaviate_response:
                 print(f"Retriever (_format_results): Weaviate query returned errors: {weaviate_response['errors']}")
            return results

        retrieved_items = weaviate_response["data"]["Get"][self.weaviate_class_name]
        for item in retrieved_items:
            result = {}
            for prop in props_to_extract: 
                if prop in item: 
                    result[prop] = item[prop]
                else:
                    result[prop] = None 

            if "_additional" in item and item["_additional"] is not None:
                for add_prop in self.DEFAULT_ADDITIONAL_PROPERTIES: 
                    if add_prop in item["_additional"]:
                        result[f"_{add_prop}"] = item["_additional"][add_prop]
            results.append(result)
        return results

    async def get_chunks_for_parent_block(self, parent_block_id: str, limit: Optional[int] = None) -> List[Dict]:
        print(f"Retriever: Fetching chunks for parent_block_id: {parent_block_id}")
        filters = {
            "operator": "Equal",
            "path": ["parent_block_id"],
            "valueText": parent_block_id
        }
        query_limit = limit if limit is not None else 50

        def _execute_query():
            query_chain = (
                self.weaviate_client.query
                .get(self.weaviate_class_name, self.DEFAULT_RETURN_PROPERTIES)
                .with_where(filters)
                .with_sort([{'path': ['sequence_in_block'], 'order': 'asc'}])
                .with_limit(query_limit)
                .with_additional(self.DEFAULT_ADDITIONAL_PROPERTIES)
            )
            response = query_chain.do()
            return self._format_results(response)

        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, _execute_query)
            print(f"Retriever: Found {len(results)} chunks for parent_block_id '{parent_block_id}'.")
            return results
        except Exception as e:
            print(f"Retriever: Error fetching chunks for parent_block_id '{parent_block_id}': {e}")
            return []


# src/data_ingestion/optimized_vector_store_manager.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
from src.data_ingestion.vector_store_manager import *

class OptimizedVectorStoreManager:
    """
    Optimized version that batches operations and uses async processing
    """
    
    @staticmethod
    async def fast_embed_and_store_chunks(client, final_text_chunks: list, batch_size: int = 50):
        """
        Optimized version with parallel embedding generation
        """
        if not final_text_chunks:
            print("No chunks to embed and store.")
            return

        print(f"Optimized embedding and storage for {len(final_text_chunks)} chunks...")
        create_weaviate_schema(client)

        # Pre-load embedding model
        embedding_model = get_embedding_model()
        
        # Process chunks in parallel batches
        loop = asyncio.get_event_loop()
        
        # Create thread pool for CPU-bound embedding work
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Split chunks into batches for parallel processing
            chunk_batches = [
                final_text_chunks[i:i+batch_size] 
                for i in range(0, len(final_text_chunks), batch_size)
            ]
            
            print(f"Processing {len(chunk_batches)} batches in parallel...")
            
            # Process batches concurrently
            batch_tasks = [
                loop.run_in_executor(
                    executor,
                    OptimizedVectorStoreManager._process_batch,
                    batch,
                    embedding_model
                )
                for batch in chunk_batches
            ]
            
            # Wait for all batches to complete embedding
            embedded_batches = await asyncio.gather(*batch_tasks)
            
            # Flatten results
            all_embedded_chunks = []
            for batch_result in embedded_batches:
                all_embedded_chunks.extend(batch_result)
        
        # Store all chunks in Weaviate
        print("Storing all embedded chunks in Weaviate...")
        client.batch.configure(batch_size=batch_size, dynamic=True, timeout_retries=3)
        
        with client.batch as batch_context:
            for chunk_data in all_embedded_chunks:
                if chunk_data['embedding'] is not None:
                    batch_context.add_data_object(
                        data_object=chunk_data['data_object'],
                        class_name=WEAVIATE_CLASS_NAME,
                        vector=chunk_data['embedding'],
                        uuid=chunk_data['uuid']
                    )
        
        print(f"✓ Optimized embedding and storage complete!")
    
    @staticmethod
    def _process_batch(chunk_batch: list, embedding_model) -> list:
        """
        Process a batch of chunks - generate embeddings in parallel
        """
        embedded_chunks = []
        
        # Extract texts for batch embedding
        texts = [chunk.get("chunk_text", "") for chunk in chunk_batch]
        valid_indices = [i for i, text in enumerate(texts) if text.strip()]
        valid_texts = [texts[i] for i in valid_indices]
        
        if not valid_texts:
            return embedded_chunks
        
        # Batch generate embeddings
        try:
            embeddings = embedding_model.encode(
                valid_texts,
                batch_size=32,
                convert_to_tensor=False,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            # Map embeddings back to chunks
            embedding_iter = iter(embeddings)
            
            for i, chunk_data in enumerate(chunk_batch):
                if i in valid_indices:
                    embedding = next(embedding_iter)
                    embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
                else:
                    embedding_list = None
                
                if embedding_list is not None:
                    # Prepare data object
                    source_file_path = chunk_data.get("source", "Unknown source")
                    file_name = chunk_data.get("filename", os.path.basename(source_file_path) if source_file_path != "Unknown source" else "unknown_filename")
                    
                    data_object = {
                        "chunk_id": str(chunk_data.get("chunk_id")),
                        "doc_id": chunk_data.get("doc_id", "unknown_doc_id"),
                        "source_path": source_file_path,
                        "original_doc_type": chunk_data.get("original_type", "unknown"),
                        "concept_type": chunk_data.get("concept_type", "general_content"),
                        "concept_name": chunk_data.get("concept_name"),
                        "chunk_text": chunk_data.get("chunk_text", ""),
                        "parent_block_id": chunk_data.get("parent_block_id"),
                        "parent_block_content": chunk_data.get("parent_block_content", ""),
                        "sequence_in_block": chunk_data.get("sequence_in_block", 0),
                        "filename": file_name
                    }
                    
                    embedded_chunks.append({
                        'embedding': embedding_list,
                        'data_object': {k: v for k, v in data_object.items() if v is not None},
                        'uuid': str(chunk_data.get("chunk_id"))
                    })
        
        except Exception as e:
            print(f"Error in batch processing: {e}")
        
        return embedded_chunks


# Convenience function to use optimized embedding
async def fast_embed_and_store_chunks(client, final_text_chunks: list, batch_size: int = 50):
    """
    Drop-in replacement for the original embed_and_store_chunks function
    """
    return await OptimizedVectorStoreManager.fast_embed_and_store_chunks(
        client, final_text_chunks, batch_size
    )