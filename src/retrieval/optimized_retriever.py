# src/retrieval/optimized_retriever.py
import asyncio
import time
from functools import lru_cache
from typing import List, Dict, Optional
import hashlib
import pickle
import os
from src.retrieval.retriever import Retriever

class OptimizedRetriever(Retriever):
    """
    Drop-in replacement for Retriever with in-memory caching and async operations.
    No external dependencies - just faster!
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # In-memory caches
        self.embedding_cache = {}      # query_hash -> embedding
        self.search_results_cache = {} # query_hash -> results
        self.max_cache_size = 1000     # Prevent memory bloat
        
        # Pre-load embedding model to avoid cold starts
        print("Pre-loading embedding model...")
        self._embedding_model_instance = self._get_embedding_model_instance()
        print("✓ Embedding model pre-loaded")
        
        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _get_cache_key(self, query_text: str, search_params: dict = None) -> str:
        """Generate cache key for query + parameters"""
        params_str = str(sorted(search_params.items())) if search_params else ""
        content = f"{query_text}:{params_str}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _manage_cache_size(self, cache_dict: dict):
        """Simple LRU cache management - remove oldest entries"""
        if len(cache_dict) > self.max_cache_size:
            # Remove oldest 20% of entries
            items_to_remove = len(cache_dict) - int(self.max_cache_size * 0.8)
            keys_to_remove = list(cache_dict.keys())[:items_to_remove]
            for key in keys_to_remove:
                del cache_dict[key]
    
    def _embed_query_cached(self, query_text: str) -> Optional[List[float]]:
        """Cached version of query embedding"""
        cache_key = self._get_cache_key(query_text)
        
        # Check cache first
        if cache_key in self.embedding_cache:
            self.cache_hits += 1
            return self.embedding_cache[cache_key]
        
        # Generate embedding
        self.cache_misses += 1
        embedding = super()._embed_query(query_text)
        
        if embedding:
            # Store in cache with size management
            self.embedding_cache[cache_key] = embedding
            self._manage_cache_size(self.embedding_cache)
        
        return embedding
    
    async def fast_semantic_search(self, 
                                  query_text: str,
                                  limit: Optional[int] = None,
                                  certainty: Optional[float] = None,
                                  **kwargs) -> List[Dict]:
        """
        Async semantic search with result caching
        """
        # Create cache key with all parameters
        search_params = {
            'limit': limit or self.default_limit,
            'certainty': certainty or self.default_semantic_certainty,
            'search_type': 'semantic'
        }
        cache_key = self._get_cache_key(query_text, search_params)
        
        # Check results cache
        if cache_key in self.search_results_cache:
            return self.search_results_cache[cache_key]
        
        # Run embedding in executor to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, 
            self._embed_query_cached, 
            query_text
        )
        
        if not embedding:
            return []
        
        # Execute search in executor
        search_params_for_weaviate = {
            'limit': search_params['limit'],
            'certainty': search_params['certainty']
        }
        
        results = await loop.run_in_executor(
            None,
            self._execute_weaviate_search,
            embedding,
            search_params_for_weaviate
        )
        
        # Cache results
        self.search_results_cache[cache_key] = results
        self._manage_cache_size(self.search_results_cache)
        
        return results
    
    def _execute_weaviate_search(self, embedding: List[float], params: dict) -> List[Dict]:
        """Execute Weaviate search synchronously (for use in executor)"""
        near_vector_filter = {"vector": embedding, "certainty": params['certainty']}
        
        try:
            query_chain = self.client.query.get(self.weaviate_class_name, self.DEFAULT_RETURN_PROPERTIES)
            query_chain = query_chain.with_near_vector(near_vector_filter)
            query_chain = query_chain.with_limit(params['limit'])
            query_chain = query_chain.with_additional(self.DEFAULT_ADDITIONAL_PROPERTIES)
            
            response = query_chain.do()
            return self._format_results(response)
        except Exception as e:
            print(f"Search error: {e}")
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
        embedding_task = loop.run_in_executor(None, self._embed_query_cached, query_text)
        
        embedding = await embedding_task
        if not embedding:
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
    
    def _execute_hybrid_search(self, query_text: str, embedding: List[float], params: dict) -> List[Dict]:
        """Execute hybrid search synchronously"""
        hybrid_params = {
            "query": query_text,
            "alpha": params['alpha'],
            "properties": self.default_hybrid_bm25_properties,
            "vector": embedding
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
            "results_cache_size": len(self.search_results_cache)
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self.embedding_cache.clear()
        self.search_results_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        print("All caches cleared")


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