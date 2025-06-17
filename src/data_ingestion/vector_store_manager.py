# src/data_ingestion/optimized_vector_store_manager.py
import asyncio
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import weaviate
from src import config

# Constants
WEAVIATE_CLASS_NAME = "MathConcept"
DEFAULT_RETURN_PROPERTIES = ["chunk_text", "source_path", "concept_type", "concept_name", "filename"]
DEFAULT_ADDITIONAL_PROPERTIES = ["distance"]

def get_weaviate_client():
    """Establishes a connection to the Weaviate instance and returns a client object."""
    weaviate_url = config.WEAVIATE_URL
    try:
        # Use the recommended way to connect from the weaviate-client library
        # For local, use connect_to_local; for remote, use connect_to_wcs or connect_to_custom
        if weaviate_url.startswith("http://") or weaviate_url.startswith("https://"):
            host = weaviate_url.replace("http://", "").replace("https://", "").split(":")[0]
            port = int(weaviate_url.split(":")[-1])
            client = weaviate.connect_to_local(host=host, port=port)
        else:
            client = weaviate.connect_to_local()
        if not client.is_ready():
            raise ConnectionError("Weaviate is not ready.")
        print("Successfully connected to Weaviate.")
        return client
    except Exception as e:
        print(f"Failed to connect to Weaviate at {weaviate_url}: {e}")
        raise

def get_embedding_model(model_name=None):
    """Initializes and returns the sentence transformer model."""
    if model_name is None:
        model_name = config.EMBEDDING_MODEL_NAME
    return SentenceTransformer(model_name)

def generate_standard_embedding(text: str) -> np.ndarray:
    """Generates a standard embedding for a given text."""
    model = get_embedding_model()
    return model.encode(text, convert_to_tensor=False, normalize_embeddings=True)

def embed_chunk_data(chunk_data: dict) -> Optional[np.ndarray]:
    """Embeds the text content of a single chunk."""
    text_to_embed = chunk_data.get("chunk_text", "").strip()
    if not text_to_embed:
        return None
    return generate_standard_embedding(text_to_embed)

def create_weaviate_schema(client):
    """Create the Weaviate schema if it doesn't exist."""
    if not client.schema.exists(WEAVIATE_CLASS_NAME):
        class_obj = {
            "class": WEAVIATE_CLASS_NAME,
            "vectorizer": "none",  # We'll provide our own vectors
            "properties": [
                {"name": "chunk_id", "dataType": ["string"]},
                {"name": "doc_id", "dataType": ["string"]},
                {"name": "source_path", "dataType": ["string"]},
                {"name": "original_doc_type", "dataType": ["string"]},
                {"name": "concept_type", "dataType": ["string"]},
                {"name": "concept_name", "dataType": ["string"]},
                {"name": "chunk_text", "dataType": ["text"]},
                {"name": "parent_block_id", "dataType": ["string"]},
                {"name": "parent_block_content", "dataType": ["text"]},
                {"name": "sequence_in_block", "dataType": ["int"]},
                {"name": "filename", "dataType": ["string"]}
            ]
        }
        client.schema.create_class(class_obj)

class VectorStoreManager:
    """
    High-performance vector store manager with optimized batch processing and async operations.
    Features:
    - Efficient batch processing of embeddings
    - Async operations for better concurrency
    - Improved error handling and reporting
    - Performance optimizations for large datasets
    """
    
    @staticmethod
    async def fast_embed_and_store_chunks(client, final_text_chunks: List[Dict], batch_size: int = 50):
        """
        Optimized version with parallel embedding generation
        
        Args:
            client: Weaviate client
            final_text_chunks: List of chunk dictionaries from chunker
            batch_size: Size of batches for processing
        """
        if not final_text_chunks:
            print("No chunks to embed and store.")
            return

        print(f"Optimized embedding and storage for {len(final_text_chunks)} chunks...")
        create_weaviate_schema(client)

        # Pre-load embedding model once
        print("Loading embedding model...")
        embedding_model = get_embedding_model()
        print(f"✓ Embedding model loaded")
        
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
            batch_tasks = []
            for i, batch in enumerate(chunk_batches):
                task = loop.run_in_executor(
                    executor,
                    VectorStoreManager._process_batch,
                    batch,
                    embedding_model,
                    i + 1
                )
                batch_tasks.append(task)
            
            # Wait for all batches to complete embedding
            print("Generating embeddings in parallel...")
            embedded_batches = await asyncio.gather(*batch_tasks)
            
            # Flatten results
            all_embedded_chunks = []
            for batch_result in embedded_batches:
                all_embedded_chunks.extend(batch_result)
        
        print(f"✓ Generated {len(all_embedded_chunks)} embeddings")
        
        # Store all chunks in Weaviate in batches
        print("Storing all embedded chunks in Weaviate...")
        await VectorStoreManager._store_chunks_in_weaviate(
            client, all_embedded_chunks, batch_size
        )
        
        print(f"✅ Optimized embedding and storage complete!")
        print(f"   Total chunks processed: {len(final_text_chunks)}")
        print(f"   Successfully stored: {len(all_embedded_chunks)}")
    
    @staticmethod
    def _process_batch(chunk_batch: List[Dict], embedding_model, batch_num: int) -> List[Dict]:
        """
        Process a batch of chunks - generate embeddings in parallel
        
        Args:
            chunk_batch: List of chunks to process
            embedding_model: Pre-loaded sentence transformer model
            batch_num: Batch number for logging
            
        Returns:
            List of chunks with embeddings and prepared data objects
        """
        print(f"  Processing batch {batch_num} ({len(chunk_batch)} chunks)...")
        
        embedded_chunks = []
        
        # Extract texts for batch embedding
        texts = []
        valid_indices = []
        
        for i, chunk in enumerate(chunk_batch):
            chunk_text = chunk.get("chunk_text", "").strip()
            if chunk_text:
                texts.append(chunk_text)
                valid_indices.append(i)
        
        if not texts:
            print(f"  Batch {batch_num}: No valid texts found")
            return embedded_chunks
        
        # Batch generate embeddings
        try:
            print(f"  Batch {batch_num}: Generating {len(texts)} embeddings...")
            embeddings = embedding_model.encode(
                texts,
                batch_size=min(32, len(texts)),  # Don't exceed available texts
                convert_to_tensor=False,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            # Convert numpy arrays to lists if needed
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            elif hasattr(embeddings, 'tolist'):
                embeddings = embeddings.tolist()
            
            print(f"  Batch {batch_num}: ✓ Embeddings generated")
            
            # Map embeddings back to chunks
            embedding_iter = iter(embeddings)
            
            for i, chunk_data in enumerate(chunk_batch):
                if i in valid_indices:
                    try:
                        embedding = next(embedding_iter)
                        
                        # Ensure embedding is a list
                        if hasattr(embedding, 'tolist'):
                            embedding_list = embedding.tolist()
                        elif isinstance(embedding, (list, tuple)):
                            embedding_list = list(embedding)
                        else:
                            embedding_list = [float(x) for x in embedding]
                            
                    except (StopIteration, Exception) as e:
                        print(f"  Warning: Could not process embedding for chunk {i}: {e}")
                        continue
                else:
                    continue  # Skip chunks with no valid text
                
                # Prepare data object for Weaviate
                try:
                    chunk_id_str = str(chunk_data.get("chunk_id", str(uuid.uuid4())))
                    
                    # Validate UUID format
                    try:
                        uuid.UUID(chunk_id_str)
                    except ValueError:
                        chunk_id_str = str(uuid.uuid4())
                    
                    source_file_path = chunk_data.get("source", "Unknown source")
                    file_name = chunk_data.get("filename")
                    if not file_name and source_file_path != "Unknown source":
                        file_name = os.path.basename(source_file_path)
                    if not file_name:
                        file_name = "unknown_filename"
                    
                    data_object = {
                        "chunk_id": chunk_id_str,
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
                    
                    # Remove None values
                    data_object = {k: v for k, v in data_object.items() if v is not None}
                    
                    embedded_chunks.append({
                        'embedding': embedding_list,
                        'data_object': data_object,
                        'uuid': chunk_id_str
                    })
                    
                except Exception as e:
                    print(f"  Warning: Could not prepare data object for chunk {i}: {e}")
                    continue
        
        except Exception as e:
            print(f"  Error in batch {batch_num} processing: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"  Batch {batch_num}: ✓ Prepared {len(embedded_chunks)} chunks for storage")
        return embedded_chunks
    
    @staticmethod
    async def _store_chunks_in_weaviate(client, embedded_chunks: List[Dict], batch_size: int):
        """
        Store embedded chunks in Weaviate using batch operations
        
        Args:
            client: Weaviate client
            embedded_chunks: List of chunks with embeddings
            batch_size: Batch size for Weaviate operations
        """
        if not embedded_chunks:
            print("No embedded chunks to store")
            return
        
        print(f"Storing {len(embedded_chunks)} chunks in Weaviate...")
        
        # Configure Weaviate batch
        client.batch.configure(
            batch_size=batch_size, 
            dynamic=True, 
            timeout_retries=3,
            connection_error_retries=3
        )
        
        processed_count = 0
        error_count = 0
        
        # Process in batches to avoid memory issues
        storage_batches = [
            embedded_chunks[i:i+batch_size] 
            for i in range(0, len(embedded_chunks), batch_size)
        ]
        
        for batch_num, storage_batch in enumerate(storage_batches, 1):
            print(f"  Storing batch {batch_num}/{len(storage_batches)} ({len(storage_batch)} chunks)...")
            
            try:
                with client.batch as batch_context:
                    for chunk in storage_batch:
                        try:
                            batch_context.add_data_object(
                                data_object=chunk['data_object'],
                                class_name=WEAVIATE_CLASS_NAME,
                                vector=chunk['embedding'],
                                uuid=chunk['uuid']
                            )
                            processed_count += 1
                            
                        except Exception as e:
                            print(f"    Warning: Failed to add chunk {chunk.get('uuid', 'unknown')}: {e}")
                            error_count += 1
                
                print(f"  ✓ Batch {batch_num} stored successfully")
                
            except Exception as e:
                print(f"  Error storing batch {batch_num}: {e}")
                error_count += len(storage_batch)
        
        # Check for batch errors
        if hasattr(client.batch, 'failed_objects') and client.batch.failed_objects:
            batch_errors = len(client.batch.failed_objects)
            print(f"⚠️  Weaviate batch errors: {batch_errors}")
            error_count += batch_errors
            
            # Show first few errors for debugging
            for i, failed_obj in enumerate(client.batch.failed_objects[:3]):
                print(f"    Error {i+1}: {failed_obj.message}")
        
        successful_count = processed_count - error_count
        print(f"✅ Storage complete:")
        print(f"   Successfully stored: {successful_count}")
        print(f"   Errors: {error_count}")


# Convenience function to replace the original
async def fast_embed_and_store_chunks(client, final_text_chunks: List[Dict], batch_size: int = 50):
    """
    Drop-in replacement for the original embed_and_store_chunks function
    
    This function provides the same interface as the original but with:
    - Parallel embedding generation
    - Batch processing for better performance
    - Async operations to avoid blocking
    
    Args:
        client: Weaviate client
        final_text_chunks: List of chunk dictionaries from chunker
        batch_size: Size of batches for processing
    """
    return await VectorStoreManager.fast_embed_and_store_chunks(
        client, final_text_chunks, batch_size
    )


# For backward compatibility, also provide a sync version
def fast_embed_and_store_chunks_sync(client, final_text_chunks: List[Dict], batch_size: int = 50):
    """
    Synchronous version of fast_embed_and_store_chunks
    """
    return asyncio.run(fast_embed_and_store_chunks(client, final_text_chunks, batch_size))


if __name__ == '__main__':
    # Test the optimized vector store manager
    print("--- Testing Optimized Vector Store Manager ---")
    
    # This would normally come from your chunker
    sample_chunks = [
        {
            "chunk_id": str(uuid.uuid4()),
            "doc_id": "test_doc",
            "source": "test.tex",
            "filename": "test.tex",
            "original_type": "latex",
            "concept_type": "section", 
            "concept_name": "Test Section",
            "parent_block_id": str(uuid.uuid4()),
            "chunk_text": "This is a test chunk for the vector store.",
            "sequence_in_block": 0,
            "parent_block_content": "This is the full block content."
        },
        {
            "chunk_id": str(uuid.uuid4()),
            "doc_id": "test_doc",
            "source": "test.tex", 
            "filename": "test.tex",
            "original_type": "latex",
            "concept_type": "section",
            "concept_name": "Test Section",
            "parent_block_id": str(uuid.uuid4()),
            "chunk_text": "This is another test chunk with different content.",
            "sequence_in_block": 1,
            "parent_block_content": "This is more block content."
        }
    ]
    
    async def test_async():
        try:
            client = get_weaviate_client()
            await fast_embed_and_store_chunks(client, sample_chunks, batch_size=10)
            print("✅ Test completed successfully!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(test_async())