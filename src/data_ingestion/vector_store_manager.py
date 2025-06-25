# src/data_ingestion/optimized_vector_store_manager.py
import asyncio
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.connect import ConnectionParams
from src import config

# Constants
WEAVIATE_CLASS_NAME = "MathConcept"
DEFAULT_RETURN_PROPERTIES = ["chunk_text", "source_path", "concept_type", "concept_name", "filename"]
DEFAULT_ADDITIONAL_PROPERTIES = ["distance"]

def get_weaviate_client():
    """Establishes a connection to the Weaviate instance and returns a client object."""
    try:
        import weaviate.classes.init as wvc
        
        # Configure connection with increased timeout and skip initial checks
        connection_params = ConnectionParams.from_url(
            url=config.WEAVIATE_URL,
            grpc_port=50051
        )
        
        # Create client with extended timeout configuration
        additional_config = wvc.AdditionalConfig(
            timeout=wvc.Timeout(
                init=60,  # 60 second init timeout
                query=30, # 30 second query timeout
                insert=30 # 30 second insert timeout
            )
        )
        
        print(f"Connecting to Weaviate at {config.WEAVIATE_URL}...")
        client = weaviate.WeaviateClient(
            connection_params,
            additional_config=additional_config,
            skip_init_checks=True  # Skip startup checks to avoid gRPC timeout
        )
        
        client.connect()
        
        # Perform a simple health check instead of is_ready()
        try:
            collections = client.collections.list_all()
            print(f"Successfully connected to Weaviate. Found {len(collections)} collections.")
        except Exception as health_check_error:
            print(f"Warning: Health check failed but connection may still work: {health_check_error}")
        
        return client
        
    except Exception as e:
        print(f"Failed to connect to Weaviate: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
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
    try:
        # Check if collection exists using v4 API
        collections = client.collections.list_all()
        if WEAVIATE_CLASS_NAME not in collections:
            # Create collection using v4 API
            from weaviate.classes.config import Configure, DataType, Property
            
            client.collections.create(
                name=WEAVIATE_CLASS_NAME,
                properties=[
                    Property(name="chunk_id", data_type=DataType.TEXT),
                    Property(name="doc_id", data_type=DataType.TEXT),
                    Property(name="source_path", data_type=DataType.TEXT),
                    Property(name="original_doc_type", data_type=DataType.TEXT),
                    Property(name="concept_type", data_type=DataType.TEXT),
                    Property(name="concept_name", data_type=DataType.TEXT),
                    Property(name="chunk_text", data_type=DataType.TEXT),
                    Property(name="parent_block_id", data_type=DataType.TEXT),
                    Property(name="parent_block_content", data_type=DataType.TEXT),
                    Property(name="sequence_in_block", data_type=DataType.INT),
                    Property(name="filename", data_type=DataType.TEXT)
                ],
                vectorizer_config=Configure.Vectorizer.none()
            )
            print(f"✓ Created collection: {WEAVIATE_CLASS_NAME}")
        else:
            print(f"✓ Collection already exists: {WEAVIATE_CLASS_NAME}")
    except Exception as e:
        print(f"Error creating schema: {e}")
        raise

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
        Store embedded chunks in Weaviate using the correct v4 API with DataObject
        
        Args:
            client: Weaviate client
            embedded_chunks: List of chunks with embeddings
            batch_size: Batch size for Weaviate operations
        """
        if not embedded_chunks:
            print("No embedded chunks to store")
            return
        
        print(f"Storing {len(embedded_chunks)} chunks in Weaviate...")
        
        processed_count = 0
        error_count = 0
        
        # Get the collection
        collection = client.collections.get(WEAVIATE_CLASS_NAME)
        
        # Process in batches to avoid memory issues
        storage_batches = [
            embedded_chunks[i:i+batch_size] 
            for i in range(0, len(embedded_chunks), batch_size)
        ]
        
        for batch_num, storage_batch in enumerate(storage_batches, 1):
            print(f"  Storing batch {batch_num}/{len(storage_batches)} ({len(storage_batch)} chunks)...")
            
            try:
                # Prepare batch data for insert_many using DataObject
                batch_data = []
                for chunk in storage_batch:
                    try:
                        # Use DataObject for objects with custom vectors
                        from weaviate.classes.data import DataObject
                        
                        data_object = DataObject(
                            properties=chunk['data_object'],
                            vector=chunk['embedding'],
                            uuid=chunk['uuid']
                        )
                        batch_data.append(data_object)
                    except Exception as e:
                        print(f"    Warning: Failed to prepare chunk {chunk.get('uuid', 'unknown')}: {e}")
                        error_count += 1
                
                if batch_data:
                    # Use insert_many for batch insertion
                    result = collection.data.insert_many(batch_data)
                    processed_count += len(batch_data)
                    print(f"  ✓ Batch {batch_num} stored successfully")
                else:
                    print(f"  Warning: No valid data in batch {batch_num}")
                    
            except Exception as e:
                print(f"  Error storing batch {batch_num}: {e}")
                error_count += len(storage_batch)
        
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