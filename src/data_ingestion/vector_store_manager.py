# src/data_ingestion/vector_store_manager.py
import os
import weaviate
from weaviate.auth import AuthApiKey # Or other auth methods
from sentence_transformers import SentenceTransformer
import time
import uuid # For chunk_id if not already present
from src import config
import json # For printing query results in main
import numpy as np # Ensure numpy is imported if using np.array

# --- Configuration for Weaviate Schema and Embeddings ---
WEAVIATE_CLASS_NAME = "MathDocumentChunk"
EMBEDDING_MODEL_NAME = config.EMBEDDING_MODEL_NAME # Get from config
embedding_model_instance = None # Will be loaded on demand

# --- Helper for loading the embedding model ---
def get_embedding_model():
    """Loads and returns the Sentence Transformer model."""
    global embedding_model_instance
    if embedding_model_instance is None:
        print(f"Loading sentence-transformer model: {EMBEDDING_MODEL_NAME}...")
        try:
            embedding_model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print("Sentence-transformer model loaded successfully.")
        except Exception as e:
            print(f"Error loading sentence-transformer model '{EMBEDDING_MODEL_NAME}': {e}")
            raise
    return embedding_model_instance

# --- Weaviate Client Initialization ---
def get_weaviate_client():
    """Initializes and returns a Weaviate client."""
    print(f"Attempting to connect to Weaviate at {config.WEAVIATE_URL}")
    try:
        client = weaviate.Client(url=config.WEAVIATE_URL)
        for i in range(5):
            if client.is_ready():
                print("Weaviate client connected successfully.")
                return client
            print(f"Weaviate client not ready. Retrying ({i+1}/5)...")
            time.sleep(2 + i)
        raise ConnectionError("Failed to connect to Weaviate after multiple retries.")
    except Exception as e:
        print(f"Failed to initialize Weaviate client: {e}")
        print("Ensure Weaviate is running and accessible at the configured URL.")
        if "Connection refused" in str(e):
            print("Common issues: Weaviate container not running, incorrect URL, or firewall blocking the port.")
        raise

# --- Schema Definition and Management ---
def create_weaviate_schema(client: weaviate.Client):
    """
    Defines and creates the class schema in Weaviate if it doesn't exist.
    """
    try:
        temp_model = get_embedding_model()
        vector_dimensionality = temp_model.get_sentence_embedding_dimension()
        if vector_dimensionality is None:
            raise ValueError("Embedding model did not return a dimension.")
        print(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded, dimension: {vector_dimensionality}.")
    except Exception as e:
        print(f"Could not load embedding model to determine dimensionality: {e}")
        print("Schema creation aborted. Ensure embedding model is valid and accessible.")
        raise

    class_obj = {
        "class": WEAVIATE_CLASS_NAME,
        "description": "A chunk of text from a mathematical document, with associated concepts.",
        "vectorizer": "none",
        "properties": [
            {"name": "chunk_id", "dataType": ["uuid"],"description": "Unique identifier for the chunk"},
            {"name": "source_path", "dataType": ["text"],"description": "Path to the source document"},
            {"name": "original_doc_type", "dataType": ["text"],"description": "Original type (e.g., latex, pdf)"},
            {"name": "concept_type", "dataType": ["text"],"description": "Type of conceptual block (e.g., section, theorem, definition, general_content)"},
            {"name": "concept_name", "dataType": ["text"],"description": "Name of concept (e.g., section title, theorem name)", "indexFilterable": True, "indexSearchable": True},
            {"name": "chunk_text", "dataType": ["text"],"description": "Text content of this chunk", "indexFilterable": False, "indexSearchable": True},
            {"name": "parent_block_content", "dataType": ["text"],"description": "Full text content of the parent conceptual block from which this chunk was derived","indexFilterable": False, "indexSearchable": False},
            {"name": "sequence_in_block", "dataType": ["int"],"description": "Order of this chunk within its parent conceptual block"},
        ],
        "vectorIndexConfig": {"distance": "cosine"},
    }

    try:
        client.schema.get(WEAVIATE_CLASS_NAME)
        print(f"Schema for class '{WEAVIATE_CLASS_NAME}' already exists.")
    except weaviate.exceptions.UnexpectedStatusCodeException as e:
        if e.status_code == 404:
            print(f"Schema for class '{WEAVIATE_CLASS_NAME}' not found. Creating now...")
            try:
                client.schema.create_class(class_obj)
                print(f"Schema for class '{WEAVIATE_CLASS_NAME}' created successfully.")
            except Exception as ce:
                print(f"Failed to create schema for class '{WEAVIATE_CLASS_NAME}': {ce}")
                raise
        else:
            print(f"Error checking schema for class '{WEAVIATE_CLASS_NAME}': {e}")
            raise
    except Exception as e:
        print(f"An unexpected error occurred while checking/creating schema: {e}")
        raise

# --- Embedding Generation ---
def generate_standard_embedding(text: str, model_instance=None) -> list[float] | None:
    """Generates embedding for a text using the standard Sentence Transformer model."""
    if not text or not text.strip():
        print("Warning: Attempted to embed empty or whitespace-only text. Returning None.")
        return None
    try:
        model_to_use = model_instance if model_instance else get_embedding_model()
        embedding = model_to_use.encode(text, convert_to_tensor=False, normalize_embeddings=True)
        if not hasattr(embedding, 'tolist'):
            print(f"Warning: Embedding for '{text[:30]}...' is not a NumPy array, type: {type(embedding)}. Attempting to convert.")
            if isinstance(embedding, list):
                return embedding
            else:
                return list(embedding)
        return embedding.tolist()
    except Exception as e:
        print(f"Error generating standard embedding for text snippet '{text[:50]}...': {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_math_embedding_placeholder(latex_math_expression: str) -> list[float] | None:
    """ Placeholder for your advanced math embedding pipeline. """
    print(f"Placeholder: Advanced math embedding for: {latex_math_expression[:100]}...")
    print("Advanced math embedding not implemented. Using standard embedding as fallback for now.")
    return generate_standard_embedding(latex_math_expression)

def embed_chunk_data(chunk_data: dict) -> list[float] | None:
    """
    Generates an embedding for a given chunk's text.
    """
    chunk_text = chunk_data.get("chunk_text", "")
    if not chunk_text:
        print(f"Warning: Chunk ID {chunk_data.get('chunk_id', 'N/A')} has no text. Skipping embedding.")
        return None
    return generate_standard_embedding(chunk_text)

# --- Data Ingestion into Weaviate ---
def embed_and_store_chunks(client: weaviate.Client, final_text_chunks: list[dict], batch_size: int = 50):
    """
    Generates embeddings for chunks and stores them in Weaviate using batch import.
    Ensures `chunk_id` is a UUID string.
    """
    if not final_text_chunks:
        print("No chunks to embed and store.")
        return

    print(f"Preparing to embed and store {len(final_text_chunks)} chunks...")
    create_weaviate_schema(client)

    client.batch.configure(
        batch_size=batch_size,
        dynamic=True,
        timeout_retries=3,
    )

    processed_count = 0
    successfully_added_count = 0
    error_count = 0
    skipped_due_to_embedding_failure_count = 0

    with client.batch as batch_context:
        for i, chunk_data in enumerate(final_text_chunks):
            current_chunk_id_str = chunk_data.get("chunk_id")
            if not current_chunk_id_str:
                current_chunk_id_str = str(uuid.uuid4())
            elif not isinstance(current_chunk_id_str, str):
                current_chunk_id_str = str(current_chunk_id_str)
            try:
                uuid.UUID(current_chunk_id_str)
            except ValueError:
                print(f"Error: Chunk ID '{current_chunk_id_str}' is not a valid UUID. Skipping chunk.")
                error_count +=1
                continue

            embedding_vector = embed_chunk_data(chunk_data)

            if embedding_vector is None:
                print(f"Skipping chunk {current_chunk_id_str} due to embedding error or empty text.")
                skipped_due_to_embedding_failure_count += 1
                continue

            data_object = {
                "chunk_id": current_chunk_id_str,
                "source_path": chunk_data.get("source_path") or chunk_data.get("source", "Unknown source"),
                "original_doc_type": chunk_data.get("original_doc_type") or chunk_data.get("original_type", "unknown"),
                "concept_type": chunk_data.get("concept_type", "general_content"),
                "concept_name": chunk_data.get("concept_name"),
                "chunk_text": chunk_data.get("chunk_text", ""),
                "parent_block_content": chunk_data.get("parent_block_content", ""),
                "sequence_in_block": chunk_data.get("sequence_in_block", 0),
            }

            batch_context.add_data_object(
                data_object=data_object,
                class_name=WEAVIATE_CLASS_NAME,
                vector=embedding_vector,
                uuid=current_chunk_id_str
            )
            processed_count +=1

            if (i + 1) % (batch_size * 2) == 0: # Print progress less frequently
                print(f"Prepared {i+1}/{len(final_text_chunks)} chunks for batching...")

    # --- Start of Batch Import Summary & Debug Prints ---
    print(f"\n--- Batch import summary ---")
    print(f"Total chunks processed for Weaviate: {processed_count}")
    print(f"Chunks skipped due to embedding failure/empty text: {skipped_due_to_embedding_failure_count}")

    # --- ADDED DEBUG PRINTS ---
    print(f"DEBUG: Type of client.batch before checking failed_objects: {type(client.batch)}")
    try:
        print(f"DEBUG: Attributes of client.batch: {dir(client.batch)}")
        if hasattr(client.batch, '_failed_objects'): # Check for the internal attribute
            print(f"DEBUG: client.batch has _failed_objects. Length: {len(client.batch._failed_objects)}")
        else:
            print("DEBUG: client.batch does NOT have _failed_objects (internal list).")
        
        if hasattr(client.batch, 'failed_objects'): # Check for the property
            print("DEBUG: client.batch HAS a 'failed_objects' attribute/property.")
        else:
            print("DEBUG: client.batch does NOT have a 'failed_objects' attribute/property.")

    except Exception as e_debug:
        print(f"DEBUG: Error inspecting client.batch: {e_debug}")
    # --- END OF ADDED DEBUG PRINTS ---

    # Check for batch errors (objects that Weaviate rejected) by using client.batch
    # This is the line that previously caused an AttributeError
    if hasattr(client.batch, 'failed_objects') and client.batch.failed_objects:
        batch_error_count = len(client.batch.failed_objects)
        print(f"Weaviate Import Errors: {batch_error_count} objects failed to import.")
        for failed_obj_idx, failed_obj in enumerate(client.batch.failed_objects):
            if failed_obj_idx < 5:
                 print(f"  Error {failed_obj_idx+1}: {failed_obj.message}")
            elif failed_obj_idx == 5:
                print(f"  ... and {batch_error_count - 5} more errors.")
                break
        error_count += batch_error_count
    elif not hasattr(client.batch, 'failed_objects'):
        print("WARNING: client.batch object does not have 'failed_objects' attribute. Cannot check for batch import errors this way.")
        # If this warning appears, it indicates a deeper issue with the client.batch object's state or type.
    else: # hasattr is true, but client.batch.failed_objects is empty or None
        print("No errors reported from Weaviate during batch import (failed_objects list is empty or None).")


    successfully_added_count = processed_count - error_count

    print(f"Successfully added to Weaviate (estimate): {max(0, successfully_added_count)}")
    print(f"Total errors encountered during storage: {error_count}")

    if error_count == 0 and processed_count > 0 :
        print("All processed chunks appear to be sent to Weaviate batch successfully.")
    elif processed_count == 0 and skipped_due_to_embedding_failure_count == len(final_text_chunks) and error_count == 0:
        print("All chunks were skipped. Nothing sent to Weaviate.")
    elif error_count > 0:
        print(f"There were {error_count} errors during Weaviate batch import. Please check logs.")


if __name__ == '__main__':
    print("--- Vector Store Manager Demo ---")
    # ... (rest of the demo code remains the same) ...
    client = None
    try:
        client = get_weaviate_client()
        dummy_chunks_for_storage = [
            {
                "chunk_id": str(uuid.uuid4()), "source_path": "doc_A.tex", "original_doc_type": "latex",
                "concept_type": "section", "concept_name": "Introduction to Vectors",
                "chunk_text": "This is the introduction to document A. It talks about vectors and their properties, such as magnitude and direction.",
                "parent_block_content": "\\section{Introduction to Vectors}\nThis is the introduction to document A. It talks about vectors and their properties, such as magnitude and direction.",
                "sequence_in_block": 0
            },
            {
                "chunk_id": str(uuid.uuid4()), "source_path": "doc_A.tex", "original_doc_type": "latex",
                "concept_type": "general_content", "concept_name": None,
                "chunk_text": "Vectors can be added using the parallelogram law. For example, $v + w = u$. They are fundamental in physics and engineering.",
                "parent_block_content": "Vectors can be added using the parallelogram law. For example, $v + w = u$. They are fundamental in physics and engineering.",
                "sequence_in_block": 0
            },
        ]
        embed_and_store_chunks(client, dummy_chunks_for_storage, batch_size=2)

        print("\n--- Querying Weaviate (Example BM25 - Keyword Search) ---")
        bm25_query = "vectors physics properties"
        query_result_bm25 = (
            client.query
            .get(WEAVIATE_CLASS_NAME, ["chunk_text", "source_path", "concept_name", "_additional {score id}"])
            .with_bm25(query=bm25_query, properties=["chunk_text^2", "concept_name"])
            .with_limit(3)
            .do()
        )
        print(f"BM25 Query result for '{bm25_query}':")
        print(json.dumps(query_result_bm25, indent=2))

    except ConnectionError as ce:
        print(f"Connection Error: Could not connect to Weaviate. {ce}")
    except Exception as e:
        print(f"An error occurred during the vector store manager demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n--- Vector Store Manager Demo Finished ---")
