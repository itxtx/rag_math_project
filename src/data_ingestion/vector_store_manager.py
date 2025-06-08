# src/data_ingestion/vector_store_manager.py
import os
import weaviate
from weaviate.auth import AuthApiKey 
from sentence_transformers import SentenceTransformer
import time
import uuid 
from src import config
import json 
import numpy as np 

WEAVIATE_CLASS_NAME = "MathDocumentChunk"
EMBEDDING_MODEL_NAME = config.EMBEDDING_MODEL_NAME 
embedding_model_instance = None 

# Default properties to return in search results
DEFAULT_RETURN_PROPERTIES = [
    "chunk_id",
    "doc_id",
    "source_path",
    "concept_type",
    "concept_name",
    "chunk_text",
    "parent_block_id",
    "parent_block_content",
    "sequence_in_block",
    "filename"
]

# Default additional properties for search
DEFAULT_ADDITIONAL_PROPERTIES = {
    "certainty": True,
    "distance": True
}

def get_embedding_model():
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

def get_weaviate_client():
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
        if "Connection refused" in str(e):
            print("Common issues: Weaviate container not running, incorrect URL, or firewall blocking the port.")
        raise

def create_weaviate_schema(client: weaviate.Client):
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
            {"name": "doc_id", "dataType": ["text"],"description": "Identifier for the source document"},
            {"name": "source_path", "dataType": ["text"],"description": "Path to the source document"},
            {"name": "original_doc_type", "dataType": ["text"],"description": "Original type (e.g., latex, pdf)"},
            {"name": "concept_type", "dataType": ["text"],"description": "Type of conceptual block"},
            {"name": "concept_name", "dataType": ["text"],"description": "Name of concept", "indexFilterable": True, "indexSearchable": True},
            {"name": "chunk_text", "dataType": ["text"],"description": "Text content of this chunk", "indexFilterable": False, "indexSearchable": True},
            # --- ENSURED parent_block_id IS IN SCHEMA ---
            {"name": "parent_block_id", "dataType": ["text"],"description": "ID of the conceptual block this chunk belongs to", "indexFilterable": True, "indexSearchable": False},
            # --- END OF ENSURE ---
            {"name": "parent_block_content", "dataType": ["text"],"description": "Full text content of the parent conceptual block","indexFilterable": False, "indexSearchable": False},
            {"name": "sequence_in_block", "dataType": ["int"],"description": "Order of this chunk within its parent conceptual block"},
            {"name": "filename", "dataType": ["text"], "description": "Original filename of the document", "indexFilterable": True, "indexSearchable": False} # Added filename
        ],
        "vectorIndexConfig": {"distance": "cosine"},
    }

    try:
        if not client.schema.exists(WEAVIATE_CLASS_NAME):
            print(f"Schema for class '{WEAVIATE_CLASS_NAME}' not found. Creating now...")
            client.schema.create_class(class_obj)
            print(f"Schema for class '{WEAVIATE_CLASS_NAME}' created successfully.")
        else:
            print(f"Schema for class '{WEAVIATE_CLASS_NAME}' already exists.")
            # Check and add parent_block_id if missing (simple check)
            current_schema = client.schema.get(WEAVIATE_CLASS_NAME)
            prop_names = [p['name'] for p in current_schema.get('properties', [])]
            
            missing_props = []
            if 'parent_block_id' not in prop_names:
                missing_props.append({
                    "name": "parent_block_id", "dataType": ["text"],
                    "description": "ID of the conceptual block this chunk belongs to",
                    "indexFilterable": True, "indexSearchable": False
                })
            if 'doc_id' not in prop_names: # Also ensure doc_id is there
                 missing_props.append({
                    "name": "doc_id", "dataType": ["text"],
                    "description": "Identifier for the source document",
                    "indexFilterable": True, "indexSearchable": False # Example config
                })
            if 'filename' not in prop_names:
                 missing_props.append({
                    "name": "filename", "dataType": ["text"],
                    "description": "Original filename of the document",
                    "indexFilterable": True, "indexSearchable": False
                })


            for prop_to_add in missing_props:
                try:
                    print(f"Attempting to add missing property '{prop_to_add['name']}' to schema '{WEAVIATE_CLASS_NAME}'.")
                    client.schema.property.create(WEAVIATE_CLASS_NAME, prop_to_add)
                    print(f"Property '{prop_to_add['name']}' added successfully.")
                except Exception as e_prop:
                    print(f"Failed to add property '{prop_to_add['name']}': {e_prop}. Manual schema update might be needed if class has data.")

    except Exception as e:
        print(f"An unexpected error occurred while checking/creating/updating schema: {e}")
        import traceback
        traceback.print_exc() 
        raise

def generate_standard_embedding(text: str, model_instance=None) -> list[float] | None:
    if not text or not text.strip():
        print("Warning: Attempted to embed empty or whitespace-only text. Returning None.")
        return None
    try:
        model_to_use = model_instance if model_instance else get_embedding_model()
        embedding = model_to_use.encode(text, convert_to_tensor=False, normalize_embeddings=True)
        if not hasattr(embedding, 'tolist'):
            if isinstance(embedding, list): return embedding
            else: return list(embedding)
        return embedding.tolist()
    except Exception as e:
        print(f"Error generating standard embedding for text snippet '{text[:50]}...': {e}")
        import traceback
        traceback.print_exc()
        return None

def embed_chunk_data(chunk_data: dict) -> list[float] | None:
    chunk_text = chunk_data.get("chunk_text", "")
    if not chunk_text:
        print(f"Warning: Chunk ID {chunk_data.get('chunk_id', 'N/A')} has no text. Skipping embedding.")
        return None
    return generate_standard_embedding(chunk_text)

def embed_and_store_chunks(client: weaviate.Client, final_text_chunks: list[dict], batch_size: int = 50):
    if not final_text_chunks:
        print("No chunks to embed and store.")
        return

    print(f"Preparing to embed and store {len(final_text_chunks)} chunks...")
    create_weaviate_schema(client) 

    client.batch.configure(batch_size=batch_size, dynamic=True, timeout_retries=3)
    processed_count = 0; error_count = 0; skipped_due_to_embedding_failure_count = 0

    with client.batch as batch_context:
        for i, chunk_data in enumerate(final_text_chunks):
            current_chunk_id_str = chunk_data.get("chunk_id", str(uuid.uuid4()))
            if not isinstance(current_chunk_id_str, str): current_chunk_id_str = str(current_chunk_id_str)
            try: uuid.UUID(current_chunk_id_str)
            except ValueError:
                print(f"Error: Chunk ID '{current_chunk_id_str}' is not valid. Skipping."); error_count +=1; continue

            embedding_vector = embed_chunk_data(chunk_data)
            if embedding_vector is None:
                skipped_due_to_embedding_failure_count += 1; continue
                     
                        
            source_file_path = chunk_data.get("source", "Unknown source") # "source" key from chunker
            file_name = chunk_data.get("filename", os.path.basename(source_file_path) if source_file_path != "Unknown source" else "unknown_filename")


            data_object = {
                "chunk_id": current_chunk_id_str,
                "doc_id": chunk_data.get("doc_id", "unknown_doc_id"), 
                "source_path": source_file_path, # Use corrected variable
                "original_doc_type": chunk_data.get("original_doc_type", "unknown"),
                "concept_type": chunk_data.get("concept_type", "general_content"),
                "concept_name": chunk_data.get("concept_name"),
                "chunk_text": chunk_data.get("chunk_text", ""),
                "parent_block_id": chunk_data.get("parent_block_id"), 
                "parent_block_content": chunk_data.get("parent_block_content", ""),
                "sequence_in_block": chunk_data.get("sequence_in_block", 0),
                "filename": file_name # Use corrected variable
            }
            data_object = {k: v for k, v in data_object.items() if v is not None}

            batch_context.add_data_object(
                data_object=data_object, class_name=WEAVIATE_CLASS_NAME,
                vector=embedding_vector, uuid=current_chunk_id_str
            )
            processed_count +=1
            if (i + 1) % (batch_size * 2) == 0: print(f"Prepared {i+1}/{len(final_text_chunks)}...")

    print(f"\n--- Batch import summary ---")
    print(f"Total chunks processed for Weaviate: {processed_count}")
    print(f"Skipped due to embedding failure: {skipped_due_to_embedding_failure_count}")
    print(f"Skipped due to invalid ID (pre-batch): {error_count}")

    batch_errors_found = 0
    if hasattr(client.batch, 'failed_objects') and client.batch.failed_objects:
        batch_errors_found = len(client.batch.failed_objects)
        print(f"Weaviate Import Errors: {batch_errors_found} objects failed to import.")
        for idx, failed_obj in enumerate(client.batch.failed_objects):
            if idx < 5: print(f"  Error {idx+1}: {failed_obj.message}")
            elif idx == 5: print(f"  ... and {batch_errors_found - 5} more errors.")
            break
    elif not hasattr(client.batch, 'failed_objects'):
        print("WARNING: client.batch object does not have 'failed_objects' attribute.")
    else:
        print("No errors reported from Weaviate during batch import (failed_objects list is empty or None).")
    
    total_errors = error_count + batch_errors_found
    successfully_added_count = processed_count - batch_errors_found

    print(f"Successfully added to Weaviate (estimate based on batch submission): {max(0, successfully_added_count)}")
    print(f"Total errors encountered during storage process: {total_errors}")
    # ... (rest of summary prints) ...

if __name__ == '__main__':
    print("--- Vector Store Manager Demo ---")
    # ... (rest of the demo) ...
