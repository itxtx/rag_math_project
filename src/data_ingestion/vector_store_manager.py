# src/data_ingestion/vector_store_manager.py
import os
import weaviate
from weaviate.auth import AuthApiKey # Or other auth methods
from sentence_transformers import SentenceTransformer
import time
import uuid # For chunk_id if not already present
from src import config
import json # For printing query results in main

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

    # Example for API Key authentication (if you set it up in Weaviate)
    # auth_config = AuthApiKey(api_key=config.WEAVIATE_API_KEY) # Assuming API_KEY is in config
    # client = weaviate.Client(url=config.WEAVIATE_URL, auth_client_secret=auth_config)

    # For anonymous access (as per docker-compose.yml default or Weaviate default)
    try:
        client = weaviate.Client(url=config.WEAVIATE_URL)
        # Check connection with retries
        # max_retries = 5 # Renamed for clarity
        # base_delay = 2  # Renamed for clarity
        for i in range(5): # Increased retries
            if client.is_ready():
                print("Weaviate client connected successfully.")
                return client
            print(f"Weaviate client not ready. Retrying ({i+1}/5)...")
            time.sleep(2 + i) # Exponential backoff might be better for real scenarios

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
    # Attempt to load the model to confirm it's valid before proceeding,
    # even though dimensionality isn't explicitly set in schema for "vectorizer":"none".
    try:
        temp_model = get_embedding_model()
        vector_dimensionality = temp_model.get_sentence_embedding_dimension() # Good for logging/validation
        if vector_dimensionality is None: # Should not happen with standard SBERT models
            raise ValueError("Embedding model did not return a dimension.")
        print(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded, dimension: {vector_dimensionality}.")
    except Exception as e:
        print(f"Could not load embedding model to determine dimensionality: {e}")
        print("Schema creation aborted. Ensure embedding model is valid and accessible.")
        raise # Re-raise the exception to halt the process if model can't load

    class_obj = {
        "class": WEAVIATE_CLASS_NAME,
        "description": "A chunk of text from a mathematical document, with associated concepts.",
        "vectorizer": "none", # Indicates that vectors will be provided manually
        "properties": [
            {"name": "chunk_id", "dataType": ["uuid"],"description": "Unique identifier for the chunk"},
            {"name": "source_path", "dataType": ["text"],"description": "Path to the source document"},
            {"name": "original_doc_type", "dataType": ["text"],"description": "Original type (e.g., latex, pdf)"},
            {"name": "concept_type", "dataType": ["text"],"description": "Type of conceptual block (e.g., section, theorem, definition, general_content)"},
            {"name": "concept_name", "dataType": ["text"],"description": "Name of concept (e.g., section title, theorem name)", "indexFilterable": True, "indexSearchable": True},
            {"name": "chunk_text", "dataType": ["text"],"description": "Text content of this chunk", "indexFilterable": False, "indexSearchable": True}, # searchable for BM25
            {"name": "parent_block_content", "dataType": ["text"],"description": "Full text content of the parent conceptual block from which this chunk was derived","indexFilterable": False, "indexSearchable": False},
            {"name": "sequence_in_block", "dataType": ["int"],"description": "Order of this chunk within its parent conceptual block"},
        ],
        "vectorIndexConfig": {
            "distance": "cosine", # Common distance metric for sentence embeddings
            # Add other index configs if needed, e.g., HNSW parameters
            # "efConstruction": 128,
            # "maxConnections": 16,
            # "ef": -1 # ef at query time, -1 for Weaviate default or specify
        },
        # "invertedIndexConfig": { # Optional: fine-tune BM25/inverted index behavior
        #    "bm25": {
        #        "b": 0.75,
        #        "k1": 1.2
        #    },
        #    "stopwords": {
        #        "preset": "en", # "none" to disable
        #        "additions": [],
        #        "removals": []
        #    }
        # }
    }

    try:
        current_schema = client.schema.get(WEAVIATE_CLASS_NAME)
        print(f"Schema for class '{WEAVIATE_CLASS_NAME}' already exists.")
        # TODO: Optionally, add schema update logic if needed, e.g., adding new properties.
        # This can be complex as some changes are not allowed or require data migration.
    except weaviate.exceptions.UnexpectedStatusCodeException as e:
        if e.status_code == 404: # Not found
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
def generate_standard_embedding(text: str) -> list[float] | None:
    """Generates embedding for a text using the standard Sentence Transformer model."""
    if not text or not text.strip():
        print("Warning: Attempted to embed empty or whitespace-only text. Returning None.")
        return None
    try:
        model = get_embedding_model()
        # Normalize to prevent issues with some models if text is purely numeric or very short
        embedding = model.encode(text, convert_to_tensor=False, normalize_embeddings=True)
        return embedding.tolist()
    except Exception as e:
        print(f"Error generating standard embedding for text snippet '{text[:50]}...': {e}")
        return None

def generate_math_embedding_placeholder(latex_math_expression: str) -> list[float] | None:
    """ Placeholder for your advanced math embedding pipeline. """
    print(f"Placeholder: Advanced math embedding for: {latex_math_expression[:100]}...")
    print("Advanced math embedding not implemented. Using standard embedding as fallback for now.")
    # In a real scenario, this would call your specialized math embedding model/service.
    # For now, it falls back to the standard text embedder.
    return generate_standard_embedding(latex_math_expression)

def embed_chunk_data(chunk_data: dict) -> list[float] | None:
    """
    Generates an embedding for a given chunk's text.
    Future: Could differentiate based on content type (e.g., pure math vs. text).
    """
    chunk_text = chunk_data.get("chunk_text", "")
    if not chunk_text:
        print(f"Warning: Chunk ID {chunk_data.get('chunk_id', 'N/A')} has no text. Skipping embedding.")
        return None

    # TODO: Implement logic to identify and use specialized math embeddings.
    # For example, if a chunk is identified as primarily LaTeX/math:
    # if chunk_data.get("is_math_dominant", False): # Add 'is_math_dominant' during preprocessing
    #     return generate_math_embedding_placeholder(chunk_text) # or a more specific math part
    # else:
    #     return generate_standard_embedding(chunk_text)

    # Current simple approach: embed the whole chunk_text using the standard model.
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
    # Ensure schema exists before attempting to add data
    create_weaviate_schema(client)

    # Configure batching. `dynamic=True` can help adjust batch size based on Weaviate's load.
    client.batch.configure(
        batch_size=batch_size,
        dynamic=True, # Recommended for Weaviate Cloud Services / when unsure of optimal size
        timeout_retries=3,
        # callback=weaviate.util.check_batch_result # Optional: for more detailed per-batch feedback
    )

    processed_count = 0
    successfully_added_count = 0
    error_count = 0
    skipped_due_to_embedding_failure_count = 0

    with client.batch as batch:
        for i, chunk_data in enumerate(final_text_chunks):
            # Ensure chunk_id is a valid UUID string, generate if missing (though expect it to be present)
            current_chunk_id_str = chunk_data.get("chunk_id")
            if not current_chunk_id_str:
                print(f"Warning: Chunk {i+1} missing 'chunk_id'. Generating one.")
                current_chunk_id_str = str(uuid.uuid4())
            elif not isinstance(current_chunk_id_str, str): # Ensure it's a string for Weaviate
                print(f"Warning: Chunk ID {current_chunk_id_str} is not a string. Converting.")
                current_chunk_id_str = str(current_chunk_id_str)
            try:
                uuid.UUID(current_chunk_id_str) # Validate if it's a valid UUID string
            except ValueError:
                print(f"Error: Chunk ID '{current_chunk_id_str}' is not a valid UUID. Skipping chunk.")
                error_count +=1
                continue


            # print(f"Processing chunk {i+1}/{len(final_text_chunks)}: ID {current_chunk_id_str}")

            embedding_vector = embed_chunk_data(chunk_data)

            if embedding_vector is None:
                print(f"Skipping chunk {current_chunk_id_str} due to embedding error or empty text.")
                skipped_due_to_embedding_failure_count += 1
                continue

            # Ensure all required fields are present and have reasonable defaults if nullable
            data_object = {
                "chunk_id": current_chunk_id_str,
                "source_path": chunk_data.get("source_path") or chunk_data.get("source", "Unknown source"),
                "original_doc_type": chunk_data.get("original_doc_type") or chunk_data.get("original_type", "unknown"),
                "concept_type": chunk_data.get("concept_type", "general_content"),
                "concept_name": chunk_data.get("concept_name"), # Can be None
                "chunk_text": chunk_data.get("chunk_text", ""),
                "parent_block_content": chunk_data.get("parent_block_content", ""),
                "sequence_in_block": chunk_data.get("sequence_in_block", 0),
            }

            batch.add_data_object(
                data_object=data_object,
                class_name=WEAVIATE_CLASS_NAME,
                vector=embedding_vector,
                uuid=current_chunk_id_str # Use the pre-generated chunk_id as Weaviate's internal ID
            )
            processed_count +=1

            if (i + 1) % (batch_size * 2) == 0: # Print progress less frequently for large imports
                print(f"Prepared {i+1}/{len(final_text_chunks)} chunks for batching...")

    # The batch is automatically sent when the `with` block exits or when it's full.
    # Now, check for errors from the batch import process.
    print(f"\n--- Batch import summary ---")
    print(f"Total chunks processed for Weaviate: {processed_count}")
    print(f"Chunks skipped due to embedding failure/empty text: {skipped_due_to_embedding_failure_count}")

    # Check for batch errors (objects that Weaviate rejected)
    if batch.failed_objects: # Accessing it after the context manager should be fine
        batch_error_count = len(batch.failed_objects)
        print(f"Weaviate Import Errors: {batch_error_count} objects failed to import.")
        for failed_obj_idx, failed_obj in enumerate(batch.failed_objects):
            if failed_obj_idx < 5: # Print details for the first few errors
                 print(f"  Error {failed_obj_idx+1}: {failed_obj.message}")
                 # print(f"  Failed Object UUID: {failed_obj.uuid}") # if available and useful
                 # print(f"  Failed Object Original: {failed_obj.object_}") # Can be verbose
            elif failed_obj_idx == 5:
                print(f"  ... and {batch_error_count - 5} more errors.")
                break
        error_count += batch_error_count
    else:
        print("No errors reported from Weaviate during batch import.")

    successfully_added_count = processed_count - error_count # Assuming processed that weren't skipped were attempted

    print(f"Successfully added to Weaviate (estimate): {max(0, successfully_added_count)}") # max(0,...) in case logic has flaw
    print(f"Total errors encountered during storage: {error_count}")

    if error_count == 0 and processed_count > 0 :
        print("All processed chunks appear to be sent to Weaviate batch successfully.")
    elif processed_count == 0 and skipped_due_to_embedding_failure_count == len(final_text_chunks):
        print("All chunks were skipped. Nothing sent to Weaviate.")
    elif error_count > 0:
        print(f"There were {error_count} errors during Weaviate batch import. Please check logs.")


if __name__ == '__main__':
    print("--- Vector Store Manager Demo ---")
    client = None
    try:
        # This assumes you have a src/config.py with:
        # EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # or your preferred model
        # WEAVIATE_URL = "http://localhost:8080"    # or your Weaviate instance URL
        # (Optional) WEAVIATE_API_KEY = "your_api_key" # if using authentication

        client = get_weaviate_client()
        # The create_weaviate_schema function is now called within embed_and_store_chunks
        # but can be called here explicitly if desired for setup before data prep.
        # create_weaviate_schema(client) # Ensure schema exists

        # Dummy chunks for demonstration, ensuring they have string UUIDs for chunk_id
        dummy_chunks_for_storage = [
            {
                "chunk_id": str(uuid.uuid4()), "source": "doc_A.tex", "original_type": "latex", # 'source' and 'original_type' used in old code
                "source_path": "doc_A.tex", "original_doc_type": "latex", # 'source_path' and 'original_doc_type' preferred by new schema
                "concept_type": "section", "concept_name": "Introduction to Vectors",
                "chunk_text": "This is the introduction to document A. It talks about vectors and their properties, such as magnitude and direction.",
                "parent_block_content": "\\section{Introduction to Vectors}\nThis is the introduction to document A. It talks about vectors and their properties, such as magnitude and direction.",
                "sequence_in_block": 0
            },
            {
                "chunk_id": str(uuid.uuid4()), "source": "doc_A.tex", "original_type": "latex",
                "source_path": "doc_A.tex", "original_doc_type": "latex",
                "concept_type": "general_content", "concept_name": None, # Example of a chunk without a specific concept name
                "chunk_text": "Vectors can be added using the parallelogram law. For example, $v + w = u$. They are fundamental in physics and engineering.",
                "parent_block_content": "Vectors can be added using the parallelogram law. For example, $v + w = u$. They are fundamental in physics and engineering.",
                "sequence_in_block": 0 # If it's the only chunk from this parent block
            },
            {
                "chunk_id": str(uuid.uuid4()), "source": "doc_B.pdf", "original_type": "pdf",
                "source_path": "doc_B.pdf", "original_doc_type": "pdf",
                "concept_type": "definition", "concept_name": "Derivative Definition", # More specific concept
                "chunk_text": "PDFs can also contain text about mathematics, like calculus. The derivative, denoted as $\\frac{dy}{dx}$ or $f'(x)$, measures the rate of change.",
                "parent_block_content": "From a section on Calculus Basics: PDFs can also contain text about mathematics, like calculus. The derivative, denoted as $\\frac{dy}{dx}$ or $f'(x)$, measures the rate of change.",
                "sequence_in_block": 1 # Assuming it's the second chunk in a larger parent block
            },
            { # Example of an empty chunk text to test skipping
                "chunk_id": str(uuid.uuid4()), "source_path": "doc_C.txt", "original_doc_type": "text",
                "concept_type": "empty_content", "concept_name": "Empty Section",
                "chunk_text": "", # Empty text
                "parent_block_content": "This section was intentionally left blank.",
                "sequence_in_block": 0
            }
        ]
        embed_and_store_chunks(client, dummy_chunks_for_storage, batch_size=2) # Small batch for demo

        # --- Querying Weaviate ---
        print("\n--- Querying Weaviate (Example BM25 - Keyword Search) ---")
        # Note: BM25 works on text properties. 'chunk_text' is searchable, 'concept_name' too.
        bm25_query = "vectors physics properties"
        query_result_bm25 = (
            client.query
            .get(WEAVIATE_CLASS_NAME, ["chunk_text", "source_path", "concept_name", "_additional {score id}"]) # score is BM25 score
            .with_bm25(query=bm25_query, properties=["chunk_text^2", "concept_name"]) # Boost chunk_text
            .with_limit(3)
            .do()
        )
        print(f"BM25 Query result for '{bm25_query}':")
        print(json.dumps(query_result_bm25, indent=2))

        print("\n--- Querying Weaviate (Example Vector Search - Semantic) ---")
        query_text_vector = "mathematical properties of vectors and their addition"
        query_embedding = generate_standard_embedding(query_text_vector)

        if query_embedding:
            vector_search_result = (
                client.query
                .get(WEAVIATE_CLASS_NAME, ["chunk_text", "source_path", "concept_name", "_additional {distance certainty id}"]) # distance & certainty
                .with_near_vector({
                    "vector": query_embedding,
                    "certainty": 0.65 # Adjust certainty threshold as needed (0.0 to 1.0)
                                      # For cosine distance, certainty = (1 + cosine_similarity) / 2
                                      # A distance of 0.0 (identical vectors) means certainty 1.0
                                      # A distance of 1.0 (orthogonal for normalized vectors) means certainty 0.5
                                      # A distance of 2.0 (opposite for normalized vectors) means certainty 0.0
                })
                .with_limit(3)
                .do()
            )
            print(f"Vector Search result for '{query_text_vector}':")
            print(json.dumps(vector_search_result, indent=2))
        else:
            print(f"Could not generate embedding for query: '{query_text_vector}'")

        print("\n--- Querying Weaviate (Hybrid Search Example) ---")
        hybrid_query = "calculus derivative definition"
        hybrid_query_embedding = generate_standard_embedding(hybrid_query)
        if hybrid_query_embedding:
            hybrid_result = (
                client.query
                .get(WEAVIATE_CLASS_NAME, ["chunk_text", "source_path", "concept_name", "_additional {score explainScore distance certainty id}"])
                .with_hybrid(
                    query=hybrid_query, # For keyword part
                    vector=hybrid_query_embedding, # For vector part
                    alpha=0.5, # 0 for pure keyword, 1 for pure vector. 0.5 is balanced.
                    properties=["chunk_text^2", "concept_name"] # properties for BM25 part
                )
                .with_limit(3)
                .do()
            )
            print(f"Hybrid Search result for '{hybrid_query}':")
            print(json.dumps(hybrid_result, indent=2))
        else:
            print(f"Could not generate embedding for hybrid query: '{hybrid_query}'")


    except ConnectionError as ce:
        print(f"Connection Error: Could not connect to Weaviate. {ce}")
        print("Please ensure Weaviate is running and accessible.")
    except Exception as e:
        print(f"An error occurred during the vector store manager demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n--- Vector Store Manager Demo Finished ---")