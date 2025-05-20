import weaviate
import weaviate.exceptions
from typing import List, Dict, Optional, Union

# Assuming your vector_store_manager and config are structured as previously discussed
# Adjust these imports if your project structure is different.
from src.data_ingestion import vector_store_manager
from src import config # May not be directly needed if vector_store_manager handles config

class Retriever:
    """
    A class to retrieve relevant text chunks from a Weaviate vector store.
    """

    DEFAULT_RETURN_PROPERTIES = [
        "chunk_text", "source_path", "original_doc_type",
        "concept_type", "concept_name", "sequence_in_block", "chunk_id",
        "parent_block_content" # Added this as it can be useful context
    ]
    DEFAULT_ADDITIONAL_PROPERTIES = ["id", "distance", "certainty", "score"] # score for BM25/hybrid

    def __init__(self,
                 weaviate_client: Optional[weaviate.Client] = None,
                 embedding_model = None, # Using 'embedding_model' to match SentenceTransformer type hint
                 weaviate_class_name: Optional[str] = None):
        """
        Initializes the Retriever.

        Args:
            weaviate_client: An optional pre-configured Weaviate client.
                             If None, a new client will be initialized.
            embedding_model: An optional pre-loaded sentence-transformer model.
                             If None, the default model will be loaded on demand.
            weaviate_class_name: The name of the Weaviate class to query.
                                 If None, uses WEAVIATE_CLASS_NAME from vector_store_manager.
        """
        try:
            self.client = weaviate_client if weaviate_client else vector_store_manager.get_weaviate_client()
            print("Retriever: Weaviate client initialized successfully.")
        except Exception as e:
            print(f"Retriever: Error initializing Weaviate client: {e}")
            # Depending on desired behavior, either raise e or set self.client to None
            # and handle it in search methods. For now, let it raise.
            raise

        # Defer loading of embedding_model until it's actually needed by a search method
        self._embedding_model = embedding_model
        self._embedding_model_instance = None # To store the loaded instance

        self.weaviate_class_name = weaviate_class_name if weaviate_class_name \
                                   else vector_store_manager.WEAVIATE_CLASS_NAME
        self.default_limit = config.DEFAULT_SEARCH_LIMIT if hasattr(config, 'DEFAULT_SEARCH_LIMIT') else 5
        self.default_semantic_certainty = config.DEFAULT_SEMANTIC_CERTAINTY if hasattr(config, 'DEFAULT_SEMANTIC_CERTAINTY') else 0.7
        self.default_hybrid_alpha = config.DEFAULT_HYBRID_ALPHA if hasattr(config, 'DEFAULT_HYBRID_ALPHA') else 0.5

        # Properties to use for BM25/keyword part of hybrid search if not specified
        self.default_hybrid_bm25_properties = ["chunk_text^2", "concept_name"]


    def _get_embedding_model_instance(self):
        """Lazy loads and returns the sentence transformer model instance."""
        if self._embedding_model_instance is None:
            if self._embedding_model is not None:
                self._embedding_model_instance = self._embedding_model
            else:
                try:
                    self._embedding_model_instance = vector_store_manager.get_embedding_model()
                    print("Retriever: Embedding model loaded successfully.")
                except Exception as e:
                    print(f"Retriever: Error loading embedding model: {e}")
                    raise
        return self._embedding_model_instance

    def _embed_query(self, query_text: str) -> Optional[List[float]]:
        """
        Embeds the query text using the configured sentence-transformer model.

        Args:
            query_text: The text to embed.

        Returns:
            A list of floats representing the embedding, or None if embedding fails.
        """
        if not query_text:
            print("Retriever: Query text is empty, cannot embed.")
            return None
        try:
            model = self._get_embedding_model_instance()
            # Assuming generate_standard_embedding handles potential errors and returns None or list[float]
            embedding = vector_store_manager.generate_standard_embedding(query_text, model_instance=model)
            return embedding
        except Exception as e:
            print(f"Retriever: Error embedding query '{query_text[:50]}...': {e}")
            return None

    def _format_results(self, weaviate_response: Dict) -> List[Dict]:
        """
        Formats the raw response from Weaviate into a list of result dictionaries.

        Args:
            weaviate_response: The raw JSON response from Weaviate's .do() call.

        Returns:
            A list of dictionaries, where each dictionary represents a retrieved chunk.
        """
        results = []
        if not weaviate_response or "data" not in weaviate_response or \
           "Get" not in weaviate_response["data"] or not weaviate_response["data"]["Get"].get(self.weaviate_class_name):
            if weaviate_response and "errors" in weaviate_response:
                 print(f"Retriever: Weaviate query returned errors: {weaviate_response['errors']}")
            return results

        retrieved_items = weaviate_response["data"]["Get"][self.weaviate_class_name]
        for item in retrieved_items:
            result = {}
            # Extract main properties
            for prop in self.DEFAULT_RETURN_PROPERTIES:
                if prop in item:
                    result[prop] = item[prop]

            # Extract _additional properties
            if "_additional" in item and item["_additional"] is not None:
                for add_prop in self.DEFAULT_ADDITIONAL_PROPERTIES:
                    if add_prop in item["_additional"]:
                        result[f"_{add_prop}"] = item["_additional"][add_prop]
            results.append(result)
        return results

    def semantic_search(self,
                        query_text: str,
                        limit: Optional[int] = None,
                        certainty: Optional[float] = None,
                        filters: Optional[Dict] = None,
                        return_properties: Optional[List[str]] = None,
                        additional_properties: Optional[List[str]] = None) -> List[Dict]:
        """
        Performs semantic (vector) search in Weaviate.

        Args:
            query_text: The query text.
            limit: The maximum number of results to return. Defaults to self.default_limit.
            certainty: The minimum certainty for results. Defaults to self.default_semantic_certainty.
            filters: Weaviate 'where' filter dictionary.
            return_properties: List of properties to return for each chunk.
            additional_properties: List of _additional properties to return (e.g., "distance", "id").

        Returns:
            A list of result dictionaries.
        """
        query_embedding = self._embed_query(query_text)
        if query_embedding is None:
            print(f"Retriever: Could not perform semantic search for '{query_text[:50]}...' due to embedding failure.")
            return []

        limit = limit if limit is not None else self.default_limit
        certainty = certainty if certainty is not None else self.default_semantic_certainty
        props_to_return = return_properties if return_properties else self.DEFAULT_RETURN_PROPERTIES
        add_props = additional_properties if additional_properties else self.DEFAULT_ADDITIONAL_PROPERTIES

        near_vector_filter = {"vector": query_embedding, "certainty": certainty}

        try:
            query_chain = self.client.query.get(self.weaviate_class_name, props_to_return)
            query_chain = query_chain.with_near_vector(near_vector_filter)
            query_chain = query_chain.with_limit(limit)

            if filters:
                query_chain = query_chain.with_where(filters)
            if add_props:
                query_chain = query_chain.with_additional(add_props)

            response = query_chain.do()
            return self._format_results(response)

        except weaviate.exceptions.WeaviateQueryException as e:
            print(f"Retriever: Weaviate query exception during semantic search: {e}")
        except Exception as e:
            print(f"Retriever: Unexpected error during semantic search: {e}")
        return []

    def hybrid_search(self,
                      query_text: str,
                      alpha: Optional[float] = None,
                      limit: Optional[int] = None,
                      filters: Optional[Dict] = None,
                      bm25_query: Optional[str] = None, # Optional separate query for BM25 part
                      bm25_properties: Optional[List[str]] = None,
                      return_properties: Optional[List[str]] = None,
                      additional_properties: Optional[List[str]] = None,
                      autocut: Optional[int] = None) -> List[Dict]:
        """
        Performs hybrid search (semantic + keyword) in Weaviate.

        Args:
            query_text: The primary query text (used for vector search part).
            alpha: Weighting factor for semantic vs. keyword search (0=keyword, 1=semantic).
                   Defaults to self.default_hybrid_alpha.
            limit: Maximum number of results. Defaults to self.default_limit.
            filters: Weaviate 'where' filter dictionary.
            bm25_query: Optional specific query string for the BM25 part. If None, query_text is used.
            bm25_properties: Properties to target for BM25 search. Defaults to self.default_hybrid_bm25_properties.
            return_properties: List of properties to return.
            additional_properties: List of _additional properties to return.
            autocut: Number of results to sharply drop off after for ranking. Experimental.

        Returns:
            A list of result dictionaries.
        """
        query_embedding = self._embed_query(query_text)
        if query_embedding is None and alpha > 0 : # Only fail if vector part is needed
             print(f"Retriever: Could not perform hybrid search for '{query_text[:50]}...' due to embedding failure for vector part.")
             return []

        alpha = alpha if alpha is not None else self.default_hybrid_alpha
        limit = limit if limit is not None else self.default_limit
        props_to_return = return_properties if return_properties else self.DEFAULT_RETURN_PROPERTIES
        add_props = additional_properties if additional_properties else self.DEFAULT_ADDITIONAL_PROPERTIES
        keyword_query_str = bm25_query if bm25_query else query_text
        target_bm25_props = bm25_properties if bm25_properties else self.default_hybrid_bm25_properties


        hybrid_params = {
            "query": keyword_query_str,
            "alpha": alpha,
            "properties": target_bm25_props
        }
        if query_embedding: # Only add vector if successfully embedded
            hybrid_params["vector"] = query_embedding


        try:
            query_chain = self.client.query.get(self.weaviate_class_name, props_to_return)
            query_chain = query_chain.with_hybrid(**hybrid_params)
            query_chain = query_chain.with_limit(limit)

            if filters:
                query_chain = query_chain.with_where(filters)
            if add_props:
                query_chain = query_chain.with_additional(add_props)
            if autocut is not None:
                query_chain = query_chain.with_autocut(autocut)


            response = query_chain.do()
            return self._format_results(response)

        except weaviate.exceptions.WeaviateQueryException as e:
            print(f"Retriever: Weaviate query exception during hybrid search: {e}")
        except Exception as e:
            print(f"Retriever: Unexpected error during hybrid search: {e}")
        return []


    def keyword_search(self,
                       query_text: str,
                       properties: Optional[List[str]] = None,
                       limit: Optional[int] = None,
                       filters: Optional[Dict] = None,
                       return_properties: Optional[List[str]] = None,
                       additional_properties: Optional[List[str]] = None) -> List[Dict]:
        """
        Performs keyword (BM25) search in Weaviate.

        Args:
            query_text: The keyword query text.
            properties: List of properties to search within (e.g., ["chunk_text", "concept_name"]).
                        Defaults to ["chunk_text", "concept_name"].
            limit: Maximum number of results. Defaults to self.default_limit.
            filters: Weaviate 'where' filter dictionary.
            return_properties: List of properties to return.
            additional_properties: List of _additional properties to return.

        Returns:
            A list of result dictionaries.
        """
        limit = limit if limit is not None else self.default_limit
        target_props = properties if properties else ["chunk_text", "concept_name"] # Sensible default
        props_to_return = return_properties if return_properties else self.DEFAULT_RETURN_PROPERTIES
        add_props = additional_properties if additional_properties else self.DEFAULT_ADDITIONAL_PROPERTIES

        bm25_params = {"query": query_text, "properties": target_props}

        try:
            query_chain = self.client.query.get(self.weaviate_class_name, props_to_return)
            query_chain = query_chain.with_bm25(**bm25_params)
            query_chain = query_chain.with_limit(limit)

            if filters:
                query_chain = query_chain.with_where(filters)
            if add_props:
                query_chain = query_chain.with_additional(add_props)

            response = query_chain.do()
            return self._format_results(response)
        except weaviate.exceptions.WeaviateQueryException as e:
            print(f"Retriever: Weaviate query exception during keyword search: {e}")
        except Exception as e:
            print(f"Retriever: Unexpected error during keyword search: {e}")
        return []

    def search(self,
               query_text: str,
               search_type: str = "semantic", # "semantic", "hybrid", "keyword"
               limit: Optional[int] = None,
               # Semantic search specific
               certainty: Optional[float] = None,
               # Hybrid search specific
               alpha: Optional[float] = None,
               bm25_query: Optional[str] = None,
               hybrid_bm25_properties: Optional[List[str]] = None, # Renamed for clarity
               autocut: Optional[int] = None,
               # Keyword search specific
               keyword_properties: Optional[List[str]] = None, # Renamed for clarity
               # Common
               filters: Optional[Dict] = None,
               return_properties: Optional[List[str]] = None,
               additional_properties: Optional[List[str]] = None
               ) -> List[Dict]:
        """
        Generic search method dispatching to specific search types.

        Args:
            query_text: The main query text.
            search_type: Type of search ("semantic", "hybrid", "keyword").
            limit: Max results.
            certainty: Min certainty for semantic search.
            alpha: Alpha for hybrid search.
            bm25_query: Specific BM25 query for hybrid.
            hybrid_bm25_properties: BM25 properties for hybrid.
            autocut: Autocut for hybrid.
            keyword_properties: Properties for keyword search.
            filters: Weaviate 'where' filter.
            return_properties: Properties to return.
            additional_properties: _additional properties to return.

        Returns:
            A list of result dictionaries.
        """
        search_type = search_type.lower()
        limit = limit if limit is not None else self.default_limit

        if search_type == "semantic":
            return self.semantic_search(query_text=query_text, limit=limit, certainty=certainty,
                                        filters=filters, return_properties=return_properties,
                                        additional_properties=additional_properties)
        elif search_type == "hybrid":
            return self.hybrid_search(query_text=query_text, alpha=alpha, limit=limit,
                                      filters=filters, bm25_query=bm25_query,
                                      bm25_properties=hybrid_bm25_properties,
                                      return_properties=return_properties,
                                      additional_properties=additional_properties,
                                      autocut=autocut)
        elif search_type == "keyword":
            return self.keyword_search(query_text=query_text, properties=keyword_properties,
                                       limit=limit, filters=filters,
                                       return_properties=return_properties,
                                       additional_properties=additional_properties)
        else:
            print(f"Retriever: Unknown search type '{search_type}'. Supported types are 'semantic', 'hybrid', 'keyword'.")
            return []

if __name__ == '__main__':
    print("--- Retriever Demo ---")
    # This demo assumes Weaviate is running and has the 'MathDocumentChunk' class
    # with some data, and the embedding model is accessible.

    # Ensure config.py has necessary values like WEAVIATE_URL, EMBEDDING_MODEL_NAME
    # For example, add to src/config.py:
    # DEFAULT_SEARCH_LIMIT = 5
    # DEFAULT_SEMANTIC_CERTAINTY = 0.70 # Adjust as needed
    # DEFAULT_HYBRID_ALPHA = 0.5

    try:
        retriever = Retriever()
        print("\nRetriever initialized.")

        # --- Test Semantic Search ---
        print("\n--- Testing Semantic Search ---")
        semantic_query = "properties of vectors"
        print(f"Query: '{semantic_query}'")
        semantic_results = retriever.semantic_search(semantic_query, limit=3, certainty=0.65)
        if semantic_results:
            print(f"Found {len(semantic_results)} semantic results:")
            for i, res in enumerate(semantic_results):
                print(f"  Result {i+1}:")
                print(f"    Chunk ID: {res.get('_id', 'N/A')}")
                print(f"    Text: {res.get('chunk_text', '')[:100]}...")
                print(f"    Source: {res.get('source_path', 'N/A')}")
                print(f"    Concept: {res.get('concept_name', 'N/A')}")
                print(f"    Certainty: {res.get('_certainty', 'N/A'):.4f}")
                print(f"    Distance: {res.get('_distance', 'N/A')}") # Ensure distance is requested if needed
        else:
            print("No semantic results found or error occurred.")

        # --- Test Hybrid Search ---
        print("\n--- Testing Hybrid Search ---")
        hybrid_query = "calculus derivative definition"
        print(f"Query: '{hybrid_query}'")
        # Note: Hybrid search needs properties for its BM25 part.
        # The default is ["chunk_text^2", "concept_name"]
        hybrid_results = retriever.hybrid_search(hybrid_query, alpha=0.5, limit=3)
        if hybrid_results:
            print(f"Found {len(hybrid_results)} hybrid results:")
            for i, res in enumerate(hybrid_results):
                print(f"  Result {i+1}:")
                print(f"    Chunk ID: {res.get('_id', 'N/A')}")
                print(f"    Text: {res.get('chunk_text', '')[:100]}...")
                print(f"    Source: {res.get('source_path', 'N/A')}")
                print(f"    Concept: {res.get('concept_name', 'N/A')}")
                print(f"    Score (hybrid): {res.get('_score', 'N/A')}") # Hybrid often returns _score
                # Certainty/distance might also be available if vector part dominates or explicitly requested
                if '_certainty' in res: print(f"    Certainty: {res['_certainty']:.4f}")
        else:
            print("No hybrid results found or error occurred.")

        # --- Test Keyword Search ---
        print("\n--- Testing Keyword (BM25) Search ---")
        keyword_query = "vectors physics"
        print(f"Query: '{keyword_query}'")
        keyword_results = retriever.keyword_search(keyword_query, properties=["chunk_text", "concept_name"], limit=3)
        if keyword_results:
            print(f"Found {len(keyword_results)} keyword results:")
            for i, res in enumerate(keyword_results):
                print(f"  Result {i+1}:")
                print(f"    Chunk ID: {res.get('_id', 'N/A')}")
                print(f"    Text: {res.get('chunk_text', '')[:100]}...")
                print(f"    Source: {res.get('source_path', 'N/A')}")
                print(f"    Concept: {res.get('concept_name', 'N/A')}")
                print(f"    Score (BM25): {res.get('_score', 'N/A')}")
        else:
            print("No keyword results found or error occurred.")

        # --- Test Semantic Search with Filters ---
        print("\n--- Testing Semantic Search with Filters ---")
        # This filter assumes you have data where concept_name is "Introduction to Vectors"
        # and the source_path ends with ".tex"
        # You might need to adjust the filter values based on your actual data.
        example_filter = {
            "operator": "And",
            "operands": [
                {
                    "path": ["concept_name"],
                    "operator": "Equal",
                    "valueText": "Introduction to Vectors" # Adjust if needed
                },
                {
                    "path": ["original_doc_type"], # Example filter
                    "operator": "Equal",
                    "valueText": "latex"
                }
            ]
        }
        print(f"Query: '{semantic_query}', Filter: concept_name='Introduction to Vectors' AND original_doc_type='latex'")
        filtered_results = retriever.semantic_search(semantic_query, limit=3, filters=example_filter)
        if filtered_results:
            print(f"Found {len(filtered_results)} filtered semantic results:")
            for i, res in enumerate(filtered_results):
                print(f"  Result {i+1}: Text: {res.get('chunk_text', '')[:60]}..., Concept: {res.get('concept_name')}, Type: {res.get('original_doc_type')}")
        else:
            print("No filtered semantic results found. Check filter criteria and data.")


    except ConnectionError as ce:
        print(f"Retriever Demo Error: Could not connect to Weaviate. {ce}")
        print("Please ensure Weaviate is running and accessible at the configured URL.")
    except Exception as e:
        print(f"An error occurred during the Retriever demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n--- Retriever Demo Finished ---")

