# src/retrieval/retriever.py
import weaviate
import weaviate.exceptions
from typing import List, Dict, Optional, Union

from src.data_ingestion import vector_store_manager
from src import config 

class Retriever:
    """
    A class to retrieve relevant text chunks from a Weaviate vector store.
    """

    DEFAULT_RETURN_PROPERTIES = [
        "chunk_text", "source_path", "original_doc_type",
        "concept_type", "concept_name", "sequence_in_block", "chunk_id",
        "parent_block_id", "parent_block_content" 
    ]
    DEFAULT_ADDITIONAL_PROPERTIES = ["id", "distance", "certainty", "score"]

    def __init__(self,
                 weaviate_client: Optional[weaviate.Client] = None,
                 embedding_model = None, 
                 weaviate_class_name: Optional[str] = None):
        try:
            self.client = weaviate_client if weaviate_client else vector_store_manager.get_weaviate_client()
            print("Retriever: Weaviate client initialized successfully.")
        except Exception as e:
            print(f"Retriever: Error initializing Weaviate client: {e}")
            raise

        self._embedding_model = embedding_model
        self._embedding_model_instance = None 

        self.weaviate_class_name = weaviate_class_name if weaviate_class_name \
                                   else vector_store_manager.WEAVIATE_CLASS_NAME
        self.default_limit = getattr(config, 'DEFAULT_SEARCH_LIMIT', 5)
        self.default_semantic_certainty = getattr(config, 'DEFAULT_SEMANTIC_CERTAINTY', 0.7)
        self.default_hybrid_alpha = getattr(config, 'DEFAULT_HYBRID_ALPHA', 0.5)
        self.default_hybrid_bm25_properties = ["chunk_text^2", "concept_name"]


    def _get_embedding_model_instance(self):
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
        if not query_text:
            print("Retriever: Query text is empty, cannot embed.")
            return None
        try:
            model = self._get_embedding_model_instance()
            embedding = vector_store_manager.generate_standard_embedding(query_text, model_instance=model)
            return embedding
        except Exception as e:
            print(f"Retriever: Error embedding query '{query_text[:50]}...': {e}")
            return None

    def _format_results(self, weaviate_response: Dict) -> List[Dict]:
        results = []
        if not weaviate_response or "data" not in weaviate_response or \
           "Get" not in weaviate_response["data"] or not weaviate_response["data"]["Get"].get(self.weaviate_class_name):
            if weaviate_response and "errors" in weaviate_response:
                 print(f"Retriever: Weaviate query returned errors: {weaviate_response['errors']}")
            return results

        retrieved_items = weaviate_response["data"]["Get"][self.weaviate_class_name]
        for item in retrieved_items:
            result = {}
            for prop in self.DEFAULT_RETURN_PROPERTIES:
                if prop in item:
                    result[prop] = item[prop]
            if "_additional" in item and item["_additional"] is not None:
                for add_prop in self.DEFAULT_ADDITIONAL_PROPERTIES:
                    if add_prop in item["_additional"]:
                        result[f"_{add_prop}"] = item["_additional"][add_prop]
            results.append(result)
        return results

    def get_all_chunks_metadata(self, properties: Optional[List[str]] = None) -> List[Dict]:
        """
        Retrieves metadata for all chunks in the specified class.
        Useful for building an initial curriculum graph.

        Args:
            properties: A list of properties to retrieve. Defaults to a minimal set
                        for curriculum building (e.g., chunk_id, source_path, concept_name, 
                        concept_type, parent_block_id, sequence_in_block).

        Returns:
            A list of dictionaries, each containing metadata for a chunk.
        """
        if properties is None:
            properties = ["chunk_id", "source_path", "concept_name", "concept_type", "parent_block_id", "sequence_in_block"]
        
        print(f"Retriever: Fetching all chunk metadata for curriculum graph (properties: {properties})...")
        try:
            # We need to fetch all, so no limit, but this could be very large.
            # For very large datasets, this approach might need optimization or pagination.
            response = (
                self.client.query
                .get(self.weaviate_class_name, properties)
                # .with_limit(10000) # Example: Add a practical limit if needed
                .do()
            )
            # _format_results is designed for search results with _additional, so adapt
            formatted_results = []
            if response and "data" in response and "Get" in response["data"] and \
               response["data"]["Get"].get(self.weaviate_class_name):
                items = response["data"]["Get"][self.weaviate_class_name]
                for item in items:
                    # Ensure all requested properties are present, defaulting to None if not
                    formatted_item = {prop: item.get(prop) for prop in properties}
                    formatted_results.append(formatted_item)
            
            print(f"Retriever: Fetched metadata for {len(formatted_results)} chunks.")
            return formatted_results
        except Exception as e:
            print(f"Retriever: Error fetching all chunk metadata: {e}")
            return []


    def get_chunks_for_parent_block(self, parent_block_id: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Retrieves all text chunks belonging to a specific parent conceptual block,
        ordered by their sequence within that block.

        Args:
            parent_block_id: The ID of the parent conceptual block.
            limit: Optional limit on the number of chunks to return.

        Returns:
            A list of chunk dictionaries, ordered by sequence_in_block.
        """
        print(f"Retriever: Fetching chunks for parent_block_id: {parent_block_id}")
        filters = {
            "operator": "Equal",
            "path": ["parent_block_id"],
            "valueText": parent_block_id
        }
        
        limit = limit if limit is not None else self.default_limit # Default limit if many chunks per block

        try:
            query_chain = (
                self.client.query
                .get(self.weaviate_class_name, self.DEFAULT_RETURN_PROPERTIES)
                .with_where(filters)
                .with_sort([{'path': ['sequence_in_block'], 'order': 'asc'}]) # Order by sequence
                .with_limit(limit) # Limit the number of chunks if necessary
                .with_additional(self.DEFAULT_ADDITIONAL_PROPERTIES)
            )
            response = query_chain.do()
            results = self._format_results(response)
            print(f"Retriever: Found {len(results)} chunks for parent_block_id '{parent_block_id}'.")
            return results
        except Exception as e:
            print(f"Retriever: Error fetching chunks for parent_block_id '{parent_block_id}': {e}")
            return []

    # semantic_search, hybrid_search, keyword_search, search methods remain the same as retriever_v1
    # ... (previous search methods from question_selector_v1's retriever context) ...
    def semantic_search(self,
                        query_text: str,
                        limit: Optional[int] = None,
                        certainty: Optional[float] = None,
                        filters: Optional[Dict] = None,
                        return_properties: Optional[List[str]] = None,
                        additional_properties: Optional[List[str]] = None) -> List[Dict]:
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
            if filters: query_chain = query_chain.with_where(filters)
            if add_props: query_chain = query_chain.with_additional(add_props)
            response = query_chain.do()
            return self._format_results(response)
        except Exception as e:
            print(f"Retriever: Error during semantic search: {e}")
        return []

    def hybrid_search(self,
                      query_text: str,
                      alpha: Optional[float] = None,
                      limit: Optional[int] = None,
                      filters: Optional[Dict] = None,
                      bm25_query: Optional[str] = None,
                      bm25_properties: Optional[List[str]] = None,
                      return_properties: Optional[List[str]] = None,
                      additional_properties: Optional[List[str]] = None,
                      autocut: Optional[int] = None) -> List[Dict]:
        query_embedding = self._embed_query(query_text)
        if query_embedding is None and (alpha is None or alpha > 0) :
             print(f"Retriever: Could not perform hybrid search for '{query_text[:50]}...' due to embedding failure for vector part.")
             return []

        alpha = alpha if alpha is not None else self.default_hybrid_alpha
        limit = limit if limit is not None else self.default_limit
        props_to_return = return_properties if return_properties else self.DEFAULT_RETURN_PROPERTIES
        add_props = additional_properties if additional_properties else self.DEFAULT_ADDITIONAL_PROPERTIES
        keyword_query_str = bm25_query if bm25_query else query_text
        target_bm25_props = bm25_properties if bm25_properties else self.default_hybrid_bm25_properties
        hybrid_params = {"query": keyword_query_str, "alpha": alpha, "properties": target_bm25_props}
        if query_embedding: hybrid_params["vector"] = query_embedding

        try:
            query_chain = self.client.query.get(self.weaviate_class_name, props_to_return)
            query_chain = query_chain.with_hybrid(**hybrid_params)
            query_chain = query_chain.with_limit(limit)
            if filters: query_chain = query_chain.with_where(filters)
            if add_props: query_chain = query_chain.with_additional(add_props)
            if autocut is not None: query_chain = query_chain.with_autocut(autocut)
            response = query_chain.do()
            return self._format_results(response)
        except Exception as e:
            print(f"Retriever: Error during hybrid search: {e}")
        return []

    def keyword_search(self,
                       query_text: str,
                       properties: Optional[List[str]] = None,
                       limit: Optional[int] = None,
                       filters: Optional[Dict] = None,
                       return_properties: Optional[List[str]] = None,
                       additional_properties: Optional[List[str]] = None) -> List[Dict]:
        limit = limit if limit is not None else self.default_limit
        target_props = properties if properties else ["chunk_text", "concept_name"]
        props_to_return = return_properties if return_properties else self.DEFAULT_RETURN_PROPERTIES
        add_props = additional_properties if additional_properties else self.DEFAULT_ADDITIONAL_PROPERTIES
        bm25_params = {"query": query_text, "properties": target_props}

        try:
            query_chain = self.client.query.get(self.weaviate_class_name, props_to_return)
            query_chain = query_chain.with_bm25(**bm25_params)
            query_chain = query_chain.with_limit(limit)
            if filters: query_chain = query_chain.with_where(filters)
            if add_props: query_chain = query_chain.with_additional(add_props)
            response = query_chain.do()
            return self._format_results(response)
        except Exception as e:
            print(f"Retriever: Error during keyword search: {e}")
        return []

    def search(self,
               query_text: str,
               search_type: str = "semantic", 
               limit: Optional[int] = None,
               certainty: Optional[float] = None,
               alpha: Optional[float] = None,
               bm25_query: Optional[str] = None,
               hybrid_bm25_properties: Optional[List[str]] = None, 
               autocut: Optional[int] = None,
               keyword_properties: Optional[List[str]] = None, 
               filters: Optional[Dict] = None,
               return_properties: Optional[List[str]] = None,
               additional_properties: Optional[List[str]] = None
               ) -> List[Dict]:
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
            print(f"Retriever: Unknown search type '{search_type}'.")
            return []

if __name__ == '__main__':
    print("--- Retriever Demo (with get_chunks_for_parent_block) ---")
    try:
        # This demo assumes Weaviate is running and has data with 'parent_block_id'
        # and 'sequence_in_block' properties.
        # You would typically get a parent_block_id from a previously identified conceptual block.
        
        # Ensure config.py has WEAVIATE_URL
        if not hasattr(config, 'WEAVIATE_URL'):
             print("Please set WEAVIATE_URL in src/config.py")
        else:
            client = vector_store_manager.get_weaviate_client() # For direct client use if needed
            ret = Retriever(weaviate_client=client)
            print("\nRetriever initialized.")

            # Example: Try to fetch chunks for a known parent_block_id from your data
            # Replace 'some_known_parent_block_id' with an actual ID from your Weaviate store.
            # This ID would correspond to a `block_id` from your `concept_tagger`.
            # For the demo to work, you need to have ingested data where chunks have this parent_block_id.
            
            # First, let's get some metadata to find a parent_block_id to test with
            all_meta = ret.get_all_chunks_metadata(properties=["parent_block_id", "chunk_id", "concept_name"])
            test_parent_block_id = None
            if all_meta and all_meta[0].get("parent_block_id"):
                test_parent_block_id = all_meta[0]["parent_block_id"]
                print(f"\nFound a test parent_block_id from existing data: {test_parent_block_id}")
            else:
                print("\nCould not find a test parent_block_id from existing data. "
                      "get_chunks_for_parent_block demo might not show results.")
                # You could manually set one if you know one exists:
                # test_parent_block_id = "your-actual-parent-block-id-here"


            if test_parent_block_id:
                print(f"\n--- Testing get_chunks_for_parent_block for ID: {test_parent_block_id} ---")
                # Fetch up to 3 chunks for this parent block
                concept_chunks = ret.get_chunks_for_parent_block(test_parent_block_id, limit=3) 
                if concept_chunks:
                    print(f"Found {len(concept_chunks)} chunks for the concept:")
                    for i, chunk in enumerate(concept_chunks):
                        print(f"  Chunk {i+1} (Sequence: {chunk.get('sequence_in_block')}):")
                        print(f"    Text: {chunk.get('chunk_text', '')[:100]}...")
                        print(f"    Chunk ID: {chunk.get('chunk_id')}")
                else:
                    print(f"No chunks found for parent_block_id: {test_parent_block_id}")
            else:
                print("Skipping get_chunks_for_parent_block demo as no test_parent_block_id was found.")
            
            print("\n--- Retriever Demo Finished ---")

    except Exception as e:
        print(f"An error occurred during the Retriever demo: {e}")
        import traceback
        traceback.print_exc()
