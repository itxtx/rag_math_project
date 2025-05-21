# src/retrieval/retriever.py
import weaviate
import weaviate.exceptions
import json # For logging raw response
from typing import List, Dict, Optional, Union

from src.data_ingestion import vector_store_manager
from src import config 

class Retriever:
    """
    A class to retrieve relevant text chunks from a Weaviate vector store.
    """

    DEFAULT_RETURN_PROPERTIES = [
        "chunk_text", "source_path", "original_doc_type", # Corrected here
        "concept_type", "concept_name", "sequence_in_block", "chunk_id",
        "parent_block_id", "parent_block_content", "doc_id", "filename" 
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

    def get_all_chunks_metadata(self, properties: Optional[List[str]] = None) -> List[Dict]:
        """
        Retrieves metadata for all chunks in the specified class.
        Useful for building an initial curriculum graph.
        """
        if properties is None:
            # --- CORRECTED PROPERTY NAME HERE ---
            properties = ["parent_block_id", "concept_name", "concept_type", 
                          "source_path", "original_doc_type", "doc_id", # Was original_type
                          "chunk_id", "sequence_in_block"] 
            # --- END OF CORRECTION ---
        
        print(f"Retriever: Fetching all chunk metadata for curriculum graph (properties: {properties})...")
        try:
            response = (
                self.client.query
                .get(self.weaviate_class_name, properties)
                .with_limit(10000) 
                .do()
            )
            
            print(f"DEBUG Retriever (get_all_chunks_metadata): Raw Weaviate response (first 500 chars): {str(response)[:500]}...")
            if response and "data" in response and "Get" in response["data"] and \
               response["data"]["Get"].get(self.weaviate_class_name):
                num_items_retrieved = len(response["data"]["Get"][self.weaviate_class_name])
                print(f"DEBUG Retriever (get_all_chunks_metadata): Number of items in response: {num_items_retrieved}")
            else:
                print("DEBUG Retriever (get_all_chunks_metadata): Response structure not as expected or no items.")

            formatted_results = self._format_results(response, requested_properties=properties)
            
            print(f"Retriever: Fetched metadata for {len(formatted_results)} chunks via get_all_chunks_metadata.")
            return formatted_results
        except Exception as e:
            print(f"Retriever: Error fetching all chunk metadata: {e}")
            import traceback
            traceback.print_exc()
            return []


    def get_chunks_for_parent_block(self, parent_block_id: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Retrieves all text chunks belonging to a specific parent conceptual block,
        ordered by their sequence within that block.
        """
        print(f"Retriever: Fetching chunks for parent_block_id: {parent_block_id}")
        filters = {
            "operator": "Equal",
            "path": ["parent_block_id"],
            "valueText": parent_block_id
        }
        
        query_limit = limit if limit is not None else 50 

        try:
            query_chain = (
                self.client.query
                .get(self.weaviate_class_name, self.DEFAULT_RETURN_PROPERTIES)
                .with_where(filters)
                .with_sort([{'path': ['sequence_in_block'], 'order': 'asc'}]) 
                .with_limit(query_limit) 
                .with_additional(self.DEFAULT_ADDITIONAL_PROPERTIES) 
            )
            response = query_chain.do()
            results = self._format_results(response) 
            print(f"Retriever: Found {len(results)} chunks for parent_block_id '{parent_block_id}'.")
            return results
        except Exception as e:
            print(f"Retriever: Error fetching chunks for parent_block_id '{parent_block_id}': {e}")
            return []

    def semantic_search(self,
                        query_text: str,
                        limit: Optional[int] = None,
                        certainty: Optional[float] = None,
                        filters: Optional[Dict] = None,
                        return_properties: Optional[List[str]] = None,
                        additional_properties: Optional[List[str]] = None) -> List[Dict]:
        query_embedding = self._embed_query(query_text)
        if query_embedding is None:
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
            return self._format_results(response, requested_properties=props_to_return)
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
            return self._format_results(response, requested_properties=props_to_return)
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
            return self._format_results(response, requested_properties=props_to_return)
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
        current_return_props = return_properties if return_properties else self.DEFAULT_RETURN_PROPERTIES
        current_add_props = additional_properties if additional_properties else self.DEFAULT_ADDITIONAL_PROPERTIES

        if search_type == "semantic":
            return self.semantic_search(query_text=query_text, limit=limit, certainty=certainty,
                                        filters=filters, return_properties=current_return_props,
                                        additional_properties=current_add_props)
        elif search_type == "hybrid":
            return self.hybrid_search(query_text=query_text, alpha=alpha, limit=limit,
                                      filters=filters, bm25_query=bm25_query,
                                      bm25_properties=hybrid_bm25_properties,
                                      return_properties=current_return_props,
                                      additional_properties=current_add_props,
                                      autocut=autocut)
        elif search_type == "keyword":
            return self.keyword_search(query_text=query_text, properties=keyword_properties,
                                       limit=limit, filters=filters,
                                       return_properties=current_return_props,
                                       additional_properties=current_add_props)
        else:
            print(f"Retriever: Unknown search type '{search_type}'.")
            return []

if __name__ == '__main__':
    print("--- Retriever Demo (with get_chunks_for_parent_block & get_all_chunks_metadata) ---")
    # ... (rest of the demo) ...
