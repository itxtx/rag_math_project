# src/retrieval/hybrid_retriever.py
import faiss
import pickle
import torch
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
from src.gnn_training.model import LinkPredictorGNN

class HybridRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print("Initializing HybridRetriever...")
        self.client = None  # For legacy test compatibility
        self.query_embedder = SentenceTransformer(model_name)
        
        # --- Load Knowledge Graph ---
        self.graph = nx.read_graphml('data/graph_db/knowledge_graph.graphml')
        self.node_list = list(self.graph.nodes())
        self.node_to_idx = {node: i for i, node in enumerate(self.node_list)}

        # --- Load Initial Text Embeddings & Build FAISS Index ---
        with open('data/embeddings/initial_text_embeddings.pkl', 'rb') as f:
            text_embeddings_dict = pickle.load(f)
        
        self.text_embeddings = np.array([text_embeddings_dict.get(node) for node in self.node_list if node in text_embeddings_dict])
        if self.text_embeddings.ndim == 2:
            self.text_index = faiss.IndexFlatIP(self.text_embeddings.shape[1])
            self.text_index.add(self.text_embeddings)
            print("Loaded text embeddings and FAISS index.")
        else:
            print("Warning: Could not create FAISS index for text embeddings due to shape mismatch.")
            self.text_index = None

        # --- Load GNN Embeddings & Build FAISS Index ---
        with open('data/embeddings/gnn_embeddings.pkl', 'rb') as f:
            gnn_embeddings_dict = pickle.load(f)

        self.gnn_embeddings = np.array([gnn_embeddings_dict.get(node) for node in self.node_list if node in gnn_embeddings_dict])
        if self.gnn_embeddings.ndim == 2:
            self.gnn_index = faiss.IndexFlatIP(self.gnn_embeddings.shape[1])
            self.gnn_index.add(self.gnn_embeddings)
            print("Loaded GNN embeddings and FAISS index.")
        else:
            print("Warning: Could not create FAISS index for GNN embeddings due to shape mismatch.")
            self.gnn_index = None
        
        # --- Load Trained GNN Model for Link Prediction ---
        self.gnn_model = LinkPredictorGNN(
            in_channels=self.text_embeddings.shape[1] if self.text_embeddings.ndim == 2 else 768,
            hidden_channels=128,
            out_channels=64
        )
        self.gnn_model.load_state_dict(torch.load('models/gnn/best_model.pt'))
        self.gnn_model.eval()
        print("Loaded trained GNN model for inference.")

    def search(self, query: str, k: int = 5) -> list:
        """Performs hybrid search and returns a list of candidate node IDs."""
        print(f"\nPerforming hybrid search for query: '{query}'")
        query_vec = self.query_embedder.encode([query], normalize_embeddings=True)

        # Search Text Index
        _, text_indices = self.text_index.search(query_vec, k)
        
        # Search GNN Index
        _, gnn_indices = self.gnn_index.search(query_vec, k)

        # Fuse results
        text_nodes = [self.node_list[i] for i in text_indices[0]]
        gnn_nodes = [self.node_list[i] for i in gnn_indices[0]]
        
        fused_node_ids = list(dict.fromkeys(text_nodes + gnn_nodes)) # Union while preserving order
        print(f"Found candidate nodes: {fused_node_ids}")
        return fused_node_ids

    def get_context_for_nodes(self, node_ids: list, depth: int = 1) -> dict:
        """
        Given a list of starting nodes, traverses the KG to get their context.
        """
        context_nodes = set(node_ids)
        for node_id in node_ids:
            # Go backwards (dependencies)
            deps = set(nx.ancestors(self.graph, node_id))
            # Go forwards (what this node proves/defines)
            consequences = set(nx.descendants(self.graph, node_id))
            context_nodes.update(deps, consequences)
        
        context_data = {
            node: self.graph.nodes[node] for node in context_nodes if node in self.graph
        }
        print(f"Expanded context to {len(context_data)} nodes.")
        return context_data
        
    def predict_related_links(self, node_id: str, top_k: int = 3):
        """Uses the trained GNN to predict new, un-linked related nodes."""
        if node_id not in self.node_to_idx:
            return []
        
        print(f"\nPredicting related links for node: '{node_id}'")
        with torch.no_grad():
            z = torch.tensor(self.gnn_embeddings)
            source_idx = self.node_to_idx[node_id]
            
            # Create pairs of (source, all_other_nodes)
            edge_index = torch.tensor([
                [source_idx] * len(self.node_list),
                list(range(len(self.node_list)))
            ])
            
            # Get similarity scores from the GNN decoder
            scores = self.gnn_model.decode(z, edge_index)
            
            # Find top k, excluding self and existing neighbors
            scores[source_idx] = -float('inf') # Exclude self
            for neighbor in self.graph.neighbors(node_id):
                if neighbor in self.node_to_idx:
                    scores[self.node_to_idx[neighbor]] = -float('inf')

            top_indices = torch.topk(scores, top_k).indices
            predicted_nodes = [self.node_list[i] for i in top_indices]
            print(f"Predicted related nodes: {predicted_nodes}")
            return predicted_nodes
