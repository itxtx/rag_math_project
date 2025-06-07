# src/gnn_training/train.py
import torch
import pickle
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from .model import LinkPredictorGNN

# --- Configuration ---
GRAPH_PATH = 'data/graph_db/knowledge_graph.graphml'
INITIAL_EMBEDDINGS_PATH = 'data/embeddings/initial_text_embeddings.pkl'
GNN_MODEL_SAVE_PATH = 'models/gnn/best_model.pt'
FINAL_EMBEDDINGS_SAVE_PATH = 'data/embeddings/gnn_embeddings.pkl'

# --- GNN Hyperparameters ---
HIDDEN_CHANNELS = 128
OUT_CHANNELS = 64
LEARNING_RATE = 0.01
EPOCHS = 200

def load_data():
    """Loads the graph and initial embeddings."""
    print("Loading knowledge graph...")
    G = nx.read_graphml(GRAPH_PATH)
    
    print("Loading initial text embeddings...")
    with open(INITIAL_EMBEDDINGS_PATH, 'rb') as f:
        initial_embeddings_dict = pickle.load(f)

    # Convert to PyG data object
    pyg_data = from_networkx(G)
    
    # Create feature matrix in the correct order
    node_order = list(G.nodes())
    embedding_dim = list(initial_embeddings_dict.values())[0].shape[0]
    
    x = torch.zeros((len(node_order), embedding_dim))
    for i, node_id in enumerate(node_order):
        if node_id in initial_embeddings_dict:
            x[i] = torch.tensor(initial_embeddings_dict[node_id])
    
    pyg_data.x = x
    return pyg_data, node_order

def prepare_link_prediction_data(data):
    """Splits edges into train/val and creates negative samples."""
    edges = data.edge_index.t().cpu().numpy()
    
    # Basic train/test split of edges
    train_edges, test_edges = train_test_split(edges, test_size=0.2, random_state=42)

    # Create negative samples (edges that do not exist in the graph)
    num_nodes = data.num_nodes
    all_possible_edges = set(map(tuple, edges))
    
    neg_edges = []
    while len(neg_edges) < len(edges):
        u, v = np.random.randint(0, num_nodes, 2)
        if u != v and (u, v) not in all_possible_edges and (v, u) not in all_possible_edges:
            neg_edges.append([u, v])
            
    neg_edges = np.array(neg_edges)
    train_neg_edges, test_neg_edges = train_test_split(neg_edges, test_size=0.2, random_state=42)

    data.train_pos_edge_index = torch.tensor(train_edges, dtype=torch.long).t().contiguous()
    data.test_pos_edge_index = torch.tensor(test_edges, dtype=torch.long).t().contiguous()
    data.train_neg_edge_index = torch.tensor(train_neg_edges, dtype=torch.long).t().contiguous()
    data.test_neg_edge_index = torch.tensor(test_neg_edges, dtype=torch.long).t().contiguous()
    
    return data

def run_training():
    """Main function to run the GNN training pipeline."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data, node_order = load_data()
    data = prepare_link_prediction_data(data).to(device)

    model = LinkPredictorGNN(
        in_channels=data.num_node_features,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUT_CHANNELS
    ).to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()

    print("Starting GNN training for link prediction...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        
        z = model.encode(data.x, data.train_pos_edge_index)
        
        # Positive edges
        pos_out = model.decode(z, data.train_pos_edge_index)
        pos_loss = criterion(pos_out, torch.ones_like(pos_out))
        
        # Negative edges
        neg_out = model.decode(z, data.train_neg_edge_index)
        neg_loss = criterion(neg_out, torch.zeros_like(neg_out))
        
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss.item():.4f}')

    print("Training finished.")

    # --- Save Artifacts ---
    # 1. Save the final GNN-aware embeddings (Use Case 1)
    print("Generating and saving final graph-aware embeddings...")
    model.eval()
    with torch.no_grad():
        final_embeddings_tensor = model.encode(data.x, data.edge_index)
    
    final_embeddings_dict = {
        node_id: emb.cpu().numpy() 
        for node_id, emb in zip(node_order, final_embeddings_tensor)
    }
    with open(FINAL_EMBEDDINGS_SAVE_PATH, 'wb') as f:
        pickle.dump(final_embeddings_dict, f)
    print(f"Saved GNN embeddings to {FINAL_EMBEDDINGS_SAVE_PATH}")

    # 2. Save the trained model for future inference (Use Case 3)
    torch.save(model.state_dict(), GNN_MODEL_SAVE_PATH)
    print(f"Saved trained GNN model to {GNN_MODEL_SAVE_PATH}")
