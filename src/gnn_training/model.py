# src/gnn_training/model.py
import torch
from torch_geometric.nn import SAGEConv

class LinkPredictorGNN(torch.nn.Module):
    """
    A GraphSAGE based GNN for learning node embeddings (encoding)
    and predicting links between them (decoding).
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # Encoder to learn node embeddings (Use Case 1)
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        """Generates final, graph-aware node embeddings."""
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        """
        Given node embeddings `z`, predicts the presence of edges.
        Used for link prediction training and inference (Use Case 3).
        """
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        # Dot product gives a score for how likely an edge is.
        return (src * dst).sum(dim=-1)
