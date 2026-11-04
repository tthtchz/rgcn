import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class RGCNEncoder(nn.Module):

    def __init__(self, num_nodes, num_relations, hidden_dim=128, out_dim=64,
                 num_layers=2, dropout=0.2, num_bases=None):
        super().__init__()

        self.embedding = nn.Embedding(num_nodes, hidden_dim)
        # nn.init.xavier_uniform_(self.node_emb.weight)

        dims = [hidden_dim] * num_layers + [out_dim]
        self.convs = nn.ModuleList([
            RGCNConv(dims[i], dims[i + 1], num_relations, num_bases=num_bases)
            for i in range(num_layers)
        ])
        self.dropout = dropout

    def forward(self, edge_index, edge_type):
        x = self.embedding.weight
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class DistMultEncoder(nn.Module):

    def __init__(self, out_dim, num_target_relations=1):
        super().__init__()
        self.embedding = nn.Embedding(num_target_relations, out_dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, z, src, dst, rel=None):
        if rel is None:
            rel = torch.zeros(src.size(0), dtype=torch.long, device=src.device)
        r = self.rel_emb(rel)
        return (z[src] * r * z[dst]).sum(dim=-1)

class RGCNLinkPredictor(nn.Module):
    def __init__(self, num_nodes, hidden_dim=128, out_dim=64,
                 num_layers=2, dropout=0.2, num_bases=None):
        super().__init__()
        self.encoder = RGCNEncoder(num_nodes, num_relations, hidden_dim,
                                   out_dim, num_layers, dropout, num_bases)
        self.decoder = DistMultEncoder(out_dim)

    def encode(self, edge_index, edge_type):
        return self.encoder(edge_index, edge_type)

    def decode(self, z, src, dst):
        return self.decoder(z, src, dst)

    def forward(self, edge_index, edge_type, src, dst):
        z = self.encode(edge_index, edge_type)
        return self.decode(z, src, dst)
