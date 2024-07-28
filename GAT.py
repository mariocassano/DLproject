import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F

class GAT(torch.nn.Module):
    def __init__(self, input, hidden_dim, heads):
        super().__init__()
        torch.manual_seed(1)
        self.conv1 = GATConv(in_channels=input, out_channels=hidden_dim, heads=heads, dropout=0.2)
        self.conv2 = GATConv(in_channels=hidden_dim*heads, out_channels=hidden_dim, heads=heads, dropout=0.2)
        self.conv3 = GATConv(in_channels=hidden_dim*heads, out_channels=2, heads=heads, dropout=0.2)

    def forward(self, x, edge_index):
        x = x.float()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x
