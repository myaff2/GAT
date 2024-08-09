import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm, SuperGATConv,TransformerConv, GATv2Conv
from torch_geometric.utils import dropout_edge
from ResGATv2Conv import ResGATv2Conv as GATv2Conv

class GAT(torch.nn.Module):
    def __init__(self, channels, feature_dim=1024, dropout=0, dropout_a=0, da_true=False, proj_dim=64):
        super(GAT, self).__init__()
        self.channels = channels
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.dropout_a = dropout_a
        self.da_true = da_true
        
        self.linear0 = nn.Linear(self.feature_dim, self.channels[0])
        self.batch0 = BatchNorm(self.channels[0])
        
        self.gat_layer1 = GATv2Conv(self.channels[0], self.channels[0], heads=8, edge_dim=2)  # GAT layer front
        self.linear1 = nn.Linear(8*self.channels[0],self.channels[0])
        self.batch01 = BatchNorm(self.channels[0])

        self.gat_layer2 = GATv2Conv(self.channels[0], self.channels[0], heads=8, edge_dim=2)  # GAT layer back
        self.linear2 = nn.Linear(8*self.channels[0],self.channels[0])
        self.batch02 = BatchNorm(self.channels[0])

        self.gat_layer3 = GATv2Conv(self.channels[0], self.channels[0], heads=8, edge_dim=2)
        self.linear3 = nn.Linear(8*self.channels[0],self.channels[0])
        self.batch03 = BatchNorm(self.channels[0])

        self.linear = nn.Linear(self.channels[0], 2)
         
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        edge_index1 = edge_index[:, edge_attr[:, 0] >= 0]
        edge_index2 = edge_index[:, edge_attr[:, 0] <= 0]
        feat_edge_attr = edge_attr[:, 1:]
        feat_edge_attr1 = edge_attr[edge_attr[:, 0]>=0, 1:]
        feat_edge_attr2 = edge_attr[edge_attr[:, 0]<=0, 1:]

        edge_index1m, _ = dropout_edge(edge_index=edge_index1, p=self.dropout_a, training=self.training if not self.da_true else True)
        edge_index2m, _ = dropout_edge(edge_index=edge_index2, p=self.dropout_a, training=self.training if not self.da_true else True)
        edge_indexm, _ = dropout_edge(edge_index=edge_index, p=self.dropout_a, training=self.training if not self.da_true else True)

        x = self.linear0(x[:, :self.feature_dim])
        x = self.batch0(x)
        x = F.relu(x)

        x = self.gat_layer1(x, edge_index1m, feat_edge_attr1)  # Apply GAT to the whole graph
        x = F.relu(x)
        x = self.linear1(x)
        x = self.batch01(x)
        x = F.relu(x)

        x = self.gat_layer2(x, edge_index2m, feat_edge_attr2)  # Apply GAT to the whole graph
        x = F.relu(x)
        x = self.linear2(x)
        x = self.batch02(x)
        x = F.relu(x)

        # x, (alpha_edge_index, alpha) = self.gat_layer3(x, edge_indexm, feat_edge_attr, return_attention_weights=True)
        x = self.gat_layer3(x, edge_indexm, feat_edge_attr)
        x = F.relu(x)
        x = self.linear3(x)
        x = self.batch03(x)
        x = F.relu(x)

        x = self.linear(x)
        x = F.softmax(x, dim = -1)[:, 1:]

        return x