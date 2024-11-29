import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm, GATv2Conv
from torch_geometric.utils import dropout_edge
from ResGATv2Conv import ResGATv2Conv as GATv2Conv

class GAT_FM(torch.nn.Module):
    def __init__(self, channel, feature_dim=1024, dropout=0, dropout_a=0, da_true=False):
        super(GAT_FM, self).__init__()
        self.channel = channel
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.dropout_a = dropout_a
        self.da_true = da_true
        
        heads = 9
        self.linear0 = nn.Linear(self.feature_dim, self.channel)
        self.batch0 = BatchNorm(self.channel)
        
        self.gat_layer1 = GATv2Conv(self.channel, self.channel, heads=heads, edge_dim=2)  # GAT layer front
        self.linear1 = nn.Linear(heads*self.channel,self.channel)
        self.batch01 = BatchNorm(self.channel)

        self.gat_layer2 = GATv2Conv(self.channel, self.channel, heads=heads, edge_dim=2)  # GAT layer back
        self.linear2 = nn.Linear(heads*self.channel,self.channel)
        self.batch02 = BatchNorm(self.channel)

        self.gat_layer3 = GATv2Conv(self.channel, self.channel, heads=heads, edge_dim=2)
        self.linear3 = nn.Linear(heads*self.channel,self.channel)
        self.batch03 = BatchNorm(self.channel)

        self.linear = nn.Linear(self.channel, 2)
         
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

        x = self.gat_layer1(x, edge_index1m, feat_edge_attr1)  
        x = F.relu(x)
        x = self.linear1(x)
        x = self.batch01(x)
        x = F.relu(x)

        x = self.gat_layer2(x, edge_index2m, feat_edge_attr2)  
        x = F.relu(x)
        x = self.linear2(x)
        x = self.batch02(x)
        x = F.relu(x)

        x = self.gat_layer3(x, edge_indexm, feat_edge_attr)
        x = F.relu(x)
        x = self.linear3(x)
        x = self.batch03(x)
        x = F.relu(x)

        x = self.linear(x)
        x = F.softmax(x, dim = -1)[:, 1:]

        return x
