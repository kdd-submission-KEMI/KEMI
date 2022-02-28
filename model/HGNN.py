import torch
import torch.nn as nn
from utils import *
import numpy as np
import networkx as nx
import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F
from base.base_model import BaseModel

class HGNN(nn.Module):
    def __init__(self, config, graph, in_size, hidden_size, out_size, rel_names):
        super().__init__()
        self.config = config
        self.graph = graph
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_size[rel], hidden_size[rel])
            for rel in rel_names}, aggregate='mean')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hidden_size[rel], out_size[rel])
            for rel in rel_names}, aggregate='mean')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h



# class HGNN(nn.Module):
#     def __init__(self, config, in_size, hidden_size, out_size, graph, rel_names):
#         super().__init__()
#         self.config = config
#         self.graph = graph
#         self.conv1 = dglnn.HeteroGraphConv({
#             rel: dglnn.GraphConv(self.config['model']['in_feats'], self.config['model']['hid_feats'])
#             for rel in rel_names}, aggregate='mean')
#         self.conv2 = dglnn.HeteroGraphConv({
#             rel: dglnn.GraphConv(self.config['model']['hid_feats'], self.config['model']['out_feats'])
#             for rel in rel_names}, aggregate='mean')
#
#     def forward(self, graph, inputs):
#         h = self.conv1(graph, inputs)
#         h = {k: F.relu(v) for k, v in h.items()}
#         h = self.conv2(graph, h)
#         return h

# class HeteroRGCNLayer(nn.Module):
#
#     def __init__(self, in_size, out_size, etypes):
#         super(HeteroRGCNLayer, self).__init__()
#         # W_r for each relation
#         self.weight = nn.ModuleDict({
#                 name : nn.Linear(in_size[name], out_size[name]) for name in etypes
#             })
#
#     def forward(self, G, feat_dict):
#         # The input is a dictionary of node features for each type
#         funcs = {}
#         for srctype, etype, dsttype in G.canonical_etypes:
#             # Compute W_r * h
#             Wh = self.weight[etype](feat_dict[srctype])
#             # Save it in graph for message passing
#             G.nodes[srctype].data['Wh_%s' % etype] = Wh
#             # Specify per-relation message passing functions: (message_func, reduce_func).
#             # Note that the results are saved to the same destination feature 'h', which
#             # hints the type wise reducer for aggregation.
#             funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
#         # Trigger message passing of multiple types.
#         # The first argument is the message passing functions for each relation.
#         # The second one is the type wise reducer, could be "sum", "max",
#         # "min", "mean", "stack"
#         G.multi_update_all(funcs, 'sum')
#         # return the updated node feature dictionary
#         return {ntype : G.nodes[ntype].data['x'] for ntype in G.ntypes}
#
# class HGNN(nn.Module):
#     def __init__(self, config, device, graph, in_size, hidden_size, out_size):
#         super(HGNN, self).__init__()
#         # Use trainable node embeddings as featureless inputs.
#         # embed_dict = {ntype : nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), in_size))
#         #               for ntype in G.ntypes}
#         # for key, embed in embed_dict.items():
#         #     nn.init.xavier_uniform_(embed)
#         # self.config = config
#         # self.embed = nn.ParameterDict(embed_dict)
#         self.config = config
#         self.graph = graph
#         self.embed = {
#             'doc': torch.FloatTensor(self.graph.nodes['doc'].data['x']).to(device),
#             'entity': torch.FloatTensor(self.graph.nodes['entity'].data['x']).to(device),
#         }
#         # doc_features = self.graph.nodes['doc'].data['x']
#         # entity_features = self.graph.nodes['entity'].data['x']
#
#         # create layers
#         self.layer1 = HeteroRGCNLayer(in_size, hidden_size, graph.etypes)
#         self.layer2 = HeteroRGCNLayer(hidden_size, out_size, graph.etypes)
#
#     def forward(self, G):
#         h_dict = self.layer1(G, self.embed)
#         h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
#         h_dict = self.layer2(G, h_dict)
#         # get paper logits
#         return h_dict

# # Create the model. The output has three logits for three classes.
# model = HeteroRGCN(G, 10, 10, 3)
#
# opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
#
# best_val_acc = 0
# best_test_acc = 0
#
# for epoch in range(100):
#     logits = model(G)
#     # The loss is computed only for labeled nodes.
#     loss = F.cross_entropy(logits[train_idx], labels[train_idx])
#
#     pred = logits.argmax(1)
#     train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
#     val_acc = (pred[val_idx] == labels[val_idx]).float().mean()
#     test_acc = (pred[test_idx] == labels[test_idx]).float().mean()
#
#     if best_val_acc < val_acc:
#         best_val_acc = val_acc
#         best_test_acc = test_acc
#
#     opt.zero_grad()
#     loss.backward()
#     opt.step()
#
#     if epoch % 5 == 0:
#         print('Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (
#             loss.item(),
#             train_acc.item(),
#             val_acc.item(),
#             best_val_acc.item(),
#             test_acc.item(),
#             best_test_acc.item(),
#         ))

