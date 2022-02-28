from base.base_model import BaseModel
import torch
import torch.nn as nn
import numpy as np
from model.HGNN import *
from model.HMN import *
from torch.distributions import Categorical

class NeighborSampler(BaseModel):
    def __init__(self, g, fanouts):
        super(NeighborSampler, self).__init__()
        self.g = g
        self.fanouts = fanouts

    def forward(self, seeds):
        blocks = []
        for fanout in self.fanouts:
            frontier = dgl.sampling.sample_neighbors(self.g, seeds, fanout, replace=True)
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks

class KEMI(BaseModel):

    def __init__(self, config, device, user_history, item_entity, doc_feature_embedding, graph):
        super(KEMI, self).__init__()
        self.config = config
        self.device = device
        self.user_history = user_history
        self.item_entity = item_entity
        self.doc_feature_embedding = doc_feature_embedding
        self.graph = graph
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.cos = nn.CosineSimilarity(dim=-1)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        # self.sampler = NeighborSampler(self.graph, self.config['model']['sample_size'])
        self.hgnn = HGNN(self.config, self.graph,
                         # (self.config['model']['doc_embedding_size'],self.config['model']['entity_embedding_size']),
                         # (self.config['model']['embedding_size'], self.config['model']['embedding_size']),
                         # (self.config['model']['embedding_size'], self.config['model']['embedding_size'])
                            {'entity_in_doc':self.config['model']['entity_embedding_size'] ,
                             'entity_relation': self.config['model']['entity_embedding_size']},
                         {'entity_in_doc': self.config['model']['embedding_size'],
                          'entity_relation': self.config['model']['embedding_size']},
                         {'entity_in_doc': self.config['model']['embedding_size'],
                          'entity_relation': self.config['model']['embedding_size']},
                         ['entity_in_doc', 'entity_relation']
                         )
        self.hmn = HMN(self.config)

        self.compress_doc = nn.Linear(self.config['model']['doc_embedding_size'],self.config['model']['embedding_size'])
        self.compress_entity = nn.Linear(self.config['model']['entity_embedding_size'], self.config['model']['embedding_size'])

    def get_item_embedding_batch(self, itemids):
        item_embeddings = []
        for newsid in itemids:
            item_embeddings.append(torch.FloatTensor(self.doc_feature_embedding[newsid]).to(self.device))
        return torch.stack(item_embeddings)

    def get_user_history(self, users):
        user_his = []
        for userid in users:
            user_his.append(self.user_history[userid])
        return user_his

    def get_item_entity(self, itemids):
        entities = []
        if type(itemids) == list:
            for i in range(len(itemids)):
                entities.append([])
                for j in range(len(itemids[i])):
                    entities[-1].append(self.item_entity[itemids[i][j]])

        else:
            for itemid in itemids:
                entities.append(self.item_entity[int(itemid)])
        return torch.Tensor(entities).to(self.device)

    def cal_score(self, user_rep, item_rep):
        logit = torch.exp(-2.0 * (1.0 - F.cosine_similarity(user_rep, item_rep, dim=-1)))
        logit = torch.clamp(logit, 1e-8, 1 - 1e-8)
        return logit

    def forward(self, users, items):

        user_his = self.get_user_history(users)

        item_entities = self.get_item_entity(items)
        user_entities = self.get_item_entity(user_his)

        doc_features = self.graph.nodes['doc'].data['x']
        entity_features = self.graph.nodes['entity'].data['x']

        #doc_features = self.compress_doc(doc_features)
        #entity_features = self.compress_entity(entity_features)

        node_features = {'doc': doc_features, 'entity': entity_features}
        logits = self.hgnn(self.graph, node_features)

        # reshape
        user_his_tensor = torch.Tensor(user_his).to(self.device)
        user_his_tensor = user_his_tensor.to(torch.long)
        user_his_reshape = user_his_tensor.reshape(user_his_tensor.shape[0] * user_his_tensor.shape[1])

        user_entities_tensor = user_entities.to(torch.long)
        user_entities_reshape = user_entities_tensor.reshape(user_entities_tensor.shape[0] * user_entities_tensor.shape[1] * user_entities_tensor.shape[2])

        item_reshape = items

        item_entities_tensor = item_entities.to(torch.long)
        item_entities_reshape = user_his_tensor.reshape(item_entities_tensor.shape[0] * item_entities_tensor.shape[1])

        user_item_rep = torch.index_select(logits['doc'], 0 ,user_his_reshape)
        user_entity_rep = torch.index_select(logits['entity'], 0, user_entities_reshape)

        item_item_rep = torch.index_select(logits['doc'], 0 ,item_reshape)
        item_entity_rep = torch.index_select(logits['entity'], 0, item_entities_reshape)

        # reshape back
        user_item_rep = user_item_rep.reshape(user_his_tensor.shape[0], user_his_tensor.shape[1], self.config['model']['embedding_size'])
        user_entity_rep = user_entity_rep.reshape(user_entities_tensor.shape[0], user_entities_tensor.shape[1], user_entities_tensor.shape[2], self.config['model']['embedding_size'])
        item_item_rep = item_item_rep
        item_entity_rep = item_entity_rep.reshape(item_entities_tensor.shape[0], item_entities_tensor.shape[1], self.config['model']['embedding_size'])
        user_rep, loss_dis = self.hmn(user_item_rep, user_entity_rep)
        item_rep, _ = self.hmn(item_item_rep, item_entity_rep)

        predict = self.cal_score(user_rep,item_rep)

        # items_h = {"doc" : items}
        # #subgraph_i = dgl.sampling.sample_neighbors(self.graph, items_h, self.config['model']['sample_size'])
        # subgraph_i = self.sampler(items_h)
        # item_fea = subgraph_i[0].srcdata['x']
        #
        # #subgraph_u = dgl.sampling.sample_neighbors(self.graph, users, self.config['model']['sample_size'])
        # item_rep = self.hgnn(subgraph_i, item_fea)
        #
        # user_items = self.get_user_history(users)
        # subgraph_u = self.sampler(user_items)
        # user_fea = subgraph_u[0].srcdata['x']
        # user_history_rep = self.hgnn(subgraph_u, user_fea)
        # user_group_rep = self.mig(user_history_rep)
        #
        # predict = self.cal_score(user_group_rep, item_rep)

        return predict, loss_dis