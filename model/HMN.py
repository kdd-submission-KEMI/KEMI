import torch
import torch.nn as nn
from base.base_model import BaseModel

class HMN(BaseModel):

    def __init__(self, config):
        super(HMN, self).__init__()
        self.config = config
        # memroy network
        self.interest_channels_i = nn.Embedding(self.config['model']['interest_num'],
                                                self.config['model']['embedding_size'])
        self.interest_channels_u = nn.Embedding(self.config['model']['interest_num'],
                                                self.config['model']['embedding_size'])

        # key matrix, to generate key
        self.key = nn.Embedding(self.config['model']['embedding_size'],
                                    self.config['model']['embedding_size'])
        # add and earse, to generate earse and add vector
        self.earse = nn.Embedding(self.config['model']['embedding_size'],
                                                self.config['model']['embedding_size'])
        self.add = nn.Embedding(self.config['model']['embedding_size'],
                                                self.config['model']['embedding_size'])
        self.cos = nn.CosineSimilarity(dim=-1)
        self.softmax = nn.Softmax(dim=-2)
        self.kl = nn.KLDivLoss()
        self.elu = nn.ELU(inplace=False)

    def forward(self, item_rep, entity_rep):
        weight_i_all = torch.zeros(item_rep.shape[0], 1, self.config['model']['interest_num'])
        loss = 0

        if len(item_rep.shape) == 2:
            item_rep = torch.unsqueeze(item_rep, 1)
            entity_rep = torch.unsqueeze(entity_rep, 1)

        for t in range(item_rep.shape[1]):
            item_i_t = torch.index_select(item_rep, 1, torch.Tensor([t]).to(torch.long))
            item_i_t = torch.transpose(item_i_t, -1, -2)
            key_i_t = torch.matmul(self.key.weight, item_i_t)

            read_weight_i_overall = torch.matmul(self.interest_channels_i.weight, key_i_t)
            read_weight_i_overall = self.softmax(read_weight_i_overall)
            read_weight_i_overall = torch.transpose(read_weight_i_overall, -1, -2)

            entity_i_t = torch.index_select(entity_rep, 1, torch.Tensor([t]).to(torch.long))
            entity_i_t = torch.transpose(entity_i_t, -1, -2)
            key_e_t = torch.matmul(self.key.weight, entity_i_t)

            read_weight_i_fine = torch.matmul(self.interest_channels_i.weight, key_e_t)
            read_weight_i_fine = self.softmax(read_weight_i_fine)

            read_weight_i_fine = torch.mean(read_weight_i_fine, dim=-1)

            weight_i = read_weight_i_fine + read_weight_i_overall

            loss = loss + self.kl(read_weight_i_overall, read_weight_i_fine)

            #earse and add vector
            earse_u_t_vector = torch.matmul(self.earse.weight, item_i_t)
            earse_weight_u = torch.transpose(weight_i, -1, -2)
            earse_u_t_vector = torch.transpose(earse_u_t_vector, -1, -2)
            earse_u_t = torch.bmm(earse_weight_u,earse_u_t_vector)
            earse_u_t = torch.squeeze(torch.mean(earse_u_t, dim=0))

            add_u_t_vector = torch.matmul(self.add.weight, item_i_t)
            add_weight_u = torch.transpose(weight_i, -1, -2)
            add_u_t_vector = torch.transpose(add_u_t_vector, -1, -2)
            add_u_t = torch.bmm(add_weight_u, add_u_t_vector)
            add_u_t = torch.squeeze(torch.mean(add_u_t, dim=0))

            item_i_t_trans = torch.transpose(item_i_t, -1, -2)
            read_weight_i_overall_trans = torch.transpose(read_weight_i_overall, -1, -2)
            write_item = torch.bmm(read_weight_i_overall_trans,item_i_t_trans)

            entity_i_t_reduce = torch.mean(entity_i_t, dim=-1)
            read_weight_i_fine_trans = torch.transpose(read_weight_i_fine, -1, -2)
            write_entity = torch.bmm(read_weight_i_fine_trans, entity_i_t_reduce)

            write_vec = write_item + write_entity
            write_vec = torch.squeeze(torch.mean(write_vec, dim=0))

            self.interest_channels_u.weight = nn.Parameter((1-earse_u_t)*(self.interest_channels_u.weight)+ add_u_t*(write_vec))

            weight_i_all = weight_i_all + weight_i#torch.squeeze(weight_i)

        # read interest
        weight_i_all_t = weight_i_all.expand(weight_i_all.shape[0], weight_i_all.shape[2], weight_i_all.shape[2])
        weight_i_all_t = weight_i_all_t/item_rep.shape[1]
        interest_rep = torch.matmul(weight_i_all_t, self.interest_channels_u.weight)


        # update interest memory
        item_i = torch.transpose(torch.mean(item_rep, dim=1, keepdim=True), -1, -2)
        key_i = torch.matmul(self.key.weight, item_i)

        read_weight_i_overall = torch.matmul(self.interest_channels_i.weight, key_i)
        read_weight_i_overall = self.softmax(read_weight_i_overall)
        read_weight_i_overall = torch.transpose(read_weight_i_overall, -1, -2)

        entity_i = torch.transpose(torch.mean(entity_rep, dim=1, keepdim=True), -1, -2)
        key_e = torch.matmul(self.key.weight, entity_i)

        read_weight_i_fine = torch.matmul(self.interest_channels_i.weight, key_e)
        read_weight_i_fine = self.softmax(read_weight_i_fine)

        read_weight_i_fine = torch.mean(read_weight_i_fine, dim=-1)

        weight_i = read_weight_i_fine + read_weight_i_overall

        earse_i_vector = torch.matmul(self.earse.weight, item_i)
        earse_weight_i = torch.transpose(weight_i, -1, -2)
        earse_i_vector = torch.transpose(earse_i_vector, -1, -2)
        earse_i = torch.bmm(earse_weight_i, earse_i_vector)
        earse_i = torch.squeeze(torch.mean(earse_i, dim=0))

        add_i_vector = torch.matmul(self.add.weight, item_i)
        add_weight_i = torch.transpose(weight_i, -1, -2)
        add_i_vector = torch.transpose(add_i_vector, -1, -2)
        add_i = torch.bmm(add_weight_i, add_i_vector)
        add_i = torch.squeeze(torch.mean(add_i, dim=0))

        item_i_trans = torch.transpose(item_i, -1, -2)
        read_weight_i_overall_trans = torch.transpose(read_weight_i_overall, -1, -2)
        write_item = torch.bmm(read_weight_i_overall_trans, item_i_trans)

        entity_i_reduce = torch.mean(entity_i, dim=-1)
        read_weight_i_fine_trans = torch.transpose(read_weight_i_fine, -1, -2)
        write_entity = torch.bmm(read_weight_i_fine_trans, entity_i_reduce)

        write_vec = write_item + write_entity
        write_vec = torch.squeeze(torch.mean(write_vec, dim=0))

        self.interest_channels_i.weight = nn.Parameter(
            (1 - earse_i) * (self.interest_channels_i.weight) + add_i * (write_vec))

        return interest_rep, loss