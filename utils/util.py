import json
import torch
import numpy as np
from pathlib import Path
import pickle
import torch
import dgl
import networkx as nx
from itertools import repeat
from collections import OrderedDict
import requests
import math
import random
import os
import tqdm
import zipfile


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def build_user_history(config, doc2id):
    print('constructing user history ...')
    with open(config['data']['datapath']+config['data']['user_history_file'], 'rb') as fo:
        user_history_dict_raw = pickle.load(fo, encoding='bytes')

    user_history_dict = {}
    # with open(config['data']['datapath'] + config['data']['doc2id_file'], 'rb') as fo:
    #     doc2id = pickle.load(fo, encoding='bytes')
    for user in user_history_dict_raw:
        if(len(user_history_dict_raw[user])>=config['model']['user_his_num']):
            user_history_dict[user] = list(map(lambda x:doc2id[str(x)], user_history_dict_raw[user][-config['model']['user_his_num']:]))
        else:
            user_history_dict[user] = list(map(lambda x: doc2id[str(x)],
                                          user_history_dict_raw[user]))
            for i in range(config['model']['user_his_num'] - len(user_history_dict[user])):
                user_history_dict[user].append(doc2id["N0"])

    return user_history_dict


def build_train(config, doc2id):
    print('constructing train ...')
    train_data = {}
    item1 = []
    item2 = []
    label = []
    # with open(config['data']['datapath'] + config['data']['doc2id_file'], 'rb') as fo:
    #     doc2id = pickle.load(fo, encoding='bytes')
    with open(config['data']['datapath'] + config['data']['train_file'], 'rb') as fo:
        train_data_list = pickle.load(fo, encoding='bytes')

    for item in train_data_list:
        item1.append(item[0])
        item2.append(doc2id[str(item[1])])
        label.append(float(item[2]))
    # fp_train = open(config['data']['datapath']+config['data']['train_file'], 'r', encoding='utf-8')
    # for line in fp_train:
    #     linesplit = line.split('\n')[0].split('\t')
    #     item1.append(linesplit[0])
    #     item2.append(doc2id[str(linesplit[1])])
    #     label.append(float(linesplit[2]))

    train_data['item1'] = item1
    train_data['item2'] = item2
    train_data['label'] = label
    return train_data

def build_val(config, doc2id):
    print('constructing val ...')
    test_data = {}
    item1 = []
    item2 = []
    label = []
    # with open(config['data']['datapath'] + config['data']['doc2id_file'], 'rb') as fo:
    #     doc2id = pickle.load(fo, encoding='bytes')
    with open(config['data']['datapath'] + config['data']['val_file'], 'rb') as fo:
        train_data_list = pickle.load(fo, encoding='bytes')
    for item in train_data_list:
        item1.append(item[0])
        item2.append(doc2id[str(item[1])])
        label.append(float(item[2]))

    test_data['item1'] = item1
    test_data['item2'] = item2
    test_data['label'] = label
    return test_data

def build_test(config, doc2id):
    print('constructing test ...')
    user_dict = {}
    item_set = set()
    with open(config['data']['datapath'] + config['data']['test_file'], 'rb') as fo:
        train_data_list = pickle.load(fo, encoding='bytes')
    for item in train_data_list:
        if item[0] not in user_dict:
            user_dict[item[0]] = []
        if item[2] == 1:
            user_dict[item[0]].append(item[1])
            item_set.add(doc2id[item[1]])
    # fp_test = open(config['data']['datapath'] + config['data']['test_file'], 'r', encoding='utf-8')
    # for line in fp_test:
    #     linesplit = line.split('\n')[0].split('\t')
    #     if linesplit[0] not in user_dict:
    #         user_dict[linesplit[0]] = []
    #     user_dict[linesplit[0]].append(linesplit[1])
    #     item_set.add(doc2id[linesplit[1]])
    for item in doc2id: # include all items in doc feature, not only in test
        item_set.add(doc2id[item])
    test_data = (user_dict, item_set)
    return test_data

# def build_infer(config):
#     '''
#     not use now
#     :param config:
#     :return:
#     '''
#     print('constructing infer ...')
#     with open(config['data']['datapath'] + config['data']['infer_file'], 'rb') as fo:
#         infer_data = pickle.load(fo, encoding='bytes')
#     return infer_data

def build_doc_feature_embedding(config):
    print('constructing doc feature embedding ...')
    # fp_test = open(config['data']['datapath'] + config['data']['entity2id_file'], 'r', encoding='utf-8')
    # entity2id_dict = {}
    # for line in fp_test:
    #     linesplit = line.split('\n')[0].split('\t')
    #     if len(linesplit)>1:
    #         entity2id_dict[linesplit[0]] = int(linesplit[1])
    # with open(config['data']['datapath'] + "/entity2id.pkl", 'wb') as fo:
    #     pickle.dump(entity2id_dict, fo)

    with open(config['data']['datapath']+config['data']['doc_feature_embedding_file'], 'rb') as fo:
        doc_embedding_feature_dict = pickle.load(fo, encoding='bytes')
    doc_embedding_feature_dict['N0'] = np.zeros(config['model']['doc_embedding_size'])
    doc2id = {}
    index = 1
    with open(config['data']['datapath'] + config['data']['entity2id_file'], 'rb') as fo:
        entity2id_dict = pickle.load(fo, encoding='bytes')
    for item in doc_embedding_feature_dict:
        doc2id[item] = index
        index = index + 1
    doc2id['N0'] = 0
    with open(config['data']['datapath'] + config['data']['doc_feature_entity_file'], 'rb') as fo:
        doc_entity_dict = pickle.load(fo, encoding='bytes')
    for item in doc_entity_dict:
        entity_set = []
        for entity in doc_entity_dict[item]:
            if entity in entity2id_dict:
                entity_set.append(entity)
        doc_entity_dict[item] = entity_set
    item_entity_dict = {}
    # for doc in doc_entity_dict:
    for doc in doc2id:
        if doc in doc_entity_dict:
            if len(doc_entity_dict[doc]) >= config['model']['item_entity_num']:
                item_entity_dict[doc2id[doc]] = list(
                    map(lambda x: entity2id_dict[x], doc_entity_dict[doc][:config['model']['item_entity_num']]))
            else:
                item_entity_dict[doc2id[doc]] = list(map(lambda x: entity2id_dict[x], doc_entity_dict[doc]))
                for i in range(config['model']['item_entity_num'] - len(item_entity_dict[doc2id[doc]])):
                    item_entity_dict[doc2id[doc]].append(0)
        else:
            item_entity_dict[doc2id[doc]] = []
            for i in range(config['model']['item_entity_num']):
                item_entity_dict[doc2id[doc]].append(0)
    return doc_embedding_feature_dict, doc2id, item_entity_dict

def build_graph(config, device, doc2id):
    print('constructing graph ...')
    # graph_set = set()
    # fp_entity_embedding = open(config['data']['datapath']+"/wikidata-graph/triple2id.txt", 'r', encoding='utf-8')
    # for line in fp_entity_embedding:
    #     linesplit = line.strip().split('\t')
    #     if len(linesplit)> 1:
    #         graph_set.add((int(linesplit[0]), int(linesplit[1]), int(linesplit[2])))
    # with open(config['data']['datapath']+"/triples.pkl", 'wb') as fo:
    #      pickle.dump(graph_set, fo)
    # entity2id_dict = {}
    # fp_entity_embedding = open(config['data']['datapath']+"/entity2id.txt", 'r', encoding='utf-8')
    # for line in fp_entity_embedding:
    #     linesplit = line.strip().split('\t')
    #     if len(linesplit)> 1:
    #         entity2id_dict[linesplit[0]] = int(linesplit[1])
    #
    # with open(config['data']['datapath']+"/entity2id.pkl", 'wb') as fo:
    #     pickle.dump(entity2id_dict, fo)

    with open(config['data']['datapath'] + config['data']['entity2id_file'], 'rb') as fo:
        entity2id_dict = pickle.load(fo, encoding='bytes')


    graph_data = {}
    with open(config['data']['datapath'] + config['data']['kg_file'], 'rb') as fo:
        graph_set = pickle.load(fo, encoding='bytes')
    head = []
    tail = []
    for item in graph_set:
        head.append(item[0])
        tail.append(item[2])
    graph_data[('entity', 'entity_relation', 'entity')] = (torch.tensor(head).to(device), torch.tensor(tail).to(device))

    with open(config['data']['datapath'] + config['data']['doc_feature_entity_file'], 'rb') as fo:
        doc_entity_dict = pickle.load(fo, encoding='bytes')
    head = []
    tail = []
    for item in doc_entity_dict:
        if item in doc2id:
            for entity in doc_entity_dict[item]:
                if entity in entity2id_dict:
                    head.append(doc2id[item])
                    tail.append(entity2id_dict[entity])
    graph_data[('entity', 'entity_in_doc', 'doc')] = (
        torch.tensor(tail).to(device), torch.tensor(head).to(device))

    graph = dgl.heterograph(graph_data)

    # with open(config['data']['datapath']+config['data']['entity_embedding_file'], 'rb') as fo:
    #     entity_embedding = pickle.load(fo, encoding='bytes')
    # graph.nodes['entity'].data['x'] = torch.FloatTensor(entity_embedding).to(device)
    entity_embedding = []

    # with open(config['data']['datapath'] + config['data']['entity_embedding_file'], 'rb') as fo:
    #     entity_embedding = pickle.load(fo, encoding='bytes')
    fp_entity_embedding = open(config['data']['datapath']+"/entity2vec.vec", 'r', encoding='utf-8')
    for line in fp_entity_embedding:
        entity_embedding.append(list(map(lambda x:float(x), line.strip().split('\t'))))
    with open(config['data']['datapath']+"/entity2vec.pkl", 'wb') as fo:
        pickle.dump(entity_embedding, fo)
    graph.nodes['entity'].data['x'] = torch.FloatTensor(entity_embedding).to(device)

    # with open(config['data']['datapath']+config['data']['doc2id_file'], 'rb') as fo:
    #     doc2id = pickle.load(fo, encoding='bytes')

    with open(config['data']['datapath']+config['data']['doc_feature_embedding_file'], 'rb') as fo:
        doc_embedding_feature_dict = pickle.load(fo, encoding='bytes')
    doc_embedding_feature_dict['N0'] = np.zeros(config['model']['doc_embedding_size'])

    doc_embedding = []
    #doc_embedding.append(np.zeros(config['model']['doc_embedding_size']))
    for doc in doc2id:
        doc_embedding.append(doc_embedding_feature_dict[doc])

    graph.nodes['doc'].data['x'] = torch.FloatTensor(doc_embedding).to(device)


    with open(config['data']['datapath']+"/graph.pkl", 'wb') as fo:
        pickle.dump(graph, fo)
    # with open(config['data']['datapath']+"/graph.pkl", 'rb') as fo:
    #      graph = pickle.load(fo, encoding='bytes')

    return graph

def build_entity_embedding(config, device):
    print('constructing embedding ...')
    with open(config['data']['datapath']+config['data']['entity_embedding_file'], 'rb') as fo:
        entity_embedding = pickle.load(fo, encoding='bytes')
    return torch.FloatTensor(entity_embedding).to(device)


def load_doc_feature(config):
    doc_embeddings = {}
    fp_news_feature = open(config['data']['datapath']+config['data']['doc_feature_entity_file'], 'r', encoding='utf-8')
    for line in fp_news_feature:
        newsid, news_embedding = line.strip().split('\t')
        news_embedding = news_embedding.split(',')
        doc_embeddings[newsid] = news_embedding
    return doc_embeddings

