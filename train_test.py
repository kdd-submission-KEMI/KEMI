from utils.util import *
import hnswlib
from utils.metrics import *
from model.KEMI import *
from torch import nn, optim
from trainer.trainer import Trainer

def train(data, device, config):
    train_dataloader, Val_data, Test_data, doc_feature_embedding, user_history, item_entity, graph = data
    model = KEMI(config, device, user_history, item_entity, doc_feature_embedding, graph).to(device)
    criterion = nn.BCEWithLogitsLoss(reduce=False)
    optimizer = optim.Adam(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
    trainer = Trainer(config, model, criterion, optimizer, device, data)
    trainer.train()

def test(Test_data, config):
    #load model
    model =  torch.load('./out/saved/models/checkpoint_model.pt')
    #test
    model.eval()
    # get all user embeddings
    user_list = list(Test_data[0])
    doc_list = list(Test_data[1])
    start_list = list(range(0, len(user_list), config['data_loader']['batch_size']))
    user_embedding = []
    doc_embedding_dict = {}
    for start in start_list:
        if start +config['data_loader']['batch_size'] <= len(user_list):
            end = start + config['data_loader']['batch_size']
            user_embedding.extend(model(user_list[start:end], doc_list[start:end])[
                                     1].cpu().data.numpy())
        else:
            user_embedding.extend(model(user_list[start:], doc_list[start:])[
                                     1].cpu().data.numpy())
    # knn search topk
    ann = hnswlib.Index(space='cosine', dim=config['model']['embedding_size'])
    ann.init_index(max_elements=len(user_list), ef_construction=200, M=16)
    for i in range(len(doc_list)):
        doc_embedding_dict[user_list[i]] = user_embedding[i]
        ann.add_items(user_embedding[i], i)
    ann.set_ef(100)
    predict_dict = {}
    for doc in Test_data:
        doc_embedding = doc_embedding_dict[doc]
        labels, distances = ann.knn_query(doc_embedding, k=10)
        predict_dict[doc] = list(map(lambda x: doc_list[x], labels[0]))
    # compute metric
    evaluate(predict_dict, Test_data)

# def infer(Infer_data, config):
#     # load model
#     model = torch.load('./out/saved/models/checkpoint_model.pt')
#     # test
#     model.eval()
#     # infer
#     model.user_history = Infer_data[0]
#     start_list = list(range(0, len(model.user_history), config['data_loader']['batch_size']))
#     user_embedding = []
#     doc_embedding = []
#     user_id_list = []
#     doc_id_list = []
#     for start in start_list:
#         if start + config['data_loader']['batch_size'] <= len(model.user_history):
#             end = start + config['data_loader']['batch_size']
#             user_embedding.extend(model(model.user_history[start:end], model.user_history[start:end])[
#                                  1].cpu().data.numpy())
#         else:
#             user_embedding.extend(model(model.user_history[start:], model.user_history[start:])[
#                                  1].cpu().data.numpy())
#     for start in start_list:
#         if start + config['data_loader']['batch_size'] <= len(model.user_history):
#             end = start + config['data_loader']['batch_size']
#             user_embedding.extend(model(model.user_history[start:end], model.user_history[start:end])[
#                                  1].cpu().data.numpy())
#         else:
#             user_embedding.extend(model(model.user_history[start:], model.user_history[start:])[
#                                  1].cpu().data.numpy())
#
#     user_data = (user_id_list, user_embedding)
#     item_data = (doc_id_list, doc_embedding)



