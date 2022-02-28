from base.base_data_loader import BaseDataLoader
from torch.utils.data import Dataset, DataLoader, RandomSampler
from utils.util import *

class DianpingDataset(Dataset):
    def __init__(self, dic_data, transform=None):
        self.dic_data = dic_data
        self.transform = transform
    def __len__(self):
        return len(self.dic_data['label'])
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'item1': self.dic_data['item1'][idx], 'item2': self.dic_data['item2'][idx], 'label': self.dic_data['label'][idx]}
        return sample

class DianpingDataLoader(BaseDataLoader):
    """
        News data loading using BaseDataLoader
    """
    def __init__(self, dataset, batch_size, shuffle=False, num_workers = 1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        super().__init__(self.dataset, batch_size, shuffle, num_workers)



def load_data(config, device):

    doc_feature_embedding, doc2id, item_entity = build_doc_feature_embedding(config)
    user_history = build_user_history(config, doc2id)
    Train_data = build_train(config, doc2id)
    Val_data = build_val(config, doc2id)
    Test_data = build_test(config, doc2id)
    graph = build_graph(config, device, doc2id)
    #entity_embedding = build_entity_embedding(config, device)
    train_data = DianpingDataset(Train_data)
    train_dataloader = DianpingDataLoader(train_data, batch_size=config['data_loader']['batch_size'])

    print("fininsh loading data!")

    return train_dataloader, Val_data, Test_data, doc_feature_embedding, user_history, item_entity, graph

