import numpy as np
from numpy import inf
import torch
from torch import nn, optim
from utils.metrics import *
from utils.pytorchtools import *
from base.base_trainer import BaseTrainer
from logger.logger import *
from model.KEMI import *

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, config, model, criterion, optimizer, device,
                 data):
        super().__init__()

        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']

        self.early_stop = cfg_trainer.get('early_stop', inf)
        if self.early_stop <= 0:
            self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        self.train_dataloader = data[0]
        self.val_data = data[1]
        self.entity_id_dict = data[4]
        self.doc_feature_embedding = data[5]
        self.entity_embedding = data[6]


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        all_loss = 0
        for step, batch in enumerate(self.train_dataloader):
            if step % 10 == 0:
                print(step)
            predict, dis_loss = self.model(
                    batch['item1'], batch['item2'])

            loss = self.criterion(predict, batch['label'].cuda().float())#+ dis_loss
            #loss = self.criterion(predict, batch['label'].float())

            loss_mean = torch.mean(loss)+ self.config['model']['dis_loss_weight']*(-dis_loss)

            all_loss = all_loss + loss_mean.data

            self.optimizer.zero_grad()
            loss_mean.backward(retain_graph=True)
            self.optimizer.step()


        torch.save(self.model.state_dict(), './out/saved/models/checkpoint.pt')

        print("all loss: " + str(all_loss))

    def _valid_epoch(self):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()

        # get all news embeddings
        y_pred = []
        start_list = list(range(0, len(self.val_data['label']), self.config['data_loader']['batch_size']))
        for start in start_list:
            if start + self.config['data_loader']['batch_size'] <= len(self.val_data['label']):
                end = start + self.config['data_loader']['batch_size']
                predict, user_rep, item_rep = self.model(
                    self.val_data['item1'][start:end], self.val_data['item2'][start:end])
                y_pred.extend(predict.cpu().data.numpy())
            else:
                predict, user_rep, item_rep = self.model(
                    self.val_data['item1'][start:], self.val_data['item2'][start:])
                y_pred.extend(predict.cpu().data.numpy())
        truth = self.val_data['label']
        auc_score = cal_auc(truth, y_pred)
        print("auc socre: "+str(auc_score))
        return auc_score

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        state_anchor = self.model.state_dict()
        filename_anchor = str(self.checkpoint_dir / 'checkpoint-anchor-epoch{}.pth'.format(epoch))
        torch.save(state_anchor, filename_anchor)
        self.logger.info("Saving checkpoint: {} ...".format(filename_anchor))
        filename_recommender = str(self.checkpoint_dir / 'checkpoint-recommender-epoch{}.pth'.format(epoch))


    def train(self):
        """
            Full training logic
        """
        logger_train = get_logger("train")

        logger_train.info("training")
        valid_scores = []
        early_stopping = EarlyStopping(patience=self.config['trainer']['early_stop'], verbose=True)
        for epoch in range(self.start_epoch, self.epochs+1):
            self._train_epoch(epoch)

            valid_socre = self._valid_epoch(epoch)
            valid_scores.append(valid_socre)

            early_stopping(valid_socre, self.model)
            if early_stopping.early_stop:
                logger_train.info("Early stopping")

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)




