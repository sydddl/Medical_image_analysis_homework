__author__ = "DeathSprout"

import os.path as osp
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
# -----------------------------------
from util import loss
from dataloader import physionet_dataset


class Trainer_object(object):

    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.model.to(self.args.device)
        self.loss = loss(class_mod = args.class_mod)
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.args.learning_rate, weight_decay = self.args.weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size= self.args.learning_rate_step, gamma=args.gamma)
        self.clip = 5

        self.trainset = DataLoader(dataset=physionet_dataset(self.args.dataset_dir, setname="train"),
                                   batch_size=self.args.batch_size, shuffle=True,
                                   num_workers=self.args.num_worker)
        self.valset = DataLoader(dataset=physionet_dataset(self.args.dataset_dir, setname="val"),
                                 batch_size=self.args.batch_size, shuffle=True,
                                 num_workers=self.args.num_worker)
        self.testset = DataLoader(dataset=physionet_dataset(self.args.dataset_dir, setname="test"),
                                  batch_size=self.args.batch_size, shuffle=True,
                                  num_workers=self.args.num_worker)

    def save_model(self, name):
        torch.save(dict(params=self.model.state_dict()), osp.join(self.args.save_path, name + '.pth'))

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)
        predict = torch.squeeze(output, dim=1)
        loss = self.loss.getloss(predict,real_val)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        return loss.item()

    def test(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        predict = torch.squeeze(output, dim=1)
        loss = self.loss.getloss(predict, real_val)
        return loss.item()
