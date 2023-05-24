__author__ = "DeathSprout"

import os
import sys
import torch
import logging

class loss():

    def __init__(self,class_mod = 2):
        if class_mod == 2: # 二分类交叉熵（Binary Cross Entropy）
            self.loss_fuction = torch.nn.BCEWithLogitsLoss()
        elif class_mod == 4:
            self.loss_fuction = torch.nn.MultiLabelSoftMarginLoss()
        else :
            raise Exception("Invalid class_mod value %d" % class_mod)

    def getloss(self,y_out,y_true):
        loss = self.loss_fuction(y_out,y_true)
        return loss

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def get_acc(y_out,y_true,class_mod = 2):
    if class_mod == 2:
        y_out = torch.where(y_out > 0.5, 1, y_out)
        y_out = torch.where(y_out < 0.5, 0, y_out)
        yes = len(torch.where(torch.eq(y_out,y_true))[0])
        acc = yes / len(y_true)
        return acc

def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    """
    日志文件
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                  "%Y-%m-%d %H:%M")
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                          "%Y-%m-%d %H:%M")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger
