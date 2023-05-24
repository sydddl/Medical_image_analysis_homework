__author__ = "DeathSprout"

import os
import numpy as np
from torch.utils.data import Dataset


class physionet_dataset(Dataset):

    def __init__(self, dataset_dir, setname):

        if setname in ['train', 'val', 'test']:
            data = np.load(os.path.join(dataset_dir, setname + '.npz'))
        else:
            raise Exception("Invalid setname value %d" % setname)
        self.data = data["x"]
        self.label = data["y"]
        print("Have load " + setname + " dataset from: " + dataset_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):  #
        data, label = self.data[i], self.label[i]
        return data, label