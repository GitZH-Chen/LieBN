import sys
import os

import numpy as np
import torch as th
import random

from torch.utils import data

class DatasetSPD(data.Dataset):
    def __init__(self, path, names):
        self._path = path
        self._names = names

    def __len__(self):
        return len(self._names)

    def __getitem__(self, item):
        x = np.load(self._path + self._names[item])[None, :, :].real
        x = th.from_numpy(x).double()
        y = int(self._names[item].split('.')[0].split('_')[-1])
        y = th.from_numpy(np.array(y)).long()
        # return x.to(device),y.to(device)
        return x, y


class DataLoaderFPHA:
    def __init__(self, data_path, batch_size):
        path_train, path_test = data_path + 'train/', data_path + 'val/'
        for filenames in os.walk(path_train):
            names_train = sorted(filenames[2])
        for filenames in os.walk(path_test):
            names_test = sorted(filenames[2])
        self._train_generator = data.DataLoader(DatasetSPD(path_train, names_train), batch_size=batch_size,
                                                shuffle='True')
        self._test_generator = data.DataLoader(DatasetSPD(path_test, names_test), batch_size=batch_size,
                                               shuffle='False')