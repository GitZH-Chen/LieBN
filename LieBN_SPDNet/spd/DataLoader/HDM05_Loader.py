import sys
import os

import numpy as np
import torch as th
import random

from torch.utils import data

device = 'cpu'
class DatasetHDM05(data.Dataset):
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
        return x.to(device), y.to(device)


class DataLoaderHDM05:
    def __init__(self, data_path, pval, batch_size):
        for filenames in os.walk(data_path):
            names = sorted(filenames[2])
        random.Random(1024).shuffle(names)
        N_test = int(pval * len(names))
        train_set = DatasetHDM05(data_path, names[N_test:])
        test_set = DatasetHDM05(data_path, names[:N_test])
        self._train_generator = data.DataLoader(train_set, batch_size=batch_size, shuffle='True')
        self._test_generator = data.DataLoader(test_set, batch_size=batch_size, shuffle='False')