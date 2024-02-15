import sys
import os

import numpy as np
import torch as th
import random

from torch.utils import data

pval=0.25 #validation percentage
ptest=0.25 #test percentage
    # th.cuda.device('cpu')

class DatasetRadar(data.Dataset):
    def __init__(self, path, names):
        self._path = path
        self._names = names
    def __len__(self):
        return len(self._names)
    def __getitem__(self, item):
        x=np.load(self._path+self._names[item])
        x=np.concatenate((x.real[:,None],x.imag[:,None]),axis=1).T
        x=th.from_numpy(x)
        y=int(self._names[item].split('.')[0].split('_')[-1])
        y=th.from_numpy(np.array(y))
        return x.float(),y.long()
class DataLoaderRadar:
    def __init__(self,data_path,pval,batch_size):
        for filenames in os.walk(data_path):
            names=sorted(filenames[2])
        random.Random().shuffle(names)
        N_val=int(pval*len(names))
        N_test=int(ptest*len(names))
        N_train=(len(names)-N_test-N_val)
        train_set=DatasetRadar(data_path,names[N_val+N_test:int(N_train)+N_test+N_val])
        test_set=DatasetRadar(data_path,names[:N_test])
        val_set=DatasetRadar(data_path,names[N_test:N_test+N_val])
        self._train_generator=data.DataLoader(train_set,batch_size=batch_size,shuffle='True')
        self._test_generator=data.DataLoader(test_set,batch_size=batch_size,shuffle='False')
        self._val_generator=data.DataLoader(val_set,batch_size=batch_size,shuffle='False')