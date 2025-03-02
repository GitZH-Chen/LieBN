"""
@author: Ziheng Chen
Please cite the paper below if you use the code:

Ziheng Chen, Yue Song, Yunmei Liu, Nicu Sebe. A Lie Group Approach to Riemannian Batch Normalization. ICLR 2024.
Ziheng Chen, Yue Song, Tianyang Xu, Zhiwu Huang, Xiao-Jun Wu, and Nicu Sebe. Adaptive Log-Euclidean metrics for SPD matrix learning TIP 2024.

Copyright (C) 2024 Ziheng Chen
All rights reserved.
"""

import torch as th
import torch.nn as nn

class LieBNBase(nn.Module):
    """
    Ziheng Chen, Yue Song, Yunmei Liu, Nicu Sebe. A Lie Group Approach to Riemannian Batch Normalization. ICLR 2024
    Input X: (batchdim,...,n,n) matrix matrices
    Output X_new: (batchdim,...,n,n) batch-normalized matrices
    arguments:
        shape: excluding the batch dim
        batchdim: the fist k dims are batch dim, such as [0],[0,1]...
        is_detach: whether use detach when calculating Frecheat statistics.
    parameters:
        self.weight: [...,n,n] manifold parameter
        self.shift: [...,1,1] scalar dispersion
    """
    def __init__(self,shape, batchdim=[0], momentum=0.1,is_detach=True):
        super().__init__()
        self.shape=shape; self.momentum = momentum; self.batchdim=batchdim
        self.eps = 1e-5;
        self.manifold = None;
        self.is_detach=is_detach

        # Handle channel vs non-channel case
        if len(self.shape) > 2:
            # --- running statistics ---
            self.register_buffer("running_mean", th.eye(self.shape[-1]).repeat(*self.shape[:-2], 1, 1))
            self.register_buffer("running_var", th.ones(*shape[:-2], 1, 1))
            # --- parameters ---
            self.shift = nn.Parameter(th.ones(*shape[:-2], 1, 1))
        else:
            # --- running statistics ---
            self.register_buffer("running_mean", th.eye(self.shape[-1]))
            self.register_buffer("running_var", th.ones(1))
            # --- parameters ---
            self.shift = nn.Parameter(th.ones(1))

        # biasing weight should be set specifically for each manifold
        # self.set_weight()

    def updating_running_statistics(self,batch_mean,batch_var):
        self.running_mean.data = self.manifold.geodesic(self.running_mean, batch_mean,self.momentum)
        self.running_var.data = (1 - self.momentum) * self.running_var + batch_var * self.momentum

    def set_weight(self):
        raise NotImplementedError

    def get_manifold(self):
        raise NotImplementedError

    def __repr__(self):
        attributes = []
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            if isinstance(value, th.Tensor):
                attributes.append(f"{key}.shape={tuple(value.shape)}")  # only show Tensor shape
            else:
                attributes.append(f"{key}={value}")

        # register_buffer
        for name, buffer in self.named_buffers(recurse=False):
            attributes.append(f"{name}={buffer.item() if buffer.numel() == 1 else tuple(buffer.shape)}")

        return f"{self.__class__.__name__}({', '.join(attributes)}) \n {self.manifold}"
