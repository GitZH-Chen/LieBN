"""
@author: Ziheng Chen
Please cite the paper below if you use the code:

Ziheng Chen, Yue Song, Yunmei Liu, Nicu Sebe. A Lie Group Approach to Riemannian Batch Normalization. ICLR 2024.
Ziheng Chen, Yue Song, Tianyang Xu, Zhiwu Huang, Xiao-Jun Wu, and Nicu Sebe. Adaptive Log-Euclidean metrics for SPD matrix learning TIP 2024.

Copyright (C) 2024 Ziheng Chen
All rights reserved.
"""

import torch as th
import geoopt
from .Geometry.Rotations import RotMatrices
from . import LieBNBase

class LieBNRot(LieBNBase):
    """ LieBN on the 3 x 3 rotations under the bi-invariant metric.
    is_left: True for left translation for centering and biasing.
    is_detach in RotMatrices is set to True by default.
    """
    def __init__(self,shape ,batchdim=[0,1],momentum=0.1,is_detach=True,
                 karcher_steps=1,is_left=True):
        super().__init__(shape,batchdim,momentum,is_detach)
        self.karcher_steps = karcher_steps;self.is_left=is_left
        self.get_manifold()
        self.set_weight()

    def set_weight(self):
        if len(self.shape) > 2:
            self.weight = geoopt.ManifoldParameter(th.eye(self.shape[-1]).repeat(*self.shape[:-2], 1, 1),
                                                   manifold=RotMatrices())
        else:
            self.weight = geoopt.ManifoldParameter(th.eye(self.shape[-1]),manifold=RotMatrices())

    def forward(self,S):
        if(self.training):
            batch_mean = self.manifold.cal_geom_mean(S,batchdim=self.batchdim)
            X_centered = self.manifold.translation(S,batch_mean,is_inverse=True,is_left=self.is_left)
            X_scaled,var = self.manifold.scaling(X_centered,shift=self.shift,batchdim=self.batchdim)
            self.updating_running_statistics(batch_mean, var)
        else:
            X_centered = self.manifold.translation(S, self.running_mean, is_inverse=True,is_left=self.is_left)
            X_scaled,_ = self.manifold.scaling(X_centered,shift=self.shift,running_var=self.running_var)
        X_normalized = self.manifold.translation(X_scaled, self.weight, is_inverse=False,is_left=self.is_left)

        return X_normalized

    def get_manifold(self):
        self.manifold = RotMatrices(karcher_steps=self.karcher_steps)
        self.manifold.is_detach = self.is_detach

