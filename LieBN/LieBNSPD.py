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
from geoopt.manifolds import SymmetricPositiveDefinite

from .Geometry.SPD.SPDMatrices import SPDLogEuclideanMetric,SPDAdaptiveLogEuclideanMetric,SPDLogCholeskyMetric,\
    SPDAffineInvariantMetric,SPDCholeskyRightInvariantMetric,tril_param_metric,bi_param_metric,single_param_metric

from . import LieBNBase

class LieBNSPD(LieBNBase):
    """  Arguments:
    metric: LEM, ALEM, LCM, AIM, CRIM, with the last one is right-invariant
    karcher_step=1: for AIM and CRIM
    power=1: power deformation for AIM, CRIM, and LCM
    (alpha, beta)=(1,0): O(n)-invariance inner product in AIM, CRIM, LEM, and ALEM.

    Note:
    - The SPD biasing paramter is optimized by AIM-based geoopt.
    - For ALEM, the paramter A is in the SPDAdaptiveLogEuclideanMetric.
    - There are two ways to calculate the varriance: 1 via d(P_i, M); 2 via d(\bar{P}_i, I).
    If the Fréchet mean are accurate, such as LEM,LCM, and ALEM, these two ways are euqivalent.
    For AIM and CRIM, as we set karcher_steps=1, the 2nd is an approximation. We adpot the 2nd one for efficiency.
    If one set karcher flow to converge, 1 is euqivalent to 2.
    - Follwoing SPDNetBN, we use detach when calculating Fréchet statistics
    """

    def __init__(self,shape ,batchdim=[0],momentum=0.1,is_detach=True,
                 metric: str ='AIM',power=1.,alpha=1.0,beta=0.,karcher_steps=1,):
        super().__init__(shape,batchdim,momentum,is_detach)
        self.metric = metric; self.power = power;self.alpha = alpha;self.beta = beta;
        self.karcher_steps = karcher_steps
        self.get_manifold()
        self.set_weight()

    def set_weight(self):
        if len(self.shape) > 2:
            self.weight = geoopt.ManifoldParameter(th.eye(self.shape[-1]).repeat(*self.shape[:-2], 1, 1),
                                                   manifold=SymmetricPositiveDefinite())
        else:
            self.weight = geoopt.ManifoldParameter(th.eye(self.shape[-1]),
                                                   manifold=SymmetricPositiveDefinite())

    def forward(self,X):
        #deformation
        X_deformed = self.manifold.deformation(X)
        weight = self.manifold.deformation(self.weight)

        if(self.training):
            # centering
            batch_mean = self.manifold.cal_geom_mean(X_deformed,batchdim=self.batchdim)
            X_centered = self.manifold.translation(X_deformed,batch_mean,is_inverse=True)

            # scaling and shifting
            # As centering is an isometry, batch variance is equal to the one of the centered data (to the identity element).
            # This is more efficient than calculating the original bach variance.
            # Note that if the mean is calculated approximatedly, such as karcher flow, cal_geom_var is also approximated
            # One can also try original bach variance.
            batch_var = self.manifold.cal_geom_var(X_centered,batchdim=self.batchdim)
            factor = self.shift / (batch_var + self.eps).sqrt()
            X_scaled = self.manifold.scaling(X_centered, factor)
            self.updating_running_statistics(batch_mean, batch_var)

        else:
            # centering, scaling and shifting
            X_centered = self.manifold.translation(X_deformed, self.running_mean, is_inverse=True)
            factor = self.shift / (self.running_var + self.eps).sqrt()
            X_scaled = self.manifold.scaling(X_centered, factor)
        #biasing
        X_normalized = self.manifold.translation(X_scaled, weight, is_inverse=False)
        # inv_deformation
        X_new = self.manifold.inv_deformation(X_normalized)

        return X_new

    def get_manifold(self):
        classes = {
            "LEM": SPDLogEuclideanMetric,
            "ALEM": SPDAdaptiveLogEuclideanMetric,
            "LCM": SPDLogCholeskyMetric,
            "AIM": SPDAffineInvariantMetric,
            "CRIM": SPDCholeskyRightInvariantMetric,
        }
        n=self.shape[-1]
        if self.metric in tril_param_metric:
            self.manifold = classes[self.metric](n=n, power=self.power,alpha=self.alpha, beta=self.beta,
                                            karcher_steps=self.karcher_steps)
        elif self.metric in bi_param_metric:
            self.manifold = classes[self.metric](n=n, alpha=self.alpha, beta=self.beta)
        elif self.metric in single_param_metric:
            self.manifold = classes[self.metric](n=n, power=self.power)
        else:
            raise NotImplementedError

        self.manifold.is_detach=self.is_detach

