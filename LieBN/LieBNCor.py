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
from geoopt.manifolds import PoincareBall

from .Geometry.Correlation.CorMatrices import CorEuclideanCholeskyMetric,CorLogEuclideanCholeskyMetric,CorOffLogMetric,CorLogScaledMetric
from .Geometry.Correlation.ppbcm_functionals import poincare_to_hemisphere

from . import LieBNBase

class LieBNCor(LieBNBase):
    """ Implemented metrics: ECM,LECM,OLM,LSM; all are bi-invariant metric with Abelian groups.
        param_mode: trivial,Riem

        Note:
        - Our experiments indicates that the Riemmannian is better than trivialization for correlation biasing parameters;
        - We do not use detach when calculating Fréchet statistics, as our experiments show it is better than detach.
        - is_inv_deformation: when it is False, this might help to simplify computations (for certain following layers).
    """

    def __init__(self,shape ,batchdim=[0],momentum=0.1,is_detach=False,
                 metric: str ='ECM',alpha=1.,param_1=0.,param_2=0.,max_iter=100,
                 is_inv_deformation=True,param_mode='Riem'):
        super().__init__(shape, batchdim=batchdim, momentum=momentum,is_detach=is_detach)
        self.metric = metric; self.alpha = alpha;self.param_1 = param_1;self.param_2 = param_2;
        self.max_iter = max_iter;self.is_inv_deformation=is_inv_deformation
        self.param_mode=param_mode
        self.get_manifold()
        self.set_weight()

    def set_weight(self):
        """Initializes (n-1) separate Poincare vectors of increasing dimensions (1 to n-1) using the Poincare Ball manifold."""
        # use poincare updates
        if self.param_mode=='Riem':
            self.weight = th.nn.ParameterList([
                geoopt.ManifoldParameter(
                    th.zeros(*self.shape[:-2], i)
                    if len(self.shape) > 2 else th.zeros(i),  # Handle channel vs non-channel case
                    manifold=PoincareBall(c=1.0, learnable=False)
                )
                for i in range(1, self.shape[-1])  # Each vector has i dimensions
            ])
        elif self.param_mode=='trivial':
            # use identical metric updates
            self.weight = th.nn.Parameter(th.randn(self.shape))

    def poincare_to_correlation(self,weight):
        """Converts stored Poincare vectors into a correlation matrix without extra memory storage."""
        # Initialize lower-triangular Cholesky factor L
        L = th.zeros(self.shape, dtype=weight[0].dtype, device=weight[0].device)
        L[...,0, 0] = 1  # Set (0,0) explicitly to 1

        # Directly compute Hemisphere representations and assign to L
        for i in range(1, self.shape[-1]):
            L[...,i, :i+1] = poincare_to_hemisphere(weight[i - 1])  # No extra storage, direct assignment
        return L @ L.transpose(-1,-2)

    def Euc2Codomain(self,weight):
        if self.metric in ['ECM','LECM']:
            weight_codomain = weight.tril(-1)
        elif self.metric == 'OLM':
            weight_codomain = weight.tril(-1) + weight.tril(-1).transpose(-1,-2)
        elif self.metric == 'LSM':
            n = self.shape[-1]
            weight_codomain = th.zeros_like(weight)
            weight_codomain[..., :n - 1, :n - 1] = weight.tril(-1)[..., 1:n , :n-1]
            # **Symmetrization**: Ensure the upper-left part of Row0 is a symmetric matrix
            weight_codomain = weight_codomain.tril() + weight_codomain.tril(-1).transpose(-1, -2)
            # **Step 2**: Compute the last column so that the sum of each row is 0
            weight_codomain[..., :-1, -1] = -th.sum(weight_codomain[..., :-1, :-1], dim=-1)
            # **Step 3**: Compute the `[n, n]` element so that the last row can be recovered symmetrically
            weight_codomain[..., -1, -1] = -th.sum(weight_codomain[..., :-1, -1], dim=-1)
            # **Step 4**: Restore the last row to be equal to the transpose of the last column
            weight_codomain[..., -1, :-1] = weight_codomain[..., :-1, -1]
        return weight_codomain

    def forward(self,X):
            #deformation
            X_deformed = self.manifold.deformation(X)
            if self.param_mode == 'Riem':
                weight = self.manifold.deformation(self.poincare_to_correlation(self.weight))
            elif self.param_mode == 'trivial':
                weight = self.Euc2Codomain(self.weight)

            if(self.training):
                # centering
                batch_mean = self.manifold.cal_geom_mean(X_deformed,batchdim=self.batchdim)
                X_centered = self.manifold.translation(X_deformed,batch_mean,is_inverse=True)

                # scaling and shifting
                # Note that as the mean is calculated in closed-form, cal_geom_var (by dist(X_i,I)) is accurate
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
            if self.is_inv_deformation:
                X_new = self.manifold.inv_deformation(X_normalized)
            else:
                X_new = X_normalized
            return X_new

    def get_manifold(self):
        # ECM,LECM,OLM,LSM
        classes = {
            "ECM": CorEuclideanCholeskyMetric,
            "LECM": CorLogEuclideanCholeskyMetric,
            "OLM": CorOffLogMetric,
            "LSM": CorLogScaledMetric,
        }

        if self.metric == 'OLM':
            self.manifold = classes[self.metric](n=self.shape[-1],
                                                 alpha=self.alpha,beta=self.param_1, gamma=self.param_2,
                                                 max_iter=self.max_iter)
        elif self.metric == 'LSM':
            self.manifold = classes[self.metric](n=self.shape[-1],
                                                 alpha=self.alpha, delta=self.param_1,zeta=self.param_2,
                                                 max_iter=self.max_iter)
        elif self.metric in ['ECM','LECM']:
            self.manifold = classes[self.metric](n=self.shape[-1])
        else:
            raise NotImplementedError

        self.manifold.is_detach = self.is_detach

