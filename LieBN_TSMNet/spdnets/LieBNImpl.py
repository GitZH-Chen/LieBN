"""
@author: Ziheng Chen
Please cite the paper below if you use the code:

Ziheng Chen, Yue Song, Yunmei Liu, Nicu Sebe. A Lie Group Approach to Riemannian Batch Normalization. ICLR 2024

Copyright (C) 2024 Ziheng Chen
All rights reserved.
"""

from builtins import NotImplementedError
from typing import Tuple
import torch as th
import torch.nn as nn

from geoopt.tensor import ManifoldParameter, ManifoldTensor
from .manifolds import SymmetricPositiveDefinite
from . import functionals
from spdnets.BaseBatchNorm import BaseBatchNorm,BatchNormDispersion,BatchNormTestStatsMode


class SPDLieBatchNormImpl(BaseBatchNorm):
    def __init__(self, shape: Tuple[int, ...] or th.Size, batchdim: int,
                 eta=1., eta_test=0.1,
                 karcher_steps: int = 1, learn_mean=False, learn_std=True,
                 dispersion: BatchNormDispersion = BatchNormDispersion.SCALAR,
                 eps=1e-5, mean=None, std=None,
                 metric='AIM',theta: float =1.,alpha: float=1.0,beta: float=0.,
                 **kwargs):
        super().__init__(eta, eta_test)
        if metric != "AIM" and metric != "LCM" and metric != "LEM" :
            raise Exception('unknown metric {}'.format(metric))
        self.metric = metric;self.theta = th.tensor(theta);self.alpha = th.tensor(alpha);self.beta = th.tensor(beta)
        self.identity = th.eye(shape[-1])

        # the last two dimensions are used for SPD manifold
        assert (shape[-1] == shape[-2])

        if dispersion == BatchNormDispersion.VECTOR:
            raise NotImplementedError()

        self.dispersion = dispersion
        self.learn_mean = learn_mean
        self.learn_std = learn_std
        self.batchdim = batchdim
        self.karcher_steps = karcher_steps
        self.eps = eps

        init_mean = th.diag_embed(th.ones(shape[:-1], **kwargs))
        init_var = th.ones((*shape[:-2], 1), **kwargs)

        self.register_buffer('running_mean', ManifoldTensor(init_mean,
                                                            manifold=SymmetricPositiveDefinite()))
        self.register_buffer('running_var', init_var)
        self.register_buffer('running_mean_test', ManifoldTensor(init_mean,
                                                                 manifold=SymmetricPositiveDefinite()))
        self.register_buffer('running_var_test', init_var)

        if mean is not None:
            self.mean = mean
        else:
            if self.learn_mean:
                self.mean = ManifoldParameter(init_mean.clone(), manifold=SymmetricPositiveDefinite())
            else:
                self.mean = ManifoldTensor(init_mean.clone(), manifold=SymmetricPositiveDefinite())

        if self.dispersion is not BatchNormDispersion.NONE:
            if std is not None:
                self.std = std
            else:
                if self.learn_std:
                    self.std = nn.parameter.Parameter(init_var.clone())
                else:
                    self.std = init_var.clone()

    @th.no_grad()
    def initrunningstats(self, X):
        self.running_mean.data = self.cal_geom_mean(X)
        self.running_mean_test.data = self.running_mean.data.clone()

        if self.dispersion is BatchNormDispersion.SCALAR:
            self.running_var.data = self.cal_geom_var(X, self.running_mean)
            self.running_var_test.data = self.running_var.data.clone()

    def forward(self, X):
        X_deformed = self.deformation(X);
        if self.learn_mean:
            weight = self.deformation(self.mean)

        if self.training:
            # compute batch mean
            batch_mean = self.cal_geom_mean(X_deformed)
            # update the running_mean, running_mean_test
            self.updating_running_means(batch_mean)
            # compute batch vars w.r.t. running_mean and running_mean_test, update the running_var and running_var_test
            if self.dispersion is not BatchNormDispersion.NONE:
                batch_var = self.cal_geom_var(X_deformed,self.running_mean)
                batch_var_test = self.cal_geom_var(X_deformed, self.running_mean_test)
                self.updating_running_var(batch_var,batch_var_test)
        else:
            if self.test_stats_mode == BatchNormTestStatsMode.BUFFER:
                pass  # nothing to do: use the ones in the buffer
            elif self.test_stats_mode == BatchNormTestStatsMode.REFIT:
                self.initrunningstats(X_deformed)
            elif self.test_stats_mode == BatchNormTestStatsMode.ADAPT:
                raise NotImplementedError()

        rm = self.running_mean if self.training else  self.running_mean_test
        if self.dispersion is BatchNormDispersion.SCALAR:
            rv = self.running_var if self.training else self.running_var_test

        # subtracting mean
        X_centered = self.Left_translation(X_deformed,rm,True)
        # scaling and shifting
        if self.dispersion is BatchNormDispersion.SCALAR:
            X_scaled = self.scaling(X_centered, rv,self.std)
        # biasing
        X_normalized = self.Left_translation(X_scaled, weight,False) if self.learn_mean else X_scaled
        # inv of deformation
        X_new = self.inv_deformation(X_normalized)

        return X_new

    def spd_power(self,X):
        if self.theta == 1.:
            X_power = X
        else:
            X_power = functionals.sym_powm.apply(X, self.theta)
        return X_power

    def inv_power(self,X):
        if self.theta == 1.:
            X_power = X
        else:
            X_power = functionals.sym_powm.apply(X, 1/self.theta)
        return X_power
    def deformation(self,X):

        if self.metric=='AIM':
            X_deformed = self.spd_power(X)
        elif self.metric == 'LEM':
            X_deformed = functionals.sym_logm.apply(X)
        elif self.metric == 'LCM':
            X_power = self.spd_power(X)
            L = th.linalg.cholesky(X_power)
            diag_part = th.diag_embed(th.log(th.diagonal(L, dim1=-2, dim2=-1)))
            X_deformed = L.tril(-1) + diag_part

        return X_deformed

    def inv_deformation(self,X):
        if self.metric=='AIM':
            X_inv_deformed = self.inv_power(X)
        elif self.metric == 'LEM':
            X_inv_deformed = functionals.sym_expm.apply(X)
        elif self.metric == 'LCM':
            Cho = X.tril(-1) + th.diag_embed(th.exp(th.diagonal(X, dim1=-2, dim2=-1)))
            spd = Cho.matmul(Cho.transpose(-1,-2))
            X_inv_deformed = self.inv_power(spd)
        return X_inv_deformed

    def cal_geom_mean(self, X):
        """Frechet mean"""
        if self.metric == 'AIM':
            mean = self.KF_AIM(X.detach())
        elif self.metric == 'LEM' or self.metric == 'LCM':
            mean = X.detach().mean(dim=self.batchdim, keepdim=True)

        return mean
    def cal_geom_var(self, X, rm):
        """Frechet variance w.r.t. rm"""
        spd = X.detach()
        if self.metric == 'AIM':
            rm_invsq = functionals.sym_invsqrtm.apply(rm)
            if self.beta == 0.:
                dists = self.alpha * th.linalg.matrix_norm(
                    functionals.sym_logm.apply(rm_invsq @ spd @ rm_invsq)).square()
            else:
                dists = self.alpha * th.linalg.matrix_norm(functionals.sym_logm.apply(rm_invsq @ spd @ rm_invsq)).square()\
                        + self.beta * th.logdet(th.linalg.solve(rm,spd)).square()

        elif self.metric == 'LEM' or self.metric == 'LCM':
            tmp = spd - rm
            if self.beta == 0.:
                dists = self.alpha * th.linalg.matrix_norm(tmp)
            else:
                item1 = th.linalg.matrix_norm(tmp)
                item2 = functionals.trace(tmp)
                dists = self.alpha * item1 + self.beta * item2.square()

        var = dists.mean(dim=self.batchdim, keepdim=True)
        if self.metric == 'AIM' or self.metric == 'LCM':
            var_final = var* (1/(self.theta**2))
        else:
            var_final = var
        return var_final.unsqueeze(dim=0)
    def Left_translation(self, X, P, is_inverse):
        """Left translation by P"""
        if self.metric == 'AIM':
            L = th.linalg.cholesky(P);
            if is_inverse:
                tmp = th.linalg.solve(L, X)
                X_new = th.linalg.solve(L.transpose(-1, -2), tmp, left=False)
            else:
                X_new = L.matmul(X).matmul(L.transpose(-1, -2))

        elif self.metric == 'LEM' or self.metric == 'LCM':
            X_new = X - P if is_inverse else X + P

        return X_new
    def scaling(self, X, var,scale):
        """Scaling by variance"""
        factor = scale / (var+self.eps).sqrt()
        if self.metric == 'AIM':
            X_new = functionals.sym_powm.apply(X, factor)
        elif self.metric == 'LEM' or self.metric == 'LCM':
            X_new = X * factor

        return X_new
    def updating_running_means(self,batch_mean):
        """updating running means"""
        with th.no_grad():
            if self.metric == 'AIM':
                self.running_mean.data = functionals.spd_2point_interpolation(self.running_mean, batch_mean,self.eta)
                self.running_mean_test.data = functionals.spd_2point_interpolation(self.running_mean_test, batch_mean,self.eta_test)
            elif self.metric == 'LEM' or self.metric == 'LCM':
                self.running_mean.data = (1-self.eta) * self.running_mean+ batch_mean * self.eta
                self.running_mean_test.data = (1 - self.eta_test) * self.running_mean_test + batch_mean * self.eta_test

    def updating_running_var(self, batch_var, batch_var_test):
        """updating running vars"""
        with th.no_grad():
            if self.dispersion is BatchNormDispersion.SCALAR:
                self.running_var = (1. - self.eta) * self.running_var + self.eta * batch_var
                self.running_var_test = (1. - self.eta_test) * self.running_var_test + self.eta_test * batch_var_test

    def KF_AIM(self,X,karcher_steps=1):
        '''
        Function which computes the Riemannian barycenter for a batch of data using the Karcher flow
        Input x is a batch of SPD matrices (batch_size,1,n,n) to average
        Output is (n,n) Riemannian mean
        '''
        batch_mean = X.mean(dim=self.batchdim, keepdim=True)
        for _ in range(karcher_steps):
            bm_sq, bm_invsq = functionals.sym_invsqrtm2.apply(batch_mean)
            XT = functionals.sym_logm.apply(bm_invsq @ X @ bm_invsq)
            GT = XT.mean(dim=self.batchdim, keepdim=True)
            batch_mean = bm_sq @ functionals.sym_expm.apply(GT) @ bm_sq
        return batch_mean
