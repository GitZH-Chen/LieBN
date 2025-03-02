"""
SPD computations under LieBN:
    @inproceedings{chen2024liebn,
        title={A Lie Group Approach to Riemannian Batch Normalization},
        author={Ziheng Chen and Yue Song and Yunmei Liu and Nicu Sebe},
        booktitle={The Twelfth International Conference on Learning Representations},
        year={2024},
        url={https://openreview.net/forum?id=okYdj8Ysru}
    }
"""
import torch as th
from .sym_functional import sym_powm,sym_logm,sym_invsqrtm2,sym_expm,sym_Glogm,sym_Gexpm
from ..Base.LieGroups import LieGroup,PullbackEuclideanMetric

tril_param_metric = {'AIM','CRIM'} # use karcher flow
bi_param_metric = {'LEM','ALEM'}
single_param_metric = {'LCM'}

spd_metrics=['AIM','LCM','LEM','ALEM','CRIM']

class SPDMatrices(LieGroup):
    """Computation for SPD data with [...,n,n]
        Following SPDNetBN, we use .detach() for mean and variance calculation
    """
    def __init__(self, n,power=1.,is_detach=True):
        super().__init__(is_detach=is_detach)
        self.n=n; self.dim = int(n * (n + 1) / 2)
        self.register_buffer('power', th.tensor(power))
        self.register_buffer('I', th.eye(n))
        if power == 0:
            raise Exception('power should not be zero with power={:.4f}'.format(power))

    def random(self,*shape):
        """
        Generates a batch of random Symmetric Positive Definite (SPD) matrices.

        Args:
            *shape: Arbitrary batch shape before the SPD matrix dimension.
        Returns:
            Tensor: A tensor of shape (*shape) containing SPD matrices.
        """
        assert len(shape) >= 2 and shape[-2] == shape[-1], "Shape must be [..., n, n] for square matrices"
        n = shape[-1]
        A = th.randn(*shape, dtype=th.double)  # Generate a random matrix
        SPD = th.einsum('...ij,...kj->...ik', A, A)  # Ensure SPD property
        SPD += th.eye(n, dtype=th.double).expand_as(SPD) * 1e-3  # Add small identity for numerical stability
        return SPD

    def inner_product(self, X, Y):
        """"Canonical inner product of X and Y, with [...,n,n]"""
        return th.einsum("...ij,...ij->...", X, Y)

    def trace(self, X):
        """"Canonical norm of X, with [...,n,n]"""
        r_trace = X.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)  # Keeps the last dimension
        return r_trace

    def spd_pow(self, S,power):
        """ computing S^{\theta}"""
        if power == 2.:
            Power_S = S.matmul(S)
        elif power == 1.:
            Power_S = S
        else:
            Power_S = sym_powm.apply(S, power)
        return Power_S

class SPDLogEuclideanMetric(PullbackEuclideanMetric,SPDMatrices):
    """Log-Euclidean Metric"""
    def __init__(self, n, alpha=1.,beta=0.):
        super().__init__(n=n, power=1.)
        if alpha <= 0 or beta <= -alpha / n:
            raise Exception('wrong alpha or beta with alpha={:.4f},beta={:.4f}'.format(alpha, beta))
        self.register_buffer('alpha', th.tensor(alpha))
        self.register_buffer('beta', th.tensor(beta))

    def deformation(self, X):
        return sym_logm.apply(X)

    def inv_deformation(self,X):
        return sym_expm.apply(X)

    def dist2Isquare(self,X):
        """"computing the O(n)-invariant Euclidean square distance to the identity (element)"""
        if self.beta == 0.:
            dist = self.alpha * th.linalg.matrix_norm(X,keepdim=True).square()
        else:
            item1 = th.linalg.matrix_norm(X,keepdim=True)
            item2 = self.trace(X)
            dist = self.alpha * item1.square() + self.beta * item2.square()
        return dist

class SPDAdaptiveLogEuclideanMetric(SPDLogEuclideanMetric):
    """Adaptive Log-Euclidean Metric"""
    def __init__(self, n, alpha=1.,beta=0.):
        super().__init__(n=n, alpha=alpha, beta=beta)
        if alpha <= 0 or beta <= -alpha / n:
            raise Exception('wrong alpha or beta with alpha={:.4f},beta={:.4f}'.format(alpha, beta))
        self.register_buffer('alpha', th.tensor(alpha))
        self.register_buffer('beta', th.tensor(beta))
        self.A = th.nn.Parameter(th.ones(n))

    def deformation(self, X):
        return sym_Glogm.apply(X,self.A)

    def inv_deformation(self,X):
        return sym_Gexpm.apply(X,self.A)

class SPDLogCholeskyMetric(PullbackEuclideanMetric,SPDMatrices):
    """Log-Cholesky Metric"""

    def __init__(self, n, power=1.):
        super().__init__(n=n, power=power)

    def deformation(self,X):
        X_power = self.spd_pow(X,self.power)
        L = th.linalg.cholesky(X_power)
        diag_part = th.diag_embed(th.log(th.diagonal(L, dim1=-2, dim2=-1)))
        X_deformed = L.tril(-1) + diag_part
        return X_deformed

    def inv_deformation(self,X):
        Cho = X.tril(-1) + th.diag_embed(th.exp(th.diagonal(X, dim1=-2, dim2=-1)))
        spd = Cho.matmul(Cho.transpose(-1,-2))
        X_inv_deformed = self.spd_pow(spd,1/self.power)
        return X_inv_deformed

    def dist2Isquare(self,X):
        return (1 / (self.power ** 2)) * (th.linalg.matrix_norm(X,keepdim=True).square())

# --- AIM and RIM, which are based on Cholesky matrix product
class SPDLieCholesky(SPDMatrices):
    """Lie-Cholesky group computations for SPD data with [...,n,n]"""
    def __init__(self, n, power=1.):
        super().__init__(n=n, power=power)

    def LieCholesky_inv(self, P):
        # return the Lie-Cholesky group inverse
        L_inv = th.linalg.inv(th.linalg.cholesky(P))
        return L_inv @ L_inv.transpose(-1, -2)

    def LieCholesky_prod(self, P, Q, is_left=True, is_inverse=False):
        '''
        Group operation for
            Left:
            P \odot Q = LQL^\top (default)
            \odot^{-1} P \odot Q = L^{-1} Q L^{-\top}
            Rgiht:
            Q \odot P = KPK^\top
            Q \odot^{-1} P = KL^{-1} (KL^{-1})^\top
        '''
        # Left
        if is_left:
            L = th.linalg.cholesky(P);
            if is_inverse:
                # \odot^{-1} P \odot Q = L^{-1} Q L^{-\top}
                L_inv = th.linalg.inv(L)
                X_new = L_inv @ Q @ L_inv.transpose(-1, -2)
            else:
                X_new = L @ Q @ L.transpose(-1, -2)
        else:
            K = th.linalg.cholesky(Q);
            if is_inverse:
                #  Q \odot^{-1} P = KL^{-1} (KL^{-1})^\top
                L = th.linalg.cholesky(P);
                K_L_inv = th.linalg.solve(L, K, left=False)
                X_new = K_L_inv @ K_L_inv.transpose(-1, -2)
            else:
                X_new = K @ P @ K.transpose(-1, -2)
        return X_new

class SPDAffineInvariantMetric(SPDLieCholesky):
    def __init__(self, n, power=1., alpha=1.0,beta=0.,karcher_steps=1):
        super().__init__(n, power)
        if alpha <= 0 or beta <= -alpha / n:
            raise Exception('wrong alpha or beta with alpha={:.4f},beta={:.4f}'.format(alpha, beta))
        self.alpha = alpha;
        self.beta = beta;
        self.karcher_steps=karcher_steps

    def logmap(self, P, Q):
        # Log_{P} (Q)
        P_sqrt, P_invsqrt = sym_invsqrtm2.apply(P)
        return P_sqrt @ sym_logm.apply(P_invsqrt @ Q @ P_invsqrt) @ P_sqrt

    def expmap(self, P, V):
        # Exp_{P} (V)
        P_sqrt, P_invsqrt = sym_invsqrtm2.apply(P)
        return P_sqrt @ sym_expm.apply(P_invsqrt @ V @ P_invsqrt) @ P_sqrt

    def geodesic(self,P,Q,t):
        P_sqrt, P_invsqrt = sym_invsqrtm2.apply(P)
        return P_sqrt @ sym_powm.apply(P_invsqrt @ Q @ P_invsqrt,t) @ P_sqrt

    #--- Computaiton of LieBN under AIM
    def deformation(self,X):
        return self.spd_pow(X,self.power)

    def inv_deformation(self,X):
        return self.spd_pow(X,1/self.power)

    def dist2Isquare(self,P):
        orgn = self.alpha * th.linalg.matrix_norm(sym_logm.apply(P)).square() + self.beta * th.logdet(P).square()
        return ((1 / (self.power ** 2)) * orgn).unsqueeze(-1).unsqueeze(-1) # keep dim

    def cal_geom_mean_(self, batch,batchdim=[0]):
        '''
        Function which computes the Riemannian barycenter for a batch of data using the Karcher flow
        Input x is a batch of SPD matrices (batch_size,1,n,n) to average
        Output is (n,n) Riemannian mean
        '''
        batch_mean = batch.mean(dim=batchdim)
        for ith in range(self.karcher_steps):
            bm_sq, bm_invsq = sym_invsqrtm2.apply(batch_mean)
            XT = sym_logm.apply(bm_invsq @ batch @ bm_invsq)
            GT = XT.mean(dim=batchdim)
            condition = GT.norm(dim=(-1, -2))
            # print(f'{ith+1}: {condition.item()}')
            if th.all(condition <= 0.1):
                # th.sum(condition>=0.1)
                # print('early stop')
                break
            batch_mean = bm_sq @ sym_expm.apply(GT) @ bm_sq
        return batch_mean

    def translation(self, X, P, is_inverse):
        X_new = self.LieCholesky_prod(P,X,is_left=True,is_inverse=is_inverse)
        return X_new

    def scaling(self, X, factor):
        return sym_powm.apply(X, factor)
class SPDCholeskyRightInvariantMetric(SPDAffineInvariantMetric):
    def __init__(self, n, power=1., alpha=1.0,beta=0.,karcher_steps=1):
        super().__init__(n,power=power,alpha=alpha,beta=beta,karcher_steps=karcher_steps)
    def sym(self,X):
        # Make a square matrix symmetrized, (A+A')
        return X+X.transpose(-1,-2)

    def tril_half_diag(self, A):
        """"[...n,n] A, strictly lower part + (1/2)*half of diagonal part"""
        return A.tril(-1) + th.diag_embed(0.5 * th.diagonal(A, dim1=-2, dim2=-1))

    def affine_logmap(self, P,Q):
        # Call the parent class's logmap
        return super().logmap(P,Q)

    def affine_expmap(self, P,V):
        # Call the parent class's expmap
        return super().expmap(P,V)

    def affine_dist2Isquare(self,P):
        '''square AIM distance to the In '''
        return super().dist2Isquare(P)
    def affine_geodesic(self,P,Q,t):
        '''square AIM geodesic '''
        return super().geodesic(P,Q,t)

    def logmap(self, P, Q):
        # Log_{P} (Q)
        L = th.linalg.cholesky(P)
        L_inv = th.linalg.inv(L)
        P_LieCholeskyInv = L_inv @ L_inv.transpose(-1,-2)
        Q_LieCholeskyInv = self.LieCholesky_inv(Q)
        V_bar = self.affine_logmap(P_LieCholeskyInv,Q_LieCholeskyInv)
        temp_tangent = P @ self.tril_half_diag(L @ V_bar @ L.transpose(-1,-2)).transpose(-1,-2)
        return -self.sym(temp_tangent)

    def expmap(self, P, V):
        # Exp_{P} (V)
        L = th.linalg.cholesky(P)
        L_inv = th.linalg.inv(L)
        L_inv_trans = L_inv.transpose(-1,-2)
        P_LieCholeskyInv = L_inv @ L_inv_trans
        tmp_tangent = self.tril_half_diag(L_inv @ V @ L_inv_trans) @ P_LieCholeskyInv
        tangent = - self.sym(tmp_tangent)
        return self.LieCholesky_inv(self.affine_expmap(P_LieCholeskyInv,tangent))

    def geodesic(self,P,Q,t):
        tmp_spd = self.affine_geodesic(self.LieCholesky_inv(P),self.LieCholesky_inv(Q),t)
        return self.LieCholesky_inv(tmp_spd)

    def dist2Isquare(self, P):
        '''square distance to the In '''
        return self.affine_dist2Isquare(self.LieCholesky_inv(P))

    def cal_geom_mean_(self, batch,batchdim=[0]):
        '''
        Function which computes the Riemannian barycenter for a batch of data using the Karcher flow
        Input x is a batch of SPD matrices (batch_size,1,n,n) to average
        Output is (n,n) Riemannian mean
        '''
        batch_mean = batch.mean(dim=batchdim)
        for ith in range(self.karcher_steps):
            tan_data = self.logmap(batch_mean,batch)
            tan_mean = tan_data.mean(dim=batchdim)
            condition = tan_mean.norm(dim=(-1, -2))
            # print(f'{ith+1}: {condition.item()}')
            if th.all(condition <= 0.1):
                # th.sum(condition>=0.1)
                # print('early stop')
                break
            batch_mean = self.expmap(batch_mean,tan_mean)
        # print(th.sum(condition >= 0.1))
        return batch_mean

    def translation(self, X, P, is_inverse):
        return self.LieCholesky_prod(P,X,is_left=False,is_inverse=is_inverse)

    def scaling(self, X, factor):
        X_LieCholeskyInv_power = sym_powm.apply(self.LieCholesky_inv(X), factor)
        return self.LieCholesky_inv(X_LieCholeskyInv_power)