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
import torch.nn as nn

from abc import ABC


class LieGroup(nn.Module):
    """LieBN for [...,n,n]: Following SPDNetBN, we use .detach() for mean and variance calculation
    """
    def __init__(self,is_detach=True):
        super().__init__()
        self.is_detach=is_detach

    def geodesic(self,P,Q,t):
        '''The geodesic from P to Q at t'''
        raise NotImplemented

    #--- Methods required in LieBN ---
    def deformation(self,X):
        raise NotImplemented

    def inv_deformation(self,X):
        raise NotImplemented
    def dist2Isquare(self,X):
        """geodesic square distance to the identity element.
        it should keep all the dim, such that cal_geom_var only eliminate the batch dim.
        This will support [bs,...,n,n] data
        """
        raise NotImplemented

    def cal_geom_mean_(self,X,batchdim=[0]):
        raise NotImplemented

    def cal_geom_mean(self,X,batchdim=[0]):
        batch = X.detach() if self.is_detach else X
        return self.cal_geom_mean_(batch,batchdim)

    def cal_geom_var(self, X,batchdim=[0]):
        """Frechet variance"""
        batch = X.detach() if self.is_detach else X
        return self.dist2Isquare(batch).mean(dim=batchdim)

    def translation(self, X, P, is_inverse):
        raise NotImplemented

    def scaling(self, X, factor):
        raise NotImplemented

    def __repr__(self):
        attributes = []
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            if isinstance(value, th.Tensor):
                attributes.append(f"{key}.shape={tuple(value.shape)}")  # 只显示 Tensor 形状
            else:
                attributes.append(f"{key}={value}")

        # 处理 register_buffer 注册的变量
        for name, buffer in self.named_buffers(recurse=False):  # 只打印当前类的 buffers
            attributes.append(f"{name}={buffer.item() if buffer.numel() == 1 else tuple(buffer.shape)}")

        return f"{self.__class__.__name__}({', '.join(attributes)})"

class PullbackEuclideanMetric(ABC):
    """The Euclidean computation in the co-domain of PullBack Euclidean Metric:
        Ziheng Chen, etal, Adaptive Log-Euclidean Metrics for SPD Matrix Learning
        For the subclass of this class, we only need to implement
            dist2Isquare, deformation, and inv_deformation
    """
    # def __init__(self, is_detach=True):
    #     super().__init__(is_detach=is_detach)

    def cal_geom_mean_(self, batch,batchdim=[0]):
        return batch.mean(dim=batchdim)

    def translation(self, X, P, is_inverse):
        X_new = X - P if is_inverse else X + P
        return X_new

    def scaling(self, X, factor):
        return X * factor

    def geodesic(self,P,Q,t):
        return (1 - t) * P + t * Q