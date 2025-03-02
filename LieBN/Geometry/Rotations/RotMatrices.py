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
from ..Base.LieGroups import LieGroup

from geoopt.manifolds import Manifold
from pytorch3d.transforms import matrix_to_axis_angle

class RotMatrices(LieGroup,Manifold):
    """
    Computation for SO3 data with size of [...,n,n]
    Following manopt, we use the Lie algebra representation for the tangent spaces
    """
    # __scaling__ = Manifold.__scaling__.copy()
    name = "Rotation"
    ndim = 2
    reversible = False
    def __init__(self,eps=1e-5,is_detach=True,karcher_steps=1):
        super().__init__(is_detach=is_detach)
        self.eps=eps;
        self.register_buffer('I', th.eye(3))
        self.karcher_steps=karcher_steps

    # === Generating random matrices ===

    def random(self, *shape):
        """Generate random 3D rotation matrices of shape [..., 3, 3]"""
        A = th.rand(shape)  # Generate a random matrix
        u, _, v = th.linalg.svd(A)  # Perform SVD

        # Ensure determinant is +1
        det_u = u.det()[..., None]  # Reshape to match the last two dimensions
        u[..., :, -1] *= th.sign(det_u)  # Flip last column where needed
        # u_new= u[..., :, -1] * th.sign(det_u)  # Flip last column where needed

        # Verify that the output is a valid SO(3) matrix
        result, _ = self._check_point_on_manifold(u)
        if not result:
            raise ValueError("SO(3) init value error")

        return u

    def rand_skrew_sym(self,n,f):
        A=th.rand(n,f,3,3)
        return A-A.transpose(-1,-2)

    # === For geoopt ===

    def quat_axis2log_axis(self,axis_quat):
        """"convert quaternion axis into matrix log axis
        https://github.com/facebookresearch/pytorch3d/issues/188
        """
        log_axis_result = axis_quat.clone()
        angle_in_2pi = log_axis_result.norm(p=2, dim=-1, keepdim=True)
        mask = angle_in_2pi > th.pi
        tmp = 1 - 2 * th.pi / angle_in_2pi
        new_values = tmp * log_axis_result
        results = th.where(mask, new_values, log_axis_result)
        return results

    def matrix2euler_axis(self,R):
        quat_vec = matrix_to_axis_angle(R)
        log_axis = self.quat_axis2log_axis(quat_vec)
        euler_axis = log_axis.div(log_axis.norm(dim=-1,keepdim=True))
        return euler_axis

    def mLog(self, R):
        """
        Note that Exp(\alpha A)=Exp(A), with \alpha = \frac{\theta-2\pi}{\theta}
        So, for a single rotation matrices, quat_axis or log_axis does not affect self.mExp.
        """
        vec = matrix_to_axis_angle(R)
        log_vec = self.quat_axis2log_axis(vec)
        skew_symmetric = self.vec2skrew(log_vec)
        return skew_symmetric

    def vec2skrew(self,vec):
        # skew_symmetric = th.zeros_like(vec).unsqueeze(-1).expand(*vec.shape, 3).contiguous()
        skew_symmetric = th.zeros(*vec.shape, 3,dtype=vec.dtype,device=vec.device)
        skew_symmetric[..., 0, 1] = -vec[..., 2]
        skew_symmetric[..., 1, 0] = vec[..., 2]
        skew_symmetric[..., 0, 2] = vec[..., 1]
        skew_symmetric[..., 2, 0] = -vec[..., 1]
        skew_symmetric[..., 1, 2] = -vec[..., 0]
        skew_symmetric[..., 2, 1] = vec[..., 0]
        return skew_symmetric
    def mExp(self, S):
        """Computing matrix exponential for skrew symmetric matrices"""
        a, b, c = S[..., 0, 1], S[..., 0, 2], S[..., 1, 2]
        theta = th.sqrt(a ** 2 + b ** 2 + c ** 2).unsqueeze(-1).unsqueeze(-1)

        S_normalized = S / theta
        S_norm_squared = S_normalized.matmul(S_normalized)
        sin_theta = th.sin(theta)
        cos_theta = th.cos(theta)
        tmp_S = self.I + sin_theta * S_normalized + (1 - cos_theta) * S_norm_squared

        S_new = th.where(theta < self.eps, S-S.detach()+self.I, tmp_S) # S+I to ensure autograd

        return S_new

    def transp(self, x, y, v):
        return v

    def inner(self, x, u, v, keepdim=False):
        if v is None:
            v = u
        return th.sum(u * v, dim=[-2, -1], keepdim=keepdim)

    def projx(self, x):
        u, s, vt = th.linalg.svd(x)
        ones = th.ones_like(s)[..., :-1]
        signs = th.sign(th.det(th.matmul(u, vt))).unsqueeze(-1)
        flip = th.cat([ones, signs], dim=-1)
        result  = u.matmul(th.diag_embed(flip)).matmul(vt)
        return result

    def proju(self,X, H):
        k = self.multiskew(H)
        return k

    def egrad2rgrad(self, x, u):
        """Map the Euclidean gradient :math:`u` in the ambient space on the tangent
        space at :math:`x`.
        """
        k = self.multiskew(x.transpose(-1, -2).matmul(u))
        return k

    def retr(self, X,U):
        Y = X + X.matmul(U)
        Q, R = th.linalg.qr(Y)
        New = th.matmul(Q, th.diag_embed(th.sign(th.sign(th.diagonal(R, dim1=-2, dim2=-1)) + 0.5)))
        return New

    def multiskew(self,A):
        return 0.5 * (A - A.transpose(-1,-2))

    def logmap(self,R,S):
        """ return skrew symmetric matrices """
        return self.mLog(R.transpose(-1,-2).matmul(S))

    def expmap(self,R,V):
        """ V is the skrew symmetric matrices """
        return R.matmul(self.mExp(V))

    def geodesic(self,R,S,t):
        """ the geodesic connecting R and s """
        vector = self.logmap(R,S)
        X_new = R.matmul(self.mExp(t*vector))
        return X_new

    def trace(self,m):
        """Computation for trace of m of [...,n,n]"""
        return th.einsum("...ii->...", m)

    def _check_point_on_manifold(
            self, x: th.Tensor, *, atol=1e-5, rtol=1e-8
        ):

        if x.shape[-1] != 3 or x.shape[-2] != 3:
            raise ValueError("Input matrices must be 3x3.")

        # Check orthogonality
        is_orthogonal = th.allclose(x @ x.transpose(-1, -2),self.I, atol=atol,rtol=rtol)

        # Check determinant
        det = th.det(x)
        is_det_one = th.allclose(det, th.tensor(1.0, device=x.device,dtype=x.dtype), atol=atol,rtol=rtol)

        # Combine both conditions
        is_SO3 = is_orthogonal & is_det_one

        return is_SO3, None

    def _check_vector_on_tangent(
            self, x: th.Tensor, u: th.Tensor, *, atol=1e-5, rtol=1e-8
    ):
        """Check whether u is a skrew symmetric matrices"""
        diff = u + u.transpose(-1,-2)
        ok = th.allclose(diff, th.zeros_like(diff), atol=atol, rtol=rtol)
        return ok, None

    def is_not_equal(self, a, b, eps=0.01):
        """ Return true if not eaqual"""
        return th.nonzero(th.abs(a - b) > eps)

    # === Computing angles ===

    def cal_roc_angel_batch(self,r,epsilon=1e-4):
        """
        Following the matlab implemetation, we set derivative=0 for tr near -1 or near 3.
        Besides, there could be cases where tr \in [-1-eps, 3+eps] due to numerical error.
        We view the cases beyond the [-1,3] as -1 or 3, and the derivative is 0.
        return:
            tr <= -1-epsilon, theta=pi (derivative is 0)
            tr >= 3-epsilon, theta=0 (derivative is 0)
        """
        assert epsilon >= 0, "Epsilon must be positive"

        mtrc = self.trace(r)

        maskpi = (mtrc + 1) <= epsilon  # tr <= -1 + epsilon
        # mtrcpi = -mtrc * maskpi * np.pi # this is different from the matlab implemetation, as its direvative is -pi
        mtrcpi = maskpi * th.pi
        maskacos = ((mtrc + 1) > epsilon) * ((3- mtrc) > epsilon) # -1+epsilon < tr < 3 - epsilon

        mtrcacos = th.acos((mtrc * maskacos - 1) / 2) * maskacos # -1+epsilon < tr < 3 - epsilon, use the acos
        results = mtrcpi + mtrcacos # for tr  -tr <= -1 + epsilon and tr >= 3-epsilon, the derivative = 0
        return results

    # === For LieBN ===

    def cal_geom_mean_(self, X, batchdim=[0,1]):
        "Karcher flow"
        init_point = X[tuple(0 for _ in batchdim)]  # Dynamically selects first elements along batchdim
        for ith in range(self.karcher_steps):
            tan_data = self.mLog(init_point.transpose(-1,-2).matmul(X))
            tan_mean = tan_data.mean(dim=batchdim)
            condition = tan_mean.norm(dim=(-1, -2))
            # print(f'{ith+1}: {condition}')
            if th.all(condition<=1e-4):
                # th.sum(condition>=0.1)
                # print('early stop')
                break
            init_point = init_point.matmul(self.mExp(tan_mean))
        return init_point

    def scaling(self, X,shift,running_var=None,batchdim=[0,1]):
        """Frechet variance"""
        Log_X = self.mLog(X)
        if running_var is None:
            Log_X_norm_square = Log_X.norm(p='fro', dim=(-2, -1), keepdim=True).square()
            var = th.mean(Log_X_norm_square, dim=batchdim)
            var = var.detach() if self.is_detach else var
        else:
            var = running_var
        factor = shift / (var + self.eps).sqrt()
        scale_Log = factor * Log_X
        X_scaled = self.mExp(scale_Log)
        return X_scaled,var

    def translation(self, X, P, is_inverse, is_left):
        """translation by P"""
        if is_left:
            X_new = P.transpose(-1, -2).matmul(X) if is_inverse else P.matmul(X)
        else:
            X_new = X.matmul(P.transpose(-1, -2)) if is_inverse else X.matmul(P)
        return X_new

