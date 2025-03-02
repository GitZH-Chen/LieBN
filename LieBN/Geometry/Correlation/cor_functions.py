import torch as th
from torch.functional import Tensor
from torch.autograd import Function
from typing import Callable, Tuple

from ..SPD.sym_functional import sym_expm,EPS, ensure_sym
def unique_diagonal_matrix_off_log(sym_mat, atol=1e-6, max_iter=100):
    """
    Finds the unique diagonal matrix of a symmetric matrix for OLM.

    Parameters:
        sym_mat (th.Tensor): Symmetric matrix or batch of symmetric matrices of shape [..., n, n].
        atol (float): Convergence tolerance.
        max_iter (int): Maximum number of iterations. O(log(n))

    Returns:
        th.Tensor: The resulting diagonal matrix or batch of diagonal matrices of shape [..., n, n].
    """
    diag_mat = th.zeros_like(sym_mat, dtype=sym_mat.dtype)  # Start with D_0 = 0 in double precision

    for i in range(max_iter):
        # Compute the matrix exponential of diag_mat + sym_mat
        exp_mat = sym_expm.apply(diag_mat + sym_mat)

        # Update diag_mat by subtracting the element-wise log of the diagonal elements of exp_mat
        new_diag_mat = diag_mat - th.diag_embed(th.log(th.diagonal(exp_mat, dim1=-2, dim2=-1)))

        # Check for convergence based on the maximum norm difference
        diff_norm = th.norm(new_diag_mat - diag_mat, dim=(-2, -1))
        if diff_norm.max() < atol:
            return new_diag_mat

        # Update diag_mat for the next iteration
        diag_mat = new_diag_mat

    # # If maximum iterations are reached, print the remaining difference norm
    # print(f"Warning: Maximum number of iterations ({max_iter}) reached. "
    #       f"The final norm difference is {diff_norm.item()}. The result may be inaccurate.")
    return diag_mat

class HolDplusFinder:
    """
    Class to find the unique positive diagonal matrix for scaling an SPD matrix
    to have unit row sums, using batch processing and a damped Newton's method.

    Parameters
    ----------
    atol : float, optional
        Convergence tolerance for the Newton method.
    max_iter : int, optional
        Maximum number of iterations for the Newton method.
    damped : bool, optional
        Whether to use a damped version of the Newton method.
    """
    def __init__(self, atol=1e-6, max_iter=100):
        self.atol = atol
        self.max_iter = max_iter

    def __call__(self, hol_mat):
        """
        Finds the unique diagonal matrix of a symmetric matrix for OLM.

        Parameters:
            hol_mat (th.Tensor): Symmetric matrix or batch of symmetric matrices of shape [..., n, n].
            self.atol (float): Convergence tolerance.
            self.max_iter (int): Maximum number of iterations. O(log(n))

        Returns:
            th.Tensor: The resulting diagonal matrix or batch of diagonal matrices of shape [..., n, n].
        """
        diag_mat = th.zeros_like(hol_mat, dtype=hol_mat.dtype)  # Start with D_0 = 0 in double precision

        for i in range(self.max_iter):
            # Compute the matrix exponential of diag_mat + hol_mat
            exp_mat = sym_expm.apply(diag_mat + hol_mat)

            # Update diag_mat by subtracting the element-wise log of the diagonal elements of exp_mat
            new_diag_mat = diag_mat - th.diag_embed(th.log(th.diagonal(exp_mat, dim1=-2, dim2=-1)))

            # Check for convergence based on the maximum norm difference
            diff_norm = th.norm(new_diag_mat - diag_mat, dim=(-2, -1))
            if diff_norm.max() < self.atol:
                return new_diag_mat

            # Update diag_mat for the next iteration
            diag_mat = new_diag_mat

        return diag_mat

class FDplus(Function):
    @staticmethod
    def value(s: Tensor) -> Tensor:
        """Compute elementwise exponential for the eigenvalues."""
        return s.exp()

    @staticmethod
    def derivative(s: Tensor) -> Tensor:
        """Compute the derivative of the elementwise exponential function."""
        return s.exp()

    @staticmethod
    def get_Loewner_matrix(s: Tensor, smod: Tensor, fun_der: Callable[[Tensor], Tensor]) -> Tensor:
        """get the Loewner matrix"""
        # Compute Loewner matrix
        L_den = s[..., None] - s[..., None].transpose(-1, -2)
        is_eq = L_den.abs() < EPS[s.dtype]
        L_den[is_eq] = 1.0
        L_num_ne = smod[..., None] - smod[..., None].transpose(-1, -2)
        L_num_ne[is_eq] = 0

        # Compute derivative for case when eigenvalues are equal
        sder = fun_der(s)

        L_num_eq = 0.5 * (sder[..., None] + sder[..., None].transpose(-1, -2))
        L_num_eq[~is_eq] = 0

        # Compose Loewner matrix
        L = (L_num_ne + L_num_eq) / L_den
        return L

    @staticmethod
    def DK(dX, L, U):
        """Daleckiĭ-Kreĭn formula for symmetric matrix function at Y
            Y = U \Sigma U^\top, eigendecomposition
            L Loewner matrix from Y
        """
        dM = U @ (L * (U.transpose(-1, -2) @ ensure_sym(dX) @ U)) @ U.transpose(-1, -2)
        return dM

    @staticmethod
    def forward(ctx, H: Tensor, HolDplusFinder):
        """
        Forward computation of Y = D^+(H) + H where D^+ is computed using HolDplusFinder.

        Parameters:
        - H: torch.Tensor, the input SPD matrix, of shape [..., n, n].
        - HolDplusFinder: callable, a function to compute D^+ given H.

        Returns:
        - Y: torch.Tensor, the result of D^+ + H, of shape [..., n, n].
        """
        # Compute D^+ and Y
        D_plus_diag = HolDplusFinder(H)
        Y = D_plus_diag + H

        # Save for backward pass
        ctx.save_for_backward(Y)

        return Y

    @staticmethod
    def backward(ctx, dldY: Tensor) -> Tuple[Tensor, None]:
        """
        Backward computation to find gradients with respect to H.

        Parameters:
        - dldY: Gradient of the loss with respect to Y, of shape [..., n, n].

        Returns:
        - dldH: Gradient of the loss with respect to H.
        """
        dldY_sym = ensure_sym(dldY)
        (Y,) = ctx.saved_tensors
        eigenvalues, eigenvectors = th.linalg.eigh(Y)

        # Step 1: Construct the H^0 matrix using the corrected einsum
        smod = FDplus.value(eigenvalues)
        L = FDplus.get_Loewner_matrix(eigenvalues, smod, FDplus.derivative)

        # Compute H^0 with the corrected einsum notation
        H0 = th.einsum('...ij,...ik,...lj,...lk,...jk->...il', eigenvectors, eigenvectors, eigenvectors, eigenvectors,
                       L)

        # Step 2: Compute the intermediate term as a diagonal matrix
        diag_dldY = th.diagonal(dldY_sym, dim1=-2, dim2=-1)  # Extract diagonal elements of dldY
        V_diag = diag_dldY.unsqueeze(-1).expand(*diag_dldY.shape, diag_dldY.size(-1))  # Expand to match shape of dldY

        # Solve for the intermediate term using `th.linalg.solve`
        intermediate_term = th.linalg.solve(H0, V_diag)
        diag_intermediate_term = th.diag_embed(th.diagonal(intermediate_term, dim1=-2, dim2=-1))

        # Step 3: Apply the differential map exp_{*, Y} using the DK formula
        item2 = FDplus.DK(diag_intermediate_term, L, eigenvectors)

        # Ensure zero diagonal as per Hol(n) constraint
        dldH_temp = dldY_sym - item2
        dldH = dldH_temp.tril(-1) + dldH_temp.tril(-1).transpose(-1,-2)

        return dldH, None

def damped_newton_method(fun, x0, fun_jac,damped,max_iter=100,atol=1e-6,verbose=False):
    """
    Find a root of a vector-valued function using the damped Newton's method.

    Parameters
    ----------
    fun : callable
        Vector-valued function.
    x0 : th.Tensor
        Initial guess.
    fun_jac : callable
        Jacobian of `fun`.

    Returns
    -------
    th.Tensor
        Solution tensor containing the root(s).
    """
    xk = x0
    for it in range(max_iter):
        fun_xk = fun(xk)

        residuals = th.norm(fun_xk, dim=-1)
        if verbose:
            print(f"Iteration {it + 1} - Residuals: max={residuals.max().item():.6f}, "
                  f"min={residuals.min().item():.6f}, "
                  f"avg={residuals.mean().item():.6f}")
        # Check convergence
        if residuals.max() <= atol:
            break

        # Compute Newton step
        jacobian = fun_jac(xk)
        y = th.linalg.solve(jacobian, fun_xk)

        # Apply damping if needed
        if damped:
            lambda_xk = th.sqrt((fun_xk * y).sum(dim=-1, keepdim=True))
        else:
            lambda_xk = 0.0

        # Update xk
        xk = xk - y / (1 + lambda_xk)

    return xk

class SPDScalingFinder:
    """
    Class to find the unique positive diagonal matrix for scaling an SPD matrix
    to have unit row sums, using batch processing and a damped Newton's method.

    Parameters
    ----------
    atol : float, optional
        Convergence tolerance for the Newton method.
    max_iter : int, optional
        Maximum number of iterations for the Newton method.
    damped : bool, optional
        Whether to use a damped version of the Newton method.
    """
    def __init__(self, atol=1e-6, max_iter=100, damped=True):
        self.atol = atol
        self.max_iter = max_iter
        self.damped = damped

    def jacobian(self, spd_matrix, diag_vec):
        """Compute the Jacobian of the objective function."""
        return spd_matrix @ diag_vec.unsqueeze(-1) - 1.0 / diag_vec.unsqueeze(-1)

    def hessian(self, spd_matrix, diag_vec):
        """Compute the Hessian of the objective function."""
        diag_inv_squared = th.diag_embed(1.0 / diag_vec ** 2)
        return spd_matrix + diag_inv_squared

    def __call__(self, spd_matrix):
        """
        Finds the unique positive diagonal matrix for scaling an SPD matrix
        to have unit row sums.

        Parameters
        ----------
        spd_matrix : th.Tensor
            Symmetric positive-definite matrix or batch of matrices, shape [..., n, n].

        Returns
        -------
        th.Tensor
            Diagonal scaling matrix, shape [..., n].
        """
        # Initial guess for the scaling vector (diagonal of D), in batch
        x0 = th.ones(spd_matrix.shape[:-1], dtype=spd_matrix.dtype, device=spd_matrix.device)

        # Define the function and Jacobian for batch processing
        func = lambda x: self.jacobian(spd_matrix, x).squeeze(-1)
        jac = lambda x: self.hessian(spd_matrix, x)

        # Use the damped Newton method to find the root in batch mode
        scaling_vector = damped_newton_method(func, x0, jac,damped=self.damped,max_iter=self.max_iter,atol=self.atol)

        return scaling_vector

class FDstar(Function):
    @staticmethod
    def forward(ctx, C, SPDScalingFinder):
        """
        Forward computation of Σ = D*(C) C D*(C) where D* is computed using SPDScalingFinder.

        Parameters:
        - C: torch.Tensor, the input SPD matrix, of shape [..., n, n].
        - SPDScalingFinder: callable, a function to compute D* given C.

        Returns:
        - Sigma: torch.Tensor, the result of D* C D*, of shape [..., n, n].
        """
        # Compute D* and Σ
        D_star_diag = SPDScalingFinder(C)  # Diagonal elements of D*
        D_star = th.diag_embed(D_star_diag)  # Convert to diagonal matrix
        Sigma = D_star @ C @ D_star  # Σ = D* C D*

        # Save variables for backward
        ctx.save_for_backward(C, Sigma, D_star_diag)

        return Sigma

    @staticmethod
    def backward(ctx, dldSigma):
        """
        Backward computation to get the gradient ∂l/∂C, based on Prop 4.5.

        Parameters:
        - dldSigma: torch.Tensor, the gradient of loss with respect to the output Σ, of shape [..., n, n].

        Returns:
        - dldC: torch.Tensor, the gradient of loss with respect to C (∂l/∂C), of shape [..., n, n].

        - Special notes on ensure_sym:
            The final gradient is not symmetric: \frac{\partial l}(\partial C) is not symmetric
            Denoting \frac{\partial l}(\partial C) as A, we have
                Assuming C = (X)_{sym} , then \frac{\partial l}(\partial X) = (A)_sym
                        (i,j) grad: (A_{ij} + A_{ji})/2, for all i, j
                Assuming C = (X)_{tril2cor} = X.tril(-1) + X.tril(-1)^T + I_n from a strictly lower triangular matrix X
                    then \frac{\partial l}(\partial X) = A.tril(-1) + (A^T).tril(-1)
                        (i,j) grad: A_{ij} + A_{ji}, for i != j
            During the network, we implicitly assume C = (X)_{sym}, that's why we have a prior and post ensure_sym.
        """
        # Retrieve saved variables
        C, Sigma, D_star_diag = ctx.saved_tensors
        dldSigma_sym = ensure_sym(dldSigma)

        # Step 2: Compute Δ = D(Σ)^{1/2} (element-wise square root of diagonal of Σ)
        Delta = th.diag_embed(th.sqrt(th.diagonal(Sigma,dim1=-2, dim2=-1)))

        # Step 3: Compute intermediate terms
        I = th.eye(C.size(-1), device=C.device, dtype=C.dtype)
        # Compute V_tilde as the vector of diagonal elements of Σ ∂l/∂Σ + ∂l/∂Σ Σ
        Sigma_prod_dldSigma_sym = Sigma @ dldSigma_sym
        v_tilde = th.diagonal(Sigma_prod_dldSigma_sym + Sigma_prod_dldSigma_sym.transpose(-1,-2), dim1=-2, dim2=-1).unsqueeze(-1)  # Shape [..., n, 1]

        # Efficiently repeat V_tilde across columns to match the shape of dldSigma_sym
        v_tilde_repeated = v_tilde.expand_as(Sigma)

        # Use th.linalg.solve to calculate (I + Σ)^{-1} (V_tilde_repeated)
        Sigma_inv_v_tilde = th.linalg.solve(I + Sigma, v_tilde_repeated)

        # Step 4: Compute gradient ∂l/∂C
        term = dldSigma_sym - ensure_sym(Sigma_inv_v_tilde)
        # term = dldSigma_sym - Sigma_inv_v_tilde
        dldC = Delta @ term @ Delta  # Final gradient as per Proposition 4.5

        # Return the gradient for C and None for non-tensor inputs
        return dldC, None

def is_rone(matrix, atol=1e-6):
    """
    Checks if the input matrix is in Row_1(n), which requires:
      1. The matrix is symmetric.
      2. Each row sums to 1.
    """
    # Check symmetry
    is_symmetric = th.allclose(matrix, matrix.transpose(-1, -2), atol=atol)
    # Sum along the last dimension (columns) to get row sums
    row_sums = matrix.sum(dim=-1)
    # Check if all row sums are close to zero within the specified tolerance
    is_row_sum_zero = th.allclose(row_sums, th.ones_like(row_sums), atol=atol)

    return is_symmetric and is_row_sum_zero