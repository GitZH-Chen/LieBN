import torch as th

@th.jit.script
def hemisphere_to_poincare(x):
    """
    Transform points from the open hemisphere HS^n to the Poincaré ball P^n.

    Parameters:
    - x: torch.Tensor of shape [..., n+1], where the last dimension
         represents the coordinates (x^T, x_{n+1}).

    Returns:
    - y: torch.Tensor of shape [..., n], transformed coordinates in the Poincaré ball.
    """
    x_T, x_n1 = x[..., :-1], x[..., -1]  # Split x into (x_T, x_n+1)
    y = x_T / (1 + x_n1.unsqueeze(-1))  # Apply the transformation: y = x_T / (1 + x_n1)
    return y

@th.jit.script
def poincare_to_hemisphere(y):
    """Map from Poincaré ball to open hemisphere."""
    norm_y = y.norm(p=2,dim=-1, keepdim=True) ** 2
    factor = 1 / (1 + norm_y)
    mapped = th.cat((2 * y * factor, (1 - norm_y) * factor), dim=-1)
    return mapped


