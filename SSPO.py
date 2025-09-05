import torch
import torch.nn.functional as F


def SSPO_fun(b_t_hat):
    """
    Args:
        b_t_hat (torch.Tensor): Initial weight vector of shape (num_assets,).

    Returns:
        torch.Tensor: Optimized sparse weight vector of shape (num_assets,).

    Main Parameters (inside function):
        max_iter (int): Maximum number of iterations for the optimization loop.
        zeta (float): Scaling factor for the final weight vector.
        lambda_ (float): Regularization parameter.
        gamma (float): Threshold for sparsity.
        eta (float): Step size for the dual update.
    """
    max_iter = 1e3
    zeta = 500
    lambda_ = 0.5
    gamma = 0.01
    eta = 0.005
    x = -b_t_hat
    tao = lambda_ / gamma
    stock_num = b_t_hat.shape[0]

    g = b_t_hat.clone()
    b = b_t_hat.clone()
    rho = 0
    I = torch.eye(stock_num)
    YI = torch.ones((stock_num, stock_num))
    yi = torch.ones_like(b_t_hat)

    for i in range(int(max_iter)):
        b = torch.linalg.solve(tao * I + eta * YI, tao * g + (eta - rho) * yi - x)
        g = F.threshold(b, gamma, 0)
        prim_res_tmp = yi * b - 1
        rho = rho + eta * prim_res_tmp

    b_tplus1_hat = zeta * b
    w_opt = simplex_projection(b_tplus1_hat)
    return w_opt


def simplex_projection(v):
    v = v.clone()
    v = torch.maximum(torch.tensor(0.0), v)
    w = v / torch.sum(v)
    return w

