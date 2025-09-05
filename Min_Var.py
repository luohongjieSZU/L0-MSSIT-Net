import numpy as np
import cvxopt as opt
from cvxopt import solvers
import torch


def min_var(returns):
    solvers.options['show_progress'] = False
    returns = (returns - returns.mean(dim=0)) / returns.std(dim=0)
    var = returns.var(dim=0)
    non_zero_variance_columns = var > 1e-8
    returns = returns[:, non_zero_variance_columns]
    cov_matrix = np.cov(returns, rowvar=False)
    n = cov_matrix.shape[1]
    P = opt.matrix(cov_matrix)
    q = opt.matrix(np.zeros(n))

    G = opt.matrix(-np.eye(n))
    h = opt.matrix(np.zeros(n))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    sol = solvers.qp(P, q, G, h, A, b)
    w = np.array(sol['x'])
    w = torch.from_numpy(w).type(torch.Tensor)
    w = w.reshape(-1)
    return w

