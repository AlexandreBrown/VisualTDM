import torch


def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=5, ratio=0.5):
    """
    Source: https://arxiv.org/abs/1903.10145
    """
    L = torch.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio)

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L
