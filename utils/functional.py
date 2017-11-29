def fold(f, l, a):
    return a if (len(l) == 0) else fold(f, l[1:], f(a, l[0]))


def f_and(x, y):
    return x and y


def f_or(x, y):
    return x or y


def parameters_allocation_check(module):
    parameters = list(module.parameters())
    return fold(f_and, parameters, True) or not fold(f_or, parameters, False)


def handle_inputs(inputs, use_cuda):
    import torch as t
    from torch.autograd import Variable

    result = [Variable(t.from_numpy(var)) for var in inputs]
    result = [var.cuda() if use_cuda else var for var in result]

    return result


def kld_coef(i, extended=False):
    import math
    if (not extended):
        return (math.tanh((i - 3500)/1000) + 1)/2
    else:
        return (math.tanh((i - 50000)/15000) +1)/2

def temp_coef(i):
    import numpy as np

    initial_temp = 1.0
    min_temp = 0.1
    ANNEAL_RATE = 0.00002
    
    return np.maximum(initial_temp * np.exp(-ANNEAL_RATE*i), min_temp)
