import math
import torch
import torch.nn.functional as F

# payoff_dict

BOC_dict = {
    'pay_func': lambda X: F.relu(torch.max(X, dim=1, keepdims=True)[0] - 1.),
    'lb2': -2.,
    'ub2': 2.,
    'lb3': -2.,
    'ub3': 2.,
    'lb4': -2.,
    'ub4': 2.,
    'lb5': -2.,
    'ub5': 2.,
    'lb20': 0.5,
    'ub20': 0.7,
    'lb50': 0.3,
    'ub50': 0.6
}

BOBC_dict = {
    'pay_func': lambda X: 1. * (torch.max(X, dim=1, keepdims=True)[0] > 1.),
    'lb2': -2.,
    'ub2': 2.,
    'lb3': -2.,
    'ub3': 2.,
    'lb4': -2.,
    'ub4': 2.,
    'lb5': -2.,
    'ub5': 2.,
    'lb20': 0.5,
    'ub20': 0.7,
    'lb50': 0.3,
    'ub50': 0.6
}

BOP_dict = {
    'pay_func': lambda X: F.relu(1. - torch.max(X, dim=1, keepdims=True)[0]),
    'lb2': -2.,
    'ub2': 2.,
    'lb3': -2.,
    'ub3': 2.,
    'lb4': -2.,
    'ub4': 2.,
    'lb5': -2.,
    'ub5': 2.,
    'lb20': 0.2,
    'ub20': 0.5,
    'lb50': 0.,
    'ub50': 0.5
}

BOBP_dict = {
    'pay_func': lambda X: 1.*(1. > torch.max(X, dim=1, keepdims=True)[0]),
    'lb2': -2.,
    'ub2': 2.,
    'lb3': -2.,
    'ub3': 2.,
    'lb4': -2.,
    'ub4': 2.,
    'lb5': -2.,
    'ub5': 2.,
    'lb20': 0.2,
    'ub20': 0.5,
    'lb50': 0.,
    'ub50': 0.5
}

WOC_dict = {
    'pay_func': lambda X: F.relu(torch.min(X, dim=1, keepdims=True)[0] - 1.),
    'lb2': -2.,
    'ub2': 2.,
    'lb3': -2.,
    'ub3': 2.,
    'lb4': -2.,
    'ub4': 2.,
    'lb5': 0.,
    'ub5': 2.,
    'lb20': 1.4,
    'ub20': 1.8,
    'lb50': 3.,
    'ub50': 3.5
}

WOBC_dict = {
    'pay_func': lambda X: 1.*(torch.min(X, dim = 1, keepdims = True)[0] > 1.),
    'lb2': -2.,
    'ub2': 2.,
    'lb3': -2.,
    'ub3': 2.,
    'lb4': -2.,
    'ub4': 2.,
    'lb5': 0.,
    'ub5': 2.,
    'lb20': 1.4,
    'ub20': 1.8,
    'lb50': 3.,
    'ub50': 3.5
}

WOP_dict = {
    'pay_func': lambda X: F.relu(1. - torch.min(X, dim=1, keepdims=True)[0]),
    'lb2': -2.,
    'ub2': 2.,
    'lb3': -2.,
    'ub3': 2.,
    'lb4': -2.,
    'ub4': 2.,
    'lb5': 0.,
    'ub5': 2.,
    'lb20': 2.2,
    'ub20': 2.5,
    'lb50': 2.,
    'ub50': 2.5
}

WOBP_dict = {
    'pay_func': lambda X: 1.*(1.> torch.min(X, dim = 1, keepdims = True)[0]),
    'lb2': -2.,
    'ub2': 2.,
    'lb3': -2.,
    'ub3': 2.,
    'lb4': -2.,
    'ub4': 2.,
    'lb5': 0.,
    'ub5': 2.,
    'lb20': 2.2,
    'ub20': 2.5,
    'lb50': 2.,
    'ub50': 2.5
}

Mex_dict = {
    'pay_func': lambda X: 1 - torch.exp( - torch.sum(X**2., dim = 1, keepdims = True) ),
    'lb2': -2.,
    'ub2': 2.,
    'lb3': -2.,
    'ub3': 2.,
    'lb4': -2.,
    'ub4': 2.,
    'lb5': -math.sqrt(3),
    'ub5': math.sqrt(3),
    'lb20': 0.,
    'ub20': 0.2,
    'lb50': 0.,
    'ub50': 0.2
}

DC_dict = {
    'pay_func': lambda X: F.relu(torch.sum(torch.abs(X), dim = 1, keepdims = True) -1),
    'lb2': -2.,
    'ub2': 2.,
    'lb3': -2.,
    'ub3': 2.,
    'lb4': -1.,
    'ub4': 1.,
    'lb5': -1.,
    'ub5': 1.,
    'lb20': -0.05,
    'ub20': 0.11,
    'lb50': -0.04,
    'ub50': 0.05
}

DP_dict = {
    'pay_func': lambda X: F.relu(1 - torch.sum(torch.abs(X), dim = 1, keepdims = True)),
    'lb2': -2.,
    'ub2': 2.,
    'lb3': -0.5,
    'ub3': 0.5,
    'lb4': -.4,
    'ub4': .4,
    'lb5': -.3,
    'ub5': .3,
    'lb20': -0.1,
    'ub20': 0.1,
    'lb50': -0.05,
    'ub50': 0.04
}

payoff_dict = {'BOC': BOC_dict, 'BOP': BOP_dict, 
               'BOBC': BOBC_dict, 'BOBP': BOBP_dict, 
               'WOC': WOC_dict, 'WOP': WOP_dict, 
               'WOBC': WOBC_dict, 'WOBP': WOBP_dict, 
               'DC': DC_dict, 'DP': DP_dict,
               'Mex': Mex_dict}