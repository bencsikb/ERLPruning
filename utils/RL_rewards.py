import torch
import numpy as np
import math

def reward_function_proposed(dmap, Tdmap, spars, Tspars, dmap_coeff, spars_coeff, device, beta=5):
    """
    :param dmap: mAP deterioration (error)
    :param Tdmap: desired mAP deterioration (maximal)
    :param spars: sparsity (percent of pruned parameters
    :param Tspars: desired sparsity
    :return: reward
    """

    print(Tdmap)

    zerotens = torch.zeros(dmap.shape).to(device)
    reward = - beta* (dmap_coeff*torch.max( (dmap-Tdmap)/(1-Tdmap), zerotens) + spars_coeff*torch.max( 1 - spars/Tspars, zerotens))

    return reward


def reward_function_purl(dmap, Tmap, spars, Tspars, map_before, device, beta=5):
    """
    :param dmap: mAP deterioration (error)
    :param Tmap: desired final mAP (after pruning)
    :param spars: sparsity (percent of pruned parameters
    :param Tspars: desired sparsity
    :return: reward
    """

    # Compute the mAP value after pruning from dmap 
    onestens = torch.ones(dmap.shape).to(device)
    map_after = (1 - dmap) * map_before

    zerotens = torch.zeros(dmap.shape).to(device)
    reward = - beta* (torch.max( 1-map_after/Tmap, zerotens) + torch.max( 1 - spars/Tspars, zerotens))

    return reward


def reward_function_amc(dmap, spars, device, init_params=63980766):
    """
    :return: reward
    """

    reward = - dmap * torch.log((spars + 5e-10) * init_params)

    return reward


def reward_function2(params, error, sparsity, baseline):


    params_tens = torch.full(error.shape, params, dtype=torch.float32)
    baseline_tens = torch.full(error.shape, baseline, dtype=torch.float32)

    reward = baseline_tens -  error * torch.log(params_tens - sparsity * params_tens)

    return reward

def reward_function3(E, S, Te, Ts, err_coef, spars_coef, device):

    zerotens = torch.zeros(E.shape).to(device)


    reward = spars_coef*S/Ts - err_coef*torch.max((E-Te)/(1-Te),zerotens)

    return reward

def reward_function4(E, S,  err_coef, spars_coef, device):

    zerotens = torch.zeros(E.shape).to(device)
    onetens = torch.ones(E.shape).to(device)
    minusonetens = torch.full(E.shape, -1.0, dtype=torch.float32).to(device)
    E_sat = torch.min(torch.max(E, minusonetens), onetens)
    S_sat = torch.min(torch.max(S, minusonetens), onetens)


    reward = -err_coef*torch.abs(torch.sin(E_sat)) - spars_coef*torch.abs(torch.sin(S_sat))

    return reward

def reward_function5(E, S,  Ts, beta, device):

    zerotens = torch.zeros(E.shape).to(device)
    onetens = torch.ones(E.shape).to(device)

    reward =  beta * (onetens - torch.max(E, zerotens)) * (S/Ts)

    return reward

def reward_function6(E, S, a, b, n, device):

    atens = torch.full(E.shape, a, dtype=torch.float32).to(device)
    btens = torch.full(E.shape, b, dtype=torch.float32).to(device)
    onetens = torch.ones(E.shape).to(device)
    minusonetens = torch.full(E.shape, -1.0, dtype=torch.float32).to(device)
    E_sat = torch.min(torch.max(E, minusonetens), onetens)
    S_sat = torch.min(torch.max(S, minusonetens), onetens)

    reward =  - atens * torch.pow(E_sat, 2*n) + btens + (- atens * torch.pow(S_sat, 2*n) + btens )

    return reward