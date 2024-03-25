import torch
import torch.nn as nn
import math
import random
import numpy as np
import math

from utils.LR_utils import list2FloatTensor

def get_discounted_reward(rewards, values, gamma):

    disc_rewards = []
    #val = values[-1]
    val = 0

    for i in reversed(range(len(rewards))):
        val = rewards[i] + gamma*val
        disc_rewards.insert(0, val)


    disc_rewards =list2FloatTensor(disc_rewards)
    out = disc_rewards - disc_rewards.mean()
    out /= disc_rewards.std()

    return out


def get_advantage(rewards, values, gamma=0.99):

    disc_rewards = get_discounted_reward(rewards, values, gamma)
    advantage = disc_rewards - values

    return advantage


class CriticLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, rewards, values, gamma=0.99):

        advantage = get_advantage(rewards, values, gamma)
        loss = advantage.pow(2).mean()
        return loss

class ActorLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, rewards, values, policies, log_probs, entropies=[], ent_coef=0.5, gamma=0.99):

        advantage = get_advantage(rewards, values, gamma)
        #loss = torch.mean(- advantage * torch.log(policies))
        if len(entropies):
            loss2  = - (log_probs * advantage + ent_coef * entropies).mean()
        else:
            loss2  = - (log_probs * advantage).mean()


        return loss2


class ActorPPOLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, rewards, values, policies, log_probs, log_probs_prev, entropies=[], ent_coef=0.5, gamma=0.99, eps=0.2):

        onetens = torch.ones(log_probs.shape).to("cuda")
        epstens = torch.full(log_probs.shape, eps, dtype=torch.float32).to("cuda")

        advantage = get_advantage(rewards, values, gamma)
        ratio = torch.exp(log_probs - log_probs_prev)
        p1 = ratio * advantage.detach()
        p2 = torch.clamp(ratio, 1-eps, 1+eps) * advantage.detach()
        loss = - (torch.min(p1, p2) +  ent_coef * entropies).mean()

        return loss

























