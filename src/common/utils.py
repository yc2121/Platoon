#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
import math
import numpy as np

def DiscreteAction(action):
    action1 = np.zeros(2)
    action1[0] = ActionValueSubchannelPlatoon25(action[0])
    action1[1] = ActionValuePower(action[1])

    if action1[1] == 0:
        action1[1] = -100
    return action1

def ActionValueSubchannel(a):
    if a > -0.5 and a <= 0.5:
        return 2
    elif a > 0.5 and a <= 1.5:
        return 3
    elif a > -1.5 and a <= -0.5:
        return 1
    elif a >= 1.5:
        return 4
    elif a <= -1.5:
        return 0

def ActionValueSubchannelPlatoon10(a):
    if a > 0 and a <= 0.3:
        return 5
    elif a > 0.3 and a <= 0.6:
        return 6
    elif a > 0.6 and a <= 0.9:
        return 7
    elif a > 0.9 and a <= 1.2:
        return 8
    elif a > 1.2:
        return 9
    elif a > -0.3 and a <= 0:
        return 4
    elif a > -0.6 and a <= -0.3:
        return 3
    elif a > -0.9 and a <= -0.6:
        return 2
    elif a > -1.2 and a <= -0.9:
        return 1
    elif a <= -1.2:
        return 0

def ActionValuePower(a):
    if a< 1.15 and a>-1.15:
        b=(a+1.15)*10
    elif a < -1.15:
        b= 0
    elif a > 1.15:
        b=23
    return b

def ActionValueSubchannelPlatoon25(a):
    if a > -0.05 and a <= 0.05:
        return 12
    elif a > 0.05 and a <= 0.15:
        return 13
    elif a > 0.15 and a <= 0.25:
        return 14
    elif a > 0.25 and a <= 0.35:
        return 15
    elif a > 0.35 and a <= 0.45:
        return 16
    elif a > 0.45 and a <= 0.55:
        return 17
    elif a > 0.55 and a <= 0.65:
        return 18
    elif a > 0.65 and a <= 0.75:
        return 19
    elif a > 0.75 and a <= 0.85:
        return 20
    elif a > 0.85 and a <= 0.95:
        return 21
    elif a > 0.95 and a <= 1.05:
        return 22
    elif a > 1.05 and a <= 1.15:
        return 23
    elif a > 1.15:
        return 24
    elif a > -0.15 and a <= -0.05:
        return 11
    elif a > -0.25 and a <= -0.15:
        return 10
    elif a > -0.35 and a <= -0.25:
        return 9
    elif a > -0.45 and a <= -0.35:
        return 8
    elif a > -0.55 and a <= -0.45:
        return 7
    elif a > -0.65 and a <= -0.55:
        return 6
    elif a > -0.75 and a <= -0.65:
        return 5
    elif a > -0.85 and a <= -0.75:
        return 4
    elif a > -0.95 and a <= -0.85:
        return 3
    elif a > -1.05 and a <= -0.95:
        return 2
    elif a > -1.15 and a <= -1.05:
        return 1
    elif a <= -1.15:
        return 0

def DiscreteActionPlatoon25(action, num):
    action1 = np.zeros(num)
    for i in range(num):
        if i % 2 == 0:
            action1[i] = ActionValueSubchannelPlatoon25(action[i])
        else:
            action1[i] = ActionValuePower(action[i])
    for i in range(num):
        if i % 2 == 1 and action1[i] == 0:
            action1[i] = -100
    return action1

def DiscreteActionPlatoon10(action, num):
    action1 = np.zeros(num)
    for i in range(num):
        if i % 2 == 0:
            action1[i] = ActionValueSubchannelPlatoon10(action[i])
        else:
            action1[i] = ActionValuePower(action[i])
    for i in range(num):
        if i % 2 == 1 and action1[i] == 0:
            action1[i] = -100
    return action1

def DiscreteActionPlatoon(action, num):
    action1 = np.zeros(num)
    for i in range(num):
        if i % 2 == 0:
            action1[i] = ActionValueSubchannel(action[i])
        else:
            action1[i] = ActionValuePower(action[i])
    for i in range(num):
        if i % 2 == 1 and action1[i] == 0:
            action1[i] = -100
    return action1

def GetAction(mu, std):
    action = torch.normal(mu, std)
    action = action.data.numpy()
    return action

def LogDensity(x, mu, std, logstd):
    var = std.pow(2)
    logDensity = -(x - mu).pow(2) / (2 * var) \
                  - 0.5 * math.log(2 * math.pi) - logstd
    return logDensity.sum(1, keepdim=True)

def FlatGrad(grads):
    gradFlatten = []
    for grad in grads:
        gradFlatten.append(grad.view(-1))
    gradFlatten = torch.cat(gradFlatten)
    return gradFlatten


def FlatHessian(hessians):
    hessiansFlatten = []
    for hessian in hessians:
        hessiansFlatten.append(hessian.contiguous().view(-1))
    hessiansFlatten = torch.cat(hessiansFlatten).data
    return hessiansFlatten


def FlatParams(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    paramsFlatten = torch.cat(params)
    return paramsFlatten


def UpdateModel(model, newParams):
    index = 0
    for params in model.parameters():
        paramsLength = len(params.view(-1))
        newParam = newParams[index: index + paramsLength]
        newParam = newParam.view(params.size())
        params.data.copy_(newParam)
        index += paramsLength


def KLDivergence(newActor, oldActor, states):
    mu, std, logstd = newActor(torch.Tensor(states))
    muOld, stdOld, logstdOld = oldActor(torch.Tensor(states))
    muOld = muOld.detach()
    stdOld = stdOld.detach()
    logstdOld = logstdOld.detach()

    # kl divergence between old policy and new policy : D( pi_old || pi_new )
    # pi_old -> mu0, logstd0, std0 / pi_new -> mu, logstd, std
    # be careful of calculating KL-divergence. It is not symmetric metric
    kl = logstdOld - logstd + (stdOld.pow(2) + (muOld - mu).pow(2)) / \
         (2.0 * std.pow(2)) - 0.5
    return kl.sum(1, keepdim=True)

def smooth(arr,r):
    smoothing_rate = r
    temp_arr = np.zeros(len(arr))
    temp_arr[0] = arr[0]
    for i in range(1, len(arr)):
        temp_arr[i] = smoothing_rate * temp_arr[i-1] + (1-smoothing_rate) * arr[i]
    return temp_arr


