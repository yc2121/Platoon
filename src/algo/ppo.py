#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from common.constant import *
from torch.autograd import Variable
from common.utils import *
from common.hparams import HyperParams as hp

def GetGAE(rewards, masks, values):
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)

    runningReturns = 0
    previousValue = 0
    runningAdvantanges = 0

    for t in reversed(range(0, len(rewards))):
        runningReturns = rewards[t] + hp.Gamma * runningReturns * masks[t]
        runningTDError = rewards[t] + hp.Gamma * previousValue * masks[t] - \
                    values.data[t]
        runningAdvantanges = runningTDError + hp.Gamma * hp.Lamda * \
                          runningAdvantanges * masks[t]

        returns[t] = runningReturns
        previousValue = values.data[t]
        advantages[t] = runningAdvantanges

    advantages = (advantages - advantages.mean()) / advantages.std()
    return returns, advantages


def SurrogateLoss(actor, advants, states, oldPolicy, actions, index):
    mu, std, logstd = actor(torch.Tensor(states))
    newPolicy = LogDensity(actions, mu, std, logstd)
    oldPolicy = oldPolicy[index]

    ratio = torch.exp(newPolicy - oldPolicy)
    surrogate = ratio * advants
    return surrogate, ratio


def TrainModel(actor, critic, memory, actorOptim, criticOptim):
    memory = np.array(memory)
    states = np.vstack(memory[:, 0])
    actions = list(memory[:, 1])
    rewards = list(memory[:, 2])
    masks = list(memory[:, 3])
    values = critic(torch.Tensor(states))

    # ----------------------------
    # step 1: get returns and GAEs and log probability of old policy
    returns, advants = GetGAE(rewards, masks, values)
    mu, std, logstd = actor(torch.Tensor(states))
    oldPolicy = LogDensity(torch.Tensor(actions), mu, std, logstd)

    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)

    # ----------------------------
    # step 2: get value loss and actor loss and update actor & critic
    actorLossRecord = []
    criticLossRecord = []
    clippedRatioRecord = []
    for epoch in range(TrainEpochNum):
        np.random.shuffle(arr)

        for i in range(n // hp.BatchSize):
            batchIndex = arr[hp.BatchSize * i: hp.BatchSize * (i + 1)]
            batchIndex = torch.LongTensor(batchIndex)
            inputs = torch.Tensor(states)[batchIndex]
            returnsSamples = returns.unsqueeze(1)[batchIndex]
            advantsSamples = advants.unsqueeze(1)[batchIndex]
            actionsSamples = torch.Tensor(actions)[batchIndex]

            loss, ratio = SurrogateLoss(actor, advantsSamples, inputs,
                                         oldPolicy.detach(), actionsSamples,
                                         batchIndex)

            values = critic(inputs)
            criticLoss = criterion(values, returnsSamples)
            criticOptim.zero_grad()
            criticLoss.backward()
            criticOptim.step()

            clippedRatio = torch.clamp(ratio,
                                        1.0 - hp.ClipRatio,
                                        1.0 + hp.ClipRatio)
            clippedLoss = clippedRatio * advantsSamples
            actorLoss = -torch.min(loss, clippedLoss).mean()

            actorOptim.zero_grad()
            actorLoss.backward()
            actorOptim.step()
            actorLossRecord.append(Variable(actorLoss))
            criticLossRecord.append(Variable(criticLoss))
            clippedRatioRecord.append(Variable(clippedRatio.mean()))
    return np.mean(actorLossRecord), np.mean(criticLossRecord), np.mean(clippedRatioRecord)







