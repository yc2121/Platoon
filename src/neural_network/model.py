#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from common.hparams import HyperParams as hp


class Actor(nn.Module):
    def __init__(self, inputDim, outputDim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(inputDim, hp.HiddenDim)
        self.fc2 = nn.Linear(hp.HiddenDim, hp.HiddenDim)
        self.fc3 = nn.Linear(hp.HiddenDim, outputDim)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc3(x)
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)
        return mu, std, logstd


class Critic(nn.Module):
    def __init__(self, inputDim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(inputDim, hp.HiddenDim)
        self.fc2 = nn.Linear(hp.HiddenDim, hp.HiddenDim)
        self.fc3 = nn.Linear(hp.HiddenDim, 1)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc3(x)
        return v