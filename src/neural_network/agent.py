#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
import torch.optim as optim
from collections import deque
from neural_network.model import Actor, Critic
from common.hparams import HyperParams as hp
from common.config import *

class Agent:
    # Agent: include all the information for a agent

    def __init__(self, platoonId, inputDim, outputDim):
        self.PlatoonId = platoonId
        self.Actor = Actor(inputDim, outputDim)
        self.Critic = Critic(inputDim)
        self.ActorOptim = optim.Adam(self.Actor.parameters(), lr=hp.ActorLR)
        self.CriticOptim = optim.Adam(self.Critic.parameters(), lr=hp.CriticLR, weight_decay=hp.L2Rate)
        self.Memory = deque()
    
    def saveModel(self):
        if not os.path.exists(SaveModelsPath):
            os.makedirs(SaveModelsPath)
        actorModelPath = SaveModelsPath + f"/actor{self.PlatoonId}.pt"
        criticModelPath = SaveModelsPath + f"/critic{self.PlatoonId}.pt"
        torch.save(self.Actor.state_dict(), actorModelPath)
        torch.save(self.Critic.state_dict(), criticModelPath)