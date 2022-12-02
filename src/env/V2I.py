#!/usr/bin/python
# -*- coding: UTF-8 -*-
import math
import numpy as np

class V2IChannels:

    # Simulator of the V2I channels
    def __init__(self):
        self.hBs = 25
        self.hMs = 1.5
        self.DecorrelationDistance = 50
        self.BSPosition = [0, 0]  # center of the grids
        self.ShadowStd = 8

    def GetPathLoss(self, position):
        distanceX = abs(position[0] - self.BSPosition[0])
        distanceY = abs(position[1] - self.BSPosition[1])
        distance = math.hypot(distanceX, distanceY)
        return 128.1 + 37.6 * np.log10(math.sqrt(distance ** 2 + (self.hBs - self.hMs) ** 2) / 1000) # + self.shadow_std * np.random.normal()

    def GetShadowing(self, deltaDistance, shadowing):
        vehNumber = len(shadowing)
        self.R = np.sqrt(0.5 * np.ones([vehNumber, vehNumber]) + 0.5 * np.identity(vehNumber))
        return np.multiply(np.exp(-1 * (deltaDistance / self.DecorrelationDistance)), shadowing) \
               + np.sqrt(1 - np.exp(-2 * (deltaDistance / self.DecorrelationDistance))) * np.random.normal(0, 8, vehNumber)
