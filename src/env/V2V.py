#!/usr/bin/python
# -*- coding: UTF-8 -*-
import math
import numpy as np

class V2VChannels:
    # Simulator of the V2V Channels

    def __init__(self):
        self.hBs = 1.5
        self.hMs = 1.5
        self.Fc = 2
        self.DecorrelationDistance = 10
        self.ShadowStd = 3

    def GetPathLoss(self, positionA, positionB):
        distanceX = abs(positionA[0] - positionB[0])
        distanceY = abs(positionA[1] - positionB[1])
        distance = math.hypot(distanceX, distanceY) + 0.001
        d_bp = 4 * (self.hBs - 1) * (self.hMs - 1) * self.Fc * (10 ** 9) / (3 * 10 ** 8)

        def LosPathLoss(distance):
            if distance <= 3:
                return 22.7 * np.log10(3) + 41 + 20 * np.log10(self.Fc / 5)
            else:
                if distance < d_bp:
                    return 22.7 * np.log10(distance) + 41 + 20 * np.log10(self.Fc / 5)
                else:
                    return 40.0 * np.log10(distance) + 9.45 - 17.3 * np.log10(self.hBs) - 17.3 * np.log10(self.hMs) + 2.7 * np.log10(self.Fc / 5)

        def NLosPathLoss(distanceX, distanceY):
            n_j = max(2.8 - 0.0024 * distanceY, 1.84)
            return LosPathLoss(distanceX) + 20 - 12.5 * n_j + 10 * n_j * np.log10(distanceY) + 3 * np.log10(self.Fc / 5)

        if min(distanceX, distanceY) < 7:
            PL = LosPathLoss(distance)
        else:
            PL = min(NLosPathLoss(distanceX, distanceY), NLosPathLoss(distanceY, distanceX))
        return PL  # + self.shadow_std * np.random.normal()

    def GetShadowing(self, deltaDistance, shadowing):
        return np.exp(-1 * (deltaDistance / self.DecorrelationDistance)) * shadowing \
               + math.sqrt(1 - np.exp(-2 * (deltaDistance / self.DecorrelationDistance))) * np.random.normal(0, 3)  # standard dev is 3 db
