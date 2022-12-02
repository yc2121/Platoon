#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import division
import numpy as np
import math
from env.vehicle import Vehicle
from env.V2I import V2IChannels
from env.V2V import V2VChannels

# np.set_printoptions(suppress=True)
np.random.seed(1234)

class Environment:
    def __init__(self):
        self.UpDownLines = [-3.5-3.5/2, -3.5/2, 3.5/2, 3.5/2+3.5]  # line_rnd: 0,1,2,3
        self.LeftRightLines = [-3.5 - 3.5/2, -3.5 / 2, 3.5 / 2, 3.5 / 2 + 3.5]  # line_rnd: 4,5,6,7
        self.RoadHeight = 100
        self.RoadWidth = 100

        self.Latency = 10  #  时延

        self.V2VChannels = V2VChannels()
        self.V2IChannels = V2IChannels()
        self.Vehicles = []

        self.Demand = []
        self.V2VShadowing = []
        self.V2IShadowing = []
        self.DeltaDistance = []
        self.V2VChannelsAbs = []
        self.V2IChannelsAbs = []

        self.V2IPowerdBm = 23  # dBm
        self.V2VPowerdBmList = [23, 15, 5, -100]  # the power levels
        self.V2IPower = 0.2
        self.CirclePowerdBm = 16  # dBm Circuit Power
        self.CirclePower = 0.04  # W
        self.Sig2dB = -114
        self.BsAntGain = 8
        self.BsNoiseFigure = 5
        self.VehAntGain = 3
        self.VehNoiseFigure = 9
        self.Sig2 = 10 ** (self.Sig2dB / 10)

        self.PlatoonNumber = 5
        self.PlatoonSize = 5
        self.RBNumber = self.PlatoonNumber * self.PlatoonSize
        self.VehNumber = self.RBNumber

        self.TimeFast = 0.001
        self.TimeSlow = 0.1  # update slow fading/vehicle Position every 100 ms
        self.TimeLatency = 0.01
        self.BandWidth = int(1e5)  # BandWidth per RB, 1 MHz
        self.DemandSize = 300  # V2V payload: 300 Bytes every 100 ms

        self.L2MInterferenceAll = np.zeros((self.PlatoonNumber, (self.PlatoonSize - 1), self.RBNumber)) + self.Sig2
        self.M2MInterferenceAll = np.zeros((self.PlatoonNumber * (self.PlatoonSize - 2), self.RBNumber)) + self.Sig2

    def AddNewVehicle(self, id, startPosition, startDirection, Velocity, platoonId, type):
        self.Vehicles.append(Vehicle(id, startPosition, startDirection, Velocity, platoonId, type))

    def AddNewPlatoonVehicles(self, n, size):
        for i in range(n):
            if i == 0:
                for j in range(size):
                    startPosition = [-9.0 - 10 * j, self.LeftRightLines[0]]
                    startDirection = '0'
                    if j == 0:
                        self.AddNewVehicle(i * size + j, startPosition, startDirection, 15, i, "Leader")
                    else:
                        self.AddNewVehicle(i * size + j, startPosition, startDirection, 15, i, "Member")
            elif i == 1:
                for j in range(size):
                    startPosition = [9.0 + 10 * j, self.LeftRightLines[2]]
                    startDirection = '10'
                    if j == 0:
                        self.AddNewVehicle(i * size + j, startPosition, startDirection, 15, i, "Leader")
                    else:
                        self.AddNewVehicle(i * size + j, startPosition, startDirection, 15, i, "Member")
            elif i == 2:
                for j in range(size):
                    startPosition = [self.UpDownLines[1], 9.0 + 10 * j]
                    startDirection = '14'
                    if j == 0:
                        self.AddNewVehicle(i * size + j, startPosition, startDirection, 0, i, "Leader")
                    else:
                        self.AddNewVehicle(i * size + j, startPosition, startDirection, 0, i, "Member")
            elif i == 3:
                for j in range(size):
                    startPosition = [self.UpDownLines[2], -9.0 - 10 * j]
                    startDirection = '6'
                    if j == 0:
                        self.AddNewVehicle(i * size + j, startPosition, startDirection, 0, i, "Leader")
                    else:
                        self.AddNewVehicle(i * size + j, startPosition, startDirection, 0, i, "Member")
            elif i == 4:
                for j in range(size):
                    startPosition = [self.UpDownLines[3], -9.0 - 10 * j]
                    startDirection = '7'
                    if j == 0:
                        self.AddNewVehicle(i * size + j, startPosition, startDirection, 15, i, "Leader")
                    else:
                        self.AddNewVehicle(i * size + j, startPosition, startDirection, 15, i, "Member")

        self.V2VShadowing = np.random.normal(0, 3, [self.VehNumber, self.VehNumber])
        self.V2IShadowing = np.random.normal(0, 8, self.VehNumber)
        self.DeltaDistance = np.asarray([c.Velocity * self.TimeSlow for c in self.Vehicles])

    def UpdateVehPositions(self, t):
        # ===============
        # This function updates the Position of each vehicle
        # ===============
        for v in self.Vehicles:
            if v.Direction == '0':
                v.Position[0] = v.Position[0] + v.Velocity / 3.6 * t
                if v.Position[0] > -7:
                    v.Position[0] = self.UpDownLines[0]
                    v.Direction = '4'
            elif v.Direction == '1':
                v.Position[0] = v.Position[0] + v.Velocity / 3.6 * t
                if v.Position[0] > 0:
                    v.Direction = '9'
            elif v.Direction == '2':
                v.Position[0] = v.Position[0] - v.Velocity / 3.6 * t
            elif v.Direction == '3':
                v.Position[0] = v.Position[0] - v.Velocity / 3.6 * t
            elif v.Direction == '4':
                v.Position[1] = v.Position[1] - v.Velocity / 3.6 * t
            elif v.Direction == '5':
                v.Position[1] = v.Position[1] - v.Velocity / 3.6 * t
            elif v.Direction == '6':
                v.Position[1] = v.Position[1] + v.Velocity / 3.6 * t
                if v.Position[1] > 0:
                    v.Direction = '13'
            elif v.Direction == '7':
                v.Position[1] = v.Position[1] + v.Velocity / 3.6 * t
                if v.Position[1] > -7:
                    v.Position[1] = self.LeftRightLines[0]
                    v.Direction = '8'
            elif v.Direction == '8':
                v.Position[0] = v.Position[0] + v.Velocity / 3.6 * t
            elif v.Direction == '9':
                v.Position[0] = v.Position[0] + v.Velocity / 3.6 * t
            elif v.Direction == '10':
                v.Position[0] = v.Position[0] - v.Velocity / 3.6 * t
                if v.Position[0] > 0:
                    v.Direction = '2'
            elif v.Direction == '11':
                    v.Position[0] = v.Position[0] - v.Velocity / 3.6 * t
                    if v.Position[0] < 7:
                        v.Position[0] = self.UpDownLines[3]
                        v.Direction = '12'
            elif v.Direction == '12':
                v.Position[1] = v.Position[1] + v.Velocity / 3.6 * t
            elif v.Direction == '13':
                v.Position[1] = v.Position[1] + v.Velocity / 3.6 * t
            elif v.Direction == '14':
                v.Position[1] = v.Position[1] - v.Velocity / 3.6 * t
                if v.Position[1] < 0:
                    v.Direction = '5'
            elif v.Direction == '15':
                v.Position[1] = v.Position[1] - v.Velocity / 3.6 * t
                if v.Position[1] < 7:
                    v.Position[1] = self.LeftRightLines[3]
                    v.Direction = '3'

    def UpdateChannelsSlowFading(self):
        """ Renew slow fading channel """

        self.V2VPathLoss = np.zeros((self.VehNumber, self.VehNumber)) + 50 * np.identity(self.VehNumber)
        self.V2IPathLoss = np.zeros((self.VehNumber))

        self.V2VChannelsAbs = np.zeros((self.VehNumber, self.VehNumber))
        self.V2IChannelsAbs = np.zeros((self.VehNumber))

        for i in range(self.VehNumber):
            for j in range(i + 1, self.VehNumber):
                self.V2VShadowing[j][i] = self.V2VShadowing[i][j] = self.V2VChannels.GetShadowing(self.DeltaDistance[i] + self.DeltaDistance[j], self.V2VShadowing[i][j])
                self.V2VPathLoss[j,i] = self.V2VPathLoss[i][j] = self.V2VChannels.GetPathLoss(self.Vehicles[i].Position, self.Vehicles[j].Position)

        self.V2VChannelsAbs = self.V2VPathLoss + self.V2VShadowing
        self.V2IShadowing = self.V2IChannels.GetShadowing(self.DeltaDistance, self.V2IShadowing)
        for i in range(self.VehNumber):
            self.V2IPathLoss[i] = self.V2IChannels.GetPathLoss(self.Vehicles[i].Position)

        self.V2IChannelsAbs = self.V2IPathLoss + self.V2IShadowing

    def UpdateChannelsFastFading(self):
        """ Renew fast fading channel """
        V2VChannelsWithFastfading = np.repeat(self.V2VChannelsAbs[:, :, np.newaxis], self.RBNumber, axis=2)
        self.V2VChannelsWithFastfading = V2VChannelsWithFastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2VChannelsWithFastfading.shape) + 1j * np.random.normal(0, 1, V2VChannelsWithFastfading.shape)) / math.sqrt(2))

        V2IChannelsWithFastfading = np.repeat(self.V2IChannelsAbs[:, np.newaxis], self.RBNumber, axis=1)
        self.V2IChannelsWithFastfading = V2IChannelsWithFastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2IChannelsWithFastfading.shape) + 1j * np.random.normal(0, 1, V2IChannelsWithFastfading.shape))/ math.sqrt(2))

    def ComputeReward(self, ActionsChannelAndPower):
        L2MSubChannel = np.zeros(self.PlatoonNumber, dtype=np.int)
        L2MPower = np.zeros(self.PlatoonNumber)
        M2MSubChannel = np.zeros(self.PlatoonNumber * (self.PlatoonSize - 2), dtype=np.int)
        M2MPower = np.zeros(self.PlatoonNumber * (self.PlatoonSize - 2))

        sumPower = 0

        j = 0
        k = 0
        for i in range(self.PlatoonNumber*(self.PlatoonSize-1)):
            if i % 4 == 0:
                L2MSubChannel[j] = int(ActionsChannelAndPower[i,0])
                L2MPower[j] = ActionsChannelAndPower[i, 1]
                j += 1
            else:
                M2MSubChannel[k] = int(ActionsChannelAndPower[i, 0])
                M2MPower[k] = ActionsChannelAndPower[i, 1]
                k += 1

        # ------------ Compute V2I rate --------------------
        V2IRate = np.zeros(self.RBNumber)
        V2IInterference = np.zeros(self.RBNumber)  # V2I interference

        j = 0
        k = 0

        for i in range(self.PlatoonNumber*(self.PlatoonSize-1+self.PlatoonSize-2)):
            if not self.ActiveLinks[i]:
                continue
            if i % (self.PlatoonSize-1+self.PlatoonSize-2) == 0:
                V2IInterference[L2MSubChannel[j]] += 10 ** ((L2MPower[j] - self.V2IChannelsWithFastfading[j * self.PlatoonSize, L2MSubChannel[j]] \
                                                               + self.VehAntGain + self.BsAntGain - self.BsNoiseFigure) / 10)
                j += 1
            elif i % (self.PlatoonSize-1+self.PlatoonSize-2) > self.PlatoonSize - 2:
                V2IInterference[M2MSubChannel[k]] += 10 ** ((M2MPower[k] -
                                                               self.V2IChannelsWithFastfading[(j - 1) * self.PlatoonSize + k % (self.PlatoonSize-2) + 1, M2MSubChannel[k]]
                                                               + self.VehAntGain + self.BsAntGain - self.BsNoiseFigure) / 10)
                k += 1

        self.V2IInterference = V2IInterference + self.Sig2
        V2ISignals = 10 ** ((self.V2IPowerdBm - self.V2IChannelsWithFastfading.diagonal() + self.VehAntGain + self.BsAntGain - self.BsNoiseFigure) / 10)
        V2IRate = np.log2(1 + np.divide(V2ISignals, self.V2IInterference))

        # ------------ Compute L2M rate --------------------
        L2MRate = np.zeros(self.PlatoonNumber)

        for n in range(self.PlatoonNumber):
            L2MAllLinkRate = np.zeros(self.PlatoonSize - 1)
            L2MAllLinkSignal = np.zeros(self.PlatoonSize - 1)
            L2MAllLinkInterference = np.zeros(self.PlatoonSize - 1)
            nPlatoonId = n * self.PlatoonSize
            nLinkId = n * (self.PlatoonSize-1+self.PlatoonSize-2)
            if self.ActiveLinks[nLinkId]:
                if L2MPower[n] != -100.0:
                    sumPower += L2MPower[n]
                for i in range(self.PlatoonSize - 1):
                    receiverDId = nPlatoonId + i + 1
                    receiverDId = int(receiverDId)
                    # P_{n}{L2M}G_{n,d}^{L2M}
                    L2MAllLinkSignal[i] = 10 ** ((L2MPower[n]
                                                     - self.V2VChannelsWithFastfading[nPlatoonId, receiverDId, L2MSubChannel[n]] + 2 * self.VehAntGain - self.VehNoiseFigure) / 10)
                    # V2I links interference to L2M links
                    L2MAllLinkInterference[i] += 10 ** ((self.V2IPowerdBm - self.V2VChannelsWithFastfading[L2MSubChannel[n], receiverDId, L2MSubChannel[n]] + 2 * self.VehAntGain - self.VehNoiseFigure) / 10)
                    # L2M interference
                    for n_ in range(self.PlatoonNumber):
                        nLinkId_ = n_ * (self.PlatoonSize - 1 + self.PlatoonSize - 2)
                        if n_ != n and L2MSubChannel[n] == L2MSubChannel[n_] and self.ActiveLinks[nLinkId_]:
                            nPlatoonId_ = n_ * self.PlatoonSize
                            #print(nPlatoonId_,L2MSubChannel[n_],L2MPower[n_])
                            L2MAllLinkInterference[i] += 10 ** ((L2MPower[n_]
                                                                    - self.V2VChannelsWithFastfading[nPlatoonId_, receiverDId, L2MSubChannel[n_]] + 2 * self.VehAntGain - self.VehNoiseFigure) / 10)
                    # M2M interference
                    for k in range(self.PlatoonNumber * (self.PlatoonSize - 2)):
                        kLinkId = int(k / (self.PlatoonSize - 2)) * (self.PlatoonSize-1+self.PlatoonSize-2) + k % (self.PlatoonSize - 2) + self.PlatoonSize - 1
                        if L2MSubChannel[n] == M2MSubChannel[k] and self.ActiveLinks[kLinkId]:
                            kId = int(k / (self.PlatoonSize - 2)) * self.PlatoonSize + (k % (self.PlatoonSize - 2)) + 1
                            kId = int(kId)
                            #print(kId, M2MPower[k])
                            L2MAllLinkInterference[i] += 10 ** ((M2MPower[k]
                                                                    - self.V2VChannelsWithFastfading[kId, receiverDId, L2MSubChannel[n]] + 2 * self.VehAntGain - self.VehNoiseFigure) / 10)
                            #print(10 ** ((M2MPower[k]- self.V2VChannelsWithFastfading[kId, receiverDId, L2MSubChannel[n]] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10))
            self.L2MInterference = L2MAllLinkInterference + self.Sig2
            L2MAllLinkRate = np.log2(1 + np.divide(L2MAllLinkSignal, self.L2MInterference))
            L2MRate[n] = np.min(L2MAllLinkRate)

        # ------------ Compute M2M rate --------------------
        M2MRate = np.zeros(self.PlatoonNumber * (self.PlatoonSize - 2))
        M2MSignal = np.zeros(self.PlatoonNumber * (self.PlatoonSize - 2))
        M2MInterference = np.zeros(self.PlatoonNumber * (self.PlatoonSize - 2))
        for k in range(self.PlatoonNumber * (self.PlatoonSize - 2)):
            kId = int(k / (self.PlatoonSize - 2)) * self.PlatoonSize + (k % (self.PlatoonSize - 2)) + 1
            kLinkId = int(k / (self.PlatoonSize - 2)) * (self.PlatoonSize - 1 + self.PlatoonSize - 2) + k % (self.PlatoonSize - 2) + self.PlatoonSize - 1

            if self.ActiveLinks[kLinkId]:
                if M2MPower[k] != -100:
                    sumPower += M2MPower[k]
                receiverKId = kId + 1
                receiverKId = int(receiverKId)
                M2MSignal[k] = 10 ** ((M2MPower[k]
                                                     - self.V2VChannelsWithFastfading[kId, receiverKId, M2MSubChannel[n]] + 2 * self.VehAntGain - self.VehNoiseFigure) / 10)
                # V2I links interference to L2M links
                M2MInterference[k] += 10 ** ((self.V2IPowerdBm - self.V2VChannelsWithFastfading[M2MSubChannel[k], receiverKId, M2MSubChannel[k]] + 2 * self.VehAntGain - self.VehNoiseFigure) / 10)
                # L2M interference
                for n in range(self.PlatoonNumber):
                    nLinkId = n * (self.PlatoonSize - 1 + self.PlatoonSize - 2)
                    if M2MSubChannel[k] == L2MSubChannel[n] and self.ActiveLinks[nLinkId]:
                        nPlatoonId = n * self.PlatoonSize
                        M2MInterference[k] += 10 ** ((L2MPower[n]
                                                                    - self.V2VChannelsWithFastfading[nPlatoonId, receiverKId, L2MSubChannel[n]] + 2 * self.VehAntGain - self.VehNoiseFigure) / 10)
                # M2M interference
                for k_ in range(self.PlatoonNumber * (self.PlatoonSize - 2)):
                    kLinkId_ = int(k_ / (self.PlatoonSize - 2)) * (self.PlatoonSize - 1 + self.PlatoonSize - 2) + k_ % (self.PlatoonSize - 2) + self.PlatoonSize - 1
                    if k_ != k and M2MSubChannel[k_] == M2MSubChannel[k] and self.ActiveLinks[kLinkId_]:
                        kId_ = int(k_ / (self.PlatoonSize - 2)) * self.PlatoonSize + (k_ % (self.PlatoonSize - 2)) + 1
                        kId_ = int(kId_)
                        M2MInterference[k] += 10 ** ((M2MPower[k_]
                                                       - self.V2VChannelsWithFastfading[kId_, receiverKId, M2MSubChannel[k]] + 2 * self.VehAntGain - self.VehNoiseFigure) / 10)
        self.M2MInterference = M2MInterference + self.Sig2
        M2MRate = np.log2(1 + np.divide(M2MSignal, self.M2MInterference))

        for l in range(self.PlatoonNumber):
            for k in range(self.PlatoonSize - 1 + self.PlatoonSize - 2):
                z = l * (self.PlatoonSize - 1 + self.PlatoonSize - 2) + k
                if k < self.PlatoonSize - 1:
                    self.Demand[z] -= L2MRate[l] * self.TimeFast * self.BandWidth
                else:
                    self.Demand[z] -= M2MRate[l * (self.PlatoonSize - 2) + k - (self.PlatoonSize - 1)] * self.TimeFast * self.BandWidth

        self.Demand[self.Demand < 0] = 0  # eliminate negative demands
        self.IndividualTimeLimit -= self.TimeFast
        self.ActiveLinks[np.multiply(self.ActiveLinks, self.Demand <= 0)] = 0  # transmission finished, turned to "inactive"

        L2MReward = L2MRate.copy() * 2
        for n in range(self.PlatoonNumber):
            nLinkId = n * (self.PlatoonSize - 1 + self.PlatoonSize - 2)
            if self.Demand[nLinkId] <= 0:
                L2MReward[n] = 10

        M2MReward = M2MRate.copy() / 3
        for k in range(self.PlatoonNumber * (self.PlatoonSize - 2)):
            kLinkId = int(k / (self.PlatoonSize - 2)) * (self.PlatoonSize - 1 + self.PlatoonSize - 2) + k % (self.PlatoonSize - 2) + self.PlatoonSize - 1
            if self.Demand[kLinkId] <= 0:
                M2MReward[k] = 1

        return V2IRate, L2MRate, M2MRate, L2MReward, M2MReward, sumPower


    def ComputeInterference(self, actions):
        L2MInterference = np.zeros((self.PlatoonNumber, (self.PlatoonSize - 1), self.RBNumber)) + self.Sig2
        M2MInterference = np.zeros((self.PlatoonNumber * (self.PlatoonSize - 2), self.RBNumber)) + self.Sig2
        channelSelection = actions.copy()[:, 0]
        powerSelection = actions.copy()[:, 1]
        # interference from V2I links
        for m in range(self.RBNumber):
            for i in range(self.PlatoonNumber):
                for j in range(self.PlatoonSize - 1):
                    if j == 0:
                        k = i * (self.PlatoonSize - 1)
                        kId = i * self.PlatoonSize
                        for d in range(self.PlatoonSize - 1):
                            receiverId = kId + d + 1
                            L2MInterference[i, d, m] += 10 ** ((self.V2IPowerdBm - self.V2VChannelsWithFastfading[m][receiverId][m] + 2 * self.VehAntGain - self.VehNoiseFigure) / 10)
                    else:
                        k = i * (self.PlatoonSize - 2) + j - 1
                        kId = i * self.PlatoonSize + j
                        receiverId = kId + 1
                        M2MInterference[k, m] += 10 ** ((self.V2IPowerdBm - self.V2VChannelsWithFastfading[m][receiverId][m] + 2 * self.VehAntGain - self.VehNoiseFigure) / 10)

        # interference from L2M links
        for n in range(self.PlatoonNumber):
            nPlatoonId = n * self.PlatoonSize
            nActionPlatoonId = n * (self.PlatoonSize - 1)
            nLinkId = n * (self.PlatoonSize - 1 + self.PlatoonSize - 2)
            for n_ in range(self.PlatoonNumber):
                n_actionPlatoonId_ = n_ * (self.PlatoonSize - 1)
                nPlatoonId_ = n_ * self.PlatoonSize
                if nPlatoonId_ == nPlatoonId or not self.ActiveLinks[nLinkId]:
                    continue
                for d in range(self.PlatoonNumber - 1):
                    receiverId = nPlatoonId + d + 1
                    L2MInterference[n, d, int(channelSelection[n_actionPlatoonId_])] += 10 ** ((powerSelection[n_actionPlatoonId_]
                                                                                   - self.V2VChannelsWithFastfading[nPlatoonId_][receiverId][int(channelSelection[n_actionPlatoonId_])] + 2 * self.VehAntGain - self.VehNoiseFigure) / 10)
            for k in range(self.PlatoonNumber * (self.PlatoonSize - 2)):
                kLinkId = int(k / (self.PlatoonSize - 2)) * (self.PlatoonSize - 1 + self.PlatoonSize - 2) + k % (
                            self.PlatoonSize - 2) + self.PlatoonSize - 1
                kId = int(k / (self.PlatoonSize - 2)) * self.PlatoonSize + (k % (self.PlatoonSize - 2)) + 1
                kId = int(kId)
                kActionPlatoonId = int(k / (self.PlatoonSize - 2)) * (self.PlatoonSize - 1) + k % (
                            self.PlatoonSize - 2) + 1
                if not self.ActiveLinks[kLinkId]:
                    continue
                for d in range(self.PlatoonSize - 1):
                    receiverId = nPlatoonId + d + 1
                    L2MInterference[n, d, int(channelSelection[kActionPlatoonId])] += 10 ** (
                                (powerSelection[kActionPlatoonId]
                                 - self.V2VChannelsWithFastfading[kId][receiverId][
                                     int(channelSelection[kActionPlatoonId])] + 2 * self.VehAntGain - self.VehNoiseFigure) / 10)

        # interference from M2M links
        for k in range(self.PlatoonNumber * (self.PlatoonSize - 2)):
            kId = int(k / (self.PlatoonSize - 2)) * self.PlatoonSize + (k % (self.PlatoonSize - 2)) + 1
            kLinkId = int(k / (self.PlatoonSize - 2)) * (self.PlatoonSize - 1 + self.PlatoonSize - 2) + k % (
                    self.PlatoonSize - 2) + self.PlatoonSize - 1
            receiverId = kId + 1
            for n in range(self.PlatoonNumber):
                nPlatoonId = n * self.PlatoonSize
                nActionPlatoonId = n * (self.PlatoonSize - 1)
                nLinkId = n * (self.PlatoonSize - 1 + self.PlatoonSize - 2)
                if not self.ActiveLinks[nLinkId]:
                    continue
                M2MInterference[k, int(channelSelection[nActionPlatoonId])]+= 10 ** (
                                (powerSelection[nActionPlatoonId]
                                 - self.V2VChannelsWithFastfading[nPlatoonId][receiverId][
                                     int(channelSelection[nActionPlatoonId])] + 2 * self.VehAntGain - self.VehNoiseFigure) / 10)
            for k_ in range(self.PlatoonNumber * (self.PlatoonSize - 2)):
                kLinkId_ = int(k_ / (self.PlatoonSize - 2)) * (self.PlatoonSize - 1 + self.PlatoonSize - 2) + k_ % (
                        self.PlatoonSize - 2) + self.PlatoonSize - 1
                kActionPlatoonId_ = int(k / (self.PlatoonSize - 2)) * (self.PlatoonSize - 1) + k % (
                        self.PlatoonSize - 2) + 1
                kId_ = int(k_ / (self.PlatoonSize - 2)) * self.PlatoonSize + (k_ % (self.PlatoonSize - 2)) + 1
                kId_ = int(kId_)
                if k_ == k or not self.ActiveLinks[kLinkId_]:
                    continue
                M2MInterference[k, int(channelSelection[kActionPlatoonId_])] += 10 ** (
                        (powerSelection[kActionPlatoonId_]
                         - self.V2VChannelsWithFastfading[kId_][receiverId][
                             int(channelSelection[kActionPlatoonId_])] + 2 * self.VehAntGain - self.VehNoiseFigure) / 10)
        self.L2MInterferenceAll = 10 * np.log10(L2MInterference)
        self.M2MInterferenceAll = 10 * np.log10(M2MInterference)

    def ObserveEachPlatoon(self, n):
        nPlatoonId = n * self.PlatoonSize
        nLinkId = n * (self.PlatoonSize - 1 + self.PlatoonSize - 2)

        V2VFast = np.zeros((self.PlatoonSize - 1, self.RBNumber, self.RBNumber))
        V2VAbs = np.zeros((self.PlatoonSize - 1, self.RBNumber))
        for i in range(self.PlatoonSize - 1):
            receiverDId = nPlatoonId + i + 1
            receiverDId = int(receiverDId)
            V2VFast[i] = (self.V2VChannelsWithFastfading[:, receiverDId, :] - self.V2VChannelsAbs[:, receiverDId] + 10) / 35
            V2VAbs[i] = (self.V2VChannelsAbs[:, receiverDId] - 80) / 60.0

        V2IFast = np.zeros((self.PlatoonSize - 1, self.RBNumber))
        V2IAbs = np.zeros(self.PlatoonSize - 1)
        V2IFast[0] = (self.V2IChannelsWithFastfading[nPlatoonId, :] - self.V2IChannelsAbs[nPlatoonId] + 10) / 35
        V2IAbs[0] = (self.V2IChannelsAbs[nPlatoonId] - 80) / 60.0
        for i in range(self.PlatoonSize - 2):
            kId = nPlatoonId + i + 1
            V2IFast[i+1] = (self.V2IChannelsWithFastfading[kId, :] - self.V2IChannelsAbs[kId] + 10) / 35
            V2IAbs[i+1] = (self.V2IChannelsAbs[kId] - 80) / 60.0

        TimeRemaining = np.asarray([self.IndividualTimeLimit[nLinkId] / self.TimeLatency])

        LoadRemaining = np.zeros((self.PlatoonSize - 1))
        for i in range(self.PlatoonSize - 1):
            if i == 0:
                LoadRemaining[i] = self.Demand[nLinkId] / self.DemandSize
            else:

                kLinkId = nLinkId + self.PlatoonSize - 2 + i
                LoadRemaining[i] = self.Demand[kLinkId] / self.DemandSize

        return np.concatenate((np.reshape(V2IFast, -1), np.reshape(V2IAbs, -1), np.reshape(V2VFast, -1), np.reshape(V2VAbs, -1), TimeRemaining, LoadRemaining))

    def ActForTrainingCompete(self, ActionsChannelAndPower):
        sumL2MAndM2MPower = 0.0001
        for i in range(self.PlatoonNumber*(self.PlatoonSize-1)):
            sumL2MAndM2MPower += 0.001 * (10 ** (ActionsChannelAndPower[i][1] / 10))
        actionsTemp = ActionsChannelAndPower.copy()
        V2IRate, L2MRate, M2MRate, L2MReward, M2MReward, sumPower = self.ComputeReward(actionsTemp)
        reward = np.zeros(self.PlatoonNumber)
        for i in range(self.PlatoonNumber):
            reward[i] += L2MReward[i]
            for j in range(self.PlatoonSize - 2):
                k = i * (self.PlatoonSize - 2) + j
                reward[i] += M2MReward[k]
        return reward

    def ActForTrainingCooperative(self, ActionsChannelAndPower):
        # compute SE/total power consumption:
        sumL2MAndM2MPower = 0.0001
        for i in range(self.PlatoonNumber*(self.PlatoonSize-1)):
            sumL2MAndM2MPower += 0.001 * (10 ** (ActionsChannelAndPower[i][1] / 10))
        actionsTemp = ActionsChannelAndPower.copy()
        V2IRate, L2MRate, M2MRate, L2MReward, M2MReward, sumPower= self.ComputeReward(actionsTemp)
        reward1 = np.zeros(self.PlatoonNumber)
        for i in range(self.PlatoonNumber):
            reward1[i] += L2MReward[i]
            for j in range(self.PlatoonSize - 2):
                k = i * (self.PlatoonSize - 2) + j
                reward1[i] += M2MReward[k]

        reward = np.sum(L2MReward) + np.sum(M2MReward)
        return reward

    def NewRandomGame(self, platoonNumber, platoonSize):
        # make a new game
        self.Vehicles = []
        self.PlatoonNumber = platoonNumber
        self.PlatoonSize = platoonSize
        self.VehNumber = platoonNumber * platoonSize
        self.RBNumber = self.VehNumber

        self.AddNewPlatoonVehicles(self.PlatoonNumber, self.PlatoonSize)
        self.UpdateChannelsSlowFading()
        self.UpdateChannelsFastFading()

        self.Demand = self.DemandSize * np.ones(self.PlatoonNumber*(self.PlatoonSize-1+self.PlatoonSize-2))
        self.IndividualTimeLimit = self.TimeLatency * np.ones(self.PlatoonNumber*(self.PlatoonSize-1+self.PlatoonSize-2))
        self.ActiveLinks = np.ones((self.PlatoonNumber*(self.PlatoonSize-1+self.PlatoonSize-2)), dtype='bool')

    def IsEpisodeEnd(self):
        for v in self.Vehicles:
            if abs(v.Position[0]) > self.RoadWidth/2 or abs(v.Position[1]) > self.RoadHeight/2:
                return True
        return False