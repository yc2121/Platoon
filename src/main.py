#!/usr/bin/python
# -*- coding: UTF-8 -*-
import time
import torch
import numpy as np
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from collections import deque
from algo.ppo import TrainModel
from common.utils import GetAction, DiscreteActionPlatoon25
from common.constant import *
from common.config import *
from env.Env_platoon import Environment
from neural_network.agent import Agent
from tools.result_record import ResultRecord

if __name__ == "__main__":
    env = Environment()
    tsRecord = ResultRecord()

    totalSteps = int(env.TimeSlow / env.TimeFast)

    # 每个排作为一个智能体
    MultiAgentList = []
    for i in range(PlatoonNumber):
        logging.info(f"Init {i}-th vehicle agent")
        MultiAgentList.append(Agent(i, InputDim, OutputDim))

    curEpisode = 0
    episodeList = []
    scoreList = []
    scoreBest = 0
    for iter in range(IterationNumber):
        env.NewRandomGame(PlatoonNumber, PlatoonSize)

        for i in range(PlatoonNumber):
            MultiAgentList[i].Actor.eval(), MultiAgentList[i].Critic.eval()
            MultiAgentList[i].Memory = deque()

        totalRewardRecordList = []
        while not env.IsEpisodeEnd():
            curEpisode += 1
            totalReward = 0

            env.UpdateVehPositions(env.TimeSlow)  # update vehicle position
            env.UpdateChannelsSlowFading()  # update channel slow fading
            env.UpdateChannelsFastFading()  # update channel fast fading
            env.Demand = env.DemandSize * np.ones(env.PlatoonNumber * (env.PlatoonSize - 1 + env.PlatoonSize - 2))
            env.IndividualTimeLimit = env.TimeLatency * np.ones(
                env.PlatoonNumber * (env.PlatoonSize - 1 + env.PlatoonSize - 2))
            env.ActiveLinks = np.ones((env.PlatoonNumber * (env.PlatoonSize - 1 + env.PlatoonSize - 2)), dtype='bool')

            for i_step in range(totalSteps):
                actionAll = np.zeros([PlatoonNumber * (PlatoonSize - 1), 2])
                actionAllIndex = 0
                stateRecordList = []
                actionRecordList = []
                for i in range(PlatoonNumber):
                    state = env.ObserveEachPlatoon(i)
                    stateRecordList.append(state)
                    mu, std, _ = MultiAgentList[i].Actor(torch.Tensor(state).unsqueeze(0))
                    action = GetAction(mu, std)[0]
                    actionDiscrete = DiscreteActionPlatoon25(action, 2*(env.PlatoonSize-1))
                    actionRecordList.append(action)

                    for j in range(env.PlatoonSize - 1):
                        actionAll[actionAllIndex, 0] = actionDiscrete[j*2]
                        actionAll[actionAllIndex, 1] = actionDiscrete[j*2+1]
                        actionAllIndex += 1

                actionAllTemp = actionAll.copy()
                reward = env.ActForTrainingCompete(actionAllTemp)
                rewardSum = np.sum(reward)

                totalReward += np.sum(reward)
                contributionBased = np.zeros(env.PlatoonNumber)
                for ctb in range(env.PlatoonNumber):
                    contributionBased[ctb] = (reward[ctb] / 13) * rewardSum

                env.UpdateChannelsFastFading()
                env.ComputeInterference(actionAllTemp)

                for i in range(PlatoonNumber):
                    nextState = env.ObserveEachPlatoon(i)
                    done = 0
                    mask = 1
                    if i_step == env.Latency - 1:
                        done = 1
                        mask = 0

                    s = stateRecordList[i]
                    a = actionRecordList[i]
                    MultiAgentList[i].Memory.append([s, a, contributionBased[i], mask])

                if i_step == env.Latency - 1:  # latency 退出
                    break

            totalRewardRecordList.append(totalReward)

        scoreAvg = np.mean(totalRewardRecordList)
        logging.info(f"iter:{iter}, episode:{curEpisode}, score:{scoreAvg}")

        if scoreBest < scoreAvg:
            scoreBest = scoreAvg
            logging.info(f"iter:{iter}, episode:{curEpisode}, score:{scoreAvg}, save current best performance model")
            if not os.path.exists(SaveModelsPath):
                os.makedirs(SaveModelsPath)
            for j in range(PlatoonNumber):
                MultiAgentList[j].saveModel()

        episodeList.append(int(curEpisode))
        scoreList.append(scoreAvg)
        trainStartTime = time.time()

        tsRecord.RecordTBData("scoreAvg", scoreAvg, iter)
        for i in range(PlatoonNumber):
            MultiAgentList[i].Actor.train(), MultiAgentList[i].Critic.train()
            actorLoss, criticLoss, ratioClip = TrainModel(MultiAgentList[i].Actor, MultiAgentList[i].Critic, MultiAgentList[i].Memory, MultiAgentList[i].ActorOptim, MultiAgentList[i].CriticOptim)
            tsRecord.RecordTBData(f"agent_{i}_actorLoss", actorLoss, iter)
            tsRecord.RecordTBData(f"agent_{i}_criticLoss", criticLoss, iter)
            tsRecord.RecordTBData(f"agent_{i}_ratioClip", ratioClip, iter)
        tsRecord.FlushTBData()
        logging.info(f"train agent time:{time.time() - trainStartTime}s")

        if iter % 10 == 1:
            logging.info(f"save score into txt, current iter:{iter}")
            tsRecord.SaveTxtData(scoreList)

    plt.plot(episodeList, scoreList, linewidth=3)
    plt.title("Performance", fontsize=19)
    plt.xlabel("Episodes", fontsize=10)
    plt.ylabel("AvgScore", fontsize=10)
    plt.tick_params(axis='both', labelsize=9)
    plt.show()

