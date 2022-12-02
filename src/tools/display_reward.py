#!/usr/bin/python
# -*- coding: UTF-8 -*-

import math
import torch
import argparse
import numpy as np
import torch.optim as optim
from model import Actor, Critic
from utils import get_action
from collections import deque
from running_state import ZFilter
from hparams import HyperParams as hp
import matplotlib.pyplot as plt
import time
import os
import pickle
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from matplotlib.pyplot import MultipleLocator

parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', type=str, default='PPO',
                    help='select one of algorithms among Vanilla_PG, NPG, TPRO, PPO')
parser.add_argument('--env', type=str, default="Humanoid-v1",
                    help='name of Mujoco environement')
parser.add_argument('--render', default=False)
args = parser.parse_args()

if args.algorithm == "PG":
    from vanila_pg import train_model
elif args.algorithm == "NPG":
    from npg import train_model
elif args.algorithm == "TRPO":
    from trpo import train_model
elif args.algorithm == "PPO":
    from ppo import train_model

def smooth(arr,r):
    smoothing_rate = r
    temp_arr = np.zeros(len(arr))
    temp_arr[0] = arr[0]
    for i in range(1, len(arr)):
        temp_arr[i] = smoothing_rate * temp_arr[i-1] + (1-smoothing_rate) * arr[i]
    return temp_arr

if __name__=="__main__":
    x1 = np.loadtxt("03_24_red.txt")[0:250]
    x2 = np.loadtxt("03_24_grey.txt")[0:250]
    x3 = np.loadtxt("03_24_blue.txt")[0:250]
    x1 = smooth(x1,0.5)
    x2 = smooth(x2,0.1)
    iter = range(250)

    sns.set(style="whitegrid", font_scale=1.1)
    sns.tsplot(time=iter, data=x1, color="red", alpha=0.85, linewidth = 3, linestyle='-', condition="CP-PPO")
    sns.tsplot(time=iter, data=x3, color="dodgerblue", alpha=0.85, linewidth = 3, linestyle='-.', condition="PPO")
    sns.tsplot(time=iter, data=x2, color="grey", alpha=0.85, linewidth = 3, linestyle='--', condition="MARL")
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 4))
    plt.ylabel("Reward", fontsize=19)
    plt.xlabel("Iteration Number", fontsize=19)
    plt.savefig("03_24.pdf", format='pdf',bbox_inches = 'tight')
    plt.show()