#!/usr/bin/python
# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
from common.config import *
from common.utils import *
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from matplotlib.pyplot import MultipleLocator

def PlotNewScore():
    result = np.loadtxt(SaveDataPath)
    print(result.size)
    iter = np.linspace(1, result.size, result.size)
    result = smooth(result,0.5)

    sns.set(style="whitegrid", font_scale=1.1)
    sns.tsplot(time=iter, data=result, color="red", alpha=0.85, linewidth = 3, linestyle='-', condition="PPO")
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 4))
    plt.ylabel("Performance", fontsize=19)
    plt.xlabel("Iteration Number", fontsize=19)
    plt.savefig(SavePDFPath, format='pdf', bbox_inches = 'tight')
    print("PlotNewScore success!")