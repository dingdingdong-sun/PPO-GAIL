#-*- coding: utf-8 -*-

from pylab import *
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import seaborn as sns

my_font = FontProperties(family='SimHei', size=12)
plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus']=False

sns.set_style("whitegrid")
sns.set_context("paper")

import warnings
warnings.filterwarnings("ignore")

std = np.load("./std_info.npy")
mid = np.median(std)
mean = np.mean(std)

fig = plt.figure()
# plt.title("过滤后SJF生成轨迹用户平均等待时间比KDE图", fontproperties=my_font)
plt.xlabel("轨迹用户平均等待时间比标准差", fontproperties=my_font)
sns.distplot(std, hist=False)
plt.axvline(mid,color='r', linestyle='--', label='mid')
plt.axvline(2*mean, color="g", linestyle='--', label='2*mean')
plt.legend()
plt.show()