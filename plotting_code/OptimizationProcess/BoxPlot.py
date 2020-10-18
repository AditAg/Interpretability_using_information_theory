# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap

font = {'family' : 'times',
        'size'   : 14}

plt.rc('font', **font)

plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14)
x = np.zeros((9, 6))

#Adadelta
x[:, 0] = []
#Adam
x[:, 1] = []
#Adagrad
x[:, 2] = []
#Gradient Descent
x[:, 3] = []
#Momentum
x[:, 4] = []
#RMSProp
x[:, 5] = []
df = pd.DataFrame(x, columns=['Adadelta', 'Adam', 'Adagrad', 
                              'Gradient Descent', 'Momentum', 'RMSProp'])
df.plot.box(grid = 'on')
plt.xticks(rotation = 15)
plt.xlabel("(b) Optimization Process", fontsize = 16)
plt.ylabel("Interpretability", fontsize = 16)
plt.tight_layout()
plt.show()