import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from textwrap import wrap

font = {'family' : 'times',
        'size'   : 14}

plt.rc('font', **font)

plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14)
samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

# Put the values obtained on running the finalWithLogEntropy.py file here for mnist and fmnist
#1Layer PWLN - MNIST
values_test = []

#1Layer PWLN - Fashion-MNIST
values_test_fmnist = []

std_test_mnist = []
mean_test_mnist = []
std_test_fmnist = []
mean_test_fmnist = []

for i in range(len(samples)):
    std_test_mnist.append(np.std(np.array(values_test[i])))
    mean_test_mnist.append(np.mean(np.array(values_test[i])))
    std_test_fmnist.append(np.std(np.array(values_test_fmnist[i])))
    mean_test_fmnist.append(np.mean(np.array(values_test_fmnist[i])))
    
plt.subplot(111)
plt.errorbar(samples, mean_test_mnist, std_test_mnist, color = 'b', linestyle = '-', marker = 'o', label = 'MNIST')

plt.errorbar(samples, mean_test_fmnist, std_test_fmnist, color = 'r', linestyle = 'dashed', marker = 'o', label = 'Fashion-MNIST')

plt.xlabel("(a) Complexity of Known Model (Width of Layer)", fontsize = 16)
plt.ylabel("Interpretability", fontsize = 16)
plt.xticks(np.arange(1, 14), [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 5120, 6144])
plt.legend(loc = 'center right', bbox_to_anchor=(1, 0.2))
plt.show()


