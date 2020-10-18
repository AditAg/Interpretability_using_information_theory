import matplotlib.pyplot as plt
import numpy as np
from textwrap import wrap


font = {'family' : 'times',
        'size'   : 18}

plt.rc('font', **font)

plt.rc('xtick', labelsize=18) 
plt.rc('ytick', labelsize=18)
samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

#1Layer PWLN
# Put the values obtained on running the finalWithLogEntropy.py file here for mnist and fmnist
values_test_mnist = []

values_test_fmnist = []

std_test_mnist, mean_test_mnist = [], []

std_test_fmnist, mean_test_fmnist = [], []


for i in range(len(samples)):
   std_test_mnist.append(np.std(np.array(values_test_mnist[i])))
   mean_test_mnist.append(np.mean(np.array(values_test_mnist[i])))
   std_test_fmnist.append(np.std(np.array(values_test_fmnist[i])))
   mean_test_fmnist.append(np.mean(np.array(values_test_fmnist[i])))
	
plt.errorbar(samples, mean_test_mnist, std_test_mnist, color = 'b', linestyle = '-', marker = 'o', label = 'MNIST')
plt.errorbar(samples, mean_test_fmnist, std_test_fmnist, color = 'r', linestyle = 'dashed', marker = 'o', label = 'Fashion-MNIST')

plt.xlabel("Percentage (%) of dataset", fontsize=18)
plt.ylabel("Interpretability", fontsize=18)
plt.xticks(np.arange(1, 13), [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.legend(loc = 'center right', bbox_to_anchor=(1, 0.2), fontsize = 16)

plt.show()


