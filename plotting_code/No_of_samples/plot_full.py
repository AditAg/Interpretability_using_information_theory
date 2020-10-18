import matplotlib.pyplot as plt
import numpy as np
from textwrap import wrap

font = {'family' : 'times',
        'size'   : 14}

plt.rc('font', **font)

plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14)
samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Put values obtained on running finalWithLogEntropy.py file for different % of the datasets for MNIST and Fashion-MNIST with model A as : 1-layer PWLN, SVM and DT as explained in the experiment details
#1Layer PWLN
values_mnist_pwln = []

values_fmnist_pwln = []

#DT
values_mnist_dt = []

values_fmnist_dt = []

#svm
values_mnist_svm = []

values_fmnist_svm = []


std_mnist_pwln = []
std_mnist_dt = []
std_mnist_svm = []
mean_mnist_pwln = []
mean_mnist_dt = []
mean_mnist_svm = []
std_fmnist_pwln = []
std_fmnist_dt = []
std_fmnist_svm = []
mean_fmnist_pwln = []
mean_fmnist_dt = []
mean_fmnist_svm = []

for i in range(len(samples)):
	std_mnist_pwln.append(np.std(np.array(values_mnist_pwln[i])))
	mean_mnist_pwln.append(np.mean(np.array(values_mnist_pwln[i])))
	std_mnist_dt.append(np.std(np.array(values_mnist_dt[i])))
	mean_mnist_dt.append(np.mean(np.array(values_mnist_dt[i])))
	std_mnist_svm.append(np.std(np.array(values_mnist_svm[i])))
	mean_mnist_svm.append(np.mean(np.array(values_mnist_svm[i])))
	
for i in range(len(samples)):
	std_fmnist_pwln.append(np.std(np.array(values_fmnist_pwln[i])))
	mean_fmnist_pwln.append(np.mean(np.array(values_fmnist_pwln[i])))
	std_fmnist_dt.append(np.std(np.array(values_fmnist_dt[i])))
	mean_fmnist_dt.append(np.mean(np.array(values_fmnist_dt[i])))
	std_fmnist_svm.append(np.std(np.array(values_fmnist_svm[i])))
	mean_fmnist_svm.append(np.mean(np.array(values_fmnist_svm[i])))

plt.subplot(131)
plt.errorbar(samples, mean_mnist_pwln, std_mnist_pwln, color = 'b', linestyle = '-', marker = 'o', label = 'MNIST')
plt.errorbar(samples, mean_fmnist_pwln, std_fmnist_pwln, color = 'r', linestyle = 'dotted', marker = 'o', label = 'Fashion-MNIST')

plt.xlabel("Percentage (%) of dataset", fontsize=16)
plt.ylabel("Interpretability", fontsize=16)
plt.xticks(np.arange(1, 13), [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.title("\n".join(wrap("(a) Single Layer ReLU network")))
plt.legend(loc = 'center right', bbox_to_anchor=(1, 0.2))

plt.subplot(132)
plt.errorbar(samples, mean_mnist_dt, std_mnist_dt, color = 'b', linestyle = '-', marker = 'o', label = 'MNIST')
plt.errorbar(samples, mean_fmnist_dt, std_fmnist_dt, color = 'r', linestyle = 'dotted', marker = 'o', label = 'Fashion-MNIST')
 
plt.xlabel("Percentage (%) of dataset", fontsize=16)
plt.xticks(np.arange(1, 13), [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.title("\n".join(wrap("(b) Decision Tree")))
plt.legend(loc = 'center right', bbox_to_anchor=(1, 0.2))

plt.subplot(133)
plt.errorbar(samples, mean_mnist_svm, std_mnist_svm, color = 'b', linestyle = '-', marker = 'o', label = 'MNIST')
plt.errorbar(samples, mean_fmnist_svm, std_fmnist_svm, color = 'r', linestyle = 'dotted', marker = 'o', label = 'Fashion-MNIST') 
 
plt.xlabel("Percentage (%) of dataset", fontsize=16)
plt.xticks(np.arange(1, 13), [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.title("\n".join(wrap("(c) SVM")))
plt.legend(loc = 'center right', bbox_to_anchor=(1, 0.2))

plt.subplots_adjust(wspace = 0.2, hspace = 0.2)
plt.show()


