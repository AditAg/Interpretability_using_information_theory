# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from textwrap import wrap


font = {'family' : 'times',
        'size'   : 12}

plt.rc('font', **font)

plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12)
samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Run ensemble experiment and put interpretability values here

#svm + ann (64) + ann (512, 256, 128)
values_test_ensemble1 = []

#svm + ann (64) + ann (512, 256, 128) + ann (500, 200)
values_test_ensemble2 = []


#svm + ann (64) + ann (500, 200, 100)
values_test_ensemble3 = []

#svm + logisitic regression + ann (64) + ann (500, 200)
values_test_ensemble4 = []

#svm + logistic regression + dt
values_test_ensemble5 = []

std_test_ensemble1, mean_test_ensemble1 = [], []

std_test_ensemble2, mean_test_ensemble2 = [], []

std_test_ensemble3, mean_test_ensemble3 = [], []

std_test_ensemble4, mean_test_ensemble4 = [], []

std_test_ensemble5, mean_test_ensemble5 = [], []

for i in range(len(samples)):
   std_test_ensemble1.append(np.std(np.array(values_test_ensemble1[i])))
   mean_test_ensemble1.append(np.mean(np.array(values_test_ensemble1[i])))
   
   std_test_ensemble2.append(np.std(np.array(values_test_ensemble2[i])))
   mean_test_ensemble2.append(np.mean(np.array(values_test_ensemble2[i])))
   
   std_test_ensemble3.append(np.std(np.array(values_test_ensemble3[i])))
   mean_test_ensemble3.append(np.mean(np.array(values_test_ensemble3[i])))
   
   std_test_ensemble4.append(np.std(np.array(values_test_ensemble4[i])))
   mean_test_ensemble4.append(np.mean(np.array(values_test_ensemble4[i])))
   
   std_test_ensemble5.append(np.std(np.array(values_test_ensemble5[i])))
   mean_test_ensemble5.append(np.mean(np.array(values_test_ensemble5[i])))
   
plt.errorbar(samples, mean_test_ensemble1, std_test_ensemble1, color = 'b', linestyle = '-', marker = 'o', label = 'svm + ann (64) + ann (512, 256, 128)')
plt.errorbar(samples, mean_test_ensemble2, std_test_ensemble2, color = 'r', linestyle = '-', marker = 'o', label = 'svm + ann (64) + ann (512, 256, 128) + ann (500, 200)')
plt.errorbar(samples, mean_test_ensemble3, std_test_ensemble3, color = 'g', linestyle = '-', marker = 'o', label = 'svm + ann (64) + ann (500, 200, 100)')
plt.errorbar(samples, mean_test_ensemble4, std_test_ensemble4, color = 'k', linestyle = '-', marker = 'o', label = 'svm + lr + ann (64) + ann (500, 200)')
plt.errorbar(samples, mean_test_ensemble5, std_test_ensemble5, color = 'y', linestyle = '-', marker = 'o', label = 'svm + logistic regression + decision tree')

plt.xlabel("Percentage (%) of dataset")
plt.ylabel("Interpretability")
plt.xticks(np.arange(1, 13), [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.legend(loc = 'center right', bbox_to_anchor=(1, 0.2))

plt.show()

