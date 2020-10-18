import matplotlib.pyplot as plt
import numpy as np
from textwrap import wrap

font = {'family' : 'times',
        'size'   : 12}

plt.rc('font', **font)

plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14)
samples = [10, 20, 30, 40, 50, 60, 70, 80, 90]


# Put values for all 3 croppings based on the different runs of finalWithLogEntropy.py file.
values_test_orignal_cropped = []
values_test_cropped_top_left = []
values_test_cropped_bottom_right = []



std_test_1, std_test_2, std_test_3 = [], [], []
mean_test_1, mean_test_2, mean_test_3 = [], [], []

values_test_orignal_cropped, values_test_cropped_top_left, values_test_cropped_bottom_right = values_test_orignal_cropped[1:], values_test_cropped_top_left[1:], values_test_cropped_bottom_right[1:]
for i in range(len(samples)):
   std_test_1.append(np.std(np.array(values_test_orignal_cropped[i])))
   mean_test_1.append(np.mean(np.array(values_test_orignal_cropped[i])))
   std_test_2.append(np.std(np.array(values_test_cropped_top_left[i])))
   mean_test_2.append(np.mean(np.array(values_test_cropped_top_left[i])))
   std_test_3.append(np.std(np.array(values_test_cropped_bottom_right[i])))
   mean_test_3.append(np.mean(np.array(values_test_cropped_bottom_right[i])))

fig = plt.figure(figsize = (8, 4))
ax = fig.add_subplot(111)

X = np.arange(1, 10)
plt.setp(ax, xticks = X, xticklabels = samples)
ax.bar(X + 0.25, mean_test_1, align = 'center', width = 0.25, alpha = 0.5, color = 'b', capsize = 5, hatch = '..', label = 'Original Cropped')
ax.bar(X + 0.0, mean_test_2, align = 'center', width = 0.25, alpha = 0.5, color = 'r', capsize = 5, hatch = '//', label = 'Cropped Top Left')
ax.bar(X - 0.25, mean_test_3, align = 'center', width = 0.25, alpha = 0.5, color = 'g', capsize = 5, hatch = '--', label = 'Cropped Bottom Right')
ax.set_ylabel("Interpretability", fontsize=16)
ax.set_xlabel("Percentage(%) of dataset", fontsize=16)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='lower right', borderaxespad=0.)
plt.tight_layout(pad=3.0)

plt.show()
