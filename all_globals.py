#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 20:53:00 2020

@author: adit
"""
# FILE CONTAINING ALL THE GLOBAL PARAMETERS 

# dataset name - values: 'mnist', 'fashion_mnist', 'stanford40'
dataset_name = 'mnist'
# Possible Model Names - "svm", "ann", "cnn", "inceptionV3", "naive_bayes", "ensemble"
# ensemble doesn't support CNNs at present for 
model_name_A = 'svm'
model_name_B = 'ann'
# No of folds for K-fold crossvaliation
no_of_folds = 5
# Whether to resize the dataset or not. If set to true, image is resized from (28, 28) in case of MNIST and Fashion-MNIST to (10, 10)
is_resized = False
# epochs to be used for training the individual models
model_A_epochs = 50
model_B_epochs = 50
learning_rate = 0.001
# Whether to convert images to grayscale or not.
is_grayscale = False
batch_size = 128
# whether to use monte-carlo dropout for neural networks (only supported in anns): Not sed in current implementation.
mc_dropout = False
# this parameter is used if we want to convert the classification task (10 digits) to a binary-classification task: 
# For converting to binary-classification task: Makes the maximally present class as 1 and the rest as 0.
is_binarized = False
# used for experimentation with Stanford40 dataset. Values: 
# 1. 'object' for crop containing actual object
# 2. 'top_left' for cropping done from top left corner
# 3. 'bottom_right' for cropping done from bottom right corner
crop_region = 'object'
# Hidden layer list for Model A: "ann"
hidden_layer_list_model_A = [256]
# Hidden layer list for Model B: "ann"
hidden_layer_list_model_A = [512, 256, 64]


# ENSEMBLE PARAMETERS
# Supported models in Ensemble: 'svm', 'dt', 'ann', 'lr', 'knn'
svm_param_list = {'C':0.01, 'kernel':'rbf', 'gamma': 'auto', 'probability':True, 'random_state':0}
ensemble_list = [("svm", svm_param_list), ("lr", ), ("dt", "gini", "best", None, 10), ("ann", [512, 256, 64])]


