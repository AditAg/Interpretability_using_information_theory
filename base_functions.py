#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 23:07:41 2019

@author: adit
"""
import numpy as np
from fns import load_mnist_dataset, load_fashionmnist, find_average, load_stanford40_dataset
from sklearn.model_selection import KFold, train_test_split
import all_globals
from model_classes import ModelA, ModelB


def get_dataset(dataset_name, is_binarized, is_resized, is_grayscale, sentiment_mode = 'original', classification_type = 'binary'):
    if (dataset_name == "mnist"):
        data_X, data_Y, test_X, test_Y = load_mnist_dataset(is_binarized, is_resized)
    elif(dataset_name == 'fashion_mnist'):
        data_X, data_Y, test_X, test_Y = load_fashionmnist(is_binarized, is_resized)
    elif(dataset_name == 'stanford40'):
        data_X, data_Y, test_X, test_Y, data_X_A, test_X_A = load_stanford40_dataset(all_globals.crop_region)
        return data_X, data_Y, test_X, test_Y, data_X_A, test_X_A
    else:
        print("Not implemented yet")
        return
    return data_X, data_Y, test_X, test_Y
        
def perform_interpretation(mc_dropout, outp_train, outp_cv, outp_test, model_A, model_B):
    probs_train_B, probs_cv_B, probs_test_B = model_B.get_output()
    preds_test_B = np.argmax(probs_test_B, axis = -1)
    print("Unique Classes in Predictions of Model B on test data", np.unique(preds_test_B))
    print("Classes in data used for training")
    
    classes_ = np.unique(preds_test_B)
    for class_ in classes_:
        indices = (preds_test_B == class_).nonzero()
        print("Class: ", class_, "No of samples: ", np.array(indices).shape)
    
    preds_train_B = np.argmax(probs_train_B, axis = -1)
    print("Unique Classes in Predictions of Model B on test data", np.unique(preds_train_B))
    print("Classes in data used for training")
    
    classes_ = np.unique(preds_train_B)
    for class_ in classes_:
        indices = (preds_train_B == class_).nonzero()
        print("Class: ", class_, "No of samples: ", np.array(indices).shape)
        
    interpretability_train, interpretability_cv, interpretability_test = model_A.calculate_interpretability(probs_train_B, probs_cv_B, probs_test_B)
    outp_train.append(interpretability_train)
    outp_cv.append(interpretability_cv)
    outp_test.append(interpretability_test)
    print("Values are:",interpretability_train, interpretability_cv, interpretability_test)
    
    return outp_train, outp_cv, outp_test


def test_kfold_cross_validation(no_samples):
    data_X, data_Y, test_X, test_Y = get_dataset(all_globals.dataset_name, all_globals.is_binarized, all_globals.is_resized, all_globals.is_grayscale)
    data_X, data_Y = data_X[:no_samples[0]], data_Y[:no_samples[0]]
    final_train, final_cv, final_test = [], [], [] 
    interpret_train = []
    interpret_test = []
    interpret_cv = []
    for itera in range(3):
        print("Iteration" + str(itera))
        kf = KFold(n_splits=all_globals.no_of_folds, shuffle = True, random_state = None)
        i = 1
        outp_train, outp_cv, outp_test = [], [], []
        for train_index, cross_validation_index in kf.split(data_X):
            print("Cross Validation fold " + str(i))
            #print("TRAIN INDEXES:", train_index, "CROSS_VALIDATION INDEXES:", cross_validation_index)
            model_B = ModelB(all_globals.model_B_epochs, all_globals.batch_size, all_globals.dataset_name, all_globals.learning_rate, all_globals.model_name_B, all_globals.is_binarized, all_globals.is_resized, all_globals.is_grayscale)
            model_A = ModelA(all_globals.model_A_epochs, all_globals.batch_size, all_globals.dataset_name, all_globals.learning_rate, all_globals.model_name_A, all_globals.is_binarized, all_globals.is_resized, all_globals.is_grayscale)
            model_B.init_model()
        
            X_train, X_cross_validation = data_X[train_index], data_X[cross_validation_index]
            Y_train, Y_cross_validation = data_Y[train_index], data_Y[cross_validation_index]
            model_B.set_dataset(X_train, Y_train, test_X, test_Y, X_cross_validation, Y_cross_validation)
            model_B.train_model()
            ##print("Model B trained")
            model_A.set_dataset(X_train, Y_train, test_X, test_Y, X_cross_validation, Y_cross_validation)
            outp_train, outp_cv, outp_test = perform_interpretation(all_globals.mc_dropout, outp_train, outp_cv, outp_test, model_A, model_B)
            i = i + 1
        print("Train ", find_average(outp_train))
        print("CV ", find_average(outp_cv))
        print("Test ", find_average(outp_test))
        final_train.extend(outp_train)
        final_cv.extend(outp_cv)
        final_test.extend(outp_test)
        final_train.append(find_average(outp_train))
        final_cv.append(find_average(outp_cv))
        final_test.append(find_average(outp_test))
        
        interpret_train.append(find_average(outp_train))
        interpret_test.append(find_average(outp_test))
        interpret_cv.append(find_average(outp_cv))

    print("Final Interpretability on CV", find_average(interpret_cv))
    print("Final interpretability on Train", find_average(interpret_train))
    print("Final interpretability on Test", find_average(interpret_test))
    final_train.append(find_average(interpret_train))
    final_cv.append(find_average(interpret_cv))
    final_test.append(find_average(interpret_test))
    
def test_cross_validation(no_samples):
    data_X, data_Y, test_X, test_Y = get_dataset(all_globals.dataset_name, all_globals.is_binarized, all_globals.is_resized, all_globals.is_grayscale)
    data_X, data_Y = data_X[:no_samples[0]], data_Y[:no_samples[0]]
    
    interpret_train = []
    interpret_test = []
    interpret_cv = []
    for itera in range(3):
        model_B = ModelB(all_globals.model_B_epochs, all_globals.batch_size, all_globals.dataset_name, all_globals.learning_rate, all_globals.model_name_B, all_globals.is_binarized, all_globals.is_resized, all_globals.is_grayscale)
        model_A = ModelA(all_globals.model_A_epochs, all_globals.batch_size, all_globals.dataset_name, all_globals.learning_rate, all_globals.model_name_A, all_globals.is_binarized, all_globals.is_resized, all_globals.is_grayscale)
        model_B.init_model()
        X_train, X_cross_validation, Y_train, Y_cross_validation = train_test_split(data_X, data_Y, test_size = 0.2, random_state = 42, stratify = data_Y)
        print("Iteration " + str(itera))
        model_B.set_dataset(X_train, Y_train, test_X, test_Y, X_cross_validation, Y_cross_validation)
        model_B.train_model()
        ##print("Model B trained")
        model_A.set_dataset(X_train, Y_train, test_X, test_Y, X_cross_validation, Y_cross_validation)
        outp_train, outp_cv, outp_test = [], [], []
        outp_train, outp_cv, outp_test = perform_interpretation(all_globals.mc_dropout, outp_train, outp_cv, outp_test, model_A, model_B)
        print("Train", find_average(outp_train))
        print("CV", find_average(outp_cv))
        print("Test", find_average(outp_test))
        interpret_train.append(find_average(outp_train))
        interpret_test.append(find_average(outp_test))
        interpret_cv.append(find_average(outp_cv))
        
    print("Final Interpretability on CV", find_average(interpret_cv))
    print("Final interpretability on Train", find_average(interpret_train))
    print("Final interpretability on Test", find_average(interpret_test))

def test_kfold_cross_validation_stanford40():
    data_X, data_Y, test_X, test_Y, data_X_A, test_X_A = get_dataset('stanford40', all_globals.is_binarized, all_globals.is_resized, all_globals.is_grayscale)
    
    interpret_train = []
    interpret_test = []
    interpret_cv = []
    for itera in range(3):
        kf = KFold(n_splits=all_globals.no_of_folds, shuffle = True, random_state = None)
        outp_train, outp_cv, outp_test = [], [], []
        indices1 = list(kf.split(data_X))
        for i in range(len(indices1)):
            model_B = ModelB(all_globals.model_B_epochs, all_globals.batch_size, all_globals.dataset_name, all_globals.learning_rate, all_globals.model_name_B, all_globals.is_binarized, all_globals.is_resized, all_globals.is_grayscale)
            model_A = ModelA(all_globals.model_A_epochs, all_globals.batch_size, all_globals.dataset_name, all_globals.learning_rate, all_globals.model_name_A, all_globals.is_binarized, all_globals.is_resized, all_globals.is_grayscale)
            model_B.init_model()
            print("Cross Validation fold " + str(i+1))
            #print("TRAIN INDEXES:", train_index, "CROSS_VALIDATION INDEXES:", cross_validation_index)
            train_index, cross_validation_index = indices1[i]
            X_train, X_cross_validation = data_X[train_index], data_X[cross_validation_index]
            Y_train, Y_cross_validation = data_Y[train_index], data_Y[cross_validation_index]
            model_B.set_dataset(X_train, Y_train, test_X, test_Y, X_cross_validation, Y_cross_validation)
            model_B.train_model()
            ##print("Model B trained")
            train_index, cross_validation_index = indices1[i]
            X_train_A, X_cross_validation_A = data_X_A[train_index], data_X_A[cross_validation_index]
            Y_train_A, Y_cross_validation_A = data_Y[train_index], data_Y[cross_validation_index]
            model_A.set_dataset(X_train_A, Y_train_A, test_X_A, test_Y, X_cross_validation_A, Y_cross_validation_A)
            outp_train, outp_cv, outp_test = perform_interpretation(all_globals.mc_dropout, outp_train, outp_cv, outp_test, model_A, model_B)
        print("Train", find_average(outp_train))
        print("CV", find_average(outp_cv))
        print("Test", find_average(outp_test))
        interpret_train.append(find_average(outp_train))
        interpret_test.append(find_average(outp_test))
        interpret_cv.append(find_average(outp_cv))

    print("Final Interpretability on CV", find_average(interpret_cv))
    print("Final interpretability on Train", find_average(interpret_train))
    print("Final interpretability on Test", find_average(interpret_test))