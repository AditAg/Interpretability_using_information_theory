import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ops import convert_one_hot, Neural_Network, DecisionTree, CNN, ensemble_model, fit_model_to_initial_dataset, Inceptionv3
from fns import convert_to_all_classes_array
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from all_globals import hidden_layer_list_model_A, hidden_layer_list_model_B, ensemble_list

#KNOWN MODEL
class ModelA(object):
    def __init__(self, epochs, batch_size, dataset_name, learning_rate, model_name, is_binarized, is_resized, is_grayscale):
        self.dataset_name = dataset_name
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.no_epochs = epochs
        self.batch_size = batch_size
        self.is_resized = is_resized
        self.is_grayscale = is_grayscale
        self.is_binarized = is_binarized
        
        if (dataset_name == 'mnist'):
            if(self.is_resized):
                self.input_image_size = (10, 10, 1)
            else:
                self.input_image_size = (28, 28, 1)
            if (is_binarized):
                self.output_classes = 2
            else:
                self.output_classes = 10
            
        elif (dataset_name == 'fashion_mnist'):
            self.input_image_size = (10, 10, 1) if (self.is_resized) else (28, 28, 1)
            self.output_classes = 2 if (self.is_binarized) else 10
            
        elif(dataset_name == 'stanford40'):
            self.input_image_size = (200, 200, 3)
            self.output_classes = 40
        
        else:
            print("Not Implemented yet")

        if (self.model_name == "svm"):
            self.classifier = svm.SVC(C = 1, kernel = 'rbf', gamma = 'auto', probability = True, random_state = 0)
            self.classifier = fit_model_to_initial_dataset(self.dataset_name, self.classifier, self.model_name, self.is_resized, self.is_grayscale)
         
        elif(self.model_name == "naive_bayes"):
            self.classifier = MultinomialNB(alpha = 1.0)
            self.classifier = fit_model_to_initial_dataset(self.dataset_name, self.classifier, self.model_name, self.is_resized, self.is_grayscale) 
             
        elif (self.model_name == "ann"):
            input_size = np.prod(self.input_image_size)
            self.hidden_layers = hidden_layer_list_model_A
            self.NN = Neural_Network(self.no_epochs, self.batch_size, self.learning_rate, input_size,
                                     self.output_classes, self.hidden_layers)
            self.NN.create_tf_model("ModelA")
            
        elif(self.model_name == "dt"):
            self.classifier = DecisionTree('gini', 'best', None, 5)
            self.classifier = fit_model_to_initial_dataset(self.dataset_name, self.classifier, "dt", self.is_resized, self.is_grayscale)
            
        elif(self.model_name == "ensemble"):
            input_size = np.prod(self.input_image_size)
            self.classifier = ensemble_model(input_size, self.no_epochs, self.batch_size, self.output_classes, self.learning_rate, "")
            self.classifier.initialize_ensemble(ensemble_list, self.dataset_name, self.is_resized, self.is_grayscale)
            print(ensemble_list)
            
        elif(self.model_name == "cnn"):
            self.CNN_classifier = CNN(self.input_image_size, self.no_epochs, self.batch_size, self.output_classes, self.learning_rate)
            self.CNN_classifier.initialize_model()
            
        elif(self.model_name == "inceptionv3"):
            self.inception_classifier = Inceptionv3(self.output_classes, self.batch_size, self.no_epochs, self.input_image_size)
            self.inception_classifier.initialize_model()
        
    def set_dataset(self, train_X, train_Y, test_X, test_Y, cv_X, cv_Y):
        self.data_X, self.data_Y, self.test_X, self.test_Y, self.cross_validation_X, self.cross_validation_Y = train_X, train_Y, test_X, test_Y, cv_X, cv_Y

    def calculate_interpretability(self, probs_model_B_train, probs_model_B_cross_validation, probs_model_B_test, probs_B_train_cf = None, probs_B_cv_cf = None, probs_B_test_cf = None):
        if(self.model_name =='ann' or self.model_name == 'svm' or self.model_name == 'dt' or self.model_name == "ensemble" or self.model_name == "naive_bayes"):
            self.data_X = self.data_X.reshape(self.data_X.shape[0], -1)
            self.cross_validation_X = self.cross_validation_X.reshape(self.cross_validation_X.shape[0], -1)
            self.test_X = self.test_X.reshape(self.test_X.shape[0], -1)
            
        initial_entropy_train = self.calculate_entropy(probs_model_B_train, 'initial', 0)
        initial_entropy_cv = self.calculate_entropy(probs_model_B_cross_validation, 'initial', 1)
        initial_entropy_test = self.calculate_entropy(probs_model_B_test, 'initial', 2)
        
        self.initialize_model(probs_model_B_train, probs_model_B_cross_validation)
        
        final_entropy_train = self.calculate_entropy(probs_model_B_train, 'final', 0)
        final_entropy_cv = self.calculate_entropy(probs_model_B_cross_validation, 'final', 1)
        final_entropy_test = self.calculate_entropy(probs_model_B_test, 'final', 2)
        
        print("Initial entropies: ", initial_entropy_train, initial_entropy_cv, initial_entropy_test)
        print("Final entropies: ", final_entropy_train, final_entropy_cv, final_entropy_test)
        print("The initial entropy on Train Dataset is :", initial_entropy_train)
        print("The initial entropy on Cross Validation Dataset is :", initial_entropy_cv)
        print("The final entropy on Train Dataset is : ", final_entropy_train)
        print("The final entropy on Cross Validation Dataset is : ", final_entropy_cv)
        if(initial_entropy_train == 0.0):
            interpret_train = 1.0
        else:
            interpret_train = (initial_entropy_train - final_entropy_train) / initial_entropy_train
        if(initial_entropy_cv == 0.0):
            interpret_cv = 1.0
        else:
            interpret_cv = (initial_entropy_cv - final_entropy_cv) / initial_entropy_cv
        if(initial_entropy_test == 0.0):
            interpret_test = 1.0
        else:
            interpret_test = (initial_entropy_test - final_entropy_test)/initial_entropy_test
        
        #re-initialize the classifier to original state for SVM and Decision Tree.
        if(self.model_name == 'svm' or self.model_name == 'dt' or self.model_name == 'naive_bayes'):
            self.classifier = fit_model_to_initial_dataset(self.dataset_name, self.classifier, self.model_name, self.is_resized, self.is_grayscale)
        elif(self.model_name == "ensemble"):
            self.classifier.fit_model(self.is_resized, self.is_grayscale)
        return interpret_train, interpret_cv, interpret_test

    def initialize_model(self, probs_model_B_train, probs_model_B_cv):
        predictions_model_B_train = np.argmax(probs_model_B_train, axis = -1)
        predictions_model_B_cv = np.argmax(probs_model_B_cv, axis = -1)
        data_train = self.data_X
        data_cv = self.cross_validation_X
            
        if (self.model_name == 'svm' or self.model_name == 'naive_bayes'):
            print(data_train.shape, predictions_model_B_train.shape)
            self.classifier.fit(data_train, predictions_model_B_train)
            print("Fitted the" + self.model_name + " to the Predictions of Model B")

        elif (self.model_name == 'ann'):
            data_train = data_train.reshape(data_train.shape[0], -1)
            print(data_train.shape, predictions_model_B_train.shape, data_cv.shape)
            self.NN.train_model(data_train, predictions_model_B_train, data_cv, predictions_model_B_cv)
            print("Trained the Neural Network Model A on Predictions of Model B")
        
        elif(self.model_name == 'dt'):
            data_train = data_train.reshape(data_train.shape[0], -1)
            data_cv = data_cv.reshape(data_cv.shape[0], -1)
            self.classifier.train_model(data_train, predictions_model_B_train, data_cv, predictions_model_B_cv)
            
        elif(self.model_name == "ensemble"):
            data_train = data_train.reshape(data_train.shape[0], -1)
            data_cv = data_cv.reshape(data_cv.shape[0], -1)
            self.classifier.train_ensemble(data_train, predictions_model_B_train, data_cv, predictions_model_B_cv)
        elif(self.model_name == "cnn"):
            self.CNN_classifier.train_model(data_train, predictions_model_B_train, data_cv, predictions_model_B_cv)
        elif(self.model_name == "inceptionv3"):
            self.inception_classifier.train_model(data_train, predictions_model_B_train, data_cv, predictions_model_B_cv)
        else:
            print("Not implemented yet")

    def calculate_entropy(self, probs_B, name, split):
        preds = np.argmax(probs_B, axis = -1)
        if(split == 0):
            data = self.data_X
            output = self.data_Y
            #print("Train split")
        elif(split == 1):
            data = self.cross_validation_X
            output = self.cross_validation_Y
            #print("Cross Validation split ")
        elif(split == 2):
            data = self.test_X
            output = self.test_Y
            #print("Test split ")
        else:
            #print("Invalid Split Value")
            return None
        if (self.model_name == 'svm' or self.model_name == 'naive_bayes'):
            probs2 = np.array(self.classifier.predict_proba(data))
            probs = convert_to_all_classes_array(probs2, self.classifier.classes_, self.output_classes)
            if(probs2.shape[-1] == self.output_classes):
                assert(probs2.all() == probs.all())
            print("Accuracy of Model A on the current split of the dataset is : ", accuracy_score(output, np.argmax(probs, axis = -1)))
            
        elif (self.model_name == 'ann'):
            probs, _, _, acc = self.NN.get_predictions(data, True, convert_one_hot(output, self.output_classes))  # These are 1X50000 arrays
            print(probs.shape)
            print("Accuracy of Model A on the" + self.dataset_name + "Training Dataset is: " + str(acc))
            probs = np.array(probs)
            if (probs.shape[0] == 1):
                probs = np.squeeze(probs, axis=0)
        
        elif(self.model_name == 'cnn' or self.model_name == 'inceptionv3'):
            if(self.model_name == 'inceptionv3'):
                probs, _, _, acc = self.inception_classifier.get_output(data, True, output)
            else:
                probs, _, _, acc = self.CNN_classifier.get_predictions(data, True, output)
            print("Accuracy of Model A on the" + self.dataset_name + "Training Dataset is: " + str(acc))
            probs = np.array(probs)
            if (probs.shape[0] == 1):
                probs = np.squeeze(probs, axis=0)
                
        elif(self.model_name == 'dt'):
            probs2 = self.classifier.predict_model(data)
            probs = convert_to_all_classes_array(probs2, self.classifier.classes_, self.output_classes)
            if(probs2.shape[-1] == self.output_classes):
                assert(probs2.all() == probs.all())
            
        elif(self.model_name == "ensemble"):
            probs = np.array(self.classifier.predict_ensemble(data, True, convert_one_hot(output, self.output_classes)))
        
        prob_A_indexes = np.argmax(probs, axis=-1)
        print("Accuracy of Model A on the current split of the predictions of Model B of the dataset is: ", accuracy_score(preds, prob_A_indexes))
        
        total_diff = 0.0
        list_diff = []
        for i in range(probs.shape[0]):
            if (prob_A_indexes[i] != preds[i]):
                list_diff.append([i, prob_A_indexes[i], preds[i]])
                    
            val = (abs(probs[i][prob_A_indexes[i]] - probs[i][preds[i]]))
            if(val <= 0):
                total_diff += 0.0
            else:
                total_diff += -1.0 * (math.log2(val))
                    
        total_diff = (total_diff) / (probs.shape[0])
        if (len(list_diff) == 0):
            print("For Model A " + str(self.model_name) + " and Model B, the final predictions on this split of the dataset are same")
        else:
            None
            
        return total_diff

    def dump_data(self, data_file, **kwargs):
        arguments = ''
        for key, val in kwargs.items():
            arguments += str(val['name'])
            arguments += "=kwargs['" + str(key) + "']['val'], "
        arguments = arguments.strip()
        arguments = arguments[:-1]
        eval("np.savez_compressed(data_file, " + arguments + ")")
        print("Data dumped")

#unknown model
class ModelB(object):
    def __init__(self, epochs, batch_size, dataset_name, learning_rate, model_name, is_binarized, is_resized, is_grayscale):
        self.dataset_name = dataset_name
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.no_epochs = epochs
        self.batch_size = batch_size
        self.is_binarized = is_binarized
        self.is_resized = is_resized
        self.is_grayscale = is_grayscale
        
        if (dataset_name == 'mnist'):
            if(self.is_resized):
                self.input_image_size = (10, 10, 1)
            else:
                self.input_image_size = (28, 28, 1)
            if (is_binarized):
                self.output_classes = 2
            else:
                self.output_classes = 10
            
        elif (dataset_name == 'fashion_mnist'):
            self.input_image_size = (10, 10, 1) if (self.is_resized) else (28, 28, 1)
            self.output_classes = 2 if (self.is_binarized) else 10
         
        elif(dataset_name == 'stanford40'):
            self.input_image_size = (200, 200, 3)
            self.output_classes = 40
        
        else:
            print("Not Implemented yet")

        if (self.model_name == 'ann'):
            input_size = np.prod(self.input_image_size)
            self.hidden_layers = hidden_layer_list_model_B
            self.NN = Neural_Network(self.no_epochs, self.batch_size, self.learning_rate, input_size,
                                     self.output_classes, self.hidden_layers, mc_dropout =  False, dropout_rate = None)

        elif (self.model_name == 'svm'):
            self.classifier = svm.SVC(C = 1, kernel = 'rbf', gamma = 'auto', probability = True, random_state = 0)
            
        elif(self.model_name == "naive_bayes"):
            self.classifier = MultinomialNB(alpha = 1.0)
                
        elif(self.model_name == 'dt'):
            self.classifier = DecisionTree('gini', 'best', None, 10)
        elif(self.model_name == 'cnn'):
            self.CNN_classifier = CNN(self.input_image_size, self.no_epochs, self.batch_size, self.output_classes, self.learning_rate)
        elif(self.model_name == 'inceptionv3'):
            self.inception_classifier = Inceptionv3(self.output_classes, self.batch_size, self.no_epochs, self.input_image_size)
        #print("Parameters Initialized")
        

    def set_dataset(self, data_X, data_Y, test_X, test_Y, cross_validation_X, cross_validation_Y):
        self.data_X, self.data_Y, self.test_X, self.test_Y, self.cross_validation_X, self.cross_validation_Y = data_X, data_Y, test_X, test_Y, cross_validation_X, cross_validation_Y
        self.no_batches = len(self.data_X) // (self.batch_size)
            
    def init_model(self):
        if (self.model_name == 'ann'):
            self.NN.create_tf_model("ModelB")

        elif (self.model_name == 'cnn'):
            self.CNN_classifier.initialize_model()

        elif(self.model_name == 'inceptionv3'):
            self.inception_classifier.initialize_model()
        elif (self.model_name == "svm" or self.model_name == "dt" or self.model_name == "naive_bayes"):
            pass
        else:
            print("Not implemented yet")

        ##print("Model Initialized")

    def train_model(self):
        if(self.model_name == 'cnn'):
            self.CNN_classifier.train_model(self.data_X, self.data_Y, self.cross_validation_X, self.cross_validation_Y)
        elif(self.model_name == 'inceptionv3'):
            self.inception_classifier.train_model(self.data_X, self.data_Y, self.cross_validation_X, self.cross_validation_Y)
        else:
            self.data_X = self.data_X.reshape(self.data_X.shape[0], -1)
            self.cross_validation_X = self.cross_validation_X.reshape(self.cross_validation_X.shape[0], -1)
            self.test_X = self.test_X.reshape(self.test_X.shape[0], -1)
            
            if (self.model_name == 'svm' or self.model_name == 'naive_bayes'):
                classes_ = np.unique(self.data_Y)
                for class_ in classes_:
                    indices = (self.data_Y == class_).nonzero()
                    print("Class: ", class_, "No of samples: ", np.array(indices).shape)
                
                classes_ = np.unique(self.test_Y)
                for class_ in classes_:
                    indices = (self.test_Y == class_).nonzero()
                    print("Class: ", class_, "No of samples: ", np.array(indices).shape)
                    
                self.classifier.fit(self.data_X, self.data_Y)
                
            elif (self.model_name == 'ann'):
                self.NN.train_model(self.data_X, self.data_Y, self.cross_validation_X, self.cross_validation_Y)
            
            elif(self.model_name == 'dt'):
                self.classifier.train_model(self.data_X, self.data_Y, self.cross_validation_X, self.cross_validation_Y)
            
            else:
                print("Not yet implemented")

    def get_output(self, mode = 'original'):
        tr_X, tr_Y = self.data_X, self.data_Y
        cv_X, cv_Y = self.cross_validation_X, self.cross_validation_Y
        te_X, te_Y = self.test_X, self.test_Y
            
        if (self.model_name == 'ann'):
            prediction_probs_train, preds_train, _, acc = self.NN.get_predictions(tr_X, True, convert_one_hot(tr_Y, self.output_classes))
            prediction_probs_train = np.array(prediction_probs_train)
            
            prediction_probs_cv, preds_cv, _, acc2 = self.NN.get_predictions(cv_X, True, convert_one_hot(cv_Y, self.output_classes))
            prediction_probs_cv = np.array(prediction_probs_cv)
            
            prediction_probs_test, preds_test, _ , acc3 = self.NN.get_predictions(te_X, True, convert_one_hot(te_Y, self.output_classes))
            prediction_probs_test = np.array(prediction_probs_test)
            
            print("Final Accuracy of Model B on current K-1 folds of the " + self.dataset_name + " Train " + mode + " dataset is :" + str(acc))
            print("Final Accuracy of Model B on the current fold of the " + self.dataset_name + " Cross Validation " + mode + " dataset is :" + str(acc2))
            print("Final Accuracy of Model B on the current fold of the " + self.dataset_name + " Test " + mode + " dataset is :" + str(acc3))
            return prediction_probs_train, prediction_probs_cv, prediction_probs_test

        elif (self.model_name == 'svm' or self.model_name == 'naive_bayes'):
            prediction_probs_train = np.array(self.classifier.predict_proba(tr_X))
            prediction_probs_cv = np.array(self.classifier.predict_proba(cv_X))
            prediction_probs_test = np.array(self.classifier.predict_proba(te_X))
            print("Final Accuracy of Model B on current K-1 folds of the " + self.dataset_name + " train " + mode + " dataset is :" + str(accuracy_score(tr_Y, np.argmax(prediction_probs_train, axis = -1))))
            print("Final Accuracy of Model B on the current fold of the " + self.dataset_name + " Cross Validation " + mode + " dataset is :" + str(accuracy_score(cv_Y, np.argmax(prediction_probs_cv, axis = -1))))
            print("Final Accuracy of Model B on the current fold of the " + self.dataset_name + " Test " + mode + " dataset is :" + str(accuracy_score(te_Y, np.argmax(prediction_probs_test, axis = -1))))
            print("Unique classes in predictions of model B on Test : ", np.unique(np.argmax(prediction_probs_test, axis = -1)))
            return prediction_probs_train, prediction_probs_cv, prediction_probs_test
        
        elif(self.model_name == 'dt'):
            prediction_probs_train = self.classifier.predict_model(tr_X)
            prediction_probs_cv = self.classifier.predict_model(cv_X)
            prediction_probs_test = self.classifier.predict_model(te_X)
            
            print("Final Accuracy of Model B on current fold of" + self.dataset_name + " CV " + mode + " Dataset is :" + str(accuracy_score(cv_Y, np.argmax(prediction_probs_cv, axis = -1))))
            return prediction_probs_train, prediction_probs_cv, prediction_probs_test
            
        elif(self.model_name == 'cnn'):
            prediction_probs_train, preds_train, _, acc = self.CNN_classifier.get_predictions(tr_X, True, tr_Y)
            
            prediction_probs_cv, preds_cv, _, acc2 = self.CNN_classifier.get_predictions(cv_X, True, cv_Y)
            
            print("Final Accuracy of Model B on current K-1 folds of the " + self.dataset_name + " Train " + mode + " dataset is :" + str(acc))
            print("Final Accuracy of Model B on the current fold of the " + self.dataset_name + " Cross Validation " + mode + " dataset is :" + str(acc2))
            
            prediction_probs_test, preds_test, _ = self.CNN_classifier.get_predictions(te_X, False, te_Y)
            
            return prediction_probs_train, prediction_probs_cv, prediction_probs_test
        
        elif(self.model_name == 'inceptionv3'):
            prediction_probs_train, preds_train, _, acc = self.inception_classifier.get_output(tr_X, True, tr_Y)
            prediction_probs_cv, preds_cv, _, acc2 = self.inception_classifier.get_output(cv_X, True, cv_Y)
            
            print("Final Accuracy of Model B on current K-1 folds of the " + self.dataset_name + " Train " + mode + " dataset is :" + str(acc))
            print("Final Accuracy of Model B on the current fold of the " + self.dataset_name + " Cross Validation " + mode + " dataset is :" + str(acc2))
            
            prediction_probs_test, preds_test, _ = self.inception_classifier.get_output(te_X, False, te_Y)
            
            return prediction_probs_train, prediction_probs_cv, prediction_probs_test
        else:
            print("Not yet implemented")
            return None, None, None
        
