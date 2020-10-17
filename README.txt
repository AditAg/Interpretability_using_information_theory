The following files are present in the folder:
1. all_globals.py -> defines all the global parameters such as dataset name, model name etc. which are used for interpretability calculation. The use of each Individual field is defined in the file itself.

2. finalWithLogEntropy.py -> base file which is to be run.

3. model_classes.py -> contains the model classes for models A and B, containing the entropy calculation logic and model configurations.

4. ops.py -> contains functions and classes implementing all basic supervised learning model architectures.

5. base_functions.py -> function implementing the Kfold cross validation for different experiments.

6. fns.py -> contains all basic utility functions like loading datasets etc.

7. data folder -> contains SVHN, MNIST datasets.

8. datasets.zip -> contains Stanford40 dataset.

Instructions for running:
1. Unzip datasets.zip 

2. Modify the global parameters as per the experiment in all_globals.py

3. Set no_samples_all as the list of no. of samples used for training of both models A and B. Based on the dataset, calculate the number of samples corresponding to 10-100% of the dataset and put the list of samples here.

4. Run python finalWithLogEntropy.py.


To get data:
For MNIST Dataset -> 
a. Download all 4 files from this link: http://yann.lecun.com/exdb/mnist/ 
b. Put the downloaded files in data/Original dataset folder 

For Stanford40 dataset ->
Download the zip file from here and extract it in the root folder.
