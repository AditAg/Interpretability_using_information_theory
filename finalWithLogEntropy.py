from base_functions import test_kfold_cross_validation, test_cross_validation, test_kfold_cross_validation_stanford40
from all_globals import dataset_name


# Main function defining the actual function to be called based on the required experimentation.
def main():
    
    # list of samples to be considered for training
    # This is used for all experiments where different % of datasets are considered.
    no_samples_all = (50000, )
    
    for sample in no_samples_all:
        print("No of Samples:", sample)
        no_samples = (sample, )
        if(dataset_name == 'stanford40'):
            test_kfold_cross_validation_stanford40()
        else:
            test_kfold_cross_validation(no_samples)

if __name__ == '__main__':
    main()
