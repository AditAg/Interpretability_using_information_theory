import os
import gzip
import numpy as np
import cv2
import scipy.io
from keras.datasets import fashion_mnist
from sklearn.preprocessing import StandardScaler
from xml.etree import ElementTree as ET
import pickle
from sklearn.model_selection import train_test_split

def unpickle(file):
 '''Load byte data from file'''
 with open(file, 'rb') as f:
  data = pickle.load(f, encoding='latin-1')
  return data

def convert_to_grayscale(array, shape):
    new_array = np.zeros((array.shape[0],) + shape)
    for i in range(array.shape[0]):
        new_array[i] = cv2.cvtColor(array[i].astype('float32'), cv2.COLOR_RGB2GRAY)
        
    return new_array

def scale_input(x):
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return x
    
def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val)/(max_val - min_val)
    return x
        
def make_binarized(y_train, y_test):
    init_shape = y_train.shape[0]
    Y = np.concatenate((y_train, y_test), axis = 0).astype(np.int)
    Y_binarized = convert_to_binary_classes(Y)
    print(Y_binarized.shape)
    y_train = Y_binarized[:init_shape]
    y_test = Y_binarized[init_shape:]
    return y_train, y_test

def shuffle_dataset(x_train, y_train, x_test, y_test):
    seed = 333
    np.random.seed(seed)
    np.random.shuffle(x_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)
    seed = 111
    np.random.seed(seed)
    np.random.shuffle(x_test)
    np.random.seed(seed)
    np.random.shuffle(y_test)
    return x_train, y_train, x_test, y_test

def resize_images(x_train, x_test, new_shape):
    x_train = np.array(x_train, dtype = 'uint8')
    new_x_train = np.zeros((x_train.shape[0], ) + new_shape)
    for i in range(x_train.shape[0]):
        new_img = cv2.resize(x_train[i], dsize = new_shape, interpolation = cv2.INTER_CUBIC)
        new_x_train[i] = new_img
        
    x_test = np.array(x_test, dtype = 'uint8')
    new_x_test = np.zeros((x_test.shape[0],) + new_shape)
    for i in range(x_test.shape[0]):
        new_img = cv2.resize(x_test[i], dsize = new_shape, interpolation = cv2.INTER_CUBIC)
        new_x_test[i] = new_img
        
    new_x_train = new_x_train.reshape(new_x_train.shape + (1,))
    new_x_test = new_x_test.reshape(new_x_test.shape + (1,))
    return new_x_train, new_x_test
    
def extract_data(filename, data_len, data_size, head_size):
    with gzip.open(filename) as bytestream:
        bytestream.read(head_size)
        buf = bytestream.read(data_size * data_len)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
    return data


def load_mnist_dataset(is_binarized, is_resized):
    file_path = os.path.join('data', 'Original dataset')
    x_train = extract_data(os.path.join(file_path, 'train-images-idx3-ubyte.gz'), 60000, 28 * 28, 16)
    y_train = extract_data(os.path.join(file_path, 'train-labels-idx1-ubyte.gz'), 60000, 1, 8)
    x_test = extract_data(os.path.join(file_path, 't10k-images-idx3-ubyte.gz'), 10000, 28 * 28, 16)
    y_test = extract_data(os.path.join(file_path, 't10k-labels-idx1-ubyte.gz'), 10000, 1, 8)
    x_train, x_test = x_train / 255, x_test / 255
    x_train = (x_train > 0.5).astype(np.int_)
    x_test = (x_test > 0.5).astype(np.int_)
    x_train = x_train.reshape((60000, 28, 28, 1))
    y_train = y_train.reshape((60000))
    x_test = x_test.reshape((10000, 28, 28, 1))
    y_test = y_test.reshape((10000))
    
    if (is_binarized):
        y_train, y_test = make_binarized(y_train, y_test)
    
    x_train, y_train, x_test, y_test = shuffle_dataset(x_train, y_train, x_test, y_test)
    
    if(is_resized):
        new_x_train, new_x_test = resize_images(x_train, x_test, (10, 10))
        return new_x_train, y_train, new_x_test, y_test
    return x_train, y_train, x_test, y_test
    
def load_fashionmnist(is_binarized, is_resized):
    ((x_train, y_train), (x_test, y_test)) = fashion_mnist.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0
    x_train = x_train.reshape((60000, 28, 28, 1))
    y_train = y_train.reshape((60000))
    x_test = x_test.reshape((10000, 28, 28, 1))
    y_test = y_test.reshape((10000))
    if(is_binarized):
        y_train, y_test = make_binarized(y_train, y_test)
    
    x_train, y_train, x_test, y_test = shuffle_dataset(x_train, y_train, x_test, y_test)
    
    if(is_resized):
        x_train, x_test = resize_images(x_train, x_test, (10, 10))
    
    return x_train, y_train, x_test, y_test


#Makes the maximally present class as 1 and the rest as 0.
def convert_to_binary_classes(labels_array):
    my_dict = {}
    for label in labels_array:
        if label not in my_dict.keys():
            my_dict[label] = 1
        else:
            my_dict[label] += 1
    max_label = max(my_dict, key=my_dict.get)
    new_labels_array = labels_array
    for i in range(len(labels_array)):
        if (labels_array[i] == max_label):
            new_labels_array[i] = 1
        else:
            new_labels_array[i] = 0
    return new_labels_array

def find_average(l):
    return sum(l)/len(l)

def convert_to_all_classes_array(probs, classes, final_no_classes):
    assert(probs.shape[-1] == len(classes))
    new_probs = np.zeros((probs.shape[0], final_no_classes), float)
    for i in range(probs.shape[0]):
        for j in range(len(classes)):
            new_probs[i][classes[j]] = probs[i][j]
    
    return new_probs

def load_svhn(grayscale):
    path2 = os.path.join('data', 'SVHN')
    path2 = os.path.join(path2, 'extra_32x32.mat')
    train_data = scipy.io.loadmat(path2)
    #print("Loaded SVHN dataset")
    X = train_data['X']
    X = np.rollaxis(X, 3)
    Y = train_data['y']
    Y[Y==10] = 0      #SVHN has classes from 1 to 10 where 10 is for class '0'
    X = X[:10000]
    Y = Y[:10000]
    if(grayscale):
        X = convert_to_grayscale(X, (32, 32))
    return X, Y

def parseXML(xmlfile):
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    for child in root:
        if(child.tag == 'object'):
            y_label = child.find('action').text
            xmax = child.find('./bndbox/xmax').text
            xmin = child.find('./bndbox/xmin').text
            ymax = child.find('./bndbox/ymax').text
            ymin = child.find('./bndbox/ymin').text
    #print(y_label, "xmin = ", xmin, "xmax = ", xmax, "ymin = ", ymin, "ymax = ", ymax)
    return y_label, int(xmin), int(xmax), int(ymin), int(ymax)
    
def get_annotation_file_name_from_img(img_name):
    file_path = os.path.join("datasets", "Stanford40")
    annotations_file_path = os.path.join(file_path, "XMLAnnotations")
    img_parts = img_name.split('.')
    img_name_parts = img_parts[0].split('_')
    length = len(img_name_parts)
    if(len(img_name_parts[length - 1]) == 1):
        img_name_parts = img_name_parts[:-1]
    img_name_new = "_"
    img_name_new = img_name_new.join(img_name_parts)
    annotation_file = os.path.join(annotations_file_path, img_name_new) + ".xml"
    return parseXML(annotation_file)
    
def get_all_classes(path):
    dict_classes = {}
    i = 0
    for files in os.walk(path):
        for file in files[2]:
            img_name = file.split('.')[0]
            class_name = img_name.split('_')[:-1]
            class_separator = "_"
            class_name = class_separator.join(class_name)
            if(class_name not in dict_classes.keys()):
                dict_classes[class_name] = i
                i = i + 1
    return dict_classes
  
def load_stanford40_dataset(crop_region = 'object'):
    data_X = []
    data_Y = []
    data_X_2 = []
    file_path = os.path.join("datasets", "Stanford40")
    images_file_path_1 = os.path.join(file_path, "JPEGImages")
    # This path contains cropped images containing the actual object where cropping is done manually instead of using the annotation file provided with the Stanford40 dataset.
    images_file_path_2 = os.path.join(file_path, "Stanford40ImagesCropped")
    dict_classes = get_all_classes(images_file_path_1)
    for i in os.walk(images_file_path_1):
        no_images = 1
        for file in i[2]:
            if(no_images > 1000):
                break
            no_images = no_images + 1
            img = cv2.imread(os.path.join(images_file_path_1, file))
            ylabel, xmin, xmax, ymin, ymax = get_annotation_file_name_from_img(file)
            y = dict_classes[ylabel]
            height = img.shape[0] - (ymax - ymin)
            if(height <= 10):
            	height = int(img.shape[0]/2)
            width = img.shape[1] - (xmax - xmin)
            if(width <=10):
            	width = int(img.shape[1]/2)
            cropped_image = img[ymin:ymax, xmin:xmax, :]
            cropped_image_2 = img[:(ymax-ymin), :(xmax-xmin),:]
            cropped_image_3 = img[height:, width:, :]
            new_image = cv2.resize(img, (200, 200), interpolation = cv2.INTER_AREA)
            new_cropped_image = cv2.resize(cropped_image, (200, 200))
            new_cropped_image_2 = cv2.resize(cropped_image_2, (200, 200))
            new_cropped_image_3 = cv2.resize(cropped_image_3, (200, 200))
            x = np.ones((10, 200, 3)).astype(np.uint8)*255
            
            # Concatenation for obtaining Stanfor40 image as present in paper.
            vertic_concat_1 = np.concatenate((new_cropped_image_2, x), axis = 0)
            vertic_concat_1 = np.concatenate((vertic_concat_1, new_cropped_image), axis = 0)
            
            vertic_concat_2 = np.concatenate((new_image, x), axis = 0)
            vertic_concat_2 = np.concatenate((vertic_concat_2, new_cropped_image_3), axis = 0)
            
            horiz_concat = np.concatenate((vertic_concat_1, np.ones((vertic_concat_1.shape[0], 10, vertic_concat_1.shape[2])).astype(np.uint8)*255), axis = 1)
            horiz_concat = np.concatenate((horiz_concat, vertic_concat_2), axis = 1)
            # cv2.imshow('concat_image', horiz_concat); cv2.waitKey(0); cv2.destroyAllWindows()
            data_X.append(new_image)
            
            if(crop_region == 'object'):
                data_X_2.append(new_cropped_image)
            elif(crop_region == 'top_left'):
                data_X_2.append(new_cropped_image_2)
            elif(crop_region == 'bottom_right'):
                data_X_2.append(new_cropped_image_3)
            else:
                print("Invalid crop region")
                return
            data_Y.append(y)
            
    nump_data_X = np.array(data_X)
    nump_data_X_2 = np.array(data_X_2)
    nump_data_Y = np.array(data_Y)
    new_data = np.hstack((nump_data_X, nump_data_X_2))
    
    data_X, test_X, data_Y, test_Y = train_test_split(new_data, nump_data_Y, test_size = 0.2, random_state = 10)
    data_X, data_Y, test_X, test_Y = shuffle_dataset(data_X, data_Y, test_X, test_Y)
    data_X, data_X_2 = np.split(data_X, 2, axis = 1)
    test_X, test_X_2 = np.split(test_X, 2, axis = 1)
    print(data_X.shape, data_X_2.shape, test_X.shape, test_X_2.shape, data_Y.shape, test_Y.shape)
    
    return data_X, data_Y, test_X, test_Y, data_X_2, test_X_2