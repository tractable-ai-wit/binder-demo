import pandas as pd
import tensorflow as tf
import zipfile

data_splits = {
                    'train': {'images_location': './datasets/cars_train_resized/', 
                              'label_mapping': 'train_labels.csv'},
                    'val': {'images_location': './datasets/cars_train_resized/', 
                              'label_mapping': 'val_labels.csv'},
                    'test': {'images_location': './datasets/cars_test_resized/', 
                              'label_mapping': 'test_labels.csv'}
                }

def get_data(split_name):
    data_locations = data_splits[split_name]
    mapping = pd.read_csv(data_locations['label_mapping'])
    mapping['filename'] = data_locations['images_location'] + mapping['filename']
    image_paths = tf.convert_to_tensor(mapping['filename'])
    labels = tf.convert_to_tensor(mapping['label'])
    return image_paths, labels

def extract_data():
    data_locations = ['./datasets/cars_test_resized.zip', 
                      './datasets/cars_train_resized.zip']
    for file in data_locations:
        with zipfile.ZipFile(file) as z:
            z.extractall('./datasets/')
    print('Data extracted!')