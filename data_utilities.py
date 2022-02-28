# Utility functions for dataset processing
# name: data_utilities.py
# author: mbwhiteh@sfu.ca
# date: 2022-02-22

import os
import numpy as np
from tensorflow import keras
import cv2
import skimage
import math
import random

TRAIN_PATH_ABS = os.path.abspath("train_filenames.txt")
DATASET_PATH_ABS = os.path.abspath("BMC-Dataset")
# Dataset label mappings for one-hot encoding
LABEL_DICT = {
    'ABE': 0,
    'ART': 1,
    'BAS': 2,
    'EBO': 3,
    'EOS': 4,
    'FGC': 5,
    'HAC': 6,
    'KSC': 7,
    'LYI': 8,
    'LYT': 9,
    'MMZ': 10,
    'MON': 11,
    'MYB': 12,
    'NGB': 13,
    'NGS': 14,
    'NIF': 15,
    'OTH': 16,
    'PEB': 17,
    'PLM': 18,
    'PMO': 19,
    'BLA': 20}

# Data Generator class for batch training 
class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, X_set, Y_set, batch_size= 32, num_classes= 21):
        self.X_set = X_set
        self.Y_set = Y_set
        self.batch_size = batch_size
        self.class_count = num_classes

    # returns the length of the sequence
    def __len__(self):
        return math.ceil(len(self.X_set) / self.batch_size)
    
    # returns the batch of data and labels
    def __getitem__(self, idx):
        batch_of_filenames = self.X_set[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_of_labels = self.Y_set[idx*self.batch_size:(idx+1)*self.batch_size]

        batch_X = list()
        # load batch of images
        for filepath in batch_of_filenames:
            abs_filepath = os.path.join(DATASET_PATH_ABS, filepath)
            try:
                img = skimage.io.imread(abs_filepath)
                image_data = img / 255.00
                batch_X.append(image_data)
            except:
                print(abs_filepath)

        return (np.array([ img for img in batch_X]), keras.utils.to_categorical(batch_of_labels, num_classes= self.class_count))

# testing of Data Generator class
if __name__ == "__main__":
    # open training filenames
    with open('train_filenames.txt') as train_fd:
        train_filenames = [f_name.strip('\n') for f_name in train_fd.readlines()]
        random.shuffle(train_filenames)
    # associated labels mapped to integer values
    train_labels = [LABEL_DICT[f_name.split('/')[0]] for f_name in train_filenames]

    data_gen = DataGenerator(train_filenames, train_labels, batch_size= 32)
    batch_x, batch_y = data_gen[0]
    # numpy array of three 250 x 250 channels
    print(batch_x[0])
    # result should be one hot encoded
    print(batch_y[0])
    print(train_filenames[0])