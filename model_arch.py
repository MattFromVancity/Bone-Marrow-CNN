# Main Deep Learning Architecture
# name: model_arch.py
# author: mbwhiteh@sfu.ca
# date: 2022-02-22

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

import numpy as np
from matplotlib import pyplot as plt
import data_utilities as DataUtil

# read training filenames from text file
with open("./train_filenames.txt", 'r+') as train_fd:
    train_fnames = [f_name.strip('\n') for f_name in train_fd.readlines()]
# format train labels
train_labels = [DataUtil.LABEL_DICT[f_name.split('/')[0]] for f_name in train_fnames]

with open("./test_filenames.txt", 'r+') as test_fd:
    test_fnames = [f_name.strip('\n') for f_name in test_fd.readlines()]
# format test labels
test_labels = [DataUtil.LABEL_DICT[f_name.split('/')[0]] for f_name in test_fnames]

# High Level Parameters
batch_size = 16

# training data generator
train_datagen = DataUtil.DataGenerator(train_fnames, train_labels, batch_size)
# testing data generator
test_datagen = DataUtil.DataGenerator(test_fnames, test_labels, batch_size)

# Start of model definition
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(250, 250, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this produces a 1D feature vector
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(21))
model.add(Activation('softmax'))

# optimizer configuration
opt = keras.optimizers.SGD(learning_rate= 0.01, momentum= 0)

model.compile(loss='categorical_crossentropy',
              optimizer= opt,
              metrics=['categorical_accuracy'])

model.summary()

model.fit(train_datagen, batch_size=16, epochs= 5)

