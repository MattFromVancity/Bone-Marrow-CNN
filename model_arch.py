# Main Deep Learning Architecture
# name: model_arch.py
# author: mbwhiteh@sfu.ca
# date: 2022-02-22

from gc import callbacks
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.regularizers import l2 as L2Norm
from keras.metrics import CategoricalAccuracy

import data_utilities as DataUtil
import random

# set the seed
random.seed(11)

# read training filenames from text file
with open("./train_filenames.txt", 'r+') as train_fd:
    train_fnames = [f_name.strip('\n') for f_name in train_fd.readlines()]
    # shuffle the training data
    random.shuffle(train_fnames)
# format train labels
train_labels = [DataUtil.LABEL_DICT[f_name.split('/')[0]] for f_name in train_fnames]

with open("./test_filenames.txt", 'r+') as test_fd:
    test_fnames = [f_name.strip('\n') for f_name in test_fd.readlines()]
# format test labels
test_labels = [DataUtil.LABEL_DICT[f_name.split('/')[0]] for f_name in test_fnames]

# High Level Parameters
batch_size = 32
lr = 0.002
epohcs = 5

initializer = keras.initializers.HeNormal()

# training data generator
train_datagen = DataUtil.DataGenerator(train_fnames, train_labels, batch_size)
# testing data generator
test_datagen = DataUtil.DataGenerator(test_fnames, test_labels, batch_size)

# Start of model definition
model = Sequential()
model.add(Conv2D(256, (3, 3), input_shape=(250, 250, 3), kernel_initializer= initializer))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(128, (3, 3), kernel_initializer= initializer))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(64, (3, 3), ))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), padding="same"))
model.add(Flatten())  # this produces a 1D feature vector
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.50))
model.add(Dense(21))
model.add(Activation('softmax'))

# optimizer configuration
opt = keras.optimizers.Adam(learning_rate= lr)

model.compile(loss='categorical_crossentropy',
              optimizer= opt,
              metrics=[CategoricalAccuracy()])

model.summary()

batch_logger = DataUtil.BatchLogger('./training_stats/batch_log.csv')

model.fit(train_datagen, batch_size = batch_size, epochs= epohcs, 
    validation_data= test_datagen, callbacks= [batch_logger], use_multiprocessing= True)

# save the model
model.save('prelim_cnn.h5')