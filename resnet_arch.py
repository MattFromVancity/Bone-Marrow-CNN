# Main Deep Learning Architecture
# name: model_arch.py
# author: mbwhiteh@sfu.ca
# date: 2022-02-22

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.regularizers import l2 as L2Norm
from tensorflow.keras.metrics import CategoricalAccuracy

import data_utilities as DataUtil
import random
import math

TRIAL_NUM = 2
OUTPUT_CLASSES = 4

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
    random.shuffle(test_fnames)
# format test labels
test_labels = [DataUtil.LABEL_DICT[f_name.split('/')[0]] for f_name in test_fnames]

# High Level Parameters
batch_size = 32
lr = 0.1
exp_decay = 0.9
epochs = 6
train_valid_split = 0.2

initializer = keras.initializers.HeNormal()

# number of samples for training and validation
train_n = len(train_fnames)
valid_n = math.ceil(train_valid_split*train_n)

# training data generator
train_datagen = DataUtil.DataGenerator(train_fnames[valid_n:train_n], train_labels[valid_n:train_n], batch_size, OUTPUT_CLASSES)
# testing data generator
validation_datagen = DataUtil.DataGenerator(train_fnames[0:valid_n], train_labels[0:valid_n], batch_size, OUTPUT_CLASSES)

# Start of model definition
model = tf.keras.applications.ResNet50(
    include_top = True,
    weights = None,
    input_tensor = None,
    input_shape = (250, 250, 3),
    pooling = None,
    classes = 4
)

# optimizer configuration
opt = keras.optimizers.Adadelta(
    learning_rate=0.1,
    rho=exp_decay
)

model.compile(loss='categorical_crossentropy',
              optimizer= opt,
              metrics=[CategoricalAccuracy()])

model.summary()

batch_logger = DataUtil.BatchLogger(f'./training_stats/bench_{exp_decay}_rho_acc_0.csv')

model.fit(train_datagen, batch_size = batch_size, epochs= epochs, 
    validation_data= validation_datagen, callbacks= [batch_logger], use_multiprocessing= False)

# save the model
model.save('resnet50.h5')