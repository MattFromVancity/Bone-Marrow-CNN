# Main Deep Learning Architecture
# name: model_arch.py
# author: mbwhiteh@sfu.ca
# date: 2022-02-22

from asyncio import FastChildWatcher
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import L2

from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.metrics import CategoricalAccuracy, CategoricalCrossentropy, Precision, Recall, FalseNegatives, TruePositives

import data_utilities as DataUtil
import random
import math
import os 

OUTPUT_CLASSES = 4
MODEL_CHECKPOINTS_FP = os.path.abspath('./tmp/MV3_Weights_20220326.hdf5')

# High Level Parameters
batch_size = 32
lr = 0.003
beta_1 = 0.9
epochs = 50
train_valid_split = 0.2

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

initializer = keras.initializers.HeNormal()

# number of samples for training and validation
train_n = len(train_fnames)
valid_n = math.ceil(train_valid_split*train_n)

# training data generator
train_datagen = DataUtil.DataGenerator(train_fnames[valid_n:train_n], train_labels[valid_n:train_n], batch_size, OUTPUT_CLASSES)
# validation data generator
validation_datagen = DataUtil.DataGenerator(train_fnames[0:valid_n], train_labels[0:valid_n], batch_size, OUTPUT_CLASSES)
# testing data generator
test_datagen = DataUtil.DataGenerator(test_fnames, test_labels, batch_size, OUTPUT_CLASSES)

# Start of model definition
model = Sequential(name= "7_Layer_Regularization_20220321")
model.add(Conv2D(256, (3, 3), input_shape=(64, 64, 3), kernel_initializer= initializer))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), kernel_initializer= initializer))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), kernel_initializer= initializer))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), kernel_initializer= initializer))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Flatten())  # this produces a 1D feature vector
model.add(Dense(64, kernel_initializer= initializer))
model.add(Activation('relu'))
model.add(Dense(32, kernel_initializer= initializer))
model.add(Activation('relu'))
model.add(Dropout(0.40))
model.add(Dense(OUTPUT_CLASSES))
model.add(Activation('softmax'))

# optimizer configuration
opt = keras.optimizers.Adam(
    learning_rate=lr,
    beta_1=beta_1
)
model.compile(loss='categorical_crossentropy',
              optimizer= opt,
              metrics=[
                  CategoricalCrossentropy(),
                  CategoricalAccuracy(),
                  Recall(thresholds=0.50, class_id=0, name='bla-recall'),
                  Recall(thresholds=0.50, class_id=1, name='lyt-recall'),
                  Recall(thresholds=0.50, class_id=2, name='ngb-recall'),
                  Recall(thresholds=0.50, class_id=3, name='ngs-recall'),
                  Precision(thresholds=0.50, class_id=0, name='bla-precision'),
                  Precision(thresholds=0.50, class_id=1, name='lyt-precision'),
                  Precision(thresholds=0.50, class_id=2, name='ngb-precision'),
                  Precision(thresholds=0.50, class_id=3, name='ngs-precision')
                  ])

model.summary()

batch_logger_cb = DataUtil.BatchLogger(os.path.abspath("./Stats/Train_Stats/MV3-20220326.csv"))
early_stopping_cb = EarlyStopping('val_loss', patience=10)

model_save_cb = keras.callbacks.ModelCheckpoint(
    filepath= MODEL_CHECKPOINTS_FP,
    save_weights_only= True,
    monitor= 'val_loss',
    mode= 'min',
    save_best_only= True
)

hist = model.fit(
    x = train_datagen, 
    batch_size = batch_size, 
    epochs= epochs, 
    validation_data= validation_datagen, 
    callbacks= [batch_logger_cb, model_save_cb, early_stopping_cb]
)
print(hist.history)

# save the model
model.load_weights(MODEL_CHECKPOINTS_FP)
model.save('./Models/MV3-20220326.hdf5')

testing_results = model.evaluate(
    x= test_datagen,
    batch_size= batch_size
)