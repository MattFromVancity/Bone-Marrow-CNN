# Main Deep Learning Architecture
# name: model_arch.py
# author: mbwhiteh@sfu.ca
# date: 2022-02-22

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.metrics import CategoricalAccuracy

import data_utilities as DataUtil
import random
import math
import os 

OUTPUT_CLASSES = 4
MODEL_CHECKPOINTS_FP = os.path.abspath('./tmp/model_weights.hdf5')

# High Level Parameters
batch_size = 32
lr = 0.001
beta_1 = 0.8
epochs = 5
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
model = Sequential()
model.add(Conv2D(256, (5, 5), input_shape=(250, 250, 3), kernel_initializer= initializer))
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
model.add(Dense(OUTPUT_CLASSES))
model.add(Activation('softmax'))

# optimizer configuration
opt = keras.optimizers.Adam(
    learning_rate=lr,
    beta_1=beta_1
)

model.compile(loss='categorical_crossentropy',
              optimizer= opt,
              metrics=[keras.metrics.CategoricalCrossentropy(), keras.metrics.CategoricalAccuracy()])

model.summary()

batch_logger_cb = DataUtil.BatchLogger(os.path.abspath("./training_stats/run1.csv"))

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
    callbacks= [batch_logger_cb]
)
print(hist.history)

# save the model
model.load_weights(MODEL_CHECKPOINTS_FP)

testing_results = model.evaluate(
    x= test_datagen,
    batch_size= batch_size,    
)

print(testing_results)