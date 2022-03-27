from tensorflow import keras
import numpy as np
import skimage
import matplotlib.pyplot as plt
model = keras.models.load_model('./Model-Version-20220318.hdf5')

# img path
path = "/Users/matt/Documents/Research/E413-Deep-Learning/BMC-Dataset/NGS/NGS_26683.jpg"
# load image
img = keras.utils.load_img(path, target_size=(250, 250))
img_resample = keras.utils.load_img(path, target_size=(128, 128))
img_resample_v2 = keras.utils.load_img(path, target_size=(64, 64))
image_data = keras.utils.img_to_array(img)*(1/255)
image_data_resample = keras.utils.img_to_array(img_resample)*(1/255)
image_data_resample_v2 = keras.utils.img_to_array(img_resample_v2)*(1/255)
keras.utils.save_img('./sample-resize-128.png', image_data_resample)
keras.utils.save_img('./sample-resize-64.png', image_data_resample_v2)