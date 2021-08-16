#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
# matplotlib.use('agg')
import numpy as np 
import pandas as pd 
import os#
from tensorflow import keras
import sklearn
# from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,Flatten
# from keras.models import Sequential
# from keras.preprocessing.image import ImageDataGenerator
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings
from sklearn.metrics import confusion_matrix
# from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.regularizers import l2
from tensorflow.keras import activations
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import sys
import skimage
from skimage.color import rgb2gray
import fnmatch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import shutil
import glob
import re
print(tf.__version__)

path1  = '/TILES_85/CANCER'
path2  = '/TILES_85/NORMAL'
path3  = '/TILES_85/INFLAMMATION'

pattern = re.compile(r'Cancer')
pattern2 = re.compile(r'Normal')
pattern3 = re.compile(r'Inflammation')
pattern4 = re.compile(r'Inflamation')

labels = []
OHE_labels =[]
for root, dirs, files in os.walk(path1, topdown = False):
    for name in files:
        if pattern.search(name):
            label = "Cancer"
            labels.append(label)
            OH_label = 0
            OHE_labels.append(OH_label)
        else:
            pass
print(len(labels))
for root, dirs, files in os.walk(path2, topdown = False):
    for name in files:    
        if pattern2.search(name):
            label2 = "Normal"
            labels.append(label2)
            OH_label2 = 1
            OHE_labels.append(OH_label2)
        else:
            pass
print(len(labels))
for root, dirs, files in os.walk(path3, topdown = False):
    for name in files:
        if pattern3.search(name):
            label3 = "Inflammation"
            labels.append(label3)
            OH_label3 = 2
            OHE_labels.append(OH_label3)
        elif pattern4.search(name):
            label4 = "Inflammation"
            labels.append(label4)
            OH_label4 = 2
            OHE_labels.append(OH_label4)
        else:
            pass
            
print(len(labels))
print(labels)

label_array = np.array(labels)
print(len(label_array))
print(label_array)
print(np.shape(label_array))

np.save('./TILES_85/MMI_class_labels.npy', label_array)

OHE_array = np.array(OHE_labels)
print(len(OHE_array))
print(OHE_array)
print(np.shape(OHE_array))

np.save('./TILES_85/MMI_OHE_class_labels.npy', OHE_array)


