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

paths  = '/Users/gabrielleosborn-lipsitz/MOVE_BACK_TO_DESKTOP/ML_Research/TILES_256/'

pattern = re.compile(r'Cancer')
pattern2 = re.compile(r'Normal')
pattern3 = re.compile(r'Inflammation')

for i in range(1):
    new_dir = os.mkdir("/Users/gabrielleosborn-lipsitz/MOVE_BACK_TO_DESKTOP/ML_Research/TILES_256/CANCER/")
    new_dir2 = os.mkdir("/Users/gabrielleosborn-lipsitz/MOVE_BACK_TO_DESKTOP/ML_Research/TILES_256/NORMAL/")
    new_dir3 = os.mkdir("/Users/gabrielleosborn-lipsitz/MOVE_BACK_TO_DESKTOP/ML_Research/TILES_256/INFLAMMATION/")


for root, dirs, files in os.walk(paths, topdown = False):
    for name in files:
        if pattern.search(name):
            shutil.move(os.path.join(root, name), "/Users/gabrielleosborn-lipsitz/MOVE_BACK_TO_DESKTOP/ML_Research/TILES_256/CANCER/")
        elif pattern2.search(name):
            shutil.move(os.path.join(root, name), "/Users/gabrielleosborn-lipsitz/MOVE_BACK_TO_DESKTOP/ML_Research/TILES_256/NORMAL/")
        elif pattern3.search(name):
            shutil.move(os.path.join(root, name), "/Users/gabrielleosborn-lipsitz/MOVE_BACK_TO_DESKTOP/ML_Research/TILES_256/INFLAMMATION/")
        else:
            shutil.move(os.path.join(root, name), "/Users/gabrielleosborn-lipsitz/MOVE_BACK_TO_DESKTOP/ML_Research/TILES_256/INFLAMMATION/")


