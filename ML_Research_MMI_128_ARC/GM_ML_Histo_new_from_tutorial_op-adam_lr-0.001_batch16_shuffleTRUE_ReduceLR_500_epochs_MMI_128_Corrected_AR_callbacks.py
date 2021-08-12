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
import random
# from sklearn.preprocessing import OneHotEncoder
print(tf.__version__)

print("TF version", tf.__version__)
print("Numpy version", np.__version__)
print("Pandas version", pd.__version__)
print("Keras version", keras.__version__)
print("Sklearn version", sklearn.__version__)
print("Skimage version", skimage.__version__)


#
input_runname = sys.argv[1]
CSV_filename = sys.argv[2]
log_D = sys.argv[3]
#

# input_runname = 'run_64_AC_Corrected'
# CSV_filename = './TILES/epoch_1'
# log_D = './TILES/tensorboard_log1'

paths  = './TILES_128_AR_Corrected/'

for root, dirs, files in os.walk(paths):
    for files in dirs:
        if files.endswith((".png")):
            print(files)
            ### Giving each of the found image files their direct path name so that they can be referenced 
imgfiles = [os.path.join(root, name)
                         for root, dirs, files in os.walk(paths)
                         for name in files
                         if name.endswith((".png"))]
# print(imgfiles)
datafiles = []
for name in imgfiles:
    images = load_img(name)
    img_array = img_to_array(images)
    gray = rgb2gray(img_array) 
    # print(np.shape(img_array))
    # print(np.shape(gray))
    datafiles.append(gray)
    
print(np.shape(datafiles))

data = datafiles

# for column in data.columns:
#     if data[column].dtype == type(object):
#         le = LabelEncoder()
#         data[column] = le.fit_transform(data[column])
#
        
### Preprocessing to train ####
Y = np.load('./TILES_128_AR_Corrected/MMI_OHE_class_labels.npy')

X = data
print('Shape of X',np.shape(X))
print('Shape of Y',np.shape(Y))
## Split the Data ##
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size = 0.2, random_state=42)
# np.mean(y_train)
print('Shape of x_train',np.shape(x_train))
print('Shape of x_val',np.shape(x_val))
print('Shape of y_train',np.shape(y_train))
print('Shape of y_val',np.shape(y_val))
#
train_images_a = np.array(x_train)
train_images = train_images_a.reshape((648, 128,128))
train_labels = np.array(y_train)

test_images_a = np.array(x_val)
test_images = test_images_a.reshape((162, 128,128))
test_labels = np.array(y_val)
#
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
#
# print(train_labels)
class_names = ['Cancer', 'Normal', 'Inflammation']

plt.figure()
plt.imshow(random.choice(train_images))
plt.colorbar()
plt.grid(False)
# plt.show()
plt.savefig('./MMI_Figures_ARC/Random_Trained_Image_'+input_runname+'.png', bbox_inches = "tight", transparent = True)
#
train_images = train_images / 255.0
test_images = test_images / 255.0
#
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    if train_labels[i] == 0:
        plt.ylabel(train_labels[i])
        plt.xlabel(class_names[0])
    elif train_labels[i] == 1:
        plt.ylabel(train_labels[i])
        plt.xlabel(class_names[1])
    elif train_labels[i] == 2:
        plt.ylabel(train_labels[i])
        plt.xlabel(class_names[2])
    # elif train_labels[i] == 3:
  #       plt.ylabel(train_labels[i])
  #       plt.xlabel(class_names[3])
  #   elif train_labels[i] == 4:
  #       plt.ylabel(train_labels[i])
  #       plt.xlabel(class_names[4])
  #   elif train_labels[i] == 5:
  #       plt.ylabel(train_labels[i])
  #       plt.xlabel(class_names[5])
  #   elif train_labels[i] == 6:
  #       plt.ylabel(train_labels[i])
  #       plt.xlabel(class_names[6])
  #   elif train_labels[i] == 7:
  #       plt.ylabel(train_labels[i])
  #       plt.xlabel(class_names[7])
    else:
        plt.ylabel(0)
        plt.xlabel(class_names[0])
# plt.show()
plt.savefig('./MMI_Figures_ARC/Images_Viewed_'+input_runname+'.png', bbox_inches = "tight", transparent = True)

gen = ImageDataGenerator(rotation_range=15,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               # shear_range=0.01,
                               # zoom_range=[0.9, 1.25],
                               horizontal_flip=True,
                               vertical_flip=True,
                               fill_mode='reflect',
                               data_format='channels_last')
                               # brightness_range=[0.5, 1.5])

model = keras.Sequential([
    keras.layers.Conv2D(32,kernel_size = (3,3), input_shape = (128,128,1),activation = tf.nn.relu, padding = "valid"),
    keras.layers.MaxPooling2D(pool_size = (2,2)),
    keras.layers.Conv2D(32,kernel_size = (3,3),  activation = tf.nn.relu, padding = "same"),
    keras.layers.MaxPooling2D(pool_size = (2,2)),
    keras.layers.Dropout(0.10),
    keras.layers.Conv2D(64,kernel_size = (3,3), activation = tf.nn.relu, padding = "same"),
    keras.layers.MaxPooling2D(pool_size = (2,2)),
    keras.layers.Dropout(0.20),
    keras.layers.Flatten(input_shape=(64,64)),
    keras.layers.Dense(128, activation=tf.nn.relu,kernel_regularizer = l2(0.001)),
    keras.layers.Dropout(0.50),
    keras.layers.Dense(128, activation=tf.nn.relu,kernel_regularizer = l2(0.001)),
    keras.layers.Dropout(0.50),
    keras.layers.Dense(4, activation=tf.nn.softmax)
])

adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8,decay = 0.0, amsgrad = True)#lr = 0.001
adamax = keras.optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0) #lr = 0.002
nadam = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)#lr = 0.002
adagrad = keras.optimizers.Adagrad(lr=0.001, epsilon=1e-6)#lr = 0.01
adadelta = keras.optimizers.Adadelta(lr=0.001, rho=0.95, epsilon=1e-6)#lr = 1.0

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.compile(optimizer=adam,
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.compile(optimizer=adamax, #lr = 0.001
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
model.compile(optimizer=adam, #lr = 0.001
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

train_images = train_images.reshape(648, 128,128, 1)
print(np.shape(train_images))

test_images = test_images.reshape(162, 128,128, 1)
print(np.shape(test_images))

import time
start = time.time()
print('Starting to train the Model!')

### CALLBACKS ###
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=10,mode = 'auto', min_lr=0.0001)#### reduces the LR when the metrics plateau
csv_logger = CSVLogger(CSV_filename)### saves all of the epoch metrics to a CSV file
tensorboard = keras.callbacks.TensorBoard(log_dir=log_D, histogram_freq=10, batch_size=16, write_graph=True, write_grads=True, write_images=True)
#callbacks=[reduce_lr, csv_logger, tensorboard],
history = model.fit_generator(gen.flow(train_images, train_labels, batch_size = 16),callbacks = [reduce_lr, csv_logger, tensorboard],validation_data=(test_images, test_labels),steps_per_epoch=len(train_images)//16, validation_steps=len(test_images)//16,shuffle = True, epochs=500, verbose = 1)
print('Model has finished training')
end = time.time()
print(end - start)

# print("Model Metrics")
# print(history.history())

## SAVING THE MODEL
model.save("./MMI_Models_ARC/Histology_IC_adam_SCC_lr-0.001_BS16_E400__NoAug_"+input_runname+".h5")
print("Saved model to disk")

model.summary()
# history = model.fit(train_images, train_labels, epochs=5, verbose=1)
print(history.history.keys())

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# plot_model(model, to_file='./MMI_Figures_ARC/Opt-Adam_lr-0.001_model__NoAug_'+input_runname+'.png')

predictions = model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])

# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model accuracy and loss')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train Acc', 'Test Acc', 'Tran Loss', 'Val Loss'], loc='upper left')
# # plt.show()
# plt.savefig('./TILES/MMI_Plots/Training_and_Testing_Accuracy_and_Loss_Opt-Adam_lr-0.001_NoAug_'+input_runname+'.png', transparent = True, bbox_inches = "tight")


fig, ax = plt.subplots(1,1, figsize = (10,10))
ax.plot(np.arange(0, 500), history.history['loss'], label="train_loss")
ax.plot(np.arange(0, 500), history.history['val_loss'], label="val_loss")
ax.plot(np.arange(0, 500), history.history['acc'], label="train_acc")
ax.plot(np.arange(0, 500), history.history['val_acc'], label="val_acc")
ax.set_title("Training Loss and Accuracy on Dataset")
ax.set_xlabel("Epoch #")
ax.set_ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
# plt.show()
plt.savefig('./MMI_Plots_ARC/Training_and_Testing_Accuracy_and_Loss_Opt-Adam_lr-0.001_NoAug_'+input_runname+'.png', transparent = True, bbox_inches = "tight")

#
