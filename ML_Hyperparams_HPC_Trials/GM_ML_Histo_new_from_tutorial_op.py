#!/usr/bin/env python

#Copied from GM_ML_Histo_new_from_tutorial_op-nadam_lr-0.01_batch16_1000_epochs.py

from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
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
print(tf.__version__)

### Input argument code added by Dirk ###
import sys

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

optimizer='adadelta' 
lr=0.01

print(f"First arguemtn is optimzer: nadam adam adamax adagrad adadelta")

if len(sys.argv) > 1:
    optimizer=sys.argv[1] 
if len(sys.argv) > 2:
    lr = float(sys.argv[2])

print(f"{lr=}")
print(f"{optimizer=}")

### End input argument code ###


data = pd.read_csv("../colorectal-histology-mnist/hmnist_64_64_L.csv")

#### Preprocessing to train ####
Y = data["label"]
data.drop(["label"],axis=1, inplace=True)
X = data
# print('Shape of X',np.shape(X))
# print('Shape of Y',np.shape(Y))
## Split the Data ##
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size = 0.2, random_state=42)
print('Shape of x_train',np.shape(x_train))
# print('Shape of x_val',np.shape(x_val))
# print('Shape of y_train',np.shape(y_train))
# print('Shape of y_val',np.shape(y_val))

train_images_a = np.array(x_train)
train_images = train_images_a.reshape((4000,64,64))
train_labels = np.array(y_train)

test_images_a = np.array(x_val)
test_images = test_images_a.reshape((1000,64,64))
test_labels = np.array(y_val)

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical

print(train_labels)
class_names = ['Tumor','Stroma','Complex','Lympho','Debris', 'Mucosa', 'Adipose', 'Empty']

def encoder(integer_encoded):
    global inverted
    values = np.array(class_names)
    #integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    print(integer_encoded)
    #binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print(onehot_encoded)
    #invert first example
    inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
    print(inverted)

print(np.shape(train_images))
print(len(train_labels))

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
# plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

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
    elif train_labels[i] == 3:
        plt.ylabel(train_labels[i])
        plt.xlabel(class_names[3])
    elif train_labels[i] == 4:
        plt.ylabel(train_labels[i])
        plt.xlabel(class_names[4])
    elif train_labels[i] == 5:
        plt.ylabel(train_labels[i])
        plt.xlabel(class_names[5])
    elif train_labels[i] == 6:
        plt.ylabel(train_labels[i])
        plt.xlabel(class_names[6])
    elif train_labels[i] == 7:
        plt.ylabel(train_labels[i])
        plt.xlabel(class_names[7])
    else:
        plt.ylabel(0)
        plt.xlabel(class_names[0])
# plt.show()

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
    keras.layers.Conv2D(32,kernel_size = (3,3), input_shape = (64,64,1),activation = tf.nn.relu, padding = "valid"),
    keras.layers.MaxPooling2D(pool_size = (2,2)),
    keras.layers.Conv2D(32,kernel_size = (3,3),  activation = tf.nn.relu, padding = "same"),
    keras.layers.MaxPooling2D(pool_size = (2,2)),
    keras.layers.Dropout(0.10),
    keras.layers.Conv2D(64,kernel_size = (3,3), activation = tf.nn.relu, padding = "same"),
    keras.layers.MaxPooling2D(pool_size = (2,2)),
    keras.layers.Dropout(0.20),
    keras.layers.Flatten(input_shape=(64, 64)),
    keras.layers.Dense(128, activation=tf.nn.relu,kernel_regularizer = l2(0.001)),
    keras.layers.Dropout(0.50),
    keras.layers.Dense(128, activation=tf.nn.relu,kernel_regularizer = l2(0.001)),
    keras.layers.Dropout(0.50),
    keras.layers.Dense(9, activation=tf.nn.softmax)
])


if optimizer=='nadam':
    opti_func = keras.optimizers.Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)#lr = 0.002

if optimizer=='adam':
    opti_func = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)#lr = 0.001

if optimizer=='adamax':
    opti_func = keras.optimizers.Adamax(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0) #lr = 0.002

if optimizer=='adagrad':
    opti_func = keras.optimizers.Adagrad(lr=lr, epsilon=1e-6)#lr = 0.01

if optimizer=='adadelta':
    opti_func = keras.optimizers.Adadelta(lr=lr, rho=0.95, epsilon=1e-6)#lr = 1.0

model.compile(optimizer=opti_func, #lr = 0.001
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

train_images = train_images.reshape(4000, 64, 64, 1)
print(np.shape(train_images))

test_images = test_images.reshape(1000, 64, 64, 1)
print(np.shape(test_images))   

import time
start = time.time()
print('Starting to train the Model!')
history = model.fit_generator(gen.flow(train_images, train_labels, batch_size = 16),validation_data=(test_images, test_labels),steps_per_epoch=len(train_images)/16, epochs=1000, verbose = 1)
print('Model has finished training')
end = time.time()
print(end - start)


## SAVING THE MODEL
model.save(f"./Models/Histology_IC_{optimizer}_SCC_lr{lr}_BS16_E1000.h5")
print("Saved model to disk")

model.summary()
# history = model.fit(train_images, train_labels, epochs=5, verbose=1)
print(history.history.keys())

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

predictions = model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])

plot_model(model, to_file=f'./Figures/Opt:{optimizer}_lr:{lr}_model.png')

fig, ax = plt.subplots(1,1, figsize = (10,10))
ax.plot(np.arange(0, 1000), history.history['loss'], label="train_loss")
ax.plot(np.arange(0, 1000), history.history['val_loss'], label="val_loss")
ax.plot(np.arange(0, 1000), history.history['acc'], label="train_acc")
ax.plot(np.arange(0, 1000), history.history['val_acc'], label="val_acc")
ax.set_title("Training Loss and Accuracy on Dataset")
ax.set_xlabel("Epoch #")
ax.set_ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
# plt.show()
fig.savefig(f'./Figures/Training_and_Testing_Accuracy_and_Loss_Opt:{optimizer}:{lr}.png', transparent = True, bbox_inches = "tight")

