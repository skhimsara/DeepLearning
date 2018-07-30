#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 13:30:41 2018

@author: sidnpoo
"""

import os
from myutils import *
import os
from tqdm import tqdm
import glob
import cv2
import imutils
import numpy as np
import keras

INPUT_IMAGE_SIZE =64
# The path to the directory where the original
# dataset was uncompressed
original_smiles_dataset_dir = '/Users/sidnpoo/Downloads/MLStuff_DoNotDelete/_DATASETS_/SmileFrownData/SMILEs/positives/positives7'
original_frowns_dataset_dir = '/Users/sidnpoo/Downloads/MLStuff_DoNotDelete/_DATASETS_/SmileFrownData/SMILEs/negatives/negatives7'



from utils import list_all_files
negative_paths = list(list_all_files(original_smiles_dataset_dir, ['.jpg']))
print('loaded', len(negative_paths), 'negative examples')
positive_paths = list(list_all_files(original_frowns_dataset_dir, ['.jpg']))
print ('loaded', len(positive_paths), 'positive examples')
examples = [(path, 0) for path in negative_paths] + [(path, 1) for path in positive_paths]

import numpy as np
from skimage.measure import block_reduce
from skimage.io import imread

def examples_to_dataset(examples, block_size=1):
    X = []
    y = []
    for path, label in examples:
        img = imread(path, as_grey=True)
        img = block_reduce(img, block_size=(block_size, block_size), func=np.mean)
        X.append(img)
        y.append(label)
    return np.asarray(X), np.asarray(y)

%time X, y = examples_to_dataset(examples)

X = X.astype(np.float32) / 255.
y = y.astype(np.int32)
print (X.dtype, X.min(), X.max(), X.shape)
print (y.dtype, y.min(), y.max(), y.shape)

X = np.expand_dims(X, axis=-1)

x_train = X[:10000]
x_test = X[10000:]
y_train = y[:10000]
y_test = y[10000:]

x_train = np.tocat

np.save('X.npy', X)
np.save('y.npy', y)



 #training parameters
batch_size = 128
maxepoches = 250
learning_rate = 0.1
lr_decay = 1e-6
lr_drop = 20
num_classes=2

y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, self.num_classes)



model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',
                        input_shape=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, COLOR_DEPTH_DIM)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.summary()

def lr_scheduler(epoch):
    return learning_rate * (0.5 ** (epoch // lr_drop))
reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

#data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images
    # (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)


#optimization details
sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['accuracy'])


# training process in a for loop with learning rate drop every 25 epoches.

historytemp = model.fit_generator(datagen.flow(x_train, y_train,
                             batch_size=batch_size),
                steps_per_epoch=x_train.shape[0] // batch_size,
                epochs=maxepoches,
                validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=1)
model.save_weights('cifar10vgg.h5')






