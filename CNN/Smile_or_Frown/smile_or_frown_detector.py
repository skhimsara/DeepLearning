#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 00:10:33 2018

@author: skhimsar
"""

import os
from myutils import *


# The path to the directory where the original
# dataset was uncompressed
original_smiles_dataset_dir = '/Users/sidnpoo/Downloads/MLStuff_DoNotDelete/_DATASETS_/SmileFrownData/SMILEs/positives/positives7'
original_frowns_dataset_dir = '/Users/sidnpoo/Downloads/MLStuff_DoNotDelete/_DATASETS_/SmileFrownData/SMILEs/negatives/negatives7'

# The directory where we will
# store our smaller dataset
base_dir = '/Users/sidnpoo/Downloads/MLStuff_DoNotDelete/_DATASETS_/SmileFrownData/dataset'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)
    
    
folder_list = create_train_val_test_dirs(base_dir, ["smiles","frowns"])

import random

random.seed(10)

frown_files = [ f for f in os.listdir(original_frowns_dataset_dir)]
random.shuffle(frown_files)

smile_files = [ f for f in os.listdir(original_smiles_dataset_dir)]
random.shuffle(smile_files)

print(len(smile_files))
print(len(frown_files))

TRAIN=0.8
VALIDATION=0.1
TEST=0.1

#split the total dataset into 3 parts for each class.
frown_train,frown_val,frown_test = split_dataset((TRAIN,VALIDATION,TEST), frown_files)
smile_train,smile_val,smile_test = split_dataset((TRAIN,VALIDATION,TEST), smile_files)

#copy them into respective train val and test folders per class
copy_files(original_frowns_dataset_dir, folder_list["frowns"][0]["train"], frown_train)
copy_files(original_frowns_dataset_dir, folder_list["frowns"][1]["validation"], frown_val)
copy_files(original_frowns_dataset_dir, folder_list["frowns"][2]["test"], frown_test)

copy_files(original_smiles_dataset_dir, folder_list["smiles"][0]["train"], smile_train)
copy_files(original_smiles_dataset_dir, folder_list["smiles"][1]["validation"], smile_val)
copy_files(original_smiles_dataset_dir, folder_list["smiles"][2]["test"], smile_test)


print('total training smile images:', len(os.listdir(folder_list["smiles"][0]["train"])))
print('total training smile images:', len(os.listdir(folder_list["smiles"][1]["validation"])))
print('total training smile images:', len(os.listdir(folder_list["smiles"][2]["test"])))
print('total training frown images:', len(os.listdir(folder_list["frowns"][0]["train"])))
print('total training frown images:', len(os.listdir(folder_list["frowns"][1]["validation"])))
print('total training frown images:', len(os.listdir(folder_list["frowns"][2]["test"])))


train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D,  BatchNormalization,  Activation
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

INPUT_IMAGE_SIZE=64
COLOR_DEPTH_DIM=1

# model 0 -- from numpy aray
#prepare the dataset from the foldersß
def prepare_dataset_as_nparray(class_path, file_pattern,  resize_width, resize_height,label_value):

X, y = prepare_dataset_as_nparray(folder_list["smiles"][2]["test"], "*.jpg",INPUT_IMAGE_SIZE,INPUT_IMAGE_SIZE, 1)
  
# we turn X and y into numpy arrays and coerce them into the right shape and range.
X,y = normalize_dataset(X,y)





# model 1 --------------------------------------------------------------------

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)


# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
        batch_size=32,
        color_mode='grayscale',
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
        batch_size=32,
        color_mode='grayscale',
        class_mode='binary')

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break




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
model.add(Dense(1, activation='sigmoid'))

print(model.get_weights())


model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=10,
      validation_data=validation_generator,
      validation_steps=50)

plot_loss_and_accuracy(history)

model.save("smile_or_frown_model1.h5")

# model 2 --------------------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)


# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
        batch_size=20,
        color_mode='grayscale',
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
        batch_size=20,
        color_mode='grayscale',
        class_mode='categorical')



from keras.optimizers import SGD

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3),  padding='same',activation='relu', 
                 input_shape=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, COLOR_DEPTH_DIM)))
model.add(Conv2D(32, (3, 3), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
#model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3),padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
sgd = SGD(lr=0.0004, decay=1e-6, momentum=0.9, nesterov=True)
#adam=keras.optimizers.Adam(lr=0.0000005)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()

    # Fit the model on the batches generated by datagen.flow().
history = model.fit_generator(train_generator,
                      validation_data=validation_generator,
                        epochs=20, 
                        verbose=1,
                        steps_per_epoch=50)


plot_loss_and_accuracy(history)

model.save("smile_or_frown_model2.h5")

#- usng pretrained VGG16 model-  Method 1
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3))
conv_base.summary()
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

train_samples=len(os.listdir(folder_list["smiles"][0]["train"]))  + len(os.listdir(folder_list["frowns"][0]["train"]))
validation_samples=len(os.listdir(folder_list["smiles"][1]["validation"])) + len(os.listdir(folder_list["frowns"][1]["validation"]))
test_samples=len(os.listdir(folder_list["smiles"][2]["test"]))  + len(os.listdir(folder_list["frowns"][2]["test"]))

train_features, train_labels = extract_features(conv_base,datagen,train_dir, train_samples, INPUT_IMAGE_SIZE,batch_size)
validation_features, validation_labels = extract_features(conv_base,datagen,validation_dir, validation_samples, INPUT_IMAGE_SIZE,batch_size)
test_features, test_labels = extract_features(conv_base,datagen,test_dir, test_samples, INPUT_IMAGE_SIZE,batch_size)

train_features     = np.reshape(train_features, (train_samples, 2 * 2 * 512))
validation_features = np.reshape(validation_features, (validation_samples, 2 * 2 * 512))
test_features = np.reshape(test_features, (test_samples, 2 * 2 * 512))



model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=2 * 2 * 512))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=1e-6),
              loss='binary_crossentropy',
              metrics=['acc']),
model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))

plot_loss_and_accuracy(history)
model.save("smile_or_frown_model3.h5")

#- usng pretrained VGG16 model-  Method 2

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

print('This is the number of trainable weights '
      'before freezing the conv base:', len(model.trainable_weights))

conv_base.trainable = False

print('This is the number of trainable weights '
      'after freezing the conv base:', len(model.trainable_weights))


train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
        batch_size=20,
        class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=1)


## usng resnet50
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

model = ResNet50(weights='imagenet')
model.summary()


def predict(model, img, target_size, top_n=3):
  """Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (width, height) tuple
    top_n: # of top predictions to return
  Returns:
    list of predicted labels and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)
  return decode_predictions(preds, top=top_n)[0]

from load_cifar10 import load_cifar10_data
X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)