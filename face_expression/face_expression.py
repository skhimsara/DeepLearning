
# coding: utf-8

# In[118]:


DATA_PATH="/Users/sidnpoo/Downloads/MLStuff_DoNotDelete/_DATASETS_/face_expresson"


# In[119]:


IMAGE_SIZE=48


# In[120]:


import pandas as pd
import tqdm 
import numpy as np


# In[121]:


#Read the csv as dataframe to understand what's in there.
df = pd.read_csv(DATA_PATH+ "/fer2013.csv")
print(df.head(10))


# In[122]:


#total number of examples
print(len(df))
#find uniques in Usage column and how many
df.Usage.value_counts()


# In[123]:


#number of unique classes
df.emotion.value_counts()


# In[124]:


#expressions = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}
expressions = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# In[125]:


def read_data(filename, entry_type):
    X=[]
    y=[]
    with open(filename , 'r') as f:
        f.readline() #skip first line
        for line in f:
                if str(entry_type) in line:
                
                    row = line.split(',')
                    y.append(int(row[0]))
                    X.append([int(p) for p in row[1].split()] )
           
            
            
    return np.array(X),np.array(y)
          
            


# In[126]:


import os

def load_data(entry_type):
    X_file  = entry_type+"_X.npy"
    y_file = entry_type+"_y.npy"
    
    if os.path.exists(X_file) and os.path.exists(y_file):
        print("Found numpy arrays saved a files..loading ")
        X,y = np.load(X_file), np.load(y_file)
    else:
        print("Numpy array file missing..")
        X, y = read_data(DATA_PATH+ "/fer2013.csv", str(entry_type))
        #save np array to file for faster processing for 2nd run

        np.save(X_file, X)
        np.save(y_file,y)
        
        #normalize the X before returning it
    X = X.astype(np.float32) / 255.0
    y = y.astype(np.int32)   
    return X , y



# In[127]:


X_train


# In[128]:


X_train,y_train = load_data("Training")

X_val,y_val = load_data("PrivateTest")

X_test,y_test = load_data("PublicTest")


#reshape to 48x48 and add a dimension
X_train = X_train.reshape(X_train.shape[0],IMAGE_SIZE,IMAGE_SIZE,1)
X_val = X_val.reshape(X_val.shape[0],IMAGE_SIZE,IMAGE_SIZE,1)
X_test = X_test.reshape(X_test.shape[0],IMAGE_SIZE,IMAGE_SIZE,1)


# In[129]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(X_train[20000].reshape(IMAGE_SIZE,IMAGE_SIZE), cmap='gray')
plt.show()
print(y_train[20000])


# In[130]:


import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D,  BatchNormalization,  Activation,ZeroPadding2D
from keras import optimizers, callbacks, layers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json


# In[131]:


# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
y_test = np_utils.to_categorical(y_test)

num_classes = y_train.shape[1]


# In[132]:


model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3),  padding='same',activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
#model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3),padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.75))
model.add(Dense(num_classes, activation='softmax'))

#model.load_weights('my_model_bestwts_mod.h5')
sgd = optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
adam=optimizers.Adam(lr=0.0001)


# In[148]:


#calls=callbacks.ModelCheckpoint(filepath="my_model_bestwts_mod.h5",monitor='val_acc',
 #                                     save_best_only=False, save_weights_only=False, mode='auto', period=1)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


# In[139]:



#model.fit(train_X, train_y, batch_size=64, epochs=15, verbose=1, validation_data=(test_X, test_y),callbacks=[calls])
#model.save_weights('my_model_weights_mod.h5')
datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=10,
        # randomly shift images horizontally
        width_shift_range=0,
        # randomly shift images vertically
        height_shift_range=0.25,
        # randomly flip images
        horizontal_flip=False,
        # randomly flip images
        vertical_flip=False)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(X_train)
model.summary()


# In[146]:


batch_size = 1024
maxepoches = 60
learning_rate = 0.2
lr_decay = 1e-6
lr_drop = 20

def lr_scheduler(epoch):
    return learning_rate * (0.5 ** (epoch // lr_drop))
reduce_lr = callbacks.LearningRateScheduler(lr_scheduler)


# In[147]:


#using image augmentation
historytemp = model.fit_generator(datagen.flow(X_train, y_train,
                             batch_size=batch_size),
                steps_per_epoch=X_train.shape[0] //batch_size,
                epochs=maxepoches,
                validation_data=(X_val, y_val),
                                  callbacks=[reduce_lr],
                                  verbose=1)
'''
model.fit(X_train, y_train, batch_size=100, epochs=60, verbose=1, validation_data=(X_val, y_val))
         # ,callbacks=[calls])
    '''


# In[149]:


#no image agumentation
model.fit(X_train, y_train, batch_size=100, epochs=60, verbose=1, validation_data=(X_val, y_val))
         # ,callbacks=[calls])

