# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 14:18:31 2022

@author: eleves
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 17:33:16 2022

@author: eleves
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 12:01:00 2021

Python code in which a CNN is designed and trained to recognise MNIST dataset elements
The trained model is created to be implemented on STM32 board (STM32F411)

@author: RJ264980
"""
import tensorflow as tf
 
from tensorflow import keras
 
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
 
from tensorflow.keras.preprocessing import image_dataset_from_directory
 
import numpy as np
import matplotlib.pyplot as plt
import os
from os import path
import time

start = time.time()

epochs = 50
batch_size = 32
# image size (Model Medium)
img_width, img_height = 1280, 720

# input shape
if keras.backend.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

PATH_DATASET = "c:/Users/eleves/Documents/CNN_C2_16_10-20220920/"

train_data_dir = os.path.join(PATH_DATASET, 'train') 
validation_data_dir = os.path.join(PATH_DATASET, 'validation')
test_data_dir = os.path.join(PATH_DATASET, 'test')

# ***********************************************************************
# ************        DATASET       *************************************
# ***********************************************************************

train_dataset = image_dataset_from_directory(train_data_dir,
                                             shuffle=True,
                                             batch_size=batch_size,
                                             image_size=(img_width, img_height),
                                             label_mode='categorical')


validation_dataset = image_dataset_from_directory(validation_data_dir,
                                                  shuffle=True,
                                                  batch_size=batch_size,
                                                  image_size=(img_width, img_height),
                                                  label_mode='categorical')


test_dataset = image_dataset_from_directory(test_data_dir,
                                            shuffle=True,
                                            batch_size=batch_size,
                                            image_size=(img_width, img_height),
                                            label_mode='categorical')


# preprocessing: input scaling (./255)
train_dataset = train_dataset.map(lambda images, labels: (images/255, labels))
validation_dataset = validation_dataset.map(lambda images, labels: (images/255, labels))
test_dataset = test_dataset.map(lambda images, labels: (images/255, labels))


# Configure the dataset for performance

#AUTOTUNE = tf.data.experimental.AUTOTUNE

#train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
#validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
#test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)




# ***********************************************************************
# **************        MODEL       *************************************
# ***********************************************************************

model_large = Sequential()
model_large.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
model_large.add(Activation('relu'))
model_large.add(MaxPooling2D(pool_size=(2, 2)))

model_large.add(Conv2D(32, (3, 3), padding='same'))
model_large.add(Activation('relu'))
model_large.add(MaxPooling2D(pool_size=(2, 2)))

model_large.add(Conv2D(64, (3, 3), padding='same'))
model_large.add(Activation('relu'))
model_large.add(MaxPooling2D(pool_size=(2, 2)))

model_large.add(Conv2D(64, (3, 3), padding='same'))
model_large.add(Activation('relu'))
model_large.add(MaxPooling2D(pool_size=(2, 2)))

model_large.add(Conv2D(32, (3, 3), padding='same'))
model_large.add(Activation('relu'))
model_large.add(MaxPooling2D(pool_size=(2, 2)))

model_large.add(Flatten())
model_large.add(Dense(64))
model_large.add(Activation('relu'))
model_large.add(Dropout(0.5))
model_large.add(Dense(2))			#because we have 2 class
model_large.add(Activation('softmax'))

model_large.summary()


# ***********************************************************************
# *******************        COMPILATION       **************************
# ***********************************************************************


model_large.compile(loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adadelta(learning_rate=1, name='Adadelta'),
            metrics=['accuracy'])



# ***********************************************************************
# *******************        TRAINING       *****************************
# ***********************************************************************


with tf.device('/device:GPU:0'):

  history = model_large.fit(
    train_dataset,
    epochs=epochs,
    validation_data=validation_dataset)



model_large.save( 'model_large_b32.h5')


# ***********************************************************************
# ********************        PLOT RESULTS        ***********************
# ***********************************************************************


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy_'+str(img_width)+' x '+str(img_height))

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss_'+str(img_width)+' x '+str(img_height))
plt.show()



# ***********************************************************************
# ***********************        TEST        ****************************
# ***********************************************************************

with tf.device('/device:GPU:0'):

  test_result = model_large.evaluate(test_dataset)

  
print("size of images: ", img_width,img_height)
print("test_result: ", test_result)


print ('Time taken for development model small {} sec\n'.format(time.time() - start))    





