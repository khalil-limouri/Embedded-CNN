# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 17:33:16 2022

@author: eleves
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



def dataset(image_size):
        
    
    train_data_dir = os.path.join(os.getcwd(), 'train')
    validation_data_dir = os.path.join(os.getcwd(), 'validation')
    test_data_dir = os.path.join(os.getcwd(), 'test')
    
    train_dataset = image_dataset_from_directory(train_data_dir,
                                                 shuffle=True,
                                                 batch_size=batch_size,
                                                 image_size=image_size,
                                                 label_mode='categorical')
    
    train_dataset = train_dataset.map(lambda images, labels: (images/255, labels))
    
    validation_dataset = image_dataset_from_directory(validation_data_dir,
                                                      shuffle=True,
                                                      batch_size=batch_size,
                                                      image_size=image_size,
                                                      label_mode='categorical')
    validation_dataset = validation_dataset.map(lambda images, labels: (images/255, labels))
    
    test_dataset = image_dataset_from_directory(test_data_dir,
                                                shuffle=True,
                                                batch_size=batch_size,
                                                image_size=image_size,
                                                label_mode='categorical')
    test_dataset = test_dataset.map(lambda images, labels: (images/255, labels))

    return (train_dataset, validation_dataset, test_dataset)

    
def build_model(image_size):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(image_size[0], image_size[1], 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))			
    model.add(Activation('softmax'))
    
    model.summary()

    return model



def train_model(image_size):
    train_dataset, validation_dataset, _ = dataset(image_size)
    model=build_model(image_size)
    model.compile(loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adadelta(learning_rate=1, name='Adadelta'),
            metrics=['accuracy'])


    history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=validation_dataset)

    return model, history

    

def plot_model(history, image_size):

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
    plt.title('Training and Validation Accuracy_'+str(image_size[0])+' x '+str(image_size[1]))
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss_'+str(image_size[0])+' x '+str(image_size[1]))
    plt.show()
    print ('Time taken for development model small {} sec\n'.format(time.time() - start))    



def test_model(test_dataset, image_size, model):
	_, _, test_dataset = dataset(image_size)
	test_result = model.evaluate(test_dataset) 
	print("size of images: ", image_size)
	print("test_result: ", test_result)
  




if __name__ == '__main__':
    
    model, history = train_model((520, 480))
    plot_model(history)
    test_model(520, 480)	
    model.save(os.path.curdir + '\model_b32\model_b32.h5')
