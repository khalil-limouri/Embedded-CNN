# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 13:35:36 2022

@author: eleves
"""

import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from os import path
import numpy as np

 
def preprocess(image):
  img_width, img_height = 320, 180
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, (img_width, img_height))
  image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
  image = image[None, ...]
  return image

# Helper function to extract labels from probability vector
def get_label(probs):
  predict=tf.math.argmax(image_probs[0], axis=-1).numpy()
  confidence=np.max(tf.nn.softmax(image_probs[0]))
  return (None, decode_predictions[predict], confidence)
loss_object = tf.keras.losses.CategoricalCrossentropy()

def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = pretrained_model(input_image)
    loss = loss_object(input_label, prediction)

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad

def display_images(image, description):
  _, label, confidence = get_label(pretrained_model.predict(image))
  plt.figure()
  plt.imshow(image[0]*0.5+0.5)
  plt.title('{} \n {} : {:.2f}% Confidence'.format(description,
                                                   label, confidence*100))
  plt.show()
  
'''a remplacer pour esca sur la carte tester
rendu readme git 17 octobre
'''

if __name__ == '__main__':

  
    pretrained_model = tf.keras.models.load_model('model_medium_b32.h5')
    
    # labels
    decode_predictions = ['esca', 'healthy']
    
    image =  np.load('train/healthy_000_cam3.npy')
    
    image = preprocess(image)
    
    image_probs = pretrained_model.predict(image)
    
    plt.figure()
    plt.imshow(image[0] * 0.5 + 0.5)  # To change [-1, 1] to [0,1]
    _, image_class, class_confidence = get_label(image_probs)
    plt.title('{} : {:.2f}% Confidence'.format(image_class, class_confidence*100))
    plt.show()

    # Get the input label of the image.
    labrador_retriever_index = 208
    label = tf.one_hot(labrador_retriever_index, image_probs.shape[-1])
    label = tf.reshape(label, (1, image_probs.shape[-1]))

    perturbations = create_adversarial_pattern(image, label)
    plt.imshow(perturbations[0] * 0.5 + 0.5);  # To change [-1, 1] to [0,1]
    
    epsilons = [0, 0.01, 0.1, 0.15]
    descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
            for eps in epsilons]
      
    for i, eps in enumerate(epsilons):
      adv_x = image + eps*perturbations
      adv_x = tf.clip_by_value(adv_x, -1, 1)
      display_images(adv_x, descriptions[i])