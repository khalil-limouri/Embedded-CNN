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

 
decode_predictions = ['esca', 'healthy']

epsilons = [0.01, 0.2, 0.5]
eps = 0
descriptions = [('Epsilon = {:0.3f}'.format(eps))
                for eps in epsilons]

def preprocess(image):
   img_width, img_height = 80, 45
   image = tf.cast(image, tf.float32)
   image = tf.image.resize(image, (img_width, img_height))
   #image = preprocess_input(image)
   image = image[None, ...]
   return image

def get_label(probs):
  predict = probs[0].argmax(axis=0)
  pourcent = np.max(tf.nn.softmax(probs[0]))
  return (None, decode_predictions[predict], pourcent)


def display_images(model, description, image):
  _, label, confidence = get_label(model.predict(image))
  plt.figure()
  plt.imshow(image[0]*0.5+0.5)
  plt.title('Classe : {} \n {}'.format(label, description))
  plt.show()
  
      
def create_adversarial_pattern(input_label, input_image):
    loss_object = tf.keras.losses.CategoricalCrossentropy()
  
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        loss = loss_object(input_label, prediction)
      
    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    return signed_grad




    
pretrained_model = tf.keras.models.load_model('model_small_b32.h5', compile = True)


image =  np.load('healthy_000_cam3.npy')  
image = preprocess(image)
display_images(pretrained_model, 'Input', image)  

image_probs = pretrained_model.predict(image)
retriever_index = 60
label = tf.one_hot(retriever_index, image_probs.shape[-1])
label = tf.reshape(label, (1, image_probs.shape[-1]))

perturbations = create_adversarial_pattern(label, image)
#plt.imshow(perturbations[0] * 0.5 + 0.5)

adv_x = image + eps*perturbations
display_images(pretrained_model, 'Epsilon = {:0.3f}'.format(eps), adv_x)

for i, eps in enumerate(epsilons):
  adv_x = image + eps*perturbations
  adv_x = tf.clip_by_value(adv_x, -1, 1)
  display_images(pretrained_model, descriptions[i], adv_x)