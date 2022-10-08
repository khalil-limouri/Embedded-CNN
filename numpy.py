# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 21:10:01 2022

@author: eleves
"""

from tensorflow.keras.preprocessing.image import load_img
import warnings
import numpy
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
  
  
# load the image via load_img function

for i in range(881):
    for j in range(1,2):
        try:
            img = load_img('train\esca_'+str(i).zfill(3)+'_cam'+str(j)+'.jpg')
            img_numpy_array = img_to_array(img)
            #numpy.save('esca_'+str(i).zfill(3)+'_cam'+str(j)+'.npy',img_numpy_array)
            # shape: (720, 1280, 3)
            print("shape:", img_numpy_array.shape)
        except FileNotFoundError:
            pass