# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 00:21:38 2018

@author: JAE
"""

from keras.layers import Input, merge, Activation
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras import Sequential

img_rows = 224
img_cols = 224
smooth = 5.0

ann = Sequential()
x = Conv2D(filters=32, kernel_size=(3, 3), input_shape=(200, 200, 1))
ann.add(x)
ann.add(Activation("relu"))
x1w = x.get_weights()[0][:, :, 0, :]
for i in range(1, 26):
    plt.subplot(5, 5, i)
    plt.imshow(x1w[:, :, i], interpolation="nearest", cmap="gray")
plt.show()