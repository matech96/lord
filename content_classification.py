#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir('..')


# In[2]:


from model.network import Converter
import matplotlib.pyplot as plt


# In[3]:


plt.rcParams["figure.figsize"] = (20,20)


# In[4]:


import os
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
import lpips_tf

from keras import backend as K
from keras import optimizers, losses, regularizers
from keras.layers import Conv2D, Dense, UpSampling2D, LeakyReLU, Activation
from keras.layers import Layer, Input, Reshape, Lambda, Flatten, Concatenate, Embedding, GaussianNoise
from keras.models import Model, load_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from keras.applications import vgg16
from keras_lr_multiplier import LRMultiplier
from assets import AssetManager

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks.callbacks import EarlyStopping, CSVLogger
from keras.callbacks.tensorboard_v1 import TensorBoard

from sklearn.preprocessing import OneHotEncoder


# In[5]:


class LORDContentClassifier:
    def __init__(self, subset=None,
                 base_dir = 'results', model_name = 'minst_10_model', data_name = 'minst_10_test'):
        assets = AssetManager(base_dir)
        data = np.load(assets.get_preprocess_file_path(data_name))
        imgs, classes, contents, n_classes = data['imgs'], data['classes'], data['contents'], data['n_classes']
        imgs = imgs.astype(np.float32) / 255.0
        if subset is not None:
            self.curr_imgs = imgs[:subset]
            self.classes = classes[:subset]
        else:
            self.curr_imgs = imgs
            self.classes = classes

        self.onehot_enc = OneHotEncoder()
        self.onehot_classes = self.onehot_enc.fit_transform(self.classes.reshape(-1,1))

        self.n_images = self.curr_imgs.shape[0]
        
        self.converter = Converter.load( assets.get_model_dir(model_name), include_encoders=True)
        self.content_codes = self.converter.content_encoder.predict(self.curr_imgs)
        
    def train_classifier(self, n_epochs):        
        self.model = Sequential()
        self.model.add(Dense(units=256, activation='relu', input_dim=self.content_codes.shape[1]))
        self.model.add(Dense(units=10, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        
        self.callbacks = [EarlyStopping(), CSVLogger('LORDContentClassifier_log.csv'), TensorBoard()]
        self.model.fit(self.content_codes, self.onehot_classes, epochs=20, validation_split=0.3, callbacks=self.callbacks)


# In[7]:


c = LORDContentClassifier()
c.train_classifier(10000)


# In[ ]:




