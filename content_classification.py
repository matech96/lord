#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os



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
from sklearn.utils import shuffle


# In[5]:


class LORDContentClassifier:
    def __init__(self, subset=None,
                 base_dir = 'results', model_name = 'minst_10_model', data_name = 'minst_10_test', include_encoders=True):
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
        self.n_classes = self.onehot_classes.shape[1]

        self.n_images = self.curr_imgs.shape[0]
        
        self.converter = Converter.load( assets.get_model_dir(model_name), include_encoders=include_encoders)
        self.content_codes = self.converter.content_encoder.predict(self.curr_imgs)
        self.class_codes = self.converter.class_encoder.predict(self.curr_imgs)
        class_adain_params = self.converter.class_modulation.predict(self.class_codes)
        self.class_adain_params = class_adain_params.reshape(class_adain_params.shape[0], -1)
        self.curr_imgs, self.classes, self.onehot_classes, self.content_codes, self.class_codes, self.class_adain_params = \
            shuffle(self.curr_imgs, self.classes, self.onehot_classes, self.content_codes, self.class_codes, self.class_adain_params)
        
    def train_content_classifier(self, n_epochs):        
        model = self.get_model(self.content_codes.shape[1])        
        callbacks = [EarlyStopping('val_accuracy', patience=10), CSVLogger('LORDContentClassifier_content.csv'), TensorBoard()]
        model.fit(self.content_codes, self.onehot_classes, epochs=n_epochs, validation_split=0.3, callbacks=callbacks)
        
    def train_class_classifier(self, n_epochs):        
        print(f'Class code size: {self.class_codes.shape[1]}')
        model = self.get_model(self.class_codes.shape[1])        
        callbacks = [EarlyStopping('val_accuracy', patience=10), CSVLogger('LORDContentClassifier_class.csv'), TensorBoard()]
        model.fit(self.class_codes, self.onehot_classes, epochs=n_epochs, validation_split=0.3, callbacks=callbacks)
        
    def get_model(self, input_dim):
        model = Sequential()
        model.add(Dense(units=256, activation='relu', input_dim=input_dim))
        model.add(Dense(units=256, activation='relu'))
        model.add(Dense(units=self.n_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model


# In[6]:


# cc = LORDContentClassifier(model_name='mnist_model_64', data_name = 'minst_10_train')
# cc = LORDContentClassifier(model_name='smallnorb_model', data_name = 'smallnorb_test')
# cc = LORDContentClassifier(model_name='smallnorb_model', data_name = 'smallnorb_strict_class_test')
# cc = LORDContentClassifier(model_name='emnist_model_frist_stage', data_name = 'emnist_test')
# cc = LORDContentClassifier(model_name='smallnorb_model_fxd', data_name = 'smallnorb_strict_class_fxd_train')
# cc = LORDContentClassifier(model_name='smallnorb_model_fxd', data_name = 'smallnorb_strict_class_fxd_test')
cc = LORDContentClassifier(model_name='smallnorb_no_adain_scale', data_name = 'smallnorb_strict_class_fxd_test')

cc.train_content_classifier(10000)
cc.train_class_classifier(10000)

