#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir('..')


# In[2]:


import numpy as np
from pathlib import Path
from scipy.io import loadmat

import tensorflow_datasets as tfds


# In[23]:


base_dir = Path('data/small_norb_lord')
os.makedirs(base_dir, exist_ok=True)


# In[10]:


import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import matplotlib

# tfds works in both Eager and Graph modes
tf.compat.v1.enable_eager_execution()

# Construct a tf.data.Dataset
dataset = tfds.load(name="smallnorb", split="train")


# In[27]:


import numpy as np
for example in dataset:
    dir_name = base_dir/f'{example["label_category"]}'/f'{example["instance"]}'
    os.makedirs(dir_name, exist_ok=True)
#     lt_rt = len(os.listdir(dir_name))
    image = example["image"]

    image_name = f'azimuth{example["label_azimuth"]}_elevation{example["label_elevation"]}_lighting{example["label_lighting"]}_donow.jpg'
    image_path = dir_name / image_name
    if os.path.exists(image_path):
        print("Error: {image_path} exists!")
    else:
        matplotlib.image.imsave(image_path, np.squeeze(image), cmap='gray')

