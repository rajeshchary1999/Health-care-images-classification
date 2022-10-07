#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop


# In[2]:


img = image.load_img("image90.jpeg")


# In[3]:


plt.imshow(img)


# In[4]:


cv2.imread("image90.jpeg").shape


# # Data

# In[5]:


train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)


# In[6]:


train_dataset = train.flow_from_directory('C:\\Users\\graje\\OneDrive\\Desktop\\cv\\basedata\\training',
                                         target_size = (200,200),
                                         batch_size = 3,
                                         class_mode = 'binary')


validation_dataset = validation.flow_from_directory('C:\\Users\\graje\\OneDrive\\Desktop\\cv\\basedata\\validation',
                                                   target_size = (200,200),
                                                   batch_size = 3,
                                                   class_mode = 'binary')


# In[7]:


train_dataset.class_indices


# # Model Building

# In[8]:


model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation = 'relu',input_shape=(200,200,3)),
                                   tf.keras.layers.MaxPool2D(2,2),
                                   #
                                   tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
                                   tf.keras.layers.MaxPool2D(2,2),
                                   #
                                   tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
                                   tf.keras.layers.MaxPool2D(2,2),
                                   ##
                                   tf.keras.layers.Flatten(),
                                   ##
                                   tf.keras.layers.Dense(512,activation='relu'),
                                   ##
                                   tf.keras.layers.Dense(1,activation='sigmoid')
                                   ])


# In[9]:


model.compile(loss='binary_crossentropy',
             optimizer = RMSprop(lr=0.001),
             metrics = ['accuracy'])


# In[10]:


model_fit = model.fit(train_dataset,
                     steps_per_epoch =3,
                     epochs = 10,
                     validation_data = validation_dataset)


# In[11]:


train_dataset.class_indices


# # Prediction

# In[12]:


y_pred = model.predict(train_dataset)


# In[13]:


y_pred[:5]


# In[14]:


y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]


# In[ ]:




