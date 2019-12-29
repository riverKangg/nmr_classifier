
# coding: utf-8

# # Install package

# In[2]:


# Install a pip package in the current Jupyter kernel
import sys
get_ipython().system('{sys.executable} -m pip install nmr_classifier')


# In[18]:


# update
get_ipython().system('{sys.executable} -m pip install --upgrade nmr_classifier==0.0.9')


# # Import packages

# In[1]:


import nmr_classifier

import numpy as np
import pandas as pd

import random

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='whitegrid',palette='deep')

import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../data"))


# # Olivatti dataset

# In[2]:


pics = np.load("../data/olivetti_faces.npy")
labels = np.load("../data/olivetti_faces_target.npy")

fig = plt.figure(figsize=(20, 10))
columns = 10
rows = 4
for i in range(1, columns*rows +1):
    img = pics[10*(i-1),:,:]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img, cmap=plt.cm.bone)
    plt.title("person {}".format(i), fontsize=16)
    plt.axis('off')

plt.show()


# # Create occlusion

# In[3]:


Xdata = pics
Ydata = labels.reshape(-1,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(Xdata, Ydata, test_size = 0.2, random_state=46)

# Occlusion rate
occlusion_percent = 0.2

# Create occlusion at random locations
for test_img in x_test:
    black_size = round(test_img.shape[1] * occlusion_percent)
    loc = random.randint(0,64-black_size)
    test_img[loc:loc+black_size,loc:loc+black_size] = np.zeros((black_size,black_size))

# Show results
fig = plt.figure(figsize=(20, 4.5))
columns = 20
rows = 4
for i in range(0, columns*rows):
    img = x_test[i,:,:]
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(img, cmap = plt.get_cmap('gray'))
    plt.ylabel(''); plt.xlabel(''); plt.yticks([]); plt.xticks([]) 
plt.suptitle("Test images", fontsize=13)
plt.show()


# # Fitting
# - finding the optimal regression coefficient

# In[2]:


train_img = np.load("../data/olivetti_faces.npy")
target = np.load("../data/olivetti_faces_target.npy")

# create test imgage
import random
num = random.randint(0,400)
test_img = train_img[num].copy()

black_size = round(test_img.shape[1]*0.4)
test_img[13:13+black_size,13:13+black_size] = np.zeros((black_size,black_size))
plt.imshow(test_img,cmap=plt.cm.bone)

train_img[num,:,:] = 0


# In[3]:


from nmr_classifier.fast_admm_nmr import fast_admm_nmr

# Define model
## At this point, you can set the parameters.
reg = fast_admm_nmr()

reg.fit(train_img, test_img)


# # classification
# - Return the most appropriate label

# In[4]:


from nmr_classifier.fast_admm_nmr_classifier import nmr_classifier

clf = nmr_classifier()
clf.fit(train_img, test_img)
clf.classifier(train_img, test_img, target)


# # classification top-n
# - Returns the n most appropriate labels

# In[5]:


from nmr_classifier.fast_admm_nmr_classifier import nmr_classifier

clf = nmr_classifier()
clf.fit(train_img, test_img)
clf.classifier_top(train_img, test_img, target, 5)

