#!/usr/bin/env python
# coding: utf-8

# In[1]:


from my_data_raw import *
import os
import argparse


# In[2]:


path = 'D:/Gaze Dataset/MIMIC-CXR & GAZE (master)/Data (new)' # change this path to the appropriate data folder
save_path = os.getcwd()

fraction = 0.01
seed = 0
downsample = 2
blur_list = [200,500,800]


# In[3]:


for blur in blur_list:
    dataMaster = DataMaster(
        path=path, fraction=fraction, seed=seed, blur=blur,
        downsample=downsample, mapping={i:[i,] for i in range(3)})
    print(save_path+'/Down%d_Blur%d'%(downsample, blur))
    dataMaster.save_all(save_path+'/Down%d_Blur%d'%(downsample, blur))
    del dataMaster

