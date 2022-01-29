#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

from my_metrics import * # for network evaluation
from my_data import * # data handle
from my_learn import * # training process handle
from my_EffCAMNetv2 import * # defines EffNet+GG-CAM
from my_gen import * # utils

import torch
from torch import nn as nn
import torch.optim as optim
import torchvision

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import copy

matplotlib.use('Agg')
plt.rcParams['axes.facecolor'] = 'white'

import argparse


# In[2]:


if __name__ == '__main__':

    # Hyper-parameters
    lr = 6e-3
    optimizer_dict = {'optimizer':optim.Adam, 'param':{}, 'name':'Adam'}
    lr_factor = 0.1
    lr_min = 1.0e-8
    epoch_max = 1024
    duration_max = 23.5*60*60 #seconds 23.5hour
    patience_reduce_lr = 40
    patience_early_stop = patience_reduce_lr*2+3
    class_num = 3
    net_type = 'S'
    batch_size = 4
    blur = 500
    down = 2

    dissimilarity_loss = nn.MSELoss
    lg_sigma_d = -16.0
    lg_sigma_c = 7.0
    lg_scale = 0.0
    bias = None

    # Training preparation
    Metrics = MetricsHandle
    Model = EffCAMNet

    name = 'ENetCAM_%s'%net_type
    folder_string = 'run'
    qH = QuickHelper(path=os.getcwd()+'/'+folder_string)
    print('New Folder name: %s'%qH.ID)
    print(folder_string)

    data_timer = QuickTimer()
    J = os.getcwd()
    path = 'D:/Google Sync/Python Code/CXR & Gaze/X_ray Image with Gaze 6/Data/hzhu_data_raw/Down%d_Blur%d'%(down,blur)
        
    dataAll = DataMaster(path=path, batch_size=batch_size)
    print('Data Preparing time: %fsec'%data_timer())

    # CNN model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Net = Model(
        net_type = net_type,
        class_num = class_num,
        device = device,
        lg_sigma_d = lg_sigma_d,
        lg_sigma_c = lg_sigma_c,
        lg_scale = lg_scale,
        bias = bias,
        dissimilarity_loss = dissimilarity_loss)
    Net.save_params(name='', path=qH())

    criterion = nn.CrossEntropyLoss()

    # Training
    netLearn = NetLearn(
        net=Net,
        dataAll=dataAll,
        criterion=criterion,
        optimizer_dict=optimizer_dict,
        lr=lr,
        lr_min=lr_min,
        lr_factor=lr_factor,
        epoch_max=epoch_max,
        duration_max=duration_max,
        patience_reduce_lr=patience_reduce_lr,
        patience_early_stop=patience_early_stop,
        device=device,
        metrics=Metrics,
        name=name,
        path=qH())

    netLearn.train()

    # Evaluation
    print(netLearn.evaluate())
    print(netLearn.interpretate())

    # End
    netLearn.remove_saved_optim()
    netLearn.remove_saved_sched()
    netLearn.save_params(name='', path=qH())
    qH.summary()


# In[ ]:




