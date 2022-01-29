import sys, json

import torch
from torch import nn as nn

import copy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages

import sklearn.metrics as M
    
class MetricsHandle:
    
    def __init__(self):
        self.data = []
        
    def __len__(self):
        return len(self.data)
    
    def add_data(self, X, Y, Y_pred):
        
        X = X.detach().clone().cpu()
        Y = Y.detach().clone().cpu()
        Y_pred = nn.Softmax(dim=1)(Y_pred.detach().clone().cpu())
        
        batch_num = X.shape[0]
        
        for i in range(batch_num):
            local_Y = Y[i:i+1]
            local_Y_pred = Y_pred[i,:]
            
            local_data = {}
            local_data['Y'] = local_Y
            local_data['Y_pred'] = local_Y_pred
            
            self.data.append(local_data)
            
    def __getitem__(self, i):
        return self.data[i]
    
    def compute_classification_report(self):
        Y_true = torch.cat([self[i]['Y'] for i in range(len(self))], dim=0).float().numpy()
        Y_score = torch.stack([self[i]['Y_pred'] for i in range(len(self))], dim=0).numpy()        
        Y_pred = np.argmax(Y_score, axis=1)

        self.classification_report = M.classification_report(Y_true, Y_pred, output_dict=True)
        
        if np.max(Y_true)==1:
            self.classification_report['ROC_AUC'] = M.roc_auc_score(Y_true, Y_score[:,1])
        else:
            self.classification_report['ROC_AUC_ovr'] = M.roc_auc_score(Y_true, Y_score, average='macro', multi_class='ovr')
            self.classification_report['ROC_AUC_ovo'] = M.roc_auc_score(Y_true, Y_score, average='macro', multi_class='ovo')
    
    def get_evaluation(self):
        if not hasattr(self, 'classification_report'):
            self.compute_classification_report()
        return self.classification_report
    
    def get_key_evaluation(self):
        Y_true = torch.cat([self[i]['Y'] for i in range(len(self))], dim=0).float().numpy()
        Y_score = torch.stack([self[i]['Y_pred'] for i in range(len(self))], dim=0).numpy()
        Y_pred = np.argmax(Y_score, axis=1)
        return M.accuracy_score(Y_true, Y_pred)
                
    def save_outputs(self, name, path):
        r = []
        for i in range(len(self)):
            item = self[i]
            r.append({'Y':item['Y'].tolist(), 'Y_pred':item['Y_pred'].tolist()})
            
        r = pd.DataFrame(r)
        r.to_csv(path+'/'+name+'.csv')
        
    def save_classification_report(self, name, path):
        if not hasattr(self, 'classification_report'):
            self.compute_classification_report()
        with open(path+'/'+name+'.json', 'w') as f:
            json.dump(self.classification_report, f, ensure_ascii=False, indent=4)