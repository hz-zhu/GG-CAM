from my_gen import *

import torch, pickle, copy
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import pandas as pd
import numpy as np
import cv2 as cv

def get_Gaussian_blur(img, blur):
    if blur%2==0: blur+=1
    r = cv.GaussianBlur(img, (blur, blur), 0, 0)
    return r

class DataPrep:
    
    def __init__(self, path, fraction, seed, mapping):
        # mapping has the form for {0:[0,1], 1:[2]} such that original label in the lists are mapped to new label in dict key
        self.path = path
        self.fraction = fraction
        self.seed = seed
        self.mapping = mapping
        self.init()
        
    def init(self):
        df = pd.read_csv(self.path+'/'+'data_summary.csv', index_col=0)
        df['folder'] = ['D_%04d'%idx for idx in df.index]

        train = {}
        valid = {}
        test = {}
        for i, key in enumerate(self.mapping):
            df_local = copy.deepcopy(df[df['Y'].isin(self.mapping[key])])
            df_local['Y_mapped'] = [key]*len(df_local)
            train[key], valid[key], test[key] = np.split(df_local, [int(0.7*len(df_local)), int(0.8*len(df_local))])
        
        self.train = pd.concat([train[key] for key in self.mapping],
                               ignore_index=True).sample(frac=self.fraction, random_state=self.seed)
        self.valid = pd.concat([valid[key] for key in self.mapping],
                               ignore_index=True).sample(frac=self.fraction, random_state=self.seed+1)
        self.test = pd.concat([test[key] for key in self.mapping],
                              ignore_index=True).sample(frac=self.fraction, random_state=self.seed+2)
    
    def __call__(self, idx):
        if idx=='Train':
            return self.train
        elif idx=='Valid':
            return self.valid
        elif idx=='Test':
            return self.test
        else:
            assert False, 'invalid idx @hzhu_data::DataPrep::__call__(self, idx)'
            
class DataHandle(Dataset):
    
    def __init__(self, path, data_info, blur, downsample):
        self.path = path
        self.data_info = data_info
        self.blur = blur
        self.downsample = downsample
        
    def init(self, index):
        row = self.data_info.iloc[index,:]
        path = self.path+'/'+row['folder']
        data = {}
        with open(path+'/cxr.pk', 'rb') as handle:
            data['cxr'] = pickle.load(handle).astype(np.float32)
        with open(path+'/gaze_dot.pk', 'rb') as handle:
            data['gaze'] = pickle.load(handle).astype(np.float32)

        with open(path+'/seg_left_lung.pk', 'rb') as handle:
            data['seg_left_lung'] = pickle.load(handle).astype(np.bool_)
        with open(path+'/seg_right_lung.pk', 'rb') as handle:
            data['seg_right_lung'] = pickle.load(handle).astype(np.bool_)
        with open(path+'/seg_mediastanum.pk', 'rb') as handle:
            data['seg_mediastanum'] = pickle.load(handle).astype(np.bool_)
        data['lung'] = data['seg_left_lung']+data['seg_right_lung']

        data['Y'] = torch.tensor(row['Y_mapped'], dtype=torch.float32).long()
        data['ID'] = row['ID']
        self.process(data)
        return data
            
    def process(self, data):
        data['gaze'] = get_Gaussian_blur(data['gaze'], self.blur)
        
        data['gaze'] = torch.tensor(data['gaze'], requires_grad=False)
        data['cxr'] = torch.tensor(data['cxr'], requires_grad=False)
        data['lung'] = torch.tensor(data['lung'], requires_grad=False)
        data['heart'] = torch.tensor(data['seg_mediastanum'], requires_grad=False)
        
        shape = data['cxr'].shape
        if shape[0]<=shape[1]:
            torch.rot90(data['cxr'], 1, [0,1])
            torch.rot90(data['gaze'], 1, [0,1])
            torch.rot90(data['lung'], 1, [0,1])
            torch.rot90(data['heart'], 1, [0,1])
        shape = data['cxr'].shape
        H = (int(3056/self.downsample/32)+1)*32*self.downsample
        W = (int(2544/self.downsample/32)+1)*32*self.downsample
        if shape[0]<=H or shape[1]<=W:
            padding_left = int((W-shape[1])/2)
            padding_right = W-shape[1]-padding_left
            padding_top = int((H-shape[0])/2)
            padding_bottom = H-shape[0]-padding_top
            data['cxr'] = F.pad(data['cxr'], (padding_left, padding_right, padding_top, padding_bottom))
            data['gaze'] = F.pad(data['gaze'], (padding_left, padding_right, padding_top, padding_bottom))
            data['lung'] = F.pad(data['lung'], (padding_left, padding_right, padding_top, padding_bottom))
            data['heart'] = F.pad(data['heart'], (padding_left, padding_right, padding_top, padding_bottom))
            
        data['gaze'] = data['gaze'][0:H+1:self.downsample*32,0:W+1:self.downsample*32]
        data['gaze'] /= data['gaze'].max()
        
        data['cxr'] = data['cxr'][0:H+1:self.downsample,0:W+1:self.downsample]
        data['cxr'] /= data['cxr'].max()
            
        data['gaze'] = data['gaze'].unsqueeze(0)
        data['cxr'] = torch.stack([data['cxr'],]*3, dim=0)
        
        data['lung'] = data['lung'][0:H+1:self.downsample*32,0:W+1:self.downsample*32]
        data['heart'] = data['heart'][0:H+1:self.downsample*32,0:W+1:self.downsample*32]
            
    def __len__(self):
        return len(self.data_info.index)
    
    def __getitem__(self, i):
        return self.init(i)
    
class DataMaster:
    
    def __init__(self, path, fraction, seed, blur, downsample, mapping):
        self.path = path
        self.fraction = fraction
        self.seed = seed
        self.blur = blur
        
        self.downsample = downsample
        self.mapping = mapping
        
        self.dataPrep = DataPrep(path=self.path, fraction=self.fraction, seed=self.seed, mapping=self.mapping)
        self.trainHandle = DataHandle(path=self.path, data_info=self.dataPrep('Train'), blur=self.blur, downsample=self.downsample)
        self.testHandle = DataHandle(path=self.path, data_info=self.dataPrep('Test'), blur=self.blur, downsample=self.downsample)
        self.validHandle = DataHandle(path=self.path, data_info=self.dataPrep('Valid'), blur=self.blur, downsample=self.downsample)
    
    def save_item(self, item, local_path):
        keys = ['gaze', 'cxr', 'lung', 'heart', 'Y']
        data = {key:item[key] for key in keys}
        torch.save(data, local_path+'/%s.pt'%item['ID'])
    
    def save_all(self, path):
        create_folder(path+'/Train')
        create_folder(path+'/Test')
        create_folder(path+'/Valid')
        
        counter = {key:0 for key in self.mapping}
        local_path = path+'/Train'
        create_folder(local_path)
        for item in self.trainHandle:
            self.save_item(item, local_path)
            if int(item['Y']) in counter:
                counter[int(item['Y'])] += 1
            else:
                counter[int(item['Y'])] = 1
        print('- Train:%d'%len(self.trainHandle))
        disp(counter)
        
        counter = {key:0 for key in self.mapping}
        local_path = path+'/Test'
        create_folder(local_path)
        for item in self.testHandle:
            self.save_item(item, local_path)
            if int(item['Y']) in counter:
                counter[int(item['Y'])] += 1
            else:
                counter[int(item['Y'])] = 1
        print('- Test:%d'%len(self.testHandle))
        disp(counter)
        
        counter = {key:0 for key in self.mapping}
        local_path = path+'/Valid'
        create_folder(local_path)
        for item in self.validHandle:
            self.save_item(item, local_path)
            if int(item['Y']) in counter:
                counter[int(item['Y'])] += 1
            else:
                counter[int(item['Y'])] = 1
        print('- Valid:%d'%len(self.validHandle))
        disp(counter)