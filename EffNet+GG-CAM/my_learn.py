import os
import pandas as pd
import numpy as np
import PIL
import matplotlib.pyplot as plt
import matplotlib
import json
matplotlib.use('Agg')

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch import optim

from my_gen import *
from my_data import *
from my_metrics import *

class NetLearn:
    
    def __init__(
        self,
        net,
        dataAll,
        criterion,
        optimizer_dict,
        lr,
        lr_min,
        lr_factor,
        epoch_max,
        duration_max,
        patience_reduce_lr,
        patience_early_stop,
        device,
        metrics,
        name,
        path):
        
        self.quickTimer = QuickTimer()
        self.net = net
        self.dataAll = dataAll
        
        self.optimizer_dict = optimizer_dict
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.duration_max = duration_max
        self.epoch_max = epoch_max
        self.criterion = criterion

        self.device = device
        self.patience_reduce_lr = patience_reduce_lr
        self.patience_early_stop = patience_early_stop
        
        self.train_loss_list = []
        self.valid_loss_list = []
        self.test_loss_list = []
        self.metrics_list = []
        self.lr_list = []
        
        self.name = name
        self.path = path
        self.ID = self.name+'_'+random_str()
        self.epoch = 0
        
        self.metrics = metrics
        
        self.set_optimizer()
        self.set_scheduler()
        
        self.model_name = 'NET.pt'
        self.optim_name = 'OPT.pt'
        self.sched_name = 'SCH.pt'
        
        self.create_save_path()
        
        print('ID:', self.ID)
        
    def set_optimizer(self):
        self.optimizer = self.optimizer_dict['optimizer'](self.net.parameters(), lr=self.lr, **self.optimizer_dict['param'])
        
    def set_scheduler(self):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.lr_factor,
            patience=self.patience_reduce_lr,
            eps=0,
            verbose=False)
        
    def train_iterate(self, dataLoader):
        self.epoch += 1
        self.net.train()
        loss_list = []
    
        for data in dataLoader:
            X = data['cxr'].to(self.device)
            Gaze = data['gaze'].to(self.device)
            Y = data['Y'].to(self.device)

            self.optimizer.zero_grad()
            Y_pred = self.net(X)
            net_list = self.net.compute_loss(y=Y, y_pred=Y_pred, y_gaze=Gaze, criteria=self.criterion)
            
            loss = net_list['loss_sum']
            loss.backward()

            self.optimizer.step()
            loss_list.append(loss.detach().clone().cpu())
            
            del data, X, Gaze, Y, Y_pred, loss, net_list
        
        return loss_list
    
    def eval_iterate(self, dataLoader):
        self.net.eval()
        loss_list = []
        loss_classification = []
        loss_dissimilarity = []
        metrics = self.metrics()
        
        with torch.no_grad():                
            for data in dataLoader:
                X = data['cxr'].to(self.device)
                Gaze = data['gaze'].to(self.device)
                Y = data['Y'].to(self.device)

                Y_pred = self.net(X)
                net_list = self.net.compute_loss(y=Y, y_pred=Y_pred, y_gaze=Gaze, criteria=self.criterion)
                           
                loss_list.append(net_list['loss_sum'].detach().clone().cpu())
                loss_classification.append(copy.deepcopy(net_list['loss_classification']))
                loss_dissimilarity.append(copy.deepcopy(net_list['loss_dissimilarity']))
                sigma_c = copy.deepcopy(net_list['sigma_c'])
                sigma_d = copy.deepcopy(net_list['sigma_d'])
                
                metrics.add_data(X, Y, self.net.linear.compute_y(Y_pred))
                
                del data, X, Gaze, Y, Y_pred, net_list

            r = {'loss_list':loss_list,
                 'metrics':metrics,
                 'loss_classification':loss_classification,
                 'loss_dissimilarity':loss_dissimilarity,
                 'sigma_c':sigma_c,
                 'sigma_d':sigma_d}
            
            return r
    
    def save_net(self, path):
        torch.save(self.net.state_dict(), path+'/'+self.model_name)
        torch.save(self.optimizer.state_dict(), path+'/'+self.optim_name)
        torch.save(self.scheduler.state_dict(), path+'/'+self.sched_name)
    
    def load_net(self, path):
        self.net.load_state_dict(torch.load(path+'/'+self.model_name))
        self.net.eval()
        self.optimizer.load_state_dict(torch.load(path+'/'+self.optim_name))
        for pg in self.optimizer.param_groups:
            if len(self.lr_list)>0:
                pg['lr'] = self.lr_list[-1]
        self.scheduler.load_state_dict(torch.load(path+'/'+self.sched_name))
        
    def create_save_path(self):
        self.save_path = self.path+'/'+self.ID
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        else:
            print('Training folder already exists!')
        
    def train(self):
        self.valid_loss_min = -np.Inf
        self.epoch_best = 0
        while self.epoch<self.epoch_max:
            time0 = time.perf_counter()
            train_loss = self.train_iterate(self.dataAll('Train'))
            eval_valid = self.eval_iterate(self.dataAll('Valid'))

            self.train_loss_list.append(np.mean(train_loss))
            self.valid_loss_list.append(np.mean(eval_valid['loss_list']))

            eval_metrics = eval_valid['metrics'].get_key_evaluation()
            self.metrics_list.append(eval_metrics)

            self.scheduler.step(self.valid_loss_list[-1])
            self.lr_list.append(self.optimizer.param_groups[0]['lr'])

            print('Epoch:%4d, loss_train:%.6f, loss_valid:%.6f [%.4e %.4e (%.3e, %.3e)], acc:%3.3f, time:%3.2fsec'%(\
                    self.epoch,\
                    self.train_loss_list[-1],\
                    self.valid_loss_list[-1],\
                    np.mean(eval_valid['loss_classification']),\
                    np.mean(eval_valid['loss_dissimilarity']),\
                    eval_valid['sigma_c'],\
                    eval_valid['sigma_d'],\
                    self.metrics_list[-1],\
                    time.perf_counter()-time0))

            del eval_valid
            del eval_metrics
            del train_loss

            if self.train_judgetment():
                break

        self.load_net(self.save_path)
        self.save_training_process()
                
    def train_judgetment(self):
        
        if self.epoch==1:
            self.save_net(self.save_path)
            self.valid_loss_min = self.valid_loss_list[-1]
        else:
            if self.valid_loss_list[-1]<self.valid_loss_min:
                self.valid_loss_min = self.valid_loss_list[-1]
                self.save_net(self.save_path)
                self.epoch_best = self.epoch
                print('- Better network saved')
                
            if self.lr_list[-1]<self.lr_list[-2]:
                print('- Learning rate reduced to %e'%self.lr_list[-1])
                self.load_net(self.save_path)
                print('- Currently best network from epoch %4d reloaded'%self.epoch_best)
                
        if self.lr_list[-1]<self.lr_min:
            print('- Early stopping: learning rate dropped below threshold at %E'%self.lr_min)
            return True
        
        if self.epoch-self.epoch_best>=self.patience_early_stop:
            print('- Early stopping: max non-improving epoch reached at %d'%(self.epoch-self.epoch_best))
            return True
        
        if self.quickTimer()>=self.duration_max:
            print('- Early stopping: Max duration reached %f>=%f (sec)'%(self.quickTimer(), self.duration_max))
            return True
    
    def save_training_process(self):
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(self.train_loss_list)
        plt.plot(self.valid_loss_list)
        plt.title('loss')

        plt.subplot(3,1,2)
        plt.plot(self.lr_list)
        plt.title('learning rate')

        plt.subplot(3,1,3)
        plt.plot(self.metrics_list)
        plt.title('metrics')
        plt.savefig(self.save_path+'/training_process.png')
        
        plt.close()
        
    def remove_saved_net(self):
        if not hasattr(self, 'model_name'):
            print("The net file does not exist")
            return
        
        if not hasattr(self, 'save_path'):
            print("The net file does not exist")
            return
        
        if os.path.exists(self.save_path+'/'+self.model_name):
            os.remove(self.save_path+'/'+self.model_name)
            print("Saved network file deleted successfully")
        else:
            print("The net file does not exist")
            
    def remove_saved_optim(self):
        if not hasattr(self, 'optim_name'):
            print("The optim file does not exist")
            return
        
        if not hasattr(self, 'save_path'):
            print("The optim file does not exist")
            return
        
        if os.path.exists(self.save_path+'/'+self.optim_name):
            os.remove(self.save_path+'/'+self.optim_name)
            print("Saved optim file deleted successfully")
        else:
            print("The optim file does not exist")
            
    def remove_saved_sched(self):
        if not hasattr(self, 'sched_name'):
            print("The sched file does not exist")
            return
        
        if not hasattr(self, 'save_path'):
            print("The sched file does not exist")
            return
        
        if os.path.exists(self.save_path+'/'+self.sched_name):
            os.remove(self.save_path+'/'+self.sched_name)
            print("Saved sched file deleted successfully")
        else:
            print("The sched file does not exist")
            
    def remove_saved(self):
        self.remove_saved_net()
        self.remove_saved_optim()
        self.remove_saved_sched()
            
    def save_params(self, name, path):
        
        attr_list = [attr for attr in dir(self) if isinstance(getattr(self, attr), (list, tuple, dict, int, float, bool)) and not attr.startswith("_")]
        content = {}
        for attr in attr_list:
            content[attr] = getattr(self, attr)
                
        with open(path+'/'+'params_%s_'%(self.__class__.__name__)+name+'.txt', 'w') as file:
            try:
                json.dump(content, file, indent=4)
            except:
                print('Exception occured at hzhu_learn::NetLearn.save_params(..): content cannot be dumped!')
                file.write(str(content))
        
    def evaluate(self):
        
        eval_test = self.eval_iterate(self.dataAll('Test'))
        eval_test['metrics'].compute_classification_report()
        eval_test['metrics'].save_classification_report('classification_report', self.save_path)
        eval_test['metrics'].save_outputs('classification_results', self.save_path)
        
        r = {key:eval_test['metrics'].classification_report[key]\
             for key in eval_test['metrics'].classification_report if 'ROC_AUC' in key}
        r['accuracy'] = eval_test['metrics'].classification_report['accuracy']
        
        return json.dumps(r, indent=4)
    
    def interpretate(self):
        self.net.eval()
        n = 1
        r = {'heart_N':0, 'lung_N':0, 'normal_N':0}
        r['heart_%d'%n] = 0
        r['lung_%d'%n] = 0
        
        with torch.no_grad():                
            for data in self.dataAll('Test'):
                X_device = data['cxr'].to(self.device)
                Y = data['Y']
                CAM = self.net.forward(X_device).clone().detach().cpu().numpy()
                
                for i in range(Y.shape[0]):
                    cam = CAM[i,Y[i],:,:]
                    heart = data['heart'][i,:,:].numpy()
                    lung = data['lung'][i,:,:].numpy()
                    idx = np.unravel_index(np.argmax(cam, axis=None), cam.shape)
                    
                    if Y[i]==1:
                        r['heart_N'] += 1
                        idx_local = index_expand(idx, heart, n)
                        for idn in idx_local:
                            if heart[idn] == True:
                                r['heart_%d'%n] += 1
                                break                        
                    elif Y[i]==2:
                        r['lung_N'] += 1

                        idx_local = index_expand(idx, lung, n)
                        for idn in idx_local:
                            if lung[idn] == True:
                                r['lung_%d'%n] += 1
                                break
                    else:
                        r['normal_N'] += 1
                
                del X_device

        interp = {}
        interp['heart_interp_%d'%n] = r['heart_%d'%n]/r['heart_N']
        interp['lung_interp_%d'%n] = r['lung_%d'%n]/r['lung_N']
        return json.dumps(interp, indent=4)
    
def index_expand(idx, image, n):
    a, b = idx
    r = []
    for i in range(a-n,a+n+1):
        for j in range(b-n,b+n+1):
            if i>=0 and i<image.shape[0] and j>=0 and j<image.shape[1]:
                r.append((i,j))
    return tuple(r)

def param_select(idx, param_pool):
    if not isinstance(param_pool, (list, tuple)):
        assert False, 'input param_pool type error @hzhu_learn::param_select(idx, param_pool)'
    N = 1
    for item in param_pool:
        if not isinstance(item, (list, tuple)):
            assert False, 'input param_pool content type error @hzhu_learn::param_select(idx, param_pool)'
        N *= len(item)
    n = len(param_pool)
    shape = tuple([len(item) for item in param_pool])
    idx_list = np.unravel_index(range(N), shape)
    
    values = []
    for i in range(n):
        value = param_pool[i][idx_list[i][idx%idx_list[i].shape[0]]]
        values.append(value)
    
    if len(values)==1: return values[0]
    else: return values