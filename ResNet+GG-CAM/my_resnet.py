import torch, json
import torch.nn as nn
from torch.nn import functional as F

from my_net import *

from torch.utils.tensorboard import SummaryWriter

def res_layer_branch(in_channels, mid_channels, out_channles, stride):
    r = nn.Sequential()
    r = conv_bn_acti(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            activation=nn.ReLU,
            stride=stride,
            sequential=r,
            padding=0,
            name='1st')
    r = conv_bn_acti(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            activation=nn.ReLU,
            stride=1,
            sequential=r,
            padding=1,
            name='2nd')
    r = conv_bn_acti(
            in_channels=mid_channels,
            out_channels=out_channles,
            kernel_size=1,
            activation=None,
            stride=1,
            sequential=r,
            padding=0,
            name='3rd')
    return r

class ResLayerA(Module):
    
    def __init__(self, in_channels, mid_channels, out_channles):
        super().__init__()
        self.left_branch = res_layer_branch(in_channels=in_channels, mid_channels=mid_channels, out_channles=out_channles, stride=2)
        self.right_branch = conv_bn_acti(
            in_channels=in_channels, out_channels=out_channles, kernel_size=1, activation=None, stride=2)
        
    def forward(self, x):
        return F.relu(self.left_branch(x)+self.right_branch(x))
    
class ResLayerB(Module):
    
    def __init__(self, in_channels, mid_channels, out_channles):
        super().__init__()
        self.left_branch = res_layer_branch(in_channels=in_channels, mid_channels=mid_channels, out_channles=out_channles, stride=1)
        
    def forward(self, x):
        return F.relu(self.left_branch(x)+x)
    
def ResBlock(in_channels, mid_channels, out_channles, repeat, sequential, name):
    
    if sequential is None: r = nn.Sequential()
    else: r = sequential

    for i in range(repeat):
        if i==0: r.add_module(name+'_'+'ResLayer%02i'%i, ResLayerA(in_channels, mid_channels, out_channles))
        else: r.add_module(name+'_'+'ResLayer%02i'%i, ResLayerB(out_channles, mid_channels, out_channles))
    
    return r
    
class ResNet(Module):
    
    def __init__(self, in_channels, class_num, filters, block_list):
        super().__init__()
        
        self.layer = conv_bn_acti(
            in_channels=in_channels, out_channels=filters, kernel_size=7, stride=2,
            activation=nn.ReLU, sequential=None, padding=3, name='input')
        
        local_in_channels = filters
        local_out_channles = local_in_channels*4
        local_mid_channels = filters
        for i, repeat in enumerate(block_list):
            self.layer = ResBlock(
                in_channels=local_in_channels, mid_channels=local_mid_channels,
                out_channles=local_out_channles, repeat=repeat, sequential=self.layer, name='ResBlock_%02i'%i)
            local_in_channels = local_out_channles
            local_mid_channels = int(local_in_channels/2)
            local_out_channles = local_out_channles*2
            
        self.linear = self.get_linear()
        
    def get_linear(self):
        return nn.Linear(in_features=2048, out_features=self.class_num)

    def forward(self, x):
        x = self.layer(x)
        x = x.mean([2,3])
        x = self.linear(x)
        
        return x
    
class ResNetPreset(ResNet):
    
    def __init__(self, net_type, in_channels, class_num, filters):
        type_block_list = {
            '18': [2, 2, 2, 2],
            '34': [3, 4, 6, 3],
            '50': [3, 4, 6, 3],
            '101': [3, 4, 23, 3],
            '152': [3, 8, 36, 3]}
        
        super().__init__(in_channels=in_channels, class_num=class_num, filters=filters, block_list=type_block_list[net_type])
        self.net_type = net_type
        self.in_channels = in_channels
        self.class_num = class_num
        self.filters = filters
        self.total_param = self.get_total_param()