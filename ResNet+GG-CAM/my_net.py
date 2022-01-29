import torch, json
import torch.nn as nn
from torch.nn import functional as F

from torch.utils.tensorboard import SummaryWriter

class Module(nn.Module):
    
    def visualize(self, x):
        # tensorboard --logdir=runs
        writer = SummaryWriter('runs/%s'%self.__class__.__name__)
        writer.add_graph(self, x)
        writer.close()
        print('tensorboard --logdir=runs')
        
    def save_params(self, path, name):
        content = {}
        self.total_param = self.get_total_param()
        for attr in dir(self):
            if attr[0]=='_':
                continue
            attr_instance = getattr(self, attr)
            if callable(attr_instance):
                continue
            if isinstance(attr_instance, (float, int, bool, list, str)):
                content[attr] = attr_instance
                
        with open(path+'/'+'params_%s_'%(self.__class__.__name__)+name+'.txt', 'w') as file:
            try:
                json.dump(content, file, indent=4)
            except:
                print('Exception occured at hzhu_resnet::%s.save_params(..): content cannot be dumped!'%self.__class__.__name__)
                file.write(str(content))
                
    def get_total_param(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def conv_bn_acti(
    in_channels,
    out_channels,
    kernel_size,
    activation=nn.ReLU,
    sequential=None,
    name='',
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
    padding_mode='zeros'):
    
    if sequential is None:
        r = nn.Sequential()
    else:
        r = sequential
    
    if len(name)==0:
        connector = ''
    else:
        connector = '_'
        
    r.add_module(
        name+connector+'conv2d',
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode))
    
    r.add_module(
        name+connector+'BN',
        nn.BatchNorm2d(out_channels))
    
    if activation is not None:
        acti = activation()
        r.add_module(
            name+connector+acti.__class__.__name__,
            acti)
    
    return r

class CAMLayer(Module):
    
    def __init__(self, in_features, out_features):
        super().__init__()
        
        uniform_limit = (1.0/in_features)**0.5
        self.weight = nn.Parameter(torch.zeros(out_features,in_features,1,1).uniform_(-uniform_limit, uniform_limit))
        self.bias = nn.Parameter(torch.zeros(out_features).uniform_(-uniform_limit, uniform_limit))
        
    def forward(self, x):
        x = x.unsqueeze(1)
        r = (self.weight*x).mean(2)
        return r
    
    def compute_y(self, raw_y):
        return raw_y.mean((2,3))+self.bias