import my_EffNetv2 as EN
from my_net import *

class EffCAMNet(EN.EffNetV2):
    
    def __init__(self, net_type, class_num, device, lg_sigma_d, lg_sigma_c, lg_scale, bias, dissimilarity_loss):
        self.init_cfgs(net_type)
        self.device = device
        self.class_num = class_num
        self.lg_sigma_d = lg_sigma_d
        self.dissimilarity_loss = dissimilarity_loss
        self.lg_sigma_c = lg_sigma_c
        self.bias = bias
        self.lg_scale = lg_scale
        super().__init__(cfgs=self.cfgs, num_classes=self.class_num, width_mult=1.)
        
        self.cam_param_init()
        self.to(self.device)
        
    def cam_param_init(self):
        if self.dissimilarity_loss is not None:
            if self.lg_sigma_c is not None:
                self.lg_sigma_classification = nn.Parameter(torch.tensor(self.lg_sigma_c, dtype=torch.float32))
            else:
                self.lg_sigma_classification = torch.tensor(0.0, dtype=torch.float32)
            
            if self.lg_sigma_d is not None:
                self.lg_sigma_dissimilarity = nn.Parameter(torch.tensor(self.lg_sigma_d, dtype=torch.float32))
            else:
                self.lg_sigma_dissimilarity = torch.tensor(0.0, dtype=torch.float32, requires_grad=False)
                
            if self.lg_scale is not None:
                self.lg_k = nn.Parameter(torch.tensor(self.lg_scale, dtype=torch.float32))
            else:
                self.lg_k = torch.tensor(0.0, dtype=torch.float32, requires_grad=False)
                
            if self.bias is not None:
                self.shift = nn.Parameter(torch.tensor(self.bias, dtype=torch.float32))
            else:
                self.shift = torch.tensor(0.0, dtype=torch.float32, requires_grad=False)
            
            self.DS_loss = self.dissimilarity_loss()
            
    def get_linear(self):
        return CAMLayer(in_features=1792, out_features=self.class_num)
    
    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.linear(x)
        return x
    
    def pred(self, x):
        y = self.forward(x)
        y = self.linear.compute_y(y)
        return y
    
    def compute_loss_classification(self, y, y_pred, criteria):
        y_pred = self.linear.compute_y(y_pred)
        return criteria(y_pred, y)
    
    def compute_loss_dissimilarity(self, y, y_pred, y_gaze):
        cam = torch.stack([y_pred[i,int(y[i]),:,:] for i in range(y.shape[0])], dim=0).unsqueeze(1)
        r = self.DS_loss(cam, y_gaze)
        del cam
        return r
    
    def compute_loss(self, y, y_pred, y_gaze, criteria):
        if self.dissimilarity_loss is None:
            loss_classification = self.compute_loss_classification(y=y, y_pred=y_pred, criteria=criteria)
            r = {'loss_sum': loss_classification, 
                'loss_classification': float(loss_classification.detach().clone().cpu()),
                'loss_dissimilarity': 0.0,
                'sigma_c': 0.0,
                'sigma_d': 0.0}
            return r
        else:
            loss_classification = self.compute_loss_classification(y=y, y_pred=y_pred, criteria=criteria)
            sigma_c = torch.exp(self.lg_sigma_classification)
            loss_classification_MTL = loss_classification/sigma_c/sigma_c+torch.log(sigma_c+1)

            loss_dissimilarity = self.compute_loss_dissimilarity(
                y=torch.sigmoid(y*torch.exp(self.lg_k)+self.shift),
                y_pred=y_pred,
                y_gaze=y_gaze)
            sigma_d = torch.exp(self.lg_sigma_dissimilarity)
            loss_dissimilarity_MTL = loss_dissimilarity/sigma_d/sigma_d/2+torch.log(sigma_d+1)

            r = {'loss_sum': loss_classification_MTL+loss_dissimilarity_MTL, 
                'loss_classification': float(loss_classification.detach().clone().cpu()),
                'loss_dissimilarity': float(loss_dissimilarity.detach().clone().cpu()),
                'sigma_c': float(sigma_c.detach().clone().cpu()),
                'sigma_d': float(sigma_d.detach().clone().cpu())}

            del loss_classification, sigma_c, loss_classification_MTL, loss_dissimilarity, sigma_d, loss_dissimilarity_MTL
            return r
        
    def init_cfgs(self, net_type):
        self.net_type = net_type
        if self.net_type == 's' or self.net_type == 'S':
            self.cfgs = [
                # t, c, n, s, SE
                [1,  24,  2, 1, 0],
                [4,  48,  4, 2, 0],
                [4,  64,  4, 2, 0],
                [4, 128,  6, 2, 1],
                [6, 160,  9, 1, 1],
                [6, 272, 15, 2, 1],]
        elif self.net_type == 'm' or self.net_type == 'M':
            self.cfgs = [
                # t, c, n, s, SE
                [1,  24,  3, 1, 0],
                [4,  48,  5, 2, 0],
                [4,  80,  5, 2, 0],
                [4, 160,  7, 2, 1],
                [6, 176, 14, 1, 1],
                [6, 304, 18, 2, 1],
                [6, 512,  5, 1, 1],]
        elif self.net_type == 'L' or self.net_type == 'l':
            self.cfgs = [
                # t, c, n, s, SE
                [1,  32,  4, 1, 0],
                [4,  64,  7, 2, 0],
                [4,  96,  7, 2, 0],
                [4, 192, 10, 2, 1],
                [6, 224, 19, 1, 1],
                [6, 384, 25, 2, 1],
                [6, 640,  7, 1, 1],]
        else:
            self.cfgs = [
                # t, c, n, s, SE
                [1,  32,  4, 1, 0],
                [4,  64,  8, 2, 0],
                [4,  96,  8, 2, 0],
                [4, 192, 16, 2, 1],
                [6, 256, 24, 1, 1],
                [6, 512, 32, 2, 1],
                [6, 640,  8, 1, 1],]