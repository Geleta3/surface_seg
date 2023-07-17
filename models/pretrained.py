from collections import OrderedDict
from turtle import forward
import torch 
from torch import nn 
import torchvision
from torchvision import models 
from torch.utils import model_zoo as model_zoo


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth', 
    'efficient-b0': 'https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth', 
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth'
}


class Encoder(nn.Module):
    def __init__(self, img_channels, conv_channels, model='resnet50'):
        super().__init__()

        if model in ['resnet50', 'R-50']:
            self.pretrained = Resnet(1, 64)
        elif model in ['mobilenet', 'mobilenet_v2', 'M']:
            self.pretrained = Mobilenet(1, 32)
        elif model in ['efficient-b0', 'E-b0']:
            self.pretrained = EfficientB0(1, 32)
        else:
            raise ValueError()
            
            
    def forward(self, img):
        feats = {}
        x, att = self.input_net(img)
        feats['layer_0'] = x
        x = self.pretrained(x)
        for f in self.pretrained.feats:
            feats[f] = self.pretrained.feats[f]
        return x, feats, att
        
class Helper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feats = {}
    
    def get_feats(self, name):
        def hook(model, input, output):
            self.feats[name] = output
        return hook 
    
    def feats_shape(self):
        feat_shape = {}
        assert self.feats.__len__(), "The model didn't recieve input yet!"
        for feat in self.feats:
            feat_shape[feat] = self.feats[feat].shape
        return feat_shape
    
    def register(self, layers):
        count_downsampling = 1
        for layer in layers:
            layer.register_forward_hook(self.get_feats(f'layer_{count_downsampling}'))
            count_downsampling += 1

class Resnet(Helper):
    def __init__(self, inp_shape, pre_shape):
        super().__init__()
        
        self.conv1 = nn.Conv2d(inp_shape, pre_shape, 3, 2, padding=1)
        model = models.resnet50(pretrained=True)
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        model_od = OrderedDict()
        for name, val in model.named_children():
            if name not in ['avgpool', 'fc', 'conv1']:
                model_od[name] = val
        self.resnet = nn.ModuleDict(model_od)
        self.register(self.resnet50_layers())
    
    def forward(self, x):
        x = self.conv1(x)
        for name in self.resnet:
            x = self.resnet[name](x)
        return x
    
    def resnet50_layers(self):
        feats = [ 
            self.resnet.relu, 
            self.resnet.layer2[0].bn1, 
            self.resnet.layer3[0].bn1, 
            self.resnet.layer4[0].bn1,
            self.resnet.layer4[0].relu
        ]
        return feats 


class EfficientB0(Helper):
    def __init__(self, inp_shape, pre_shape):
        super().__init__()
        
        self.conv1 = nn.Conv2d(inp_shape, pre_shape, 3, 2, padding=1)
        model = models.efficientnet_b0(pretrained=True)
        model.load_state_dict(model_zoo.load_url(model_urls['efficient-b0']))
        model_od = OrderedDict()
        for name, val in model.named_children():
            if name not in ['avgpool', 'classifier']:
                model_od[name] = val
                
        self.efficient = nn.Sequential(*(list(model_od['features'].children())[1:]))
        self.register(self.efficient_layers())
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.efficient(x)
        return x
    
    def efficient_layers(self):
        
        feats = [
            self.efficient[1][0].block[0][2],
            self.efficient[2][0].block[0][2],
            self.efficient[3][0].block[0][2], 
            self.efficient[5][0].block[0][2], 
            self.efficient[7][2]
        ]
        return feats 


class Mobilenet(Helper):
    def __init__(self, inp_shape, pre_shape):
        super().__init__()
        
        self.conv1 = nn.Conv2d(inp_shape, pre_shape, 3, 2, padding=1)
        
        model = models.mobilenet_v2(pretrained=True)
        model.load_state_dict(model_zoo.load_url(model_urls['mobilenet_v2']))
        model_od = OrderedDict()
        for name, val in model.named_children():
            if name not in ['avgpool', 'classifier']:
                model_od[name] = val

        self.mobile = nn.Sequential(*(list(model_od['features'].children())[1:]))
        self.register(self.mobile_layers())
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.mobile(x)
        return x
    
    def mobile_layers(self):

        feats = [
            self.mobile[1].conv[0][2],
            self.mobile[3].conv[0][2],
            self.mobile[6].conv[0][2], 
            self.mobile[13].conv[0][2], 
            self.mobile[17][2]
        ]
        return feats 


    
if __name__ == '__main__':
    img = torch.randn(1, 256, 224, 224)
    model = Resnet(256, 64)
    out = model(img)
    
    for f in model.feats:
        print(f, model.feats[f].shape)
  
    
    