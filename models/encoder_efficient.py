
from typing import OrderedDict
import torch 
from torch import nn 
import torchvision
from torchvision import models 
from torch.utils import model_zoo as model_zoo

# intermidiate_feat = {}
# def get_feats(name):
#     def hook(model, input, output):
#         intermidiate_feat[name] = output
#     return hook 

model_urls = {
    'b0': 'https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth',
    'b1': 'https://download.pytorch.org/models/efficientnet_b1-c27df63c.pth',               # v2 
    'b2': 'https://download.pytorch.org/models/efficientnet_b2_rwightman-bcdf34b7.pth',
    'b3': 'https://download.pytorch.org/models/efficientnet_b3_rwightman-cf984f9c.pth', 
    'b4': 'https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth',
    'b5': 'https://download.pytorch.org/models/efficientnet_b5_lukemelas-b6417697.pth', 
    'b6': 'https://download.pytorch.org/models/efficientnet_b6_lukemelas-c76e70fd.pth', 
    'b7': 'https://download.pytorch.org/models/efficientnet_b7_lukemelas-dcc49843.pth', 
    
}

class EfficientNet(nn.Module):
    def __init__(self, model='b0') -> None:
        super().__init__()
        
        self.feats = {}
        if model == 'b0':
            self.efficient = models.efficientnet_b0()
        elif model == 'b1':
            self.efficient = models.efficientnet_b1()
        elif model == 'b2':
            self.efficient = models.efficientnet_b2()
        elif model == 'b3':
            self.efficient = models.efficientnet_b3()
        elif model == 'b4':
            self.efficient = models.efficientnet_b4()
        elif model == 'b5':
            self.efficient = models.efficientnet_b5()
        elif model == 'b6':
            self.efficient = models.efficientnet_b6()
        elif model == 'b7':
            self.efficient = models.efficientnet_b7()
        else:
            raise ValueError('Model should be [b0, b1, b2, b3, b4, b5, b6, b7]')
            
        self.efficient.load_state_dict(model_zoo.load_url(model_urls[model]))
        
        self.efficient_od = OrderedDict()
        for name, layer in self.efficient.named_children():
            if name not in  ['avgpool', 'classifier']:
                self.efficient_od[name] = layer
                 
        count_downsampling = 0
        for idx, layers in enumerate(self.efficient_od['features']):
            if idx in [1, 2, 3, 5, 8]:
                layers.register_forward_hook(self.get_feats(f'feat_{count_downsampling}'))
                count_downsampling += 1
                # print('cn', count_downsampling)
                
        assert count_downsampling == 5, f"Need five strides of downsampling but got{count_downsampling}"
        
    def forward(self, x):
        for name in self.efficient_od:
            x = self.efficient_od[name](x)
        return x
    
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

##########################################
# Table EfficientNet-B0:
# 0 - 2x    torch.Size([1, 32, 112, 112])
# 1 - 2x    torch.Size([1, 16, 112, 112])
# 2 - 4x    torch.Size([1, 24, 56, 56])
# 3 - 8x    torch.Size([1, 40, 28, 28])
# 4 - 16x   torch.Size([1, 80, 14, 14])
# 5 - 16x   torch.Size([1, 112, 14, 14])
# 6 - 32x   torch.Size([1, 192, 7, 7])
# 7 - 32x   torch.Size([1, 320, 7, 7])
# 8 - 32x   torch.Size([1, 1280, 7, 7])
#############################################
# [1, 2, 3, 5, 8]

if __name__ == '__main__':
    # check 
    model = EfficientNet('b1')
    img = torch.randn(1, 3, 224, 224)
    out = model(img)
    print(model.feats_shape())







