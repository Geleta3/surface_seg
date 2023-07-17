from unicodedata import ucd_3_2_0
import torch 
from torch import nn 
import torch.nn.functional as F
from collections import OrderedDict


class Conv(nn.Module):
    def __init__(self, inp_channel, out_channel, use_batch_norm=True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inp_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        
        if use_batch_norm:
            self.net = nn.Sequential(
                self.conv1, self.bn1, self.act1, self.conv2, self.bn2, self.act2 
            )
        else:
            self.net = nn.Sequential(
                self.conv1, self.act1, self.conv2, self.act2
            )
    
    def forward(self, x):
        return self.net(x)

class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel, use_batch_norm=True, max_pool=True):
        super().__init__()
        
        if max_pool:
            self.net = nn.Sequential(
                nn.MaxPool2d(2), 
                Conv(in_channel, out_channel, use_batch_norm=use_batch_norm)
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1), 
                Conv(out_channel, out_channel, use_batch_norm=use_batch_norm) #  kernel_size=3,
                )
    
    def forward(self, x):
        return self.net(x)
        
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel, skip_channel, use_batch_norm=True, up_sample=True):
        super().__init__()
        
        if up_sample:
            self.up = nn.Upsample(scale_factor=2,  mode='bilinear',  align_corners=False)
            self.conv2 = Conv(in_channel+skip_channel, out_channel, use_batch_norm=use_batch_norm)
        else:
            self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)
            self.conv2 = Conv(out_channel+skip_channel, out_channel, use_batch_norm=use_batch_norm)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        
        return self.conv2(x)
    

class Encoder(nn.Module):
    def __init__(self, img_channel=3, channels=[], use_batch_norm=True, max_pool=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(img_channel, channels[0], kernel_size=3, stride=1, padding=1), 
            nn.ReLU()
        )
        
        layers = OrderedDict()
        layers[f'layer_0'] = self.conv 
        for i in range(len(channels)-1):
            layers[f'layer_{i+1}'] = Downsample(channels[i], channels[i+1], 
                                                use_batch_norm=use_batch_norm, 
                                                max_pool=max_pool)
        self.net = nn.ModuleDict(layers)
    
    def forward(self, x):

        middle_output = OrderedDict()
        for layer in self.net:
            x = self.net[layer](x)
            middle_output[layer] = x 
        return x, middle_output

 
class Decoder(nn.Module):
    def __init__(self, channels, skip_channels, up_sample=True, use_batch_norm=True):
        super().__init__()
        
        layers = OrderedDict()
        for i in range (len(channels)-1):#(len(channels)-1, -1, -1):
            layers[f'layer_{len(channels)-2-i}'] = Upsample(channels[i], channels[i+1], skip_channels[i], 
                                                            use_batch_norm=use_batch_norm, up_sample=up_sample)
        
        self.layers = nn.ModuleDict(layers)
    
    def forward(self, x1, x2):
        for layer in self.layers:
            x1 = self.layers[layer](x1, x2[layer])
        return x1  



if __name__ == '__main__':
    dummy = torch.randn(1, 1, 224, 224)
    model = Model(img_channel=1, 
                  enc_channels=[64, 256, 512, 256, 512], 
                  dec_channels=[512, 256, 256, 512, 64], 
                  classes=3, 
                  use_batch_norm=True, 
                  up_sample=True, 
                  use_max_pool=True)
    
    out = model(dummy)
    print(out.shape)
        