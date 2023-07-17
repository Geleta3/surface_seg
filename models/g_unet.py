
import torch 
from torch import nn 
import torch.nn.functional as F
from collections import OrderedDict
from .unet import Downsample, Conv 
from typing import List 
# from .image_processing import MultiInputImage


class InputBlock(nn.Module):
    def __init__(self, conv_channels):
        super().__init__()
 
        convs = [nn.Conv2d(conv_channels[i], conv_channels[i+1], 3, stride=1, padding=1) 
                 for i in range(len(conv_channels)-1)]
        bn = [nn.BatchNorm2d(conv_channels[i]) for i in range(1, len(conv_channels))]
        act = [nn.ReLU() for _ in range(len(conv_channels)-1)]

        net = []
        for i in range(len(conv_channels)-1):
            net.append(convs[i])
            net.append(bn[i])
            net.append(act[i])
        self.net = nn.Sequential(*net)
        
    def forward(self, x):
        return self.net(x)


class MultiInputNet(nn.Module):
    def __init__(self, img_channels: List, conv_channels: List):
        super().__init__()
        
        self.img_channels = img_channels
        inp_conv = [nn.Conv2d(img_channels[i], conv_channels[0], kernel_size=3, padding=1)
                    for i in range(len(img_channels))]
        middle_conv = [InputBlock(conv_channels=conv_channels) for _ in range(len(img_channels))]
        
        net = []
        for i in range(len(img_channels)):
            net.append(nn.Sequential(inp_conv[i], middle_conv[i]))
            
        self.net = nn.ModuleList(net)
        self.out = nn.ModuleList(
            [nn.Conv2d(conv_channels[-1], 1, 3, padding=1) for _ in range(len(img_channels))]
        )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x: List[torch.Tensor]):
        feats = []
        s = 0
        for idx, layer in enumerate(self.net):
            feats.append(layer(x[:, s:s+self.img_channels[idx], :, :]))
            s += self.img_channels[idx]
            
        out = []
        for idx, layer in enumerate(self.out):
            out.append(layer(feats[idx]))
            
        out = torch.cat(out, dim=1)
        att_weight = self.softmax(out)
        
        attened_feats = []
        for i, feat in enumerate(feats):
            attened_feats.append(feat*att_weight[:, i, :, :].unsqueeze(1))

        return torch.cat(attened_feats, dim=1)


class SigmoidAttention(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, groups=channel),  # Depthwise
            nn.Conv2d(channel, channel, 1, padding=0),                  # Pointwise 
            nn.Conv2d(channel, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        atten_weight = self.net(x)
        return atten_weight*x


class SoftmaxAttention(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, groups=channel), 
            nn.Conv2d(channel, channel, 1, padding=0),
            nn.Conv2d(channel, 1, kernel_size=3, padding=1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        atten_weight = self.net(x)
        return atten_weight*x
    
    
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel, skip_channel, 
                 use_batch_norm=True, up_sample=True, att_mode='concat', first_att='sig'):   # mode in ['concat', 'add', 'seq']
        super().__init__()
        
        self.att_mode = att_mode
        self.first_att = first_att  #When mode == seq
        self.sigmoid_att = SigmoidAttention(channel=skip_channel)
        self.softmax_att = SoftmaxAttention(channel=skip_channel)
        
        if 'concat' in att_mode: 
            skip_channel = 2*skip_channel
        if up_sample:
            self.up = nn.Upsample(scale_factor=2,  mode='bilinear',  align_corners=False)
            self.conv2 = Conv(in_channel+skip_channel, out_channel, use_batch_norm=use_batch_norm)
        else:
            self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)
            self.conv2 = Conv(out_channel+skip_channel, out_channel, use_batch_norm=use_batch_norm)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Attention on skip_feats. 
        if 'concat' in self.att_mode:
            sig_out = self.sigmoid_att(x2)
            soft_out = self.softmax_att(x2)
            x2 = torch.cat([sig_out, soft_out], dim=1)
        elif 'add' in self.att_mode:
            sig_out = self.sigmoid_att(x2)
            soft_out = self.softmax_att(x2)
            x2 = sig_out + soft_out
        elif 'seq' in self.att_mode:
            if 'sig' in self.first_att:
                sig_out = self.sigmoid_att(x2)
                x2 = self.softmax_att(sig_out)
            elif 'soft' in self.first_att:
                soft_out = self.softmax_att(x2)
                x2 = self.sigmoid_att(soft_out)
                  
        x = torch.cat([x1, x2], dim=1)
        
        return self.conv2(x)
    
    
class Encoder(nn.Module):
    def __init__(self, img_channels, channels, inp_conv_channels, 
                 use_batch_norm=True, max_pool=True, 
                 ):
        super().__init__()
        
        self.input_net = MultiInputNet(img_channels=img_channels, 
                                       conv_channels=inp_conv_channels)
        
        channels[0] = inp_conv_channels[-1]*len(img_channels)
        
        self.conv = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1), 
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
        # x = self.multi_inp_img(x)
        x = self.input_net(x)

        middle_output = OrderedDict()
        for layer in self.net:
            x = self.net[layer](x)
            middle_output[layer] = x 
        return x, middle_output

 
class Decoder(nn.Module):
    def __init__(self, channels, skip_channels, up_sample=True, use_batch_norm=True, 
                 att_mode='concat', first_att='sig'):
        super().__init__()
        
        # if att_mode == 'concat':
        #     skip_channels = [sc*2 for sc in skip_channels]
            
        layers = OrderedDict()
        for i in range (len(channels)-1):#(len(channels)-1, -1, -1):
            layers[f'layer_{len(channels)-2-i}'] = Upsample(channels[i], channels[i+1], skip_channels[i], 
                                                            use_batch_norm=use_batch_norm, up_sample=up_sample, 
                                                            att_mode=att_mode, first_att=first_att)
        
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
        