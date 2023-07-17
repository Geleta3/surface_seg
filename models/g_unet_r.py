
import torch 
from torch import nn 
import torch.nn.functional as F
from collections import OrderedDict
from .unet import Downsample, Conv 
from typing import List 
# from .image_processing import MultiInputImage

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)



class InputBlock(nn.Module):
    def __init__(self, conv_channels):
        super().__init__()
 
        convs = [nn.Conv2d(conv_channels[i], conv_channels[i+1], 3, stride=1, padding=1) 
                 for i in range(len(conv_channels)-1)]
        bn = [nn.BatchNorm2d(conv_channels[i]) for i in range(1, len(conv_channels))]
        act = [nn.ReLU() for _ in range(len(conv_channels))]

        net = []
        for i in range(len(conv_channels)-1):
            net.append(convs[i])
            net.append(bn[i])
            net.append(act[i])
        self.net = nn.Sequential(*net)
        
        self.net.apply(self.init_weight)
        
    def forward(self, x):
        return self.net(x)
    
    def init_weight(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            #nn.init.xavier_uniform(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)

class MultiInputNet_(nn.Module):
    def __init__(self, img_channels: List, conv_channels):
        super().__init__()
         
        self.img_channels = img_channels
        inp_conv = [nn.Conv2d(img_channels[i], conv_channels[0], kernel_size=3, padding=1)
                    for i in range(len(img_channels))]
        relu = [nn.ReLU() for i in range(len(img_channels))]
        middle_conv = [InputBlock(conv_channels=conv_channels) for _ in range(len(img_channels))]
        
        net = []
        for i in range(len(img_channels)):
            net.append(nn.Sequential(inp_conv[i], relu[i], middle_conv[i]))
            
        self.net = nn.ModuleList(net)
        
        cat_channel = conv_channels[-1]*2
        self.attention = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(cat_channel, cat_channel, 3, padding=1, groups=cat_channel),
                    nn.Conv2d(cat_channel, cat_channel, 1), 
                    nn.Conv2d(cat_channel, 1, 3, padding=1), 
                    nn.Tanh()
                    )
                for _ in range(len(img_channels)-1)
             ]
        )
    
    def forward(self, x):
        feats = []
        s = 0
        for idx, layer in enumerate(self.net):
            feats.append(layer(x[:, s:s+self.img_channels[idx], :, :]))
            s += self.img_channels[idx]
        
        attention_weights = []
        attended_feat = [feats[0]]
        for idx, layer in enumerate(self.attention):
            feat = torch.cat([feats[0] , feats[idx+1]], dim=1)
            weight = layer(feat)
            attention_weights.append(weight)
            attended_feat.append(feats[idx+1]*weight)
            
        out = torch.cat(attended_feat, dim=1)
       
        return out, attention_weights     


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
            
        if len(feats) == 1:
            # No need of attention 
            return feats[0], s
            
        out = []
        for idx, layer in enumerate(self.out):
            out.append(layer(feats[idx]))
            
        out = torch.cat(out, dim=1)
        att_weight = self.softmax(out)
        
        attened_feats = []
        for i, feat in enumerate(feats):
            attened_feats.append(feat*att_weight[:, i, :, :].unsqueeze(1))

        return torch.cat(attened_feats, dim=1), att_weight
    

class SigmoidAttention(nn.Module):
    def __init__(self, skip_channel, dec_channel):
        super().__init__()

        channel = skip_channel + dec_channel
        # self.net = nn.Sequential(
        #     nn.Conv2d(channel, channel, 3, padding=1, groups=channel),  # Depthwise
        #     nn.Conv2d(channel, channel, 1, padding=0),                  # Pointwise 
        #     nn.Conv2d(channel, 1, kernel_size=3, padding=1),
        #     nn.Sigmoid()
        # )
        # self.net = nn.Sequential(
        #     nn.Conv2d(channel, skip_channel, 3, padding=1), 
        #     nn.Conv2d(skip_channel, skip_channel, 1), 
        #     nn.Conv2d(skip_channel, 1, 1), 
        #     nn.Sigmoid()
        # )
        # self.net = nn.Sequential(
        #     nn.Conv2d(channel, channel, 1), 
        #     # nn.Conv2d(skip_channel, skip_channel, 1), 
        #     nn.Conv2d(channel, 1, 1), 
        #     nn.Sigmoid()
        # )
        
        self.net = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, groups=channel),  # Depthwise
            nn.Conv2d(channel, channel, 1, padding=0),                  # Pointwise 
            nn.Conv2d(channel, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )
        # self.net = nn.Sequential(
        #     nn.Conv2d(channel, channel, 3, padding=1, groups=channel),  # Depthwise
        #     nn.Conv2d(channel, channel, 1, padding=0),                  # Pointwise 
        #     nn.Conv2d(channel, channel, 3, padding=1, groups=channel),  # Depthwise
        #     nn.Conv2d(channel, channel, 1, padding=0),                  # Pointwise 
        #     nn.Conv2d(channel, 1, kernel_size=3, padding=1),
        #     nn.Sigmoid()
        # )
        
        # self.net = nn.Sequential(
        #     nn.Conv2d(channel, 1, kernel_size=3, padding=1),
        #     nn.Sigmoid()
        # )
        
    def forward(self, dec_feat, enc_feat):
        x = torch.cat([dec_feat, enc_feat], dim=1)
        atten_weight = self.net(x)
        return atten_weight #*x


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

        if up_sample:
            self.up = nn.Upsample(scale_factor=2,  mode='bilinear',  align_corners=False)
            self.conv2 = Conv(in_channel+skip_channel, out_channel, use_batch_norm=use_batch_norm)
            ch = in_channel
        else:
            self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)
            self.conv2 = Conv(out_channel+skip_channel, out_channel, use_batch_norm=use_batch_norm)
            ch = out_channel
            
        self.sigmoid_att = SigmoidAttention(skip_channel=skip_channel, dec_channel=ch)   
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        att_weight = self.sigmoid_att(dec_feat=x1, enc_feat=x2)
        attended_skip = x2*att_weight
        x = torch.cat([attended_skip, x1], dim=1)
        x = self.conv2(x)
        return x, att_weight
    
    
class Encoder(nn.Module):
    def __init__(self, img_channels, channels, inp_conv_channels, 
                 use_batch_norm=True, max_pool=True, 
                 ):
        super().__init__()
        
        self.img_channels = img_channels 
        if len(img_channels) >= 1:
            self.input_net = MultiInputNet(img_channels=img_channels, 
                                       conv_channels=inp_conv_channels)
        
            channels[0] = inp_conv_channels[-1]*len(img_channels)
            
            self.conv = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1), 
            nn.ReLU()
        )
        else:
            #channels[0] = 1
            self.conv = nn.Sequential(
            nn.Conv2d(1, channels[1], kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1), 
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
        
        if len(self.img_channels) >= 1:
            x, img_attention = self.input_net(x)
        else:
            x, img_attention = x[:, 0:1, :, :], 0

        middle_output = OrderedDict()
        for layer in self.net:
            x = self.net[layer](x)
            middle_output[layer] = x 
        return x, middle_output, img_attention

 
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
        skip_attentions = []
        for layer in self.layers:
            x1, skip_att = self.layers[layer](x1, x2[layer])
            skip_attentions.append(skip_att)
        return x1, skip_attentions



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
        