from audioop import bias
from bisect import bisect
from tokenize import group
import torch 
from torch import nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Tuple 


# [num_row, num_col] - kernel_size
# filters - out_channel 
# padding - same==1
class Conv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size, padding=1, 
            stride=1, name=None):
        self.conv = nn.Conv2d(in_channels, out_channels, 
                            kernel_size=kernel_size, 
                            stride=stride, 
                            padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        return self.act(self.bn(x))

class Conv2dBn(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels, 
                kernel_size,
                padding= 0, #'valid',
                activation=None,
                use_batchnorm=False,
                stride=1,
                
                data_format=None,
                dilation_rate=(1, 1),
                
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                
                **kwargs
        ):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size, padding=padding, stride=stride, 
                              bias = not(use_batchnorm))
        if use_batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)
        if activation:
            if activation == 'relu':
                self.act =  nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        return self.act(self.bn(x))


class Conv3x3BnReLU(nn.Module):
    def __init__(self, in_channels,
                 out_channels, 
                use_batchnorm,
                kernel_initializer='glorot_uniform',
                stride=1,
                name=None,
                **kwargs) -> None:
        super().__init__()
        
        self.net = Conv2dBn(in_channels=in_channels,
                                 out_channels=out_channels, 
                                 kernel_size=3,
                                 padding=1,
                                 stride=stride, 
                                 use_batchnorm=use_batchnorm,
                                 activation='relu',
                                 kernel_initializer=kernel_initializer,
                                 name=name,
                                 **kwargs)
        
    def forward(self, x):
        return self.net(x)
        

class DecoderTransposeX2Block_my3(nn.Module):
    def __init__(self, in_channels, 
                 out_channels, 
                 skip_channel = 112, 
                 use_batchnorm=False, 
                type_='bos'):
        super().__init__()
        
        self.use_batchnorm = use_batchnorm
        self.type_ = type_ 
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, 
                              kernel_size=4, stride=2, padding=1, 
                              bias= not use_batchnorm)
        if use_batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)
        
        self.act = nn.ReLU()
        
        # concatenate
        
        # out_channels = out_channels + skip_channel
        if 'DSEB' in type_:
            if '3x3' in type_:
                # Depthwise 
                channel = skip_channel
                self.skip__ = nn.Conv2d(channel, channel, 
                                kernel_size=3, stride=1, 
                                padding=0, groups=channel)
            if '5x5' in type_:
                channel = skip_channel 
                self.skip__ = nn.Conv2d(channel, channel, 
                                kernel_size=5, stride=1, 
                                padding=0, groups=channel)
            if '7x7' in type_:
                channel = skip_channel 
                self.skip__ = nn.Conv2d(channel, channel, 
                                kernel_size=7, stride=1, 
                                padding=0, groups=channel)
        
            self.conv2 = nn.Conv2d(skip_channel, 1, kernel_size=1, 
                                padding=1)
            
            self.sigmoid = nn.Sigmoid()
        
            # Lambda
        
        self.out = Conv3x3BnReLU(in_channels=skip_channel + out_channels,
                          out_channels=out_channels,
                          use_batchnorm=use_batchnorm)
        
    def forward(self, x, skip=None):
        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.bn(x)
        x = self.act(x)
        if skip is not None and 'nor' in self.type_:
            x = torch.cat([x, skip], dim=1)
        elif skip is not None and 'DSEB' in self.type_:
            bs, c, h, w = skip.shape
            skip_ = self.skip__(skip)
            skip_ = self.sigmoid(self.conv2(skip_))
            repeat = skip_.repeat(1, c, 1, 1)
            y = skip*repeat
            x = torch.cat([x, y], dim=1)
        
        x = self.out(x)
        return x 

class Decoder(nn.Module):
    def __init__(self, 
            classes=1,
            inp_shape = 192, 
            encoder_output = 1280, 
            use_batchnorm=True,
            skip_channels = [], 
            type_='bos', 
            ends_with_maxpool=False) -> None:
        super().__init__()
        
        self.classes = classes
        if isinstance(inp_shape, Tuple):
            inp_shape = inp_shape[0]

        self.inp_shape = inp_shape
        
        self.ends_with_maxpool = ends_with_maxpool
        self.type_ = type_
        decoder_filters = [encoder_output, 256, 128, 64, 32]
        
        # add center block if previous operation was maxpooling (for vgg models)
        # if isinstance(backbone.layers[-1], nn.MaxPool2d()):
        if ends_with_maxpool:
            self.conv1 = nn.Sequential(
                Conv3x3BnReLU(encoder_output, 512, use_batchnorm), 
                Conv3x3BnReLU(512, encoder_output, use_batchnorm)
                )
        
        # self.decoder = []
        self.decoder = OrderedDict()
        
        for i in range(4):
            self.decoder[f'feat_{5-i-1}'] = DecoderTransposeX2Block_my3(decoder_filters[i], 
                                                                      decoder_filters[i+1], 
                                                                      use_batchnorm=use_batchnorm, 
                                                                      skip_channel=skip_channels[i],
                                                                      type_=type_)
            
        
        self.decoder = nn.ModuleDict(self.decoder)
        
        if 'mout' in type_:
            self.conv_bn_relu = Conv3x3BnReLU(sum(decoder_filters[1:]), 16, use_batchnorm)
            self.m_out = nn.Conv2d(16, out_channels=classes, kernel_size=3, 
                                padding=1, bias=True)
            # Activation. 
        else:
            self.out = nn.Conv2d(16, classes, kernel_size=3, padding=1, bias=True)
            # Activation. 
        
    def forward(self, x, skip_inputs):
        if self.ends_with_maxpool:
            x = self.conv1(x)
        output = []
        for i, feat_name in enumerate(self.decoder):
            # print(feat_name)
            x = self.decoder[feat_name](x, skip_inputs[f'{feat_name[:-1]}{5-(i+1)-1}'])
            bs, c, h, w = x.shape
            out = x
            if 'mout' in self.type_:
                kernel = self.inp_shape//h
                if kernel>1:
                    out = F.upsample(x, scale_factor=kernel, mode='nearest')
                # print('OUT: ', out.shape, x.shape)
                output.append(out)
        
        if 'mout' in self.type_:
            out = torch.cat(output, dim=1)
            out = self.conv_bn_relu(out)
            out = self.m_out(out)
        else:
            out = self.out(out)

            
        return out 



