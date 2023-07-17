import torch 
from torch import nn 
from .encoder_efficient import EfficientNet
from .decoder_DSEB import Decoder as DSEBDecoder
from .unet import Encoder as UEncoder, Decoder as UDecoder
from .g_unet import Encoder as GEncoder, Decoder as GDecoder
from .g_unet_mod import Encoder as GMEncoder, Decoder as GMDecoder
from .g_unet_ss import Encoder as SSEncoder, Decoder as SSDecoder 
from .g_unet_r import Encoder as REncoder, Decoder as RDecoder
from .pretrained import Encoder as PreEncoder

class GRModel(nn.Module):
    def __init__(self, 
                enc_channels, 
                dec_channels, 
                classes, 
                img_channels, 
                inp_conv_channels, 
                use_batch_norm=True, max_pool=True, 
                up_sample=True, 
                att_mode='concat', first_att='sig'      
    ):
        super().__init__()
        self.encoder = REncoder(img_channels=img_channels,
                                channels=enc_channels, 
                                
                                inp_conv_channels=inp_conv_channels, 
                                use_batch_norm=use_batch_norm, 
                                max_pool=max_pool)

        skip_channels = [enc_channels[i] for i in range(len(enc_channels)-2, -1, -1)]
        self.decoder = RDecoder(channels=dec_channels, skip_channels=skip_channels, 
                                up_sample=up_sample, use_batch_norm=use_batch_norm, 
                                att_mode=att_mode, first_att=first_att)
        
        self.out_layer = nn.Conv2d(dec_channels[-1], classes, kernel_size=1)#, stride=1, padding=1)
        
    def forward(self, x):
        x, middle_output, img_attentions = self.encoder(x)
        x, skip_attentions = self.decoder(x, middle_output)
        out = self.out_layer(x)
        return out, skip_attentions, img_attentions

class PreModel(nn.Module):
    def __init__(self, 
                 model_name, 
                enc_channels, 
                dec_channels, 
                classes, 
                img_channels, 
                inp_conv_channels, 
                
                use_batch_norm=True, max_pool=True, 
                up_sample=True, 
                att_mode='concat', first_att='sig'      
    ):
        super().__init__()
        self.encoder = PreEncoder(  img_channels=img_channels,  
                                    conv_channels=inp_conv_channels, 
                                    model=model_name
                               )

        skip_channels = [enc_channels[i] for i in range(len(enc_channels)-2, -1, -1)]
        self.decoder = RDecoder(channels=dec_channels, skip_channels=skip_channels, 
                                up_sample=up_sample, use_batch_norm=use_batch_norm, 
                                att_mode=att_mode, first_att=first_att)
        
        self.out_layer = nn.Conv2d(dec_channels[-1], classes, kernel_size=1)#, stride=1, padding=1)
        
    def forward(self, x):
        x, middle_output, img_attentions = self.encoder(x)
        x, skip_attentions = self.decoder(x, middle_output)
        out = self.out_layer(x)
        return out, skip_attentions, img_attentions

class GSSModel(nn.Module):
    def __init__(self, 
                enc_channels, 
                dec_channels, 
                classes, 
                img_channels, 
                inp_conv_channels, 
                use_batch_norm=True, max_pool=True, 
                up_sample=True, 
                att_mode='concat', first_att='sig'      
    ):
        super().__init__()
        self.encoder = SSEncoder(img_channels=img_channels,
                                channels=enc_channels, 
                                
                                inp_conv_channels=inp_conv_channels, 
                                use_batch_norm=use_batch_norm, 
                                max_pool=max_pool)

        skip_channels = [enc_channels[i] for i in range(len(enc_channels)-2, -1, -1)]
        self.decoder = SSDecoder(channels=dec_channels, skip_channels=skip_channels, 
                                up_sample=up_sample, use_batch_norm=use_batch_norm, 
                                att_mode=att_mode, first_att=first_att)
        
        self.out_layer = nn.Conv2d(dec_channels[-1], classes, kernel_size=1)#, stride=1, padding=1)
        
    def forward(self, x):
        x, middle_output = self.encoder(x)
        x = self.decoder(x, middle_output)
        out = self.out_layer(x)
        return out


class GMModel(nn.Module):
    def __init__(self, 
                enc_channels, 
                dec_channels, 
                classes, 
                img_channels, 
                inp_conv_channels, 
                use_batch_norm=True, max_pool=True, 
                up_sample=True, 
                att_mode='concat', first_att='sig'      
    ):
        super().__init__()
        self.encoder = GMEncoder(img_channels=img_channels,
                                channels=enc_channels, 
                                
                                inp_conv_channels=inp_conv_channels, 
                                use_batch_norm=use_batch_norm, 
                                max_pool=max_pool)

        skip_channels = [enc_channels[i] for i in range(len(enc_channels)-2, -1, -1)]
        self.decoder = GMDecoder(channels=dec_channels, skip_channels=skip_channels, 
                                up_sample=up_sample, use_batch_norm=use_batch_norm, 
                                att_mode=att_mode, first_att=first_att)
        
        # classes = classes if classes == 1 else classes + 1      # Binary vs Multi-class
        self.out_layer = nn.Conv2d(dec_channels[-1], classes, kernel_size=1)#, stride=1, padding=1)
        
    def forward(self, x):
        x, middle_output = self.encoder(x)
        x = self.decoder(x, middle_output)
        out = self.out_layer(x)
        return out

class GModel(nn.Module):
    def __init__(self, 
                enc_channels, 
                dec_channels, 
                classes, 
                img_channels, 
                inp_conv_channels, 
                use_batch_norm=True, max_pool=True, 
                up_sample=True, 
                att_mode='concat', first_att='sig'      
    ):
        super().__init__()
        self.encoder = GEncoder(img_channels=img_channels,
                                channels=enc_channels, 
                                
                                inp_conv_channels=inp_conv_channels, 
                                use_batch_norm=use_batch_norm, 
                                max_pool=max_pool)

        skip_channels = [enc_channels[i] for i in range(len(enc_channels)-2, -1, -1)]
        self.decoder = GDecoder(channels=dec_channels, skip_channels=skip_channels, 
                                up_sample=up_sample, use_batch_norm=use_batch_norm, 
                                att_mode=att_mode, first_att=first_att)
        
        # classes = classes if classes == 1 else classes + 1      # Binary vs Multi-class
        self.out_layer = nn.Conv2d(dec_channels[-1], classes, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x, middle_output = self.encoder(x)
        x = self.decoder(x, middle_output)
        out = self.out_layer(x)
        return out
        
class UModel(nn.Module):
    def __init__(self, img_channel, 
                 enc_channels, 
                 dec_channels, 
                 use_batch_norm, 
                 classes, 
                 up_sample=False, 
                 use_max_pool=True):
        super().__init__()
        
        self.encoder = UEncoder(img_channel=img_channel,
                               channels=enc_channels,
                               use_batch_norm=use_batch_norm, 
                               max_pool=use_max_pool)
        
        skip_channels = [enc_channels[i] for i in range(len(enc_channels)-2, -1, -1)]
        self.decoder = UDecoder(channels=dec_channels, 
                               skip_channels=skip_channels,
                               up_sample=up_sample, 
                               use_batch_norm=use_batch_norm)

        classes = classes if classes == 1 else classes + 1      # Binary vs Multi-class
        self.out_layer = nn.Conv2d(dec_channels[-1], classes, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x, middle_output = self.encoder(x)
        x = self.decoder(x, middle_output)
        out = self.out_layer(x)
        return out 


class DSEBModel(nn.Module):
    def __init__(self, enc_model='b0', 
                classes=1, 
                inp_shape = 192, 
                use_batchnorm=True,
                type_='DSEB_mout_3x3', 
                skip_channels = [112, 40, 24, 16], # 1280, 
                ends_with_maxpool=False, 
                device='cuda') -> None:
        super().__init__()

        if classes > 1: classes = classes + 1
        self.device = device
        self.encoder = EfficientNet(model=enc_model)
        
        self.decoder = DSEBDecoder(classes=classes, 
                                inp_shape=inp_shape, 
                                use_batchnorm=use_batchnorm,
                                type_=type_, 
                                skip_channels=skip_channels, 
                                ends_with_maxpool=ends_with_maxpool).to(device)
    
    def forward(self, x):
        enc_feat = self.encoder(x)
        out = self.decoder(enc_feat, self.encoder.feats)
        return out 
        
    
if __name__ == '__main__':
    # check 
    model = Model(inp_shape=192, type_='DSEB_mout_3x3', classes=4)
    img = torch.randn(2, 3, 192, 192)
    out = model(img)
    print('OUT - ', out.shape)
    print('Congratulation!')
    
    