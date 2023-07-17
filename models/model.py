import torch 
from torch import nn 

from .unet import Encoder as UEncoder, Decoder as UDecoder
from .pretrained import Encoder as PreEncoder


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

if __name__ == '__main__':
    # check 
    model = Model(inp_shape=192, type_='DSEB_mout_3x3', classes=4)
    img = torch.randn(2, 3, 192, 192)
    out = model(img)
    print('OUT - ', out.shape)
    print('Congratulation!')
    
    