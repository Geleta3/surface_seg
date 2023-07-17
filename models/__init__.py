from .model import *


def get_model(config, device):
    
    ds = config.dataset
        
    if config.model.lower() in ['pretrained']:
        model = PreModel(model_name=config.pretrained_model, 
                        img_channels=config.multi_img_channels,
                        enc_channels=config.pre_enc_channels,
                        dec_channels=config.pre_dec_channels,
                        classes=config.classes[ds], 
                        use_batch_norm=config.use_batch_norm,
                        up_sample=config.up_sample,
                        inp_conv_channels=config.inp_conv_channels, 
                        att_mode=config.att_mode, 
                        first_att=config.first_att)
        
    elif 'unet' in config.model.lower():
        model = UModel(img_channel=config.img_channel[ds], 
                       enc_channels=config.enc_channels,
                       dec_channels=config.dec_channels,
                       use_batch_norm=config.use_batch_norm,
                       classes=config.classes[ds],
                       up_sample=config.up_sample,
                       use_max_pool=config.use_max_pool)
        
    return model  
