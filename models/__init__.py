from .model import *


def get_model(config, device):
    
    ds = config.dataset
    if config.model.lower() in ['multi-inp', 'multiunet', 'multi-unet']:
        model = GRModel(img_channels=config.multi_img_channels,
                       enc_channels=config.enc_channels,
                       dec_channels=config.dec_channels,
                       classes=config.classes[ds], 
                       use_batch_norm=config.use_batch_norm,
                       up_sample=config.up_sample,
                       inp_conv_channels=config.inp_conv_channels, 
                       att_mode=config.att_mode, 
                       first_att=config.first_att)
        
    elif config.model.lower() in ['pretrained']:
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
        
    elif config.model.lower() in ['gunet', 'gnet', 'g_unet']:
      
        model = GModel(img_channels=config.multi_img_channels,
                       enc_channels=config.enc_channels,
                       dec_channels=config.dec_channels,
                       classes=config.classes[ds], 
                       use_batch_norm=config.use_batch_norm,
                       up_sample=config.up_sample,
                       inp_conv_channels=config.inp_conv_channels, 
                       att_mode=config.att_mode, 
                       first_att=config.first_att)
    
    elif config.model.lower() in ['gssunet', 'ssunet', 'ss_unet']:
        model = GSSModel(img_channels=config.multi_img_channels,
                enc_channels=config.enc_channels,
                dec_channels=config.dec_channels,
                classes=config.classes[ds], 
                use_batch_norm=config.use_batch_norm,
                up_sample=config.up_sample,
                inp_conv_channels=config.inp_conv_channels, 
                att_mode=config.att_mode, 
                first_att=config.first_att)
        
    elif config.model.lower() in ['gmunet', 'gmnet', 'gm_unet']:
        model = GMModel(img_channels=config.multi_img_channels,
                enc_channels=config.enc_channels,
                dec_channels=config.dec_channels,
                classes=config.classes[ds], 
                use_batch_norm=config.use_batch_norm,
                up_sample=config.up_sample,
                inp_conv_channels=config.inp_conv_channels, 
                att_mode=config.att_mode, 
                first_att=config.first_att)
                
    elif 'DSEB'.lower() in config.model.lower():
        model = dseb_model(config, device)[0]   # Ignore model name
    elif 'unet' in config.model.lower():
        model = UModel(img_channel=config.img_channel[ds], 
                       enc_channels=config.enc_channels,
                       dec_channels=config.dec_channels,
                       use_batch_norm=config.use_batch_norm,
                       classes=config.classes[ds],
                       up_sample=config.up_sample,
                       use_max_pool=config.use_max_pool)
        
    return model  

def dseb_model(config, device):
    model_idx = config.model_idx
    model_name = config.dseb_model_name
    classes = config.classes[config.dataset]
    
    if model_idx == 13:
        model = DSEBModel(enc_model=model_name, 
                      inp_shape=config.img_dim, 
                      use_batchnorm=True, 
                      skip_channels=config.skip_channels, 
                      classes=classes,
                      type_='DSEB_mout_3x3', 
                      device=device)
        model_name = 'DSEB_EUNET_3x3_mout'
    elif model_idx == 15:
        model = DSEBModel(enc_model=model_name,
                      inp_shape=config.img_dim, 
                      use_batchnorm=True, 
                      skip_channels=config.skip_channels, 
                      classes=classes,
                      type_='DSEB_mout_5x5', 
                      device=device)
        model_name = 'DSEB_EUNET_5x5_mout'
    elif model_idx == 17:
        model = DSEBModel(enc_model=model_name,
                      inp_shape=config.img_dim, 
                      use_batchnorm=True, 
                      skip_channels=config.skip_channels, 
                      classes=classes, 
                      type_='DSEB_mout_7x7', 
                      device=device)
        model_name = 'DSEB_EUNET_7x7_mout'
    else:
        raise ValueError(f"Given model index - {model_idx} but it should be one of [13, 15, 17]")
    
    return model, model_name 
