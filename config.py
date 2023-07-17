
class Config:
    def __init__(self):
        
        self.model = 'unet'     # ['unet',  pretrained]
        
        # data 
        self.dataset = 'magnetic'   # [magnetic, mt, neu, dagm]
        
        # common 
        self.batch_size = 1
        self.epochs = 1000
        self.init_lr = 0.0001
        self.train_valid_split = 0.5
        self.img_dim = (192, 192)
        self.log_dir = f'log_{self.dataset}'
        self.epoch_resume = False
        self.make_binary = True #True 
        self.pretrained_model = 'mobilenet'  # resnet50, mobilenet, efficient-b0
        
        
        if self.make_binary:
            self.classes = {
                'mt': 1,
                'neu': 1, 
                'magnetic': 1, 
                'dagm': 1
            }
        else:
            self.classes = {
                'mt': 1,
                'neu': 3, 
                'magnetic': 5, 
                'dagm': 10
            }
      
        self.root = {
                'mt': './data/mt/', 
                'neu': './data/NEU-ds', 
                'magnetic': './data/Magnetic-ds', 
                'dagm': './data/DAGM'
            }
        self.img_channel = {
            'mt': 1,
            'neu': 1,
            'magnetic':1, 
            'dagm': 1
        }
        
        # UNET. 
        self.use_max_pool = False
        self.up_sample = False
        self.use_batch_norm = True
        self.enc_channels = [64, 64, 128, 128, 256, 512] 
        self.dec_channels = [512, 256, 128, 128, 64, 128]

        # Pretrain
        if self.pretrained_model == 'resnet50':
            self.pre_enc_channels = [256, 64, 128, 256, 512, 2048]
            self.pre_dec_channels = [2048, 512, 256, 256, 128, 128]
        elif self.pretrained_model == 'efficient-b0':
            self.pre_enc_channels = [256, 96, 144, 240, 672, 1280]
            self.pre_dec_channels = [1280, 512, 256, 256, 128, 128]
        elif self.pretrained_model == 'mobilenet':
            self.pre_enc_channels = [256, 96, 144, 192, 576, 1280]
            self.pre_dec_channels = [1280, 512, 256, 256, 128, 128]
        
        
    
    