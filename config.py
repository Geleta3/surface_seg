is_linux = False

class Config:
    def __init__(self):
        
        self.model = 'multi-inp'     # ['unet', 'dseb', gunet, 'gmunet', ssunet, multi-inp, pretrained]
        
        # data 
        self.dataset = 'magnetic'   # [magnetic, mt, neu, dagm]
        
        # common 
        self.batch_size = 2
        self.epochs = 1000
        self.init_lr = 0.0001
        self.loss_threshold = 0.15
        self.mid_loss = 0.15
        self.train_valid_split = 0.5
        self.img_dim = (192, 192)
        self.log_dir = f'log_{self.dataset}'
        self.epoch_resume = False
        self.make_binary = True #True 
        self.pretrained_model = 'mobilenet'  # resnet50, mobilenet, efficient-b0
        
        
        if self.make_binary:
            self.classes = {
                'mt': 1,
                'neu': 3, 
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
        
        if is_linux:
            self.root = {
            'mt': '/home/geleta/surface seg/data/mt/', 
            'neu': '/home/geleta/surface seg/data/NEU-ds', 
            'magnetic': '/home/geleta/surface seg/data/Magnetic-ds', 
            'dagm': '/home/geleta/surface seg/data/DAGM'
            }
        else:
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
        

        
        # dseb 
        self.dseb_model_name = 'b0'
        self.names = [f'Class{i}' for i in range(1, 7)]
        self.dseb_encoder_output = 1280
        self.model_idx = 13
        
        # UNET. 
        self.use_max_pool = False
        self.up_sample = False
        self.use_batch_norm = True
        self.enc_channels = [64, 64, 128, 128, 256, 512] #[256, 256, 256, 256, 512, 1024] #[256, 128, 256, 256] # #[256, 128, 256] #[128, 256, 256, 512, 1024] #[] #[128, 256, 256, 512, 1024] #[128, 256, 256, 512, 512] #[128, 512, 512, 256, 256] #[128, 256, 256, 512, 1024] #[64, 256, 512, 256, 512]
        self.dec_channels = [512, 256, 128, 128, 64, 128]#[1024, 512, 256, 256, 256, 256] #[256, 256, 128, 256] #[1024, 512, 256, 256, 256, 128] #[256, 128, 128] #[1024, 512, 256, 256, 128] #[1024, 512, 256, 256, 256, 128] #[1024, 512, 256, 256, 128] #[512, 512, 256, 256, 256] #[256, 256, 256, 512, 256] #[1024, 512, 256, 256, 128] # [512, 256, 256, 512, 64]
        
        # GUNET
        self.preprocess = True 
        self.att_mode = 'seq'
        self.first_att = 'soft'
        self.multi_img_channels = []
        self.inp_conv_channels = [64, 64, 64] #[32, 64, 64] #[32, 64, 64] #[16, 64, 64] #[16, 64, 64] [8, 16, 16] 
        
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
        
        
        # Conditional
        if self.model == 'dseb':
            self.skip_channels = [112, 40, 24, 16]
        elif self.model == 'unet':
            self.skip_channels = [256, 512, 256, 64]    # 
        
    
    