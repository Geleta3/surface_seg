import torch 
from torch.utils.tensorboard import SummaryWriter
from typing import List 
import numpy as np
import os 
import time;


class Logger:
    def __init__(self, log_dir="", num_img2save=50):
        localtime = time.asctime( time.localtime(time.time()))
        localtime = f'{localtime}'.replace(':', '_')
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, localtime), exist_ok=True)
        
        log_dir = os.path.join(log_dir, localtime)
        self.num_img2save = num_img2save
        self.logger = SummaryWriter(log_dir)
        self.loss_step = 0
    
    def add_scalar(self, tag, value, step=None):
        if isinstance(value, List):
            for val in value:
                self.logger.add_scalar(tag, val, self.loss_step)
                self.loss_step += 1
        else:
            self.logger.add_scalar(tag, value, step)
            
    def save_img(self, tag, img, step):

        if 'image' in tag:
            image = self.preprocess_img(img)
        elif 'mask' in tag:
            image = self.preprocess_mask(img) >= 1  # saves as binary image. 
        elif 'pred' in tag:
            image = self.preprocess_mask(img) >= 1  # saves as binary image. 
        else:
            raise 'Unknown Tag Format! Tags should have either ["image", "mask", "pred"]'
        
        image = image.squeeze()
        if len(image.shape) == 3:
            self.logger.add_image(tag, image, step, dataformats='CHW')
        elif len(image.shape) == 2:
            self.logger.add_image(tag, image, step, dataformats='HW')
        else:
            raise "Unknow Image format!"
    
    def preprocess_img(self, img):
        if len(img.shape) == 4:
            return img[0] # The first element of Batch Image. 
        return img.type(torch.uint8)
    
    def preprocess_mask(self, mask):
        assert len(mask.shape) >= 3, 'Mask shape should be either [BCHW] or [CHW]'
        mask = mask.detach().cpu().numpy()
        if mask.shape[-3] > 1:  # Multi-Class. 
            mask = mask.argmax(-3)*(255/mask.shape[-3]) #torch.argmax(mask, -3)*(255/mask.shape[-3]).cpu().numpy().astype('uint8')
            mask = np.expand_dims(mask, -3)
        else:
            mask = mask > 0.5
        mask = mask.astype('uint8')     #CHW
        if len(mask.shape) == 4:
            mask = mask[0]          # The first element of Batch Mask or Pred. 
        return mask 
            
