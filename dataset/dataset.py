from torch.utils.data import Dataset, DataLoader
from PIL import Image 
import torch 
import numpy as np 
import math 
from typing import Tuple, List


class Dataset(Dataset):
    def __init__(self, classes, img_size=224, img_channel=3,
                 make_binary=True, transform=None,):
        
        self.transform = transform
        self.classes = classes
        self.img_channel = img_channel
        self.make_binary = make_binary
        
        if isinstance(img_size, (Tuple, List)):
            assert img_size[0] == img_size[1], 'Not supported for different height and width!'
            self.img_size = img_size[0]
        else:
            self.img_size = img_size
        
        self.cls_idx = []   # if the classes are in the folder. 
        
        
    def __getitem__(self, idx):
        if self.img_channel == 1:
            image = Image.open(self.img_paths[idx]).convert('L')
        else:
            image = Image.open(self.img_paths[idx]).convert('RGB')
            
        if self.classes == 1:
            mask = Image.open(self.mask_paths[idx]).convert('1')
        else:
            mask = Image.open(self.mask_paths[idx])
        
        if len(self.cls_idx) == len(self.img_paths):
            cls = self.cls_idx[idx] # Classes specified as Folder. 
            mask = mask.convert('1')
        else:
            cls = None 
            
        image = np.array(self.resize(image, resize_to=self.img_size), np.uint8)
        mask = np.array(self.resize(mask, resize_to=self.img_size), np.uint8)
            
        if len(image.shape) == 2: # Gray or Binary
            image = image[:, :, None] # add channel. 
            w, h, c = image.shape 
        elif len(image.shape) == 3: # RGB 
            w, h, c = image.shape
        
        if self.transform is not None:
            image, mask = self.transform(image, mask)
            
        if w < h:
            r, l = math.floor((self.img_size-w)/2), math.ceil((self.img_size-w)/2)
            image = np.pad(image, ((r, l), (0, 0), (0, 0)))
            mask = np.pad(mask, ((r, l), (0, 0) ))
        elif w > h:
            r, l = math.floor((self.img_size-h)/2), math.ceil((self.img_size-h)/2)
            image = np.pad(image, ((0, 0), (r, l), (0, 0)))
            mask = np.pad(mask, ((0, 0), (r, l)))
        
        if self.classes > 1 and not self.make_binary:
            one_hot = np.zeros([self.classes, self.img_size, self.img_size])
            if cls is None:
                for i in range(self.classes):   
                    one_hot[i] = (mask == i)
            else:
                one_hot[cls] = mask 
                # one_hot[0] = (mask == 0)
            mask = one_hot
        else:
            mask = mask[None, :, :] >= 1
        
        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.as_tensor(mask)
    
        
        if self.preprocess:
            return image, mask, np.array(image_channels)
        return image, mask 

    def resize(self, img, resize_to):
        w, h = img.size 
        scale = max(w, h)/resize_to
        if w > h:
            h = int(h//scale)
            img = img.resize((self.img_size, h))
        elif w == h:
            img = img.resize((self.img_size, self.img_size))
        else:
            w = int(w//scale)
           
            img = img.resize((w, self.img_size))
        return img 
    
    def __len__(self):
        return len(self.img_paths)


