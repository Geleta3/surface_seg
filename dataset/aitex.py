from torch.utils.data import Dataset, DataLoader
from PIL import Image 
import torch 
import numpy as np 
import math 
from typing import Tuple, List
import os 


class AITEXDataset(Dataset):
    def __init__(self, root, classes, img_size=224, 
                 img_channel=3, transform=None,):
        
        self.transform = transform
        self.classes = classes
        self.img_channel = img_channel
        self.img_size = img_size
        
        self.img_paths = []
        self.mask_paths = []
        not_found = []
        mask_root = os.path.join(root, "Mask_images")
        img_root = os.path.join(root, 'Defect_images')
        for mask in os.listdir(mask_root):
            img_name = mask[:-9] + '.png'
            if os.path.exists(os.path.join(img_root, img_name)):
                self.img_paths.append(os.path.join(img_root, img_name))
                self.mask_paths.appenf(os.path.join(mask_root, mask))
            else:
                not_found.append(mask)

        if len(not_found) > 1:
            print(f'{len(not_found)} masks have no images')
        if len(self.img_paths) < 1:
            raise 'No images found for training Stopped !'
            
        
    def __getitem__(self, idx):
        if self.img_channel == 1:
            image = Image.open(self.img_paths[idx]).convert('L')
        else:
            image = Image.open(self.img_paths[idx]).convert('RGB')
            
        if self.classes == 1:
            mask = Image.open(self.mask_paths[idx]).convert('1')
        else:
            mask = Image.open(self.mask_paths[idx])
            
        image = np.array(self.resize(image, resize_to=self.img_size), np.uint8)
        mask = np.array(self.resize(mask, resize_to=self.img_size), np.uint8)
            
        if len(image.shape) == 2: # Gray or Binary
            image = image[:, :, None] # add channel. 
            w, h, c = image.shape 
        elif len(image.shape) == 3: # RGB 
            w, h, c = image.shape
        
        if isinstance(self.img_size, (Tuple, List)):
            new_h, new_w = self.img_size[0], self.img_size[1]
        else:
            new_h, new_w = self.img_size, self.img_size
            
        if w < h:
            r, l = math.floor((new_w-w)/2), math.ceil((new_w-w)/2)
            image = np.pad(image, ((r, l), (0, 0), (0, 0)))
            mask = np.pad(mask, ((r, l), (0, 0) ))
        elif w > h:
            r, l = math.floor((new_h-h)/2), math.ceil((new_h-h)/2)
            image = np.pad(image, ((0, 0), (r, l), (0, 0)))
            mask = np.pad(mask, ((0, 0), (r, l)))
        
        if self.classes > 1:
            one_hot = np.zeros([self.classes+1, self.img_size, self.img_size])
            for i in range(self.classes):   
                one_hot[i] = (mask == i)

            mask = one_hot
        else:
            mask = mask[None, :, :]
            
        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.as_tensor(mask)
        
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        
        if self.preprocess:
            return image, mask, np.array(image_channels)
        return image, mask 

    def resize(self, img, resize_to):
        
        if resize_to is None:
            return img 
        w, h = img.size 
        scale = max(w, h)/resize_to
        if isinstance(self.img_size, (Tuple, List)):
            new_h, new_w = self.img_size[0], self.img_size[1]
        else:
            new_h, new_w = self.img_size, self.img_size
        
        if w > h:
            h = int(h//scale)
            img = img.resize((new_w, h))
        elif w == h:
            img = img.resize((new_w, new_h))
        else:
            w = int(w//scale)
           
            img = img.resize((w, new_h))
        return img 
    
    def __len__(self):
        return len(self.img_paths)


