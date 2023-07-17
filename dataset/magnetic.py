import os 
import numpy as np 
from random import shuffle
from .dataset import Dataset


class MagneticDataset(Dataset):
    def __init__(self, 
                 root, 
                 classes=5, 
                 img_size=224, 
                 img_channel=1, 
                 mode='train',
                 train_valid_split=0.8,  
                 transform=None,
                 *args,
                 **kwargs):
        super().__init__(classes=classes, img_size=img_size, 
                         img_channel=img_channel, transform=transform,
                         *args, **kwargs)
        
        cls_names = ['MT_Blowhole', 'MT_Break', 'MT_Crack', 'MT_Fray',  'MT_Uneven']    # 'MT_Free',
        self.classes = len(cls_names)
        roots = [os.path.join(root, cls, 'Imgs') for cls in cls_names]
        
        self.cls_idx = []
        self.img_paths = []
        self.mask_paths = []
        not_found = []
        
        print(f'Loading data for {mode}...')
        for idx, cls in enumerate(roots):
            img_files = os.listdir(cls)
            img_files = [img_file for img_file in img_files if 'jpg' in img_file ]
            for img in img_files:
                base_name = img[:-3]
                mask_path = os.path.join(cls, base_name + 'png')
                if os.path.exists(mask_path):
                    self.img_paths.append(os.path.join(cls, base_name + 'jpg'))
                    self.mask_paths.append(mask_path)
                    self.cls_idx.append(idx)
                else:
                    not_found.append(img)
        
        index_shuffle = list(range(len(self.img_paths)))
        shuffle(index_shuffle)
        
        self.img_paths  = [self.img_paths[idx] for idx in index_shuffle]
        self.mask_paths = [self.mask_paths[idx] for idx in index_shuffle]
        self.cls_idx    = [self.cls_idx[idx] for idx in index_shuffle]
        
        if len(not_found) > 1:
            print(f'{len(not_found)} images have no masks!')
            
        if len(self.img_paths) < 1:
            raise 'All images have no masks! Stopping !'
        
        train_split = int(len(self.img_paths)*train_valid_split)
        if mode in ['train', 'training']:
            self.img_paths = self.img_paths[:train_split]
            self.mask_paths = self.mask_paths[:train_split]
            self.cls_idx = self.cls_idx[:train_split]
            
        elif mode in ['valid', 'validate', 'validation', 'validating']:
            self.img_paths = self.img_paths[train_split:]
            self.mask_paths = self.mask_paths[train_split:]
            self.cls_idx = self.cls_idx[train_split:]
        else:
            raise ValueError ('mode should be in [train or valid]')


