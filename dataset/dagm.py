import os 
import numpy as np 
from random import shuffle
from .dataset import Dataset


class DAGMDataset(Dataset):
    def __init__(self, 
                 root, 
                 classes=10, 
                 img_size=224, 
                 img_channel=1, 
                 mode='train',
                 transform=None,
                 *args,
                 **kwargs):
        super().__init__(classes=classes, img_size=img_size, 
                         img_channel=img_channel, transform=transform,
                         *args, **kwargs)
        
        cls_names = [f'Class{i}' for i in range(1, 7)]
        self.classes = len(cls_names)
        if mode in ['train', 'training']:
            roots = [os.path.join(root, cls, 'Train') for cls in cls_names]
        elif mode in ['valid', 'validate', 'validation', 'validating']:
            roots = [os.path.join(root, cls, 'Test') for cls in cls_names]
        else:
            raise 'Mode should be in ["train", "valid"]'
        
        self.cls_idx = []
        self.img_paths = []
        self.mask_paths = []
        not_found = []
        
        print(f'Loading data for {mode}...')
        for idx, cls in enumerate(roots):
            mask_root = os.path.join(cls, 'Label')
            mask_imgs = [mask_img for mask_img in os.listdir(mask_root) if 'PNG' in mask_img]
            for mask_file in mask_imgs:
                img_file = mask_file[:-10] + '.PNG'
                img_path = os.path.join(cls, img_file)
                if os.path.exists(img_path):
                    self.img_paths.append(img_path)
                    self.mask_paths.append(os.path.join(mask_root, mask_file))
                    self.cls_idx.append(idx)
                else:
                    not_found.append(mask_file)

        
        index_shuffle = list(range(len(self.img_paths)))
        shuffle(index_shuffle)
        
        self.img_paths  = [self.img_paths[idx] for idx in index_shuffle]
        self.mask_paths = [self.mask_paths[idx] for idx in index_shuffle]
        self.cls_idx    = [self.cls_idx[idx] for idx in index_shuffle]
        
        if len(not_found) > 1:
            print(f'{len(not_found)} masks have no matching images!')
            
        if len(self.img_paths) < 1:
            raise 'All masks have no matching images! Stopping !'
        
        print(f'Images for {mode}: ', len(self.img_paths))



