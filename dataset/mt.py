import os
from .dataset import Dataset
from PIL import Image 
import numpy as np 

class MT_Dataset(Dataset):
    def __init__(self, 
                 root, 
                 classes, 
                 mode='train', 
                 img_channel=1, 
                 img_size=224, 
                 transform=None, 
                 train_valid_split=0.8, 
                 *args, 
                 **kwargs):
        super().__init__(classes=classes, img_channel=img_channel, 
                         img_size=img_size, transform=transform, 
                         *args, **kwargs)
         
        self.classes = classes 
        self.transform = transform
        
        imgs_root = os.path.join(root, 'images')
        masks_root = os.path.join(root, 'masks')
        img_files = os.listdir(imgs_root)
        
        print(f'Loading data for {mode}...')
        self.img_paths = []
        self.mask_paths = []
        not_found = []
        for img in img_files:
            mask_name = img[:-3] + 'png'
            mask_path = os.path.join(masks_root, mask_name)
            if os.path.exists(mask_path):
                self.img_paths.append(os.path.join(imgs_root, img))
                self.mask_paths.append(mask_path)
            else:
                not_found.append(img)
                
        # Filter before splitting 
        self.filter_defects()
        train_size = int(len(self.img_paths)*train_valid_split)
        if mode in ['train', 'training']:
            self.img_paths = self.img_paths[:train_size]
            self.mask_paths = self.mask_paths[:train_size]
        elif mode in ['valid', 'validating', 'validation', 'test', 'testing']:
            self.img_paths = self.img_paths[train_size:]
            self.mask_paths = self.mask_paths[train_size:]
        else:
            raise f"Unknown mode {mode}"
        
        if len(not_found) > 0:
            print(f'{len(not_found)} images have no masks. Training on found masks')

        print(f'Defected images: {len(self.img_paths)}')
        
    def filter_defects(self):
        print('Filtering defected surfaces...')
        new_img_path = []
        new_mask_path = []
        defect_less = []
        for idx, mask_file in enumerate(self.mask_paths):
            mask = Image.open(mask_file).convert('1')
            mask = np.array(mask, np.uint8)
            classes, count = np.unique(mask, return_counts=True)
            cls_count = dict(zip(classes, count))
            if 1 in cls_count:
                if cls_count[1] > 0:
                    new_img_path.append(self.img_paths[idx])
                    new_mask_path.append(self.mask_paths[idx])
            else:
                defect_less.append(mask)
                
        self.img_paths = new_img_path
        self.mask_paths = new_mask_path
        self.defect_less = defect_less
        
        
