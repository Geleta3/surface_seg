import os
import warnings
from .dataset import Dataset


class NEU_Dataset(Dataset):
    def __init__(self, 
                 root, 
                 mode='train', 
                 classes=3, 
                 img_size=224,
                 img_channel=1,  
                 transform=None, 
                 *args, 
                 **kwargs):
        super().__init__(classes=classes, img_size=img_size, 
                         img_channel=img_channel, transform=transform,
                         *args, **kwargs)
        
        img_root = os.path.join(root, "images")
        mask_root = os.path.join(root, 'annotations')
        
        if mode in ['train', 'training']:
            img_root = os.path.join(img_root, 'training')
            mask_root = os.path.join(mask_root, 'training')
        elif mode in ['valid', 'test', 'validation', 'testing', 'validating']:
            img_root = os.path.join(img_root, 'test')
            mask_root = os.path.join(mask_root, 'test')
        else:
            raise TypeError('Unknown mode')

        print(f'Loading data for {mode}...')
        # check mask for each image 
        self.img_paths = []
        self.mask_paths = []
        not_found = []
        for img in os.listdir(img_root):
            mask_name = img[:-3] + 'png'
            mask_path = os.path.join(mask_root, mask_name)
            if os.path.exists(mask_path):
                self.img_paths.append(os.path.join(img_root, img))
                self.mask_paths.append(mask_path)
            else:
                not_found.append(img)

        if len(not_found) > 1:
            warnings.warn(f'{len(not_found)} images have no mask. Training on found mask')
            print(f'Images for {mode} is {len(self.img_paths)}')
            
        if len(self.img_paths) < 1:
            raise "No image have a matching mask!"
        


        