from random import shuffle
import torch 
from torch.utils.data import DataLoader, Dataset
from PIL import Image 
import os 
import numpy as np 
import math 


class Dataset(Dataset):
    def __init__(self, root, 
                 img_size=224, 
                 classes=1,     # Binary. Yours.  Without Background
                 transforms=None) -> None:
        super().__init__()
        
        self.img_size = img_size 
        self.classes = classes 
        self.transforms = transforms
        
        imgs_root = os.path.join(root, 'images')
        masks_root = os.path.join(root, 'masks')
        img_files = os.listdir(imgs_root)
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
                
        if len(not_found) > 0:
            print(f'{len(not_found)} images have no masks. Training on found masks')
    
    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx])
        mask = Image.open(self.mask_paths[idx])
        
        image = np.array(self.resize(image, resize_to=self.img_size), np.float32)
        mask = np.array(self.resize(mask, resize_to=self.img_size), np.uint8)
        
        if len(image.shape) == 2: # Gray or Binary
            image = image[:, :, None] # add channel. 
            w, h, c = image.shape 
        elif len(image.shape) == 3: # RGB 
            w, h, c = image.shape
            
        if w < h:
            r, l = math.floor((224-w)/2), math.ceil((224-w)/2)
            image = np.pad(image, ((r, l), (0, 0), (0, 0)))
            mask = np.pad(mask, ((r, l), (0, 0) ))
        elif w > h:
            r, l = math.floor((224-h)/2), math.ceil((224-h)/2)
            image = np.pad(image, ((0, 0), (r, l), (0, 0)))
            mask = np.pad(mask, ((0, 0), (r, l)))
        
        if self.classes > 1:
            one_hot = np.zeros([self.classes+1, self.img_size, self.img_size])
            for i in range(self.classes):   
                one_hot[i] = (mask == i)
            mask = one_hot
            
        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.as_tensor(mask)
        if self.transforms is not None:
            image, mask = self.transform(image, mask)
            return image, mask 
        
        return image, mask 
    
    def resize(self, img, resize_to):
        w, h = img.size 
        scale = max(w, h)/resize_to
        if w > h:
            h = int(h//scale)
            img = img.resize((224, h))
        elif w == h:
            img = img.resize((224, 224))
        else:
            w = int(w//scale)
           
            img = img.resize((w, 224))
        return img 
    
    def __len__(self):
        return len(self.img_paths)

if __name__ == '__main__':
    # check 
    path = 'D:\\Jiruu\\upwork\\Semantic Segmentation\\sem-seg\\mt'
    dataset = Dataset(path, classes=3)
    image, mask = dataset[1]
    dataloader = DataLoader(dataset, 16, shuffle=True)
    print('Dataset: ', image.shape, mask.shape)
    image, mask = next(iter(dataloader))
    print('Dataloader, Batch-size = 16: ', image.shape, mask.shape)