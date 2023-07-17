import torch 
from torch import nn 
import cv2 
import numpy as np

  
# [0.1, 1, 1](JET) , [1, 1, 1](TURBO) , [1, 0.6, 1](HSV)
class MultiInputImage:
    def __init__(self, choice=0):
        self.img_format = [
            cv2.COLORMAP_TURBO,
            cv2.COLORMAP_JET,
            cv2.COLORMAP_HSV, 
            cv2.COLORMAP_WINTER, 
            cv2.COLORMAP_RAINBOW, 
            cv2.COLORMAP_SPRING, 
            cv2.COLORMAP_TWILIGHT_SHIFTED, 
            cv2.COLORMAP_COOL, 
            cv2.COLORMAP_OCEAN, 
            cv2.COLORMAP_PARULA
        ]
        scale = Scale(choice=choice)
        self.scale = scale.scale
        # self.scale = [
        #     np.array([1, 1, 1])[None, None, :], 
        #     np.array([0.1, 1, 1])[None, None, :],
        #     np.array([1, 0.6, 1])[None, None, :]
        # ]
        
        self.img_channels = []
    
    def change(self, img):
        images = [img]
        self.img_channels.append(img.shape[-1])
        for i in range(len(self.img_format)):
            #image = (img * self.scale[i]).astype('uint8')
            image = (img * np.random.randn(3)[None, None, :]).astype('uint8')
            images.append(image)
            #images.append(cv2.applyColorMap(image, colormap=self.img_format[i]))
            self.img_channels.append(images[i+1].shape[-1])
        images = np.concatenate(images, axis=-1)
        return images 
    
    def __call__(self, img):
        return self.change(img)
    
    def img_channels(self):
        return self.img_channels
    
class Scale:
    def __init__(self, choice=0):
        if choice == 0:
            self.scale = [
                np.array([1, 1, 1])[None, None, :], 
                np.array([0.1, 1, 1])[None, None, :],
                np.array([1, 0.6, 1])[None, None, :],
                np.array([1, 1, 1])[None, None, :], 
                np.array([0.3, 1, 1])[None, None, :],
                np.array([1, 0.9, 1])[None, None, :],
                np.array([1, 1, 0.5])[None, None, :], 
                np.array([0.8, 1, 1])[None, None, :],
                np.array([1, 0.1, 1])[None, None, :],
                np.array([1, 0.5, 0.5])[None, None, :],
                
            ]
        elif choice == 1:
            self.scale = [
                np.array([1, 0.5, 0.25])[None, None, :], 
                np.array([1, 0.5, 0.25])[None, None, :],
                np.array([1, 0.5, 0.25])[None, None, :], 
                np.array([1, 0.5, 0.25])[None, None, :], 
                np.array([1, 0.5, 0.25])[None, None, :],
                np.array([1, 0.5, 0.25])[None, None, :], 
                np.array([1, 0.5, 0.25])[None, None, :], 
                np.array([1, 0.5, 0.25])[None, None, :],
                np.array([1, 0.5, 0.25])[None, None, :], 
                np.array([1, 0.5, 0.25])[None, None, :], 
            ]
        elif choice == 2:
            self.scale = [
                np.array([0.6, 1, 0.3])[None, None, :], 
                np.array([0.2, 0.7, 1])[None, None, :],
                np.array([1, 0.6, 1])[None, None, :], 
                np.array([0.6, 1, 0.3])[None, None, :], 
                np.array([0.2, 0.7, 1])[None, None, :],
                np.array([1, 0.6, 1])[None, None, :], 
                np.array([0.6, 1, 0.3])[None, None, :], 
                np.array([0.2, 0.7, 1])[None, None, :],
                np.array([1, 0.6, 1])[None, None, :], 
                np.array([1, 0.6, 1])[None, None, :], 
            ]
        elif choice == 3:
            self.scale = [
                np.array([0.9, 0.4, 1])[None, None, :], 
                np.array([0.5, 0.3, 1])[None, None, :],
                np.array([1, 0.6, 0.1])[None, None, :]
            ]
        elif choice == 4:
            self.scale = [
                np.array([1, 0.1, 0.5])[None, None, :], 
                np.array([0.9, 0.4, 1])[None, None, :],
                np.array([0.5, 0.6, 1])[None, None, :]
            ]
        elif choice == 5:
            self.scale = [
                np.array([0.33, 0.32, 0.66])[None, None, :], 
                np.array([0.45, 0.54, 0.8])[None, None, :],
                np.array([0.92, 0.6, 0.2])[None, None, :]
            ]
        
