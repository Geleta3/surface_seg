import torch 
import cv2 
from typing import Any
 
# potential. 
# Gray - RGB
# Local thresholding 
# Gray - Jet 
# Gray - HSV
# Image enhancement - []

# RGB - Gray [0.3R + 0.59G + 0.11B]

class MultiInputImage:
    def __init__(self, ):
        
        self.img_channels = []
        pass
    
    def __call__(self, img, *args: Any, **kwds: Any) -> Any:
        pass
    
    def img_channels(self):
        return 

if __name__ == '__main__':
    from PIL import Image 
    import numpy as np
    import matplotlib.pyplot as plt 
    img_file = 'D:\\Jiruu\\upwork\\Semantic Segmentation\\surface_segmentation\\data\\NEU-ds\\images\\training\\000201.jpg'
    img = Image.open(img_file).convert('L')
    img = np.array(img, 'uint8')
    print('Img Shape: ', img.shape)

    scale = np.array([1, 0.6, 1])[None, None, :]    # [0.1, 1, 1](JET) , [1, 1, 1](TURBO) , [1, 0.6, 1](HSV)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) 
    img = (img * scale).astype('uint8')
    # img = cv2.applyColorMap(img, colormap = cv2.COLORMAP_TURBO) #
    # img = cv2.applyColorMap(img, colormap = cv2.COLORMAP_TURBO) #
    
    
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) 
    # img = (img * scale).astype('uint8')
    # img = cv2.applyColorMap(img, colormap=cv2.COLORMAP_JET)
    img = cv2.applyColorMap(img, colormap=cv2.COLORMAP_HSV)
    # img = cv2.applyColorMap(img, colormap=cv2.COLORMAP_RAINBOW)
    # img = cv2.applyColorMap(img, colormap=cv2.COLORMAP_TWILIGHT_SHIFTED)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) 
    # img = cv2.applyColorMap(img, colormap = cv2.COLORMAP_HOT)
    # img = cv2.applyColorMap(img, colormap = cv2.COLORMAP_TURBO) #
    # img = cv2.applyColorMap(img, colormap = cv2.COLORMAP_DEEPGREEN) 
    # img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    print('After Conversion: ', img.shape)
    # img = Image.fromarray(img)
    # img.show()
    # plt.imshow(img, cmap='jet')
    # plt.show()
    
    cv2.imshow("Image", img)
    cv2.waitKey(5000)