import torchvision.transforms as T
import torch 


class MultiInputImage:
    def __init__(self):
        self.transforms = [
            T.ColorJitter(brightness=.5, hue=.3), 
            T.RandomPosterize(bits=2), 
            T.RandomAdjustSharpness(sharpness_factor=2), 
            T.RandomEqualize(), 
            # T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), 
            # T.RandomInvert(), 
            # T.RandomSolarize(threshold=192.0), 
            # T.RandomAutocontrast(), 
        ]
        
    
    def change(self, img):
        images = [img]
        for i, tra in enumerate(self.transforms):
            images.append(tra(img))
            self.img_channels.append(images[i+1].shape[-1])
        return torch.cat(images, dim=0)
        pass 
    
    def __call__(self, img):
        return self.change(img)
    
    def img_channels(self):
        return self.img_channels

if __name__ == '__main__':
    from PIL import Image 
    path = 'D:\\Jiruu\\upwork\\Semantic Segmentation\\surface_segmentation\\data\\mt\\images\\exp0_num_461.jpg'
    img = Image.open(path)
    img.show()
    img = T.ToTensor()(img)
    img = img.repeat(3, 1, 1) #*torch.tensor([[[0.1]], [[0.9]], [[0.8]]])
    # print(img.shape)
    # img = torch.randn(224, 224)
    out = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))(img)
    print(out == img)
    out = T.ToPILImage()(img)
    
    out.show()
    
    # print(out.shape)
    