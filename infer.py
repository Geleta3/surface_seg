from models.model import Model 
from config import Config
from dataset import MT_Dataset, NEU_Dataset
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt 


if __name__ == '__main__':
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classes = config.classes[config.dataset]
    path = config.root[config.dataset]
    model_path = f'D:/Jiruu/upwork/Semantic Segmentation/DSEB-EUnet/saved_model_{config.dataset}/model_15'
    
    dataset = NEU_Dataset(path, mode='valid')
    dataloader = DataLoader(dataset, 1, shuffle=True)
    model = Model(classes=classes, device=device, inp_shape=224, )
    model = model.eval()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    for idx, (image, mask) in  enumerate(dataloader):
        pred = model(image)
        pred = (torch.argmax(pred, dim=1).squeeze().numpy()*255/classes).astype('uint8')
        mask = (torch.argmax(mask, dim=1).squeeze().numpy()*255/classes).astype('uint8')
        plt.imshow(mask, cmap='jet')
        plt.show()
        plt.imshow(pred, cmap='jet')
        plt.show()
    

