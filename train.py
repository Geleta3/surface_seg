from audioop import avg
import os 
import torch 
import numpy as np 
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchvision.transforms as T

from dataset import get_dataset
from evaluate import evaluate
from models import get_model
from utils.loss import Loss 
from config import Config
import warnings
from utils.logger import Logger 
import random

warnings.simplefilter('ignore')
torch.manual_seed(1234)
np.random.seed(4321)
random.seed(1234)

def train_one_epoch(model, epoch, loss, optimizer, dataloader, device, classes):
    model = model.train()
    total_loss = []
    loss_20 = []
    for idx, (batch) in enumerate(dataloader):
        image, mask = batch[0], batch[1]
        if len(batch) > 2:
            img_channels = batch[2]
        image = image/255.0
        image, mask = image.to(device), mask.to(device)
        optimizer.zero_grad()
        if classes == 1:
            out, skip_att, img_att = model(image)
            pred = torch.sigmoid(out)
        else:
            pred, skip_att, img_att = model(image)
          
        m_loss = loss(pred, mask)
        m_loss.backward()
        optimizer.step()
        total_loss.append(m_loss.item())
        loss_20.append(m_loss.item())
        if idx%20 == 0:
            loss_ = round(sum(loss_20)/len(loss_20), 7)
            print(f'Epoch: {epoch}, Step: {idx}/{len(dataloader)}, Loss: {loss_}')
            loss_20 = []
            
            
    return total_loss 


if __name__ == "__main__":
    
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    classes = config.classes[config.dataset]
    
    ########
    train_ds = get_dataset(config=config, mode='train', make_binary=config.make_binary)
    valid_ds = get_dataset(config=config, mode='valid', make_binary=config.make_binary)
    
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    valid_dl = DataLoader(valid_ds, batch_size=2, shuffle=False, num_workers=0)
    ########

    model = get_model(config, device=device)
    model = model.to(device)
        
    if config.epoch_resume:
        path = os.path.join('saved_model', config.dataset, f'model_{config.model}_{len(config.multi_img_channels)}.pth')
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
        
    loss = Loss(classes=classes, loss='CEL')
        
    optimizer = Adam(params=model.parameters(), lr=config.init_lr)
    rl_scheduler = ReduceLROnPlateau(optimizer=optimizer, 
                                     patience=15, 
                                     verbose=True, 
                                     threshold=0.0001)
                                    
                                    
    logger = Logger(log_dir=config.log_dir, num_img2save=30)
    
    # Tensorboard. 
    os.makedirs(f'saved_model', exist_ok=True)
    os.makedirs(f'./saved_model/{config.dataset}', exist_ok=True)
    
    min_loss = 1e10
    mid_loss = 1e10
    best_iou = 0
    patience = 5
    count = 0
    entropy = []
    
    for i in range(start_epoch, config.epochs):
        total_loss = train_one_epoch(model=model, 
                                    epoch=i, 
                                    loss = loss, 
                                    optimizer=optimizer, 
                                    dataloader=train_dl, 
                                    device=device, 
                                    classes=config.classes[config.dataset])
        logger.add_scalar(f'Loss', value=total_loss)
        # Eavaluate
        iou, f1_s,  AP = evaluate(  model, 
                                    epoch=i, 
                                    dataloader=valid_dl, 
                                    device=device, 
                                    classes=config.classes[config.dataset],
                                    logger=logger, 
                                    return_mean=True)

        logger.add_scalar(f'mIOU', iou, step=i)
        logger.add_scalar(f'f1_s', f1_s, step=i)
        logger.add_scalar(f'AP', AP, step=i)
        
        avg_loss = sum(total_loss)/len(total_loss)
        rl_scheduler.step(avg_loss)
            
        if iou > best_iou:
            torch.save({
                        'model':model.state_dict(), 
                        'epoch': i,
                        'metrics': {'iou': iou, 'F1': f1_s, 'AP': AP},  
                        'loss': avg_loss, 
                        'img_channel': len(config.multi_img_channels), 
                        'Batch size': config.batch_size, 
                        'lr': config.init_lr}, 
                f'./saved_model/{config.dataset}/model_{config.model}_{len(config.multi_img_channels)}.pth')
            best_iou = iou 
            
        entropy.append(best_iou - iou)
        avg_entropy = sum(entropy)/len(entropy)
        print('BEST IOU SO FAR: ', best_iou)
        print('AVG ENTROPY: ', avg_entropy)

    