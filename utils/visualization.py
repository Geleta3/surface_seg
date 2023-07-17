from turtle import color
import numpy as np
import torch 
import matplotlib.pyplot as plt
from utils.utils import area_ratio
from dataset import get_dataset
from config import Config

# plt.style.use('ggplot')

def plot_bar(values, names, dataset=['']):#, x_label, y_label):
    plt.figure(figsize=(6, 5))
    color = ['r', 'b', 'g']
    x = np.arange(len(values[0]))
    for idx, v in enumerate(values):#enumerate(zip(names, values)):
        plt.bar(x, v, width=0.25, label=f"{dataset[idx]}", color=color[idx])
        x = x+0.25
        
    plt.xticks(np.arange(len(values[0]))+0.175, names[0])
    plt.xlabel('mIOU - [Defected_Area/Image_Area]',  fontsize = 10)
    plt.ylabel('Count', fontsize = 10)
    plt.legend()
    plt.show()

    
def plot_bar_(values, names, dataset=''):#, x_label, y_label):
    plt.figure(figsize=(7, 6))
    plt.bar(names, values, width=0.8, label=f"{dataset}")
    plt.xlabel('Defected_Area/Image_Area',  fontsize = 16, fontweight='bold')
    plt.ylabel('Count', fontsize = 16, fontweight='bold')
    plt.xticks(names, fontsize=15)
    plt.yticks(values, fontsize=15)
    plt.legend()
    plt.show()

    

def multi_im_show(image, mask, img_channels=[]):
    sub_plots = [231, 232, 233, 234]
    idx = 0
    for im_ch, sub_p in zip(img_channels, sub_plots):
        img = image[idx:idx+im_ch]
        img = img.permute(1, 2, 0).numpy().astype('uint8').squeeze()
        plt.subplot(sub_p)
        if im_ch == 1:
            plt.imshow(img, 'gray')
        else:
            plt.imshow(img)
        plt.xticks(color='w')
        plt.yticks(color='w')
        idx += im_ch
    plt.subplot(235)
    plt.imshow(mask.squeeze())
    plt.xticks(color='w')
    plt.yticks(color='w')
    
    plt.show()
    
    

def avg_area_range(dataset, ranges=[[0.0, 0.01], [0.01, 0.1], [0.1, 0.2], [0.2, 1]]):
    ds = dataset
    areas = []
    print(f'Calculating avg_areas for ranges of {ranges}...')
    sample_mask = dict(zip([f'{r}' for r in ranges], [[] for _ in range(len(ranges))]))
    
    for i in range(len(ds)):
        img, mask = ds[i][0], ds[i][1]
        a_r = area_ratio(mask.float()).item()
        for r in ranges:
            if len(sample_mask[f'{r}']) < 5:
                if r[0] < a_r < r[1]:
                    sample_mask[f'{r}'].append(mask)
            else:
                continue
        areas.append(round(a_r, 5))

    areas = sorted(areas)
    range_areas = {}
    for r in ranges:
        range_areas[f'{r}'] = []
        for a in areas:
            if r[0] < a < r[1]:
                range_areas[f'{r}'].append(a)
            if a > r[1]:
                break 
    avg_areas = [sum(range_areas[r])/(len(range_areas[r] )+ 1e-6) for r in range_areas] 
    labels  = [r.replace(',', ' -') for r in range_areas]
    count = [len(range_areas[r]) for r in range_areas]     
      
    # ds_name = config.dataset
    
    return avg_areas, labels, count, sample_mask

def avg_area(divide_to=5):
    config = Config()
    ds = get_dataset(config=config, mode='train')
    areas = []
    print(f'Calculating avg_areas by dividing into {divide_to}')
    for i in range(len(ds)):
        img, mask = ds[i][0], ds[i][1]
        a_r = area_ratio(mask.float()).item()
        areas.append(round(a_r, 3))

    areas = sorted(areas)
    portion = int(len(ds)//5)
    points  = [portion*i for i in range(divide_to)] + [len(ds)-1]

    avg_areas = []
    labels = []
    for i in range(len(points)-1):
        avg_areas.append(sum(areas[points[i]:points[i+1]])/(points[i+1] - points[i]))
        labels.append(f'{areas[points[i]]}-{areas[points[i+1]]}')
    
    ds_name = config.dataset
    count = [points[i+1]-points[i] for i in range(len(points)-1)]
    
    return avg_areas, labels, count, ds_name 


if __name__ =='__main__':
    area, labels, ds_name = avg_area()

    
        
        
