from dataset.dagm import DAGMDataset
from dataset.mt import MT_Dataset
from dataset.neu import NEU_Dataset
from dataset.magnetic import MagneticDataset
from dataset.aitex import AITEXDataset


def get_dataset(config, mode, transform=None, *args, **kwargs):
    name = config.dataset
    if name == 'mt':
        dataset = MT_Dataset(root=config.root['mt'], 
                             classes=config.classes['mt'], 
                             mode=mode, 
                             img_size=config.img_dim, 
                             img_channel=config.img_channel['mt'], 
                             train_valid_split=config.train_valid_split,
                             preprocess=config.preprocess,
                             transform=transform,
                             *args, **kwargs
                             )
    elif name == 'neu':
        dataset = NEU_Dataset(root=config.root['neu'], 
                            mode=mode, 
                            img_size=config.img_dim, 
                            classes=config.classes['neu'],
                            img_channel=config.img_channel['neu'], 
                            preprocess=config.preprocess, 
                            *args, **kwargs)
    elif name == 'magnetic':
        dataset = MagneticDataset(root=config.root['magnetic'], 
                                  classes=config.classes['magnetic'], 
                                  mode=mode, 
                                  img_size=config.img_dim, 
                                  img_channel=config.img_channel['magnetic'], 
                                  train_valid_split=config.train_valid_split, 
                                  preprocess=config.preprocess, 
                                  *args, **kwargs)
    elif name == 'dagm':
        dataset = DAGMDataset(root=config.root['dagm'], 
                            classes=config.classes['dagm'],
                            img_size=config.img_dim,
                            img_channel=config.img_channel['dagm'],
                            mode=mode,
                            transform=transform,
                            preprocess=config.preprocess,
                            *args, **kwargs)
    elif name == 'aitex':
        dataset = AITEXDataset(root=config.root['aitex'],
                               classes=config.classes['aitex'],
                               img_size=config.img_dim,
                               img_channel=config.img_channel['aitex'],
                               preprocess=config.preprocess,
                               transform=transform, 
                               *args, **kwargs)
        
    else:
        raise ValueError('Dataset should be one of  ["mt", "neu", "magnetic"]')
    
    return dataset
