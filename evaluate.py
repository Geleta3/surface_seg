import torch 
import torch.nn.functional as F 
import numpy as np 
from utils.utils import pred2onehot
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.metrics import jaccard_score, confusion_matrix, average_precision_score

np.random.seed(4321)

def evaluate(model, epoch, dataloader, device, classes, logger, return_mean=True): 
    model.eval()
    print('Validating...')
    metrics_list = ['iou', 'f1',  'AP']    # 'roc_auc',
    metrics_result = dict(zip(metrics_list, [[] for _ in range(len(metrics_list))]))
    
    np.random.seed(0)
    num_to_save = logger.num_img2save
    indexes = np.random.randint(0, len(dataloader.dataset), num_to_save)
    step = 0
    for id, (batch) in enumerate(dataloader):
        image, mask = batch[0], batch[1]
        if len(batch) > 2:
            img_channels = batch[2]
        image, mask = image.to(device), mask.to(device)
        image = image/255
        if classes == 1:
            out, skip_att, img_att = model(image)
            pred = torch.sigmoid(out)
        else:
            out, skip_att, img_att = model(image)
            pred = torch.softmax(out, dim=1)
        
        metrics = Metrics(y_pred=pred, y_true=mask, 
                          n_class=classes, bg_index=0)
        
        iou = metrics.IOU(return_mean=return_mean).tolist()
        f1  = metrics.F1(return_mean=return_mean).tolist()
        AP  = metrics.AP().tolist()
        # print('IOU: ', iou, f1, AP)
        for idx, metric in enumerate([iou, f1, AP]):
            metrics_result[metrics_list[idx]].append(metric)
 
        if id in indexes:
            step += 1
            logger.save_img(tag=f'Epoch{epoch}/image', img=image[:, 0:img_channels[0,0], :, :], step=step)
            logger.save_img(tag=f'Epoch{epoch}/mask', img=mask, step=step)
            logger.save_img(tag=f'Epoch{epoch}/pred', img=pred, step=step)
            
    
    mr = metrics_result
    avg_iou = sum(mr["iou"])/len(mr["iou"])*100
    avg_f1 = sum(mr["f1"])/len(mr["f1"])*100
    avg_AP = sum(mr["AP"])/len(mr["AP"])*100
    
    print(f'Epoch: {epoch}')
    print('###############')
    print(f'#IOU: {avg_iou} F1: {avg_f1}', f'AP: {avg_AP}') 
    # Not_Impl {sum(mr["roc_auc"])/len(mr["roc_auc"]) ROC_AUC: {sum(mr["roc_auc"])/len(mr["roc_auc"])} 
    print('###############')
        
    return avg_iou, avg_f1, avg_AP #metrics_result['iou'], metrics_result['f1'], metrics_result['AP']



class Metrics:
    def __init__(self, y_pred, y_true, n_class=1, bg_index=0, epsilon=1e-6):
        
        assert y_pred.ndim == y_true.ndim == 4, 'Excpect the output shape to be BCHW'
        assert y_pred.shape == y_true.shape 
        
        self.y_pred = y_pred
        self.y_true = y_true 
        self.n_class = n_class+1 if n_class > 1 else n_class
        self.epsilon = epsilon
        self.bg_index = bg_index    

        bs, c, h, w = y_pred.shape
        if c > 1:
          long_tens = y_pred.permute(0, 2, 3, 1).argmax(-1).reshape(-1)
          self.y_pred_onehot = F.one_hot(long_tens, num_classes=c).reshape(bs, h, w, c)
          self.y_pred_onehot = self.y_pred_onehot.permute(0, 3, 1, 2)
        else:
          self.y_pred_onehot = (self.y_pred > 0.4).float()
        
        y_true = y_true.float()
        self.tp = (self.y_pred_onehot*y_true).sum(dim=[0, 2, 3])
        self.fp = ((1-y_true)*self.y_pred_onehot).sum(dim=[0, 2, 3])
        self.fn = (y_true * (1 - self.y_pred_onehot)).sum(dim=[0, 2, 3])
    
    def IOU(self, return_mean=True):
        intersection = (self.y_pred_onehot * self.y_true).sum(dim=[0, 2, 3])
        union = self.y_true.sum(dim=[0, 2, 3]) + self.y_pred_onehot.sum(dim=[0, 2, 3]) - intersection
        ious = intersection/(union+1e-6)
        if return_mean:
            return ious.mean().squeeze()
        return ious.squeeze()
    
    def F1(self, return_mean=True):
        precision = self.precision(return_mean=False)
        recall = self.recall(return_mean=False)
        if return_mean and self.n_class > 1:
            f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
            f1 = f1.squeeze()[1:]   # filter background 
            return f1.mean().squeeze()
        return (2* (precision*recall) / (precision + recall + self.epsilon)).squeeze()
    
    def AP(self):
        bs, c, h, w = self.y_pred.shape
        y_true, y_pred = self.y_true.permute(0, 2, 3, 1), self.y_pred.permute(0, 2, 3, 1)
        y_true, y_pred = y_true.reshape(-1, c), y_pred.reshape(-1, c)
        y_true, y_pred = y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
        return average_precision_score(y_true, y_pred)
    
    def AUC(self):
        return 
    
    def sensetivity(self):
        return 
    
    def precision(self, return_mean=True):
        if return_mean and self.n_class > 1:
            prec = self.tp/(self.tp + self.fp + self.epsilon)
            prec = prec.squeeze()[1:].mean()
            return prec 
        return self.tp/(self.tp + self.fp + self.epsilon)
    
    def recall(self, return_mean=True):
        if return_mean and self.n_class > 1:
            rec = self.tp/(self.tp + self.fn + self.epsilon)
            rec = rec.squeeze()[1:].mean()
            return rec 
        return self.tp/(self.tp + self.fn + self.epsilon)    
