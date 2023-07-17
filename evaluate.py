import torch 
import torch.nn.functional as F 
import numpy as np 
from utils.utils import pred2onehot
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.metrics import jaccard_score, confusion_matrix, average_precision_score

np.random.seed(4321)

def IOU(y_pred, y_true):
    assert y_true.shape == y_pred.shape, "Target and prediction shape should be equal to calculate IOU"
    intersection = (y_pred*y_true).sum()
    union = sum(y_true) + sum(y_pred) - intersection
    return intersection/(union+1e-8)

def MultiIOU(y_pred, y_true):
    assert y_pred.shape == y_true.shape, 'Target and prediction shape should be equal to calculate IOU'
    assert len(y_pred.shape) == len(y_true.shape) == 4, 'Calculating Multi-class IOU except BCHW shape'
    
    return 

def result_treshold_05_jaccard_score_f1score(sonuc, y_test, pred_test):
    
    y_pred, y_true = pred_test, y_test
    assert y_pred.shape == y_true.shape, "Pred and GT should have the same size and Shape should be BCHW"
    bs, c, h, w = y_pred.shape

    #print(y_true.shape, y_pred.shape)

    y_pred, y_true = y_pred.permute(0, 2, 3, 1), y_true.permute(0, 2, 3, 1)     # BCHW -> BHWC
    
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    y_pred_binary = pred2onehot(y_pred)
    
    y_pred, y_true, y_pred_binary = y_pred.reshape(-1, c), y_true.reshape(-1, c), y_pred_binary.reshape(-1, c)
    y_pred, y_true, y_pred_binary = y_pred.squeeze(), y_true.squeeze(), y_pred_binary.squeeze()

    if c > 1:
        average = 'micro'
    else:
        average = None

    average = None
    AP = average_precision_score(y_true, y_pred, average=average) #*10000//1
    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    # roc_a
    
    f1_s = f1_score(y_true, y_pred_binary, average=average) [1]# * 10000//1
    iou = IOU(y_pred=y_pred_binary, y_true=y_true)
    
    # iou = jaccard_score(y_true, y_pred_binary, average=average)# * 10000//1
    
    # sonuc.append('my_iou_score')
    # sonuc.append(str(iou))
    # sonuc.append('my_f1-score')
    # sonuc.append(str(f1_s))
    # sonuc.append('my_roc_auc')
    # sonuc.append(str(roc_auc))
    # sonuc.append('AP_{}'.format(AP))
    
    return iou, f1_s, roc_auc,  AP

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
        
        # if classes == 1:
        #     iou = metrics.IOU(return_mean=return_mean).item()
        #     f1  = metrics.F1(return_mean=return_mean).item()
        #     AP  = metrics.AP(return_mean=return_mean).tolist()
        # else:
        
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





def IOU_(output, target, K, ignore_index=255, device='cpu'):
    """
    author: zacario li
    date: 2020-04-02
    
    Reference: 
    https://github.com/zacario-li/Segmentation-based-deep-learning-approach-for-surface-defect-detection/blob/master/utils/common.py

    """
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K-1) 
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K-1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.to(device), area_union.to(device), area_target.to(device)

# def IOU(y_pred, y_true):
#     assert y_true.shape == y_pred.shape, "Target and prediction shape should be equal to calculate IOU"
#     intersection = (y_pred*y_true).sum()
#     union = sum(y_true) + sum(y_pred)
#     return intersection/union

def f1_loss(y_true:torch.Tensor, y_pred:torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
        
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1


if __name__ == '__main__':
    y_pred = torch.randn(2, 3, 224, 224)
    y_true = torch.ones(2, 3, 224, 224)
    metrics = Metrics(y_pred, y_true, n_class=3, bg_index=0)
    print('IOU: ', metrics.IOU(False).tolist())
    print('AP: ', metrics.AP())
    print('F1: ', metrics.F1(True))
