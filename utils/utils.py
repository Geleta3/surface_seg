import torch 
import numpy as np 


def area_ratio(mask, channel_dim=0):
    if channel_dim == 0:
        c, h, w = mask.shape
    elif channel_dim == 2 or channel_dim == -1:
        h, w, c = mask.shape
    segmentation = torch.sum(mask)
    overall_area = (h*w) #*c
    
    return segmentation/overall_area
        
    
def pred2onehot(pred):
    # BHWC
    bs, h, w, c = pred.shape 
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if c == 1:
        return pred > 0.5
    pred = np.argmax(pred, axis=-1)[:, :, :]#, None]
    one_hot = np.zeros([bs, h, w, c])
    for i in range(c):
        one_hot[:, :, :, i] = (pred == i)
    
    return one_hot
        
def result_treshold_05_jaccard_score_f1score(sonuc, y_test, pred_test):
    
    assert isinstance([y_pred, y_true], torch.Tensor), "Only  input tensor supported"
    assert y_pred.shape == y_true.shape, "Pred and GT should have the same size and Shape should be BCHW"
    bs, c, h, w = y_pred.shape
    
    y_pred, y_true = y_pred.permute(0, 2, 3, 1), y_true.permute(0, 2, 3, 1)     # BCHW -> BHWC
    
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    y_pred_binary = pred2onehot(y_pred)
    
    y_pred, y_true = y_pred.reshape(-1, c), y_true.reshape(-1, c)
    
    # Calculate Metrics
    AP = average_precision_score(y_true, y_pred)*10000//1
    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    
    f1_s = f1_score(y_true, y_pred_binary) * 10000//1
    iou = jaccard_score(y_true, y_pred_binary) * 10000//1
    
    sonuc.append('my_iou_score')
    sonuc.append(str(iou))
    sonuc.append('my_f1-score')
    sonuc.append(str(f1_s))
    sonuc.append('my_roc_auc')
    sonuc.append(str(roc_auc))
    sonuc.append('AP_{}'.format(AP))
    
    return sonuc 

        
        
        
        