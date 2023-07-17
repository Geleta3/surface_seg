from torch import nn 
import torch as torch



class Loss(nn.Module):
    def __init__(self, classes=1, loss='CEL'):
        super().__init__()
        self.classes = classes 
        if classes == 1:
            self.loss = nn.BCELoss()
        
        if classes > 1:
            if loss.lower() in ['cel', 'cross_entropy_loss', 'celoss']:
                self.loss = nn.CrossEntropyLoss()
    
    def forward(self, y_pred, y_true):
        # Excpect BCHW 
        bs, c, h, w = y_pred.shape
        if c == 1:
            dice_loss = self.dice_coef(y_true, y_pred)
        elif c > 1:
            dice_loss = self.dice_coef(y_true, torch.softmax(y_pred, 1))
            
        y_pred = y_pred.permute(0, 2, 3, 1).reshape(-1, c)
        
        if self.classes == 1:
            y_true = torch.as_tensor(y_true, dtype=torch.float32)
            y_pred = y_pred.reshape(-1)
            y_true = y_true.reshape(-1)
        else:
            
            y_true = torch.argmax(y_true, dim=1).reshape(-1)

        celoss = self.loss(y_pred, y_true)
            
        return  dice_loss + celoss # +
    
    def new_loss(self, y_true, y_pred, smooth=0.2, epoch=0):
        bs, c, h, w = y_true.shape 
        x_grid = torch.arange(w).reshape(1, 1, 1, w).repeat(bs, c, h, 1)/w
        y_grid = torch.arange(h).reshape(1, 1, h, 1).repeat(bs, c, 1, w)/h
        
        y_t_x, y_t_y = y_true*x_grid, y_true*y_grid
        y_p_x, y_p_y = y_pred*x_grid, y_pred*y_grid
    
        x_d = torch.sum(y_t_x) - torch.sum(y_p_x) 
        y_d = torch.sum(y_t_y) - torch.sum(y_p_y) 
        
        return 2 - (self.sech(x_d) + self.sech(y_d))
    
    def sech(self, x):
        return 2/(torch.exp(x) + torch.exp(-x))
    
        
    def new_loss(self, y_true, y_pred, smooth=0.2):
        bs, c, h, w = y_true.shape 
        x_grid = torch.arange(w).reshape(1, 1, 1, w).repeat(bs, c, h, 1)/w
        y_grid = torch.arange(h).reshape(1, 1, h, 1).repeat(bs, c, 1, w)/h
        
        y_t_x, y_t_y = y_true*x_grid, y_true*y_grid
        y_p_x, y_p_y = y_pred*x_grid, y_pred*y_grid
        
        x_inter = torch.sum(y_t_x*y_p_x)
        y_inter = torch.sum(y_t_y*y_p_y)
        
        x_union = torch.sum(y_t_x) + torch.sum(y_p_x)
        y_union = torch.sum(y_t_y) + torch.sum(y_p_y)
        
        x_dice = 2*x_inter/x_union
        y_dice = 2*y_inter/y_union
        
        return 2 - (x_dice + y_dice)
    
    def dice_coef(self, y_true, y_pred, smooth=1):
      
        y_true_f = torch.flatten(y_true)
        y_pred_f = torch.flatten(y_pred)
        intersection = torch.sum(y_true_f * y_pred_f)
        return 1 - (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)







