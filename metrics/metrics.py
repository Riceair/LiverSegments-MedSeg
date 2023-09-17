import torch

def dice_coeff(pred : torch.tensor, target : torch.tensor):
    smooth = 1
    intersect = (pred * target).sum()
    denom = (target+pred).sum()
    dice = (2. * intersect + smooth) / (denom + smooth)
    return dice.mean()

def dice_loss(pred : torch.tensor, target : torch.tensor):
    return 1 - dice_coeff(pred, target)

def dice_coeff_multicat(pred : torch.tensor, target : torch.tensor, smooth=1e-7):
    '''
    Dice coefficient for multi categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    pred = pred[:, 1:]
    target = target[:, 1:]
    intersect = (target * pred).sum()
    denom = (target + pred).sum()
    dice = (2. * intersect / (denom + smooth))
    return dice.mean()

def dice_multicat_loss(pred : torch.tensor, target : torch.tensor):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - dice_coeff_multicat(target, pred)