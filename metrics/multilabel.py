import torch

def dice_coeff(pred, target, smooth = 1.):    
    # 計算交集
    intersection = torch.sum(pred * target)
    # 計算分母
    denominator = torch.sum(pred) + torch.sum(target)
    # 計算Dice係數
    dice_coefficient = (2. * intersection + smooth) / (denominator + smooth)
        
    return dice_coefficient

def dice_coeff_no_bg(pred, target, smooth = 1.):
    pred = pred[:, 1:]
    target = target[:, 1:]
    intersection = torch.sum(pred * target)
    denominator = torch.sum(pred) + torch.sum(target)
    # 計算Dice係數
    dice_coefficient = (2. * intersection + smooth) / (denominator + smooth)
        
    return dice_coefficient

def dice_all_labels(pred, target, smooth = 1.):
    num_classes = pred.size(1)
    label_dice = []

    for i in range(num_classes):
        # 取出第i個類別的預測和真實分割Mask
        pred_i = pred[:, i, :, :]
        target_i = target[:, i, :, :]
        
        # 計算第i個類別的Dice損失
        dice_i = dice_coeff(pred_i, target_i, smooth)
        label_dice.append(dice_i.item())
    return label_dice

def dice_loss(pred, target, smooth = 1.):
    return 1 - dice_coeff(pred, target, smooth)

def multilabel_dice(pred, target, smooth = 1.):
    num_classes = pred.size(1)
    total_dice = 0.

    for i in range(num_classes):
        # 取出第i個類別的預測和真實分割Mask
        pred_i = pred[:, i, :, :]
        target_i = target[:, i, :, :]
        
        # 計算第i個類別的Dice損失
        dice_i = dice_coeff(pred_i, target_i, smooth)
        
        # 將第i個類別的Dice損失加到總體損
        total_dice += dice_i
    return total_dice / num_classes

def multilabel_dice_loss(pred, target, smooth = 1.):
    num_classes = pred.size(1)
    total_loss = 0.
    
    for i in range(num_classes):
        # 取出第i個類別的預測和真實分割Mask
        pred_i = pred[:, i, :, :]
        target_i = target[:, i, :, :]
        
        # 計算第i個類別的Dice損失
        dice_loss_i = dice_loss(pred_i, target_i, smooth)
        
        # 將第i個類別的Dice損失加到總體損
        total_loss += dice_loss_i
    return total_loss / num_classes