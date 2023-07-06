import torch
import torch.nn.functional as F



class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        # calculate binary cross entropy with logits
        logpt = F.binary_cross_entropy(input, target, reduction='none')
        pt = torch.exp(-logpt)

        # calculate focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * logpt

        # apply reduction
        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
        elif self.reduction == 'sum':
            focal_loss = torch.sum(focal_loss)

        return focal_loss


class dice_loss(torch.nn.Module):
    def __init__(self, smooth=1.0, reduction='mean'):
        super(dice_loss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, input, target):
        # calculate dice loss
        intersection = torch.sum(input * target)
        dice_loss = 1 - (2. * intersection + self.smooth) / (torch.sum(input) + torch.sum(target) + self.smooth)

        # apply reduction
        if self.reduction == 'mean':
            dice_loss = torch.mean(dice_loss)
        elif self.reduction == 'sum':
            dice_loss = torch.sum(dice_loss)

        return dice_loss


class focal_dice_loss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, smooth=1.0, reduction='mean', _lamda=1):
        super(focal_dice_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.reduction = reduction
        self._lamda = _lamda

    def forward(self, input, target):
        # calculate binary cross entropy with logits
        logpt = F.binary_cross_entropy(input, target, reduction='none')
        pt = torch.exp(-logpt)

        # calculate focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * logpt

        # calculate dice loss
        intersection = torch.sum(input * target)
        dice_loss = 1 - (2. * intersection + self.smooth) / (torch.sum(input) + torch.sum(target) + self.smooth)

        # apply reduction
        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
            dice_loss = torch.mean(dice_loss)
        elif self.reduction == 'sum':
            focal_loss = torch.sum(focal_loss)
            dice_loss = torch.sum(dice_loss)

        return focal_loss + self._lamda * dice_loss


class bce_focal_loss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', _lamda=1):
        super(bce_focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self._lamda = _lamda

    def forward(self, input, target):
        # calculate binary cross entropy with logits
        bce = torch.nn.BCELoss()
        Bce = bce(input, target)
        logpt = F.binary_cross_entropy(input, target, reduction='none')
        pt = torch.exp(-logpt)

        # calculate focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * logpt

        # apply reduction
        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
        elif self.reduction == 'sum':
            focal_loss = torch.sum(focal_loss)

        return focal_loss + self._lamda * Bce


class bce_dice_loss(torch.nn.Module):
    def __init__(self, smooth=1.0, reduction='mean'):
        super(bce_dice_loss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, input, target):
        # calculate binary cross entropy with logits
        logpt = F.binary_cross_entropy(input, target, reduction='none')

        # calculate dice loss
        intersection = torch.sum(input * target)
        dice_loss = 1 - (2. * intersection + self.smooth) / (torch.sum(input) + torch.sum(target) + self.smooth)

        # apply reduction
        if self.reduction == 'mean':
            dice_loss = torch.mean(dice_loss)
        elif self.reduction == 'sum':
            dice_loss = torch.sum(dice_loss)

        return logpt + dice_loss


class generalized_dice_loss(torch.nn.Module):
    def __init__(self, smooth=1.0, reduction='mean'):
        super(generalized_dice_loss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, input, target):
        num = input.shape[1]
        total_intersec = 0
        total_sum = 0
        for i in range(num):
            w = torch.sum(target[:, i, :, :, :])
        # calculate generalized dice loss
            intersection = torch.sum(input * target)
            total_intersec += intersection * w
            total_sum += (torch.sum(input) + torch.sum(target)) * w
        dice_loss = 1 - (2. * total_intersec + self.smooth) / (total_sum + self.smooth)

        # apply reduction
        if self.reduction == 'mean':
            dice_loss = torch.mean(dice_loss)
        elif self.reduction == 'sum':
            dice_loss = torch.sum(dice_loss)

        return dice_loss


class mse_focal_loss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', _lamda=0.5):
        super(mse_focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self._lamda = _lamda

    def forward(self, input, target, loc, loc_gt):
        # calculate binary cross entropy with logits
        mse = torch.nn.MSELoss()
        Mse = mse(loc, loc_gt)
        logpt = F.binary_cross_entropy(input, target, reduction='none')
        pt = torch.exp(-logpt)

        # calculate focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * logpt

        # apply reduction
        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
        elif self.reduction == 'sum':
            focal_loss = torch.sum(focal_loss)

        return (1 - self._lamda) * focal_loss + self._lamda * Mse
