
import torch

def dice_score(preds, targets, smooth=1e-6):
    inter = (preds * targets).sum(dim=(2, 3))
    return ((2 * inter + smooth) /
            (preds.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + smooth)).mean().item()

def iou_score(preds, targets, smooth=1e-6):
    inter = (preds * targets).sum(dim=(2, 3))
    union = (preds + targets - preds * targets).sum(dim=(2, 3))
    return ((inter + smooth) / (union + smooth)).mean().item()
