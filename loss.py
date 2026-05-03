
import torch
import torch.nn as nn

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.bce    = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs    = torch.sigmoid(logits)
        inter    = (probs * targets).sum(dim=(2, 3))
        dice     = 1 - (2 * inter + self.smooth) / (
                       probs.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + self.smooth)
        return bce_loss + dice.mean()
