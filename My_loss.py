
import torch
import torch.nn as nn

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class Fu_loss(nn.Module):

    def __init__(self):
        super(Fu_loss, self).__init__()
        self.loss_fn  = nn.L1Loss()

    def forward(self, x, y):
        x = torch.fft.fft2(x)
        y = torch.fft.fft2(y)
        loss = self.loss_fn(x,y)

        return loss

