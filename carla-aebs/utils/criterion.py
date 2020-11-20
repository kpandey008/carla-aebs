import torch
import torch.nn as nn


class VAELoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1.0, use_bce=True):
        super(VAELoss, self).__init__()
        if reduction not in ['mean', 'sum']:
            raise ValueError('Valid values for the reduction param are `mean`, `sum`')
        self.alpha = alpha
        self.use_bce = use_bce
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction='mean')
        self.bce = nn.BCELoss(reduction='mean')
    
    def forward(self, x, decoder_out, mu, logvar):
        # Reconstruction Loss:
        # TODO: Try out the bce loss for reconstruction
        if self.use_bce:
            reconstruction_loss = self.bce(decoder_out, x)
        else:
            reconstruction_loss = self.mse(decoder_out, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = reconstruction_loss + self.alpha * kl_loss
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss
