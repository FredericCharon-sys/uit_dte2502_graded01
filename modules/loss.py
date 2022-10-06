import torch

import torch.nn as nn
from torch.nn import functional as F


class PHOSCLoss(nn.Module):
    def __init__(self, phos_w=4.5, phoc_w=1):
        super().__init__()

        self.phos_w = phos_w
        self.phoc_w = phoc_w

    def forward(self, y: dict, targets: torch.Tensor):

        """
        y is a dictionary, containing the outputs from our model. It has two entries, one for PHOS, containing the
        PHOS-layer output and one for PHOC, containing the output from the PHOC layers.

        The target Tensor contains the PHOSC vector we stored in our dataset. That's why it has to be split into
            two parts, the PHOS and the PHOC  vector.
        """

        phos_loss = self.phos_w * F.mse_loss(y['phos'], targets[:, :165])

        phoc_loss = self.phoc_w * F.binary_cross_entropy(y['phoc'], targets[:, 165:])

        loss = phos_loss + phoc_loss
        return loss
