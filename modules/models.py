import torch

import torch.nn as nn

from modules.pyramidpooling import TemporalPyramidPooling

from timm.models.registry import register_model

__all__ = [
    'PHOSCnet_temporalpooling'
]


class PHOSCnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(

            # For each Convolutional layer in TensorFlow add one in Pytorch as well. Here we have to make sure we
            #   follow the (channel_in, channel_out, kernel) structure. The activation function is added after.
            # 2x Conv(64)
            nn.Conv2d(3, 64, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding='same'),
            nn.ReLU(),

            nn.MaxPool2d((2, 2), 2),

            # 2x Conv(128)
            nn.Conv2d(64, 128, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), padding='same'),
            nn.ReLU(),

            nn.MaxPool2d((2, 2), 2),

            # 6x Conv(256)
            nn.Conv2d(128, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),

            # 3x Conv(512)
            nn.Conv2d(256, 512, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding='same'),
            nn.ReLU(),

        )

        self.temporal_pool = TemporalPyramidPooling([1, 2, 5])

        # the model works a bit different for PHOS and PHOC, that's why the we add one function for each
        self.phos = nn.Sequential(
            nn.Flatten(),
            # 2x Linear(4096)
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # Output needs to have 165 nodes
            nn.Linear(4096, 165),
            nn.ReLU()
        )

        self.phoc = nn.Sequential(
            nn.Flatten(),
            # 2x Linear(4096)
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # Output needs to have 604 nodes
            nn.Linear(4096, 604),
            # We have to use the Sigmoid function for the output, since we have a classification problem for PHOC
            #   and the result has to be in between 0 and 1
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> dict:
        x = self.conv(x)
        x = self.temporal_pool(x)

        return {'phos': self.phos(x), 'phoc': self.phoc(x)}


@register_model
def PHOSCnet_temporalpooling(**kwargs):
    return PHOSCnet()


if __name__ == '__main__':
    model = PHOSCnet()

    x = torch.randn(5, 50, 250, 3).view(-1, 3, 50, 250)

    y = model(x)

    print(y['phos'].shape)
    print(y['phoc'].shape)
