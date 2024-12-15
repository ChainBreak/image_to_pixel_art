import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, down_blocks_channels):
        super().__init__()

        self.down_blocks_channels = down_blocks_channels[::-1]

        self.model = nn.Sequential( *self.yield_model_parts() )

    def yield_model_parts(self):
        last_channels = 3

        for channels in self.down_blocks_channels:
            yield nn.ConvTranspose2d(
                in_channels=last_channels,
                out_channels=channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
            yield nn.BatchNorm2d(channels)
            yield nn.LeakyReLU(True)
            last_channels = channels

        yield nn.Conv2d(
            in_channels=last_channels,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0,
        )
    
    def forward(self, x):
        return self.model(x)