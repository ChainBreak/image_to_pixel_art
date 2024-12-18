import torch.nn as nn
from .encoder import ResidualBlock

class Decoder(nn.Module):
    def __init__(self, num_channels, num_blocks):
        super().__init__()

        self.num_channels = num_channels[::-1]
        self.num_blocks = num_blocks[::-1]

        self.model = nn.Sequential( *self.yield_model_parts() )

    def yield_model_parts(self):
        last_channels = 3

        for channels, blocks in zip(self.num_channels, self.num_blocks):
            
            # Initial conv
            yield nn.Conv2d(
                in_channels=last_channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            yield nn.BatchNorm2d(channels)
            yield nn.SiLU(False)

            for _ in range(blocks):
                yield ResidualBlock(channels)
            
            yield nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

            last_channels = channels
            # yield nn.ConvTranspose2d(
            #     in_channels=channels,
            #     out_channels=channels,
            #     kernel_size=4,
            #     stride=2,
            #     padding=1,
            #     bias=False,
            # )
            # yield nn.BatchNorm2d(channels)
            # yield nn.SiLU(True)

            last_channels = channels

        yield nn.Conv2d(
            in_channels=channels,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0,
        )
    
    def forward(self, x):
        x = x.contiguous()
        return self.model(x)