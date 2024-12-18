import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, num_channels, num_blocks):
        super().__init__()

        self.num_channels = num_channels
        self.num_blocks = num_blocks

        self.model = nn.Sequential( *self.yield_model_parts() )

    def yield_model_parts(self):
        last_channels = self.num_channels[0]

        # Initial conv
        yield nn.Conv2d(
            in_channels=3,
            out_channels=last_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        yield nn.BatchNorm2d(last_channels)
        yield nn.SiLU(False)

        for channels, blocks in zip(self.num_channels, self.num_blocks):
            # Down conv
            yield nn.Conv2d(
                in_channels=last_channels,
                out_channels=channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
            yield nn.BatchNorm2d(channels)
            yield nn.SiLU(False)

            for _ in range(blocks):
                yield ResidualBlock(channels)

            last_channels = channels
    
    def forward(self, x):
        return self.model(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.SiLU(inplace=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)
        return out