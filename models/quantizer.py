
import torch.nn as nn
import torch

class Quantizer(nn.Module):
    def __init__(self, channels, color_pallete):
        super().__init__()

        self.num_colors = len(color_pallete)
        self.color_pallete_rgb = torch.tensor(self.hex_list_to_rgb_list(color_pallete), dtype=torch.float32) / 255
        self.color_pallete_rgb = self.color_pallete_rgb.reshape(1, self.num_colors, 3, 1, 1)
        self.color_pallete_rgb = nn.Parameter(self.color_pallete_rgb, requires_grad=False)

        self.channels_to_rgb_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=self.num_colors,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        #bchw
        x = self.channels_to_rgb_conv(x)
        b,c,h,w = x.shape

        x = x.reshape(b,c,1,h,w)
        x = torch.softmax(x, dim=1)
        
        x = x * self.color_pallete_rgb

        x = x.sum(dim=1)
        # x = self.quantize(x)
        return x

    def hex_list_to_rgb_list(self, hex_list):
        return [self.hex_to_rgb(h[2:]) for h in hex_list]
    
    def hex_to_rgb(self, hex):
        hex = hex.lstrip('#')

        # Ignore the first two characters and treat the others as rgb
        return [ int(hex[i:i+2], 16) for i in (0, 2, 4) ]
    