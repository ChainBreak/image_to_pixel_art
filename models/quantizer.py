
import torch.nn as nn
import torch

class Quantizer(nn.Module):
    def __init__(self, channels, color_pallete):
        super().__init__()

        self.num_colors = len(color_pallete)
        self.color_pallete_rgb = torch.tensor(self.hex_list_to_rgb_list(color_pallete), dtype=torch.float32) / 255
        
        self.color_pallete_rgb = nn.Parameter(self.color_pallete_rgb, requires_grad=False)

        self.channels_to_color_prob_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=self.num_colors,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        
        x = self.channels_to_color_prob_conv(x)

        color_prob = torch.softmax(x, dim=1)
        
        blended_color = self.get_weighted_sum_of_colors(color_prob, self.color_pallete_rgb)

        discrete_color = self.sample_discrete_color(color_prob, self.color_pallete_rgb)
 
        pixel_art = (discrete_color - blended_color).detach() + blended_color

        return pixel_art
    
    def get_weighted_sum_of_colors(self, color_prob, color_pallete_rgb):
        b,c,h,w = color_prob.shape

        color_prob = color_prob.reshape(b,c,1,h,w)

        color_pallete_rgb = color_pallete_rgb.reshape(1, self.num_colors, 3, 1, 1)

        blended_color = (color_prob * color_pallete_rgb).sum(dim=1)

        return blended_color
    
    def sample_discrete_color(self, color_prob, color_pallete_rgb):
        b,c,h,w = color_prob.shape

        color_prob = color_prob.permute(0,2,3,1).reshape(-1, c)

        color_index = torch.multinomial(color_prob, 1).reshape(-1)

        discrete_color = color_pallete_rgb[color_index]

        discrete_color = discrete_color.reshape(b,h,w,3).permute(0,3,1,2)

        return discrete_color

    def hex_list_to_rgb_list(self, hex_list):
        return [self.hex_to_rgb(h[2:]) for h in hex_list]
    
    def hex_to_rgb(self, hex):
        hex = hex.lstrip('#')

        # Ignore the first two characters and treat the others as rgb
        return [ int(hex[i:i+2], 16) for i in (0, 2, 4) ]
    