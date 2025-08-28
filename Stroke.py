import torch
from torch import nn
import torch.nn.functional as F

class LightweightStrokeModule(nn.Module):

    def __init__(self, in_channels, out_channels=32, cat=True):
        super().__init__()
        self.cat = cat
        #  Directional gradient detection
        self.direction_conv = nn.Conv2d(in_channels, 4, 5, padding=2, bias=False)
        self._init_direction_filters(in_channels)

        # Stroke feature extraction
        self.stroke_conv = nn.Sequential(
            nn.Conv2d(4, out_channels // 2, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels // 2, affine=True),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU()
        )

        # Spatial attention module
        self.attn = nn.Sequential(
            nn.Conv2d(out_channels, 1, 1),
            nn.Sigmoid()
        )

    def _init_direction_filters(self, in_channels):
        """Initialize directional filters for edge/stroke detection."""
        # Horizontal direction
        kernel_h = torch.tensor([
            [-1, -1, -1, -1, -1],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [-1, -1, -1, -1, -1]
        ]).float()

        # Vertical direction
        kernel_v = kernel_h.t()

        # 45-degree direction
        kernel_45 = torch.tensor([
            [-1, -1, 0, 1, 1],
            [-1, 0, 1, 1, 0],
            [0, 1, 1, 0, -1],
            [1, 1, 0, -1, -1],
            [1, 0, -1, -1, -1]
        ]).float()

        # 135-degree direction
        kernel_135 = torch.flip(kernel_45, [1])

        # Stack all kernels
        kernels = torch.stack([kernel_h, kernel_v, kernel_45, kernel_135])
        kernels = kernels.unsqueeze(1).repeat(1, in_channels, 1, 1)  # [4, C_in, 5, 5]

        # Initialize convolutional weight
        self.direction_conv.weight.data = kernels
        self.direction_conv.weight.requires_grad = False  # Fix the filter weights (non-trainable)

    def forward(self, x):

        # Extract directional features
        directions = self.direction_conv(x)
        # Extract stroke features
        stroke_feat = self.stroke_conv(directions)

        # Apply spatial attention
        attn_map = self.attn(stroke_feat)
        stroke_weighted = stroke_feat * attn_map  #


        if self.cat:
            # Concatenate original input x with stroke features
            return torch.cat([x, stroke_weighted], dim=1)  # [B, C_in + out_channels, H, W]
        else:
            return stroke_weighted  # [B, out_channels, H, W]