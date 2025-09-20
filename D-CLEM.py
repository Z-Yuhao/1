import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d  # 可变形卷积模块


class D_CLEM(nn.Module):
    """Direction perception cross layer feature enhancement module"""

    def __init__(self, in_channels, out_channels, deform_groups=4, use_residual=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_residual = use_residual and (in_channels == out_channels)

        # 1. Direction aware deformable convolution branch
        self.deform_conv = DeformConv2d(
            in_channels, in_channels, kernel_size=3, padding=1,
            groups=deform_groups, bias=False
        )
        self.offset_conv = nn.Conv2d(
            in_channels, 2 * 3 * 3, kernel_size=3, padding=1, bias=True
        )  # Generate 2 * K * K offsets (K=3)

        # 2. Cross layer dense connection (fusion of current layer and previous layer features)
        self.cross_layer_conv = nn.Conv2d(
            in_channels * 2, in_channels, kernel_size=1, bias=False
        )  # Input as current layer+previous layer features (doubling the number of channels)

        # 3. Context Gated Attention
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.SiLU(),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 4. Output convolution (adjust the number of channels)
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def forward(self, x, x_prev=None):
        """
        Args:
        x: Current layer feature map (B, C, H, W)
        X_rev: Previous layer feature map (B, C, H, W), used for cross layer connections (set to None if none)
        """
        # Direction aware feature extraction
        offset = self.offset_conv(x)
        x_dir = self.deform_conv(x, offset)  # Output and input dimensions are consistent

        # Cross layer dense connection (if there are previous layer features)
        if x_prev is not None:
            x_dense = torch.cat([x_dir, x_prev], dim=1)  # double..
            x_dense = self.cross_layer_conv(x_dense)  # Restore..
        else:
            x_dense = x_dir

        # Context Gated Attention
        attn = self.gate(x_dense)
        x_gated = x_dense * attn  # Channel space joint weighting

        # Output convolution (keeping dimensions unchanged)
        out = self.out_conv(x_gated)

        # residual connection
        if self.use_residual:
            out = x + out  # in_channels==out_channels
        return out