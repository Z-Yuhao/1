import torch
import torch.nn as nn
import torch.nn.functional as F


class VCSPPF(nn.Module):
    """VCSPPF"""

    def __init__(self, in_channels, out_channels, reduction=4, scales=[1, 3, 5]):
        super().__init__()

        # 1. Feature compression layer (connecting upstream feature output)
        self.horizontal_compress = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.SiLU()
        )

        # 2. Multi scale lateral feature branching (simulating the lateral pooling of raw SPPF)
        self.horizontal_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels // reduction, in_channels // reduction,
                          kernel_size=3, padding=s, dilation=s),
                nn.BatchNorm2d(in_channels // reduction),
                nn.SiLU()
            )
            for s in scales  #Simulating horizontal multi-scale with different expansion rates
        ])

        # 3. Vertical feature compensation branch (core vertical enhancement)
        self.vertical_compensation = nn.Sequential(
            nn.Conv2d(in_channels // reduction, in_channels // reduction,
                      kernel_size=(7, 1), padding=(3, 0)),  # 纵向7x1卷积
            nn.BatchNorm2d(in_channels // reduction),
            nn.SiLU()
        )

        # 4. Channel attention (suppressing horizontal noise and enhancing vertical targets)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels // reduction, in_channels // (reduction * 2), kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(in_channels // (reduction * 2), in_channels // reduction, kernel_size=1),
            nn.Sigmoid()
        )

        # 5. Feature fusion (horizontal+vertical compensation feature concatenation)
        self.fusion = nn.Sequential(
            nn.Conv2d((len(scales) + 1) * (in_channels // reduction), out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        """
        Args:
        x: The input feature maps (B, C, H, W) are usually feature maps with rich horizontal features output by DABConv
        Returns:
        Out: Feature maps after vertical compensation (B, out_channels, H, W)
        """
        #Horizontal feature compression
        x_hor = self.horizontal_compress(x)

        #Horizontal multi-scale feature extraction (simulating SPPF's horizontal pooling)
        hor_features = [branch(x_hor) for branch in self.horizontal_branches]

        #Vertical feature compensation (core operation: enhancing vertical information)
        ver_feature = self.vertical_compensation(x_hor)

        #Channel attention: Suppress horizontal background and highlight vertical targets
        attn = self.attention(x_hor)
        attn_hor_features = [f * attn for f in hor_features]  #Apply attention to horizontal features
        attn_hor_features.append(ver_feature)  #Add vertical compensation feature (total length: len (scales)+1)

        #Feature fusion: horizontal features (with attention)+vertical compensation features
        concat = torch.cat(attn_hor_features, dim=1)
        return self.fusion(concat)