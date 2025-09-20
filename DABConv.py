import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


class DABConv(nn.Module):
    """DABConv"""

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=None,
            dilation=1,
            groups=1,
            bias=True,
            deformable_groups=1,
            use_corner_offset=True,
            use_spatial_attn=True,
            debug=False
    ):
        super().__init__()

        # padding
        if padding is None:
            padding = (kernel_size - 1) // 2

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.debug = debug

        # 1. Standard Convolutional Branch (maintains the same output size as the original Conv)
        self.std_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias
        )

        # 2. Offset prediction branch
        if isinstance(kernel_size, int):
            kh, kw = kernel_size, kernel_size
        else:
            kh, kw = kernel_size

        offset_channels = 2 * kh * kw #Two offsets for each position (x, y)

        self.offset_conv = nn.Conv2d(
            in_channels,
            offset_channels * deformable_groups,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True
        )

        # Initialize offset to 0
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

        # 3. Corner attention prediction (if enabled)
        self.corner_conv = None
        if use_corner_offset:
            self.corner_conv = nn.Conv2d(
                in_channels,
                4,  #Attention weights for 4 corner points
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=True
            )
            nn.init.constant_(self.corner_conv.weight, 0)
            nn.init.constant_(self.corner_conv.bias, 0)

        # 4. Modulation weight prediction (optional)
        self.modulator_conv = None
        if use_spatial_attn:
            modulator_channels = kh * kw * deformable_groups

            # If corner attention is enabled, add 4 channels for corner attention
            if self.corner_conv is not None:
                modulator_channels += 4

            self.modulator_conv = nn.Conv2d(
                in_channels,
                modulator_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=True
            )
            nn.init.constant_(self.modulator_conv.weight, 0)
            nn.init.constant_(self.modulator_conv.bias, 1)  # init to 1

        # 5. Deformable convolutional layer
        self.deform_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias
        )

        # 6. Output fusion
        self.fusion = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

        # 7. Mapping matrix from corner to position
        if use_corner_offset:
            # Define the mapping from corner points to convolutional kernel positions
            corner_to_position = torch.zeros(4, kh * kw)
            corner_to_position[0, 0] = 1.0  # Top left
            corner_to_position[1, 2] = 1.0  # Top right
            corner_to_position[2, 6] = 1.0  # Lower left
            corner_to_position[3, 8] = 1.0  # Lower right
            # Buffer zone, enabling it to automatically follow the data type of the model
            self.register_buffer('corner_to_position', corner_to_position)

    def forward(self, x):
        # 1. Standard convolution result
        x_std = self.std_conv(x)

        # 2. Calculate offset
        offset = self.offset_conv(x)

        # 3. Calculate modulation weights (if enabled)
        if self.modulator_conv is not None:
            modulator = self.modulator_conv(x)

            # If there is corner attention, incorporate it into modulation weights
            if self.corner_conv is not None:
                corner_attn = torch.sigmoid(self.corner_conv(x))

                # Separate standard modulation weights and corner modulation weights
                kh, kw = self.kernel_size, self.kernel_size
                std_modulator, corner_modulator = torch.split(
                    modulator,
                    [kh * kw * self.deformable_groups, 4],
                    dim=1
                )

                # Map corner attention to 9 positions of the convolutional kernel
                # Use registered buffer to ensure consistent data types
                corner_weights = torch.einsum('bchw,ck->bkhw', corner_modulator, self.corner_to_position)
                corner_weights = corner_weights.reshape(corner_weights.size(0), self.deformable_groups, kh * kw,
                                                        corner_weights.size(2), corner_weights.size(3))
                corner_weights = corner_weights.reshape(corner_weights.size(0), -1,
                                                        corner_weights.size(3), corner_weights.size(4))

                # Combining standard modulation weights and corner modulation weights
                modulator = torch.sigmoid(std_modulator) * torch.sigmoid(corner_weights)
            else:
                modulator = torch.sigmoid(modulator)
        else:
            modulator = None

        # 4. Application of deformable convolution
        b, c, h, w = x.size()

        # debug information
        if self.debug:
            print(f"输入尺寸: {x.shape}")
            print(f"标准卷积输出尺寸: {x_std.shape}")
            print(f"偏移量形状: {offset.shape}")
            if modulator is not None:
                print(f"调制权重形状: {modulator.shape}")
                print(f"调制权重数据类型: {modulator.dtype}")
        x_def = deform_conv2d(
            input=x,
            offset=offset,
            weight=self.deform_conv.weight,
            bias=self.deform_conv.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=modulator
        )

        if self.debug:
            print(f"可变形卷积输出尺寸: {x_def.shape}")


        x_fused = torch.cat([x_std, x_def], dim=1)
        x_out = self.fusion(x_fused)

        if self.debug:
            print(f"最终输出尺寸: {x_out.shape}")

        return x_out