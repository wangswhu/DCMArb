"""Shared convolution and attention layers used by DCMArb."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    bias=True,
    dilation=1,
    groups=1,
):
    padding = ((kernel_size - 1) // 2) * dilation
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
        dilation=dilation,
        groups=groups,
    )


class CAB(nn.Module):
    """Channel Attention Block (CAB)."""

    def __init__(self, in_channels, reduction=4):
        super().__init__()
        hidden_channels = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x))
        return x * self.sigmoid(attention)


class SAB(nn.Module):
    """Spatial Attention Block (SAB)."""

    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(
            2,
            1,
            kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.conv1(torch.cat([avg_out, max_out], dim=1))
        return x * self.sigmoid(attention)


class SelectionBranch(nn.Module):
    """HFIS selection branch composed of CAB and SAB."""

    def __init__(
        self,
        conv,
        n_feats,
        kernel_size,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
        res_scale=1,
    ):
        super().__init__()
        modules = []
        for index in range(2):
            modules.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                modules.append(nn.BatchNorm2d(n_feats))
            if index == 0:
                modules.append(act)

        modules.extend((CAB(n_feats, 16), SAB()))
        self.body = nn.Sequential(*modules)
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x).mul(self.res_scale)


def mean_channels(feature):
    if feature.ndim != 4:
        raise ValueError(f"Expected a 4D tensor, got shape {tuple(feature.shape)}.")
    spatial_sum = feature.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (feature.size(2) * feature.size(3))


def stdv_channels(feature):
    if feature.ndim != 4:
        raise ValueError(f"Expected a 4D tensor, got shape {tuple(feature.shape)}.")
    feature_mean = mean_channels(feature)
    variance = (
        (feature - feature_mean)
        .pow(2)
        .sum(3, keepdim=True)
        .sum(2, keepdim=True)
        / (feature.size(2) * feature.size(3))
    )
    return variance.pow(0.5)


class BSConvU(nn.Module):
    """Pointwise followed by depthwise convolution."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=True,
        padding_mode="zeros",
        with_ln=False,
        bn_kwargs=None,
    ):
        super().__init__()
        self.with_ln = with_ln
        self.bn_kwargs = {} if bn_kwargs is None else bn_kwargs
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.dw = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, feature):
        return self.dw(self.pw(feature))


def round_scale_interpolate(image, scale_factor):
    """Bicubically resize an image using rounded spatial dimensions."""

    _, _, height, width = image.shape
    if isinstance(scale_factor, torch.Tensor):
        if scale_factor.numel() != 1:
            raise ValueError("scale_factor must contain a single value.")
        scale_factor = scale_factor.item()

    scale_factor = float(scale_factor)
    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}.")

    new_height = round(height * scale_factor)
    new_width = round(width * scale_factor)
    return F.interpolate(
        image,
        size=(new_height, new_width),
        mode="bicubic",
        align_corners=False,
    )
