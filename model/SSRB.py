"""Spatial relation block used in DCMArb."""

import torch
import torch.nn as nn
from mamba_ssm import Mamba


class SSRB(nn.Module):
    """Shifted-window Spatial Relation Block (SSRB).

    The input shape is [B, H*W, C]. Channels are split into two halves:
    one branch scans local windows in the horizontal partition layout, and the
    other scans local windows in the vertical partition layout. The two halves
    are concatenated back after window reversal.
    """

    def __init__(self, dim, input_resolution, split_size=(2, 4), shift_size=(1, 2), expand_ratio=4.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SSRB requires an even channel dimension, got dim={dim}.")

        self.dim = dim
        self.input_resolution = tuple(input_resolution)
        self.split_size = tuple(split_size)
        self.shift_size = tuple(shift_size)
        half_dim = dim // 2

        self.mamba_h = Mamba(d_model=half_dim, d_state=16, d_conv=4, expand=expand_ratio)
        self.mamba_v = Mamba(d_model=half_dim, d_state=16, d_conv=4, expand=expand_ratio)

        self.locality_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        if L != H * W:
            raise ValueError(f"Input token length {L} does not match H*W={H * W}.")
        if C != self.dim:
            raise ValueError(f"Input channel {C} does not match dim={self.dim}.")
        self._check_window_size(H, W, self.split_size)
        self._check_window_size(H, W, self.split_size[::-1])

        shortcut = x
        x = x.view(B, H, W, C)
        x_h, x_v = torch.chunk(x, chunks=2, dim=-1)

        if any(s > 0 for s in self.shift_size):
            shifted_h = torch.roll(x_h, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
            shifted_v = torch.roll(x_v, shifts=(-self.shift_size[1], -self.shift_size[0]), dims=(1, 2))
        else:
            shifted_h = x_h
            shifted_v = x_v

        out_h = self._run_window_mamba(shifted_h, self.split_size, self.mamba_h)
        out_v = self._run_window_mamba(shifted_v, self.split_size[::-1], self.mamba_v)

        if any(s > 0 for s in self.shift_size):
            out_h = torch.roll(out_h, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
            out_v = torch.roll(out_v, shifts=(self.shift_size[1], self.shift_size[0]), dims=(1, 2))

        attended_x = torch.cat([out_h, out_v], dim=-1).view(B, H * W, C)
        local = self.locality_conv(shortcut.view(B, H, W, C).permute(0, 3, 1, 2))
        local = local.flatten(2).transpose(1, 2)
        return self.proj(attended_x + local)

    def _run_window_mamba(self, x, window_size, mamba):
        B, H, W, C = x.shape
        windows = self.window_partition(x, window_size)
        windows = windows.view(-1, window_size[0] * window_size[1], C)
        windows = mamba(windows)
        windows = windows.view(-1, window_size[0], window_size[1], C)
        return self.window_reverse(windows, H, W, B, window_size)

    @staticmethod
    def _check_window_size(H, W, window_size):
        if H % window_size[0] != 0 or W % window_size[1] != 0:
            raise ValueError(
                f"Input size ({H}, {W}) must be divisible by window size {window_size}. "
                "Please adjust lr_size or split_size."
            )

    @staticmethod
    def window_partition(x, window_size):
        B, H, W, C = x.shape
        wh, ww = window_size
        x = x.view(B, H // wh, wh, W // ww, ww, C)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous()

    @staticmethod
    def window_reverse(windows, H, W, B, window_size):
        wh, ww = window_size
        x = windows.view(B, H // wh, W // ww, wh, ww, -1)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


SwinSpatialMamba = SSRB


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssrb = SSRB(dim=120, input_resolution=(32, 32), split_size=(2, 16), shift_size=(0, 0)).to(device)
    x = torch.randn(2, 32 * 32, 120, device=device)
    y = ssrb(x, (32, 32))
    print(y.shape)
