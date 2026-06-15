"""Scale-aware meta-learned upsampler block used in DCMArb."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SMUB(nn.Module):
    """Scale-Aware Meta-Learned Upsampler Block (SMUB).

    The block retains the coordinate encoding, dual-path meta prediction,
    mixture-of-experts kernel generation, and residual 3D reconstruction
    described in the manuscript. Expert kernels are factorized into spectral
    and spatial components and routed continuously across sub-pixel phases.
    """
    def __init__(
        self,
        channels,
        num_experts=4,
        kernel=16,
        bias=False,
        phase_temperature=0.01,
        spectral_stride=2,
    ):
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be positive, got {channels}.")
        if num_experts < 1:
            raise ValueError(f"num_experts must be positive, got {num_experts}.")
        if kernel < 1:
            raise ValueError(f"kernel must be positive, got {kernel}.")
        if phase_temperature <= 0:
            raise ValueError(
                f"phase_temperature must be positive, got {phase_temperature}."
            )
        if spectral_stride < 1:
            raise ValueError(
                f"spectral_stride must be positive, got {spectral_stride}."
            )

        self.bias = bias
        self.num_experts = num_experts
        self.channels = channels
        self.kernel = min(kernel, max(1, channels - 1))
        self.phase_temperature = phase_temperature
        self.spectral_stride = spectral_stride

        # Each expert represents an effective 5 x 3 x 3 dynamic 3D kernel
        # through a spectral-spatial low-rank factorization.
        self.compress_spectral = nn.Parameter(
            torch.empty(num_experts, self.kernel, 1, 5, 1, 1)
        )
        self.compress_spatial = nn.Parameter(
            torch.empty(num_experts, self.kernel, 1, 1, 3, 3)
        )
        self.expand_spatial = nn.Parameter(
            torch.empty(num_experts, self.kernel, 1, 1, 3, 3)
        )
        self.expand_spectral = nn.Parameter(
            torch.empty(num_experts, self.kernel, 1, 5, 1, 1)
        )

        if bias:
            self.compress_bias = nn.Parameter(torch.zeros(num_experts, self.kernel))
            self.expand_bias = nn.Parameter(torch.zeros(num_experts))
        else:
            self.register_parameter("compress_bias", None)
            self.register_parameter("expand_bias", None)

        self.meta_routing = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(True),
            nn.Linear(64, num_experts),
            nn.Softmax(dim=1),
        )
        self.offset_head = nn.Sequential(
            nn.Conv2d(3, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 2, 1, 1, 0, bias=True),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for weight in (
            self.compress_spectral,
            self.compress_spatial,
            self.expand_spatial,
            self.expand_spectral,
        ):
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        # Start from a stable residual interpolation while retaining gradients
        # through the complete compress-expand path.
        with torch.no_grad():
            self.expand_spectral.mul_(0.1)

        final_offset = self.offset_head[-1]
        nn.init.zeros_(final_offset.weight)
        nn.init.zeros_(final_offset.bias)

    @staticmethod
    def _scale_to_float(scale):
        if isinstance(scale, torch.Tensor):
            if scale.numel() != 1:
                raise ValueError("SMUB expects one shared scale per mini-batch.")
            scale = scale.detach().item()
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"scale must be finite and positive, got {scale}.")
        return scale

    def _coordinate_encoding(self, x, scale):
        _, _, h, w = x.shape
        rh, rw = round(h * scale), round(w * scale)
        if rh < 1 or rw < 1:
            raise ValueError(
                f"scale={scale} produces an invalid output size {(rh, rw)}."
            )

        dtype = self.compress_spectral.dtype
        device = x.device
        hr_h = torch.arange(rh, device=device, dtype=dtype)
        hr_w = torch.arange(rw, device=device, dtype=dtype)

        mapped_h = (hr_h + 0.5) / scale - 0.5
        mapped_w = (hr_w + 0.5) / scale - 0.5
        coor_h = mapped_h - torch.floor(mapped_h + 1e-3) - 0.5
        coor_w = mapped_w - torch.floor(mapped_w + 1e-3) - 0.5

        inv_scale = torch.full(
            (rh, rw), 1.0 / scale, device=device, dtype=dtype
        )
        meta_input = torch.stack(
            (
                inv_scale,
                coor_h[:, None].expand(rh, rw),
                coor_w[None, :].expand(rh, rw),
            ),
            dim=0,
        ).unsqueeze(0)

        sample_count = min(rh, max(1, round(scale)))
        routing_input = torch.stack(
            (
                torch.full(
                    (sample_count,), 1.0 / scale, device=device, dtype=dtype
                ),
                coor_h[:sample_count],
            ),
            dim=1,
        )
        return meta_input, routing_input, coor_h

    def _dense_routing(self, routing_input, coor_h):
        phase_routes = self.meta_routing(routing_input)
        phase_centers = routing_input[:, 1]

        # Circular sub-pixel distance treats -0.5 and 0.5 as adjacent phases.
        distance = (coor_h[:, None] - phase_centers[None, :]).abs()
        distance = torch.minimum(distance, 1.0 - distance)
        phase_basis = torch.softmax(
            -(distance.square()) / self.phase_temperature, dim=1
        )

        dense_routes = phase_basis @ phase_routes
        return dense_routes / dense_routes.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def _compress(self, feature, routes):
        latent = None
        for expert_idx in range(self.num_experts):
            bias = (
                self.compress_bias[expert_idx]
                if self.compress_bias is not None
                else None
            )
            expert_latent = F.conv3d(
                feature,
                self.compress_spectral[expert_idx],
                bias=bias,
                stride=(self.spectral_stride, 1, 1),
                padding=(2, 0, 0),
            )
            expert_latent = F.conv3d(
                expert_latent,
                self.compress_spatial[expert_idx],
                padding=(0, 1, 1),
                groups=self.kernel,
            )
            weighted = expert_latent * routes[:, expert_idx : expert_idx + 1]
            latent = weighted if latent is None else latent + weighted
        return latent

    def _expand(self, latent, routes):
        compressed_bands = latent.shape[2]
        base_bands = (compressed_bands - 1) * self.spectral_stride + 1
        output_padding = self.channels - base_bands
        if not 0 <= output_padding < self.spectral_stride:
            raise RuntimeError(
                "Invalid spectral reconstruction geometry: "
                f"channels={self.channels}, compressed_bands={compressed_bands}, "
                f"stride={self.spectral_stride}."
            )

        correction = None
        for expert_idx in range(self.num_experts):
            expert_latent = F.conv3d(
                latent,
                self.expand_spatial[expert_idx],
                padding=(0, 1, 1),
                groups=self.kernel,
            )
            bias = (
                self.expand_bias[expert_idx : expert_idx + 1]
                if self.expand_bias is not None
                else None
            )
            expert_correction = F.conv_transpose3d(
                expert_latent,
                self.expand_spectral[expert_idx],
                bias=bias,
                stride=(self.spectral_stride, 1, 1),
                padding=(2, 0, 0),
                output_padding=(output_padding, 0, 0),
            )
            weighted = expert_correction * routes[:, expert_idx : expert_idx + 1]
            correction = weighted if correction is None else correction + weighted
        return correction

    def forward(self, x, scale):
        if x.ndim != 4:
            raise ValueError(f"SMUB expects a 4D tensor, got shape {tuple(x.shape)}.")
        if x.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} feature channels, got {x.shape[1]}."
            )

        scale = self._scale_to_float(scale)
        meta_input, routing_input, coor_h = self._coordinate_encoding(x, scale)
        offset = self.offset_head(meta_input)

        row_routes = self._dense_routing(routing_input, coor_h)
        routes = row_routes.transpose(0, 1)[None, :, :, None]

        initial = grid_sample(x, offset, scale)
        latent = self._compress(initial.unsqueeze(1), routes)
        correction = self._expand(latent, routes).squeeze(1)
        return initial + correction

    @property
    def routing(self):
        return self.meta_routing

    @property
    def offset(self):
        return self.offset_head


Upsampler = SMUB


def grid_sample(x, offset, scale):
    """Resample ``x`` on the HR grid corrected by learned LR-pixel offsets."""

    scale = SMUB._scale_to_float(scale)
    b, _, h, w = x.shape
    rh, rw = round(scale * h), round(scale * w)
    if offset.shape != (1, 2, rh, rw) and offset.shape != (b, 2, rh, rw):
        raise ValueError(
            "offset must have shape (1 or batch, 2, round(H*s), round(W*s)); "
            f"got {tuple(offset.shape)}."
        )

    dtype = x.dtype
    device = x.device
    hr_h = torch.arange(rh, device=device, dtype=dtype)
    hr_w = torch.arange(rw, device=device, dtype=dtype)
    mapped_h = (hr_h + 0.5) / scale - 0.5
    mapped_w = (hr_w + 0.5) / scale - 0.5
    grid_y, grid_x = torch.meshgrid(mapped_h, mapped_w, indexing="ij")

    if w > 1:
        grid_x = 2.0 * grid_x / (w - 1) - 1.0
        offset_x = 2.0 * offset[:, 0].to(dtype=dtype) / (w - 1)
    else:
        grid_x = torch.zeros_like(grid_x)
        offset_x = torch.zeros_like(offset[:, 0], dtype=dtype)

    if h > 1:
        grid_y = 2.0 * grid_y / (h - 1) - 1.0
        offset_y = 2.0 * offset[:, 1].to(dtype=dtype) / (h - 1)
    else:
        grid_y = torch.zeros_like(grid_y)
        offset_y = torch.zeros_like(offset[:, 1], dtype=dtype)

    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
    grid = grid.expand(b, -1, -1, -1).clone()
    grid[..., 0] += offset_x.expand(b, -1, -1)
    grid[..., 1] += offset_y.expand(b, -1, -1)
    return F.grid_sample(
        x,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
