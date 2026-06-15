import sys
import os
# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from common import *
from einops import rearrange
from SSRB import SSRB
from timm.layers import DropPath
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from einops import rearrange
from SMUB import SMUB

save_dir = r" "

class DCMArb(nn.Module):
    """Decoupled-Collaborative Mamba for arbitrary-scale HSI SR."""
    def __init__(self,
                 inp_channels=31,
                 dim=90,
                 input_resolution=[32, 32],
                 depths=[1, 1, 1],
                 expand_ratio=2,
                 bias=False,
                 drop_path_rate=0.1
                 ):
        super(DCMArb, self).__init__()
        self.dim = dim
        self.conv_first = nn.Conv2d(inp_channels, dim, 3, 1, 1)  # shallow featrure extraction
        self.num_layers = depths
        self.hfis_stages = nn.ModuleList()
        print(f"dim: {self.dim}, depths: {self.num_layers}")

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for i_layer in range(len(self.num_layers)):
            layer = HFIS(dim=dim,
                         depth=depths[i_layer],
                         input_resolution=input_resolution,
                         expand_ratio=expand_ratio,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                         bias=bias,
                         stage_idx=i_layer)
            self.hfis_stages.append(layer)

        self.skip_conv = default_conv(inp_channels, dim, 3)
        self.smub = SMUB(dim)

        self.conv = default_conv(dim, dim, 3)
        self.tail = default_conv(dim, inp_channels, 3)

    def forward(self, inp_img, scale):
        # if not self.training and inp_img.shape[0] == 1:
        #     os.makedirs(save_dir, exist_ok=True)
        #     torch.save(inp_img.detach().cpu(), f"{save_dir}/input_lr.pt")
        lms = round_scale_interpolate(inp_img, scale)
        f1 = self.conv_first(inp_img)


        # if not self.training and inp_img.shape[0] == 1:
        #     os.makedirs(save_dir, exist_ok=True)
        #     torch.save(inp_img.detach().cpu(), f"{save_dir}/global_input_f1.pt")

        x = f1
        for i in range(len(self.num_layers)):
            x = self.hfis_stages[i](x)

        # if not self.training and x.shape[0] == 1:
        #     torch.save(x.detach().cpu(), f"{save_dir}/global_after_3_stages.pt")


        x = self.conv(x + f1)
        # if not self.training and x.shape[0] == 1:
        #     torch.save(x.detach().cpu(), f"{save_dir}/global_final_hfis.pt")
        x = self.smub(x, scale)
        x = x + self.skip_conv(lms)
        x = self.tail(x)
        # if not self.training and x.shape[0] == 1:
        #     torch.save(x.detach().cpu(), f"{save_dir}/output_sr.pt")
        return x


class HFIS(nn.Module):
    """Heterogeneous Feature Integration Stage (HFIS)."""
    def __init__(self,
                 dim=90,
                 input_resolution=[32, 32],
                 split_size=(2, 16),
                 depth=1,
                 expand_ratio=2,
                 drop_path=0.1,
                 bias=False,
                 use_dynamic_shift=True,
                 use_recursive=True,
                 stage_idx=0):
        super(HFIS, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.split_size = split_size
        self.depth = depth
        self.expand_ratio = expand_ratio
        self.use_dynamic_shift = use_dynamic_shift
        self.use_recursive = use_recursive
        self.stage_idx = stage_idx

        if isinstance(drop_path, float):
            self.drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]
        else:
            self.drop_path = drop_path

        self.mdes_branch = nn.ModuleList()
        self.selection_branch = SelectionBranch(default_conv, dim, 1, res_scale=0.1)

        for i_layer in range(depth):
            if use_dynamic_shift:
                split_h = max(1, split_size[0] // (2 ** (i_layer // 2)))
                split_w = max(1, split_size[1] // (2 ** (i_layer // 2)))
                current_split = (split_h, split_w)
                shift_h = current_split[0] // 2 if i_layer % 2 == 1 else 0
                shift_w = current_split[1] // 2 if i_layer % 2 == 1 else 0
                current_shift = [shift_h, shift_w]
            else:
                current_split = split_size
                current_shift = [0, 0] if (i_layer % 2 == 0) else [split_size[0] // 2, split_size[1] // 2]

            self.mdes_branch.append(MDES(
                dim=dim,
                input_resolution=input_resolution,
                drop_path=self.drop_path[i_layer],
                split_size=current_split,
                shift_size=current_shift,
                expand_ratio=expand_ratio,
                bias=bias
            ))

        self.conv = nn.Conv2d(dim, dim, 1)

    def recursive_forward(self, x, layer_idx=0):
        if layer_idx >= self.depth:
            return x
        else:
            return self.recursive_forward(self.mdes_branch[layer_idx](x), layer_idx + 1)

    def forward(self, x):
        x1 = x
        x2 = self.selection_branch(x)

        if self.use_recursive:
            x1 = self.recursive_forward(x1)
        else:
            for i in range(self.depth):
                x1 = self.mdes_branch[i](x1)

        out = self.conv(x1) + x2
        out = x + out

        # if not self.training and x.shape[0] == 1:
        #     import os
        #     os.makedirs(save_dir, exist_ok=True)
        #     torch.save(x.detach().cpu(), f"{save_dir}/input_f0_stage{self.stage_idx}.pt")
        #     torch.save(x1.detach().cpu(), f"{save_dir}/mdes_x1_stage{self.stage_idx}.pt")
        #     torch.save(x2.detach().cpu(), f"{save_dir}/attn_x2_stage{self.stage_idx}.pt")
        #     torch.save(out.detach().cpu(), f"{save_dir}/fused_out_stage{self.stage_idx}.pt")
        # ==========================================================

        return out

class ICB(nn.Module):
    """
    Inter-Band Contextualization Block (ICB).
    - ASC: Adaptive Spectral Companding
    - S-Mamba: spectral selective Mamba
    - SDD: local-global spectral dependency distillation
    - Optional: lightweight high-frequency stabilization (Laplacian)
    """
    def __init__(
        self,
        dim,                         # 输入的光谱维度
        bias=True,
        k_ratio=0.5,                 # Mamba特征比例
        expand_ratio=2.0,
        k_min=8,                     # 最少保留通道
        sd_kernel=3,                 # SDD中的局部卷积
        beta=10.0, tau=0.05          # companding形状控制
    ):
        super().__init__()

        self.dim = dim
        self.k = max(int(k_ratio * dim), k_min)
        self.k_min = k_min
        self.beta = beta
        self.tau = tau

        # ----------------------------
        # Mamba backbone
        # ----------------------------
        self.mamba = Mamba(
            d_model=self.k,
            d_state=16,
            d_conv=4,
            expand=expand_ratio
        )

        # ----------------------------
        # ASC projection
        # ----------------------------
        self.feature_proj = nn.Conv2d(dim, self.k, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(self.k, dim, kernel_size=1, bias=bias)

        # ----------------------------
        # SDD: Local conv across spectral dimension
        # ----------------------------
        padding = (sd_kernel - 1) // 2
        self.spectral_local_conv = nn.Conv1d(
            self.k, self.k,
            kernel_size=sd_kernel,
            padding=padding,
            bias=True
        )

        # SDD weight MLP (maps cosine similarity → fusion weight)
        self.sdd_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # optional HF stabilization
        self.hf_gate = nn.Parameter(torch.tensor(0.1), requires_grad=True)

        self._init_weights()

    def _init_weights(self):
        # weight init
        nn.init.kaiming_normal_(self.feature_proj.weight, a=0.2)
        if getattr(self.feature_proj, "bias", None) is not None:
            nn.init.zeros_(self.feature_proj.bias)

        nn.init.kaiming_normal_(self.project_out.weight, a=0.2)
        if getattr(self.project_out, "bias", None) is not None:
            nn.init.zeros_(self.project_out.bias)

        nn.init.kaiming_normal_(self.spectral_local_conv.weight, a=0.2)
        if getattr(self.spectral_local_conv, "bias", None) is not None:
            nn.init.zeros_(self.spectral_local_conv.bias)


    # ==============================================================
    # 1. ASC
    # ==============================================================
    def spectral_density_estimate(self, x):
        B, C, H, W = x.shape
        if C <= 1:
            return x.new_zeros(B)

        diff = torch.abs(x[:, 1:, :, :] - x[:, :-1, :, :])  # [B, C-1, H, W]
        d = diff.mean(dim=[1,2,3])
        d = F.softplus(d)

        mean_d = d.detach().mean().clamp(min=1e-6)
        d = d / (mean_d + 1e-6)

        return d

    def companding_alpha(self, d):
        a = torch.sigmoid(self.beta * (d - self.tau))
        alpha = 0.5 + 0.5 * a
        return alpha


    def forward(self, x):
        B, C, H, W = x.shape
        d = self.spectral_density_estimate(x)
        alpha = self.companding_alpha(d)
        k_eff = (alpha * self.k).long().clamp(min=self.k_min)

        feat = self.feature_proj(x)  # [B, k, H, W]

        channel_idx = torch.arange(self.k, device=x.device).unsqueeze(0).repeat(B, 1)
        k_eff_exp = k_eff.unsqueeze(1)

        small_factor = 0.05
        mask = (channel_idx < k_eff_exp).float().unsqueeze(-1).unsqueeze(-1)
        mask = mask + (~(channel_idx < k_eff_exp)).float().unsqueeze(-1).unsqueeze(-1) * small_factor
        feat = feat * mask

        seq = rearrange(feat, 'b c h w -> b (h w) c')  # [B, L, k]
        y_global = self.mamba(seq)                     # [B, L, k]

        y_global_t = rearrange(y_global, 'b l c -> b c l')
        y_local_t = self.spectral_local_conv(y_global_t)
        y_local = rearrange(y_local_t, 'b c l -> b l c')

        eps = 1e-6
        dot = (y_local * y_global).sum(-1, keepdim=True)
        n1 = y_local.norm(dim=-1, keepdim=True).clamp(min=eps)
        n2 = y_global.norm(dim=-1, keepdim=True).clamp(min=eps)
        cos_sim = dot / (n1 * n2)

        w = self.sdd_mlp(cos_sim)
        y_fused = w * y_global + (1 - w) * y_local

        out_feat = rearrange(y_fused, 'b (h w) c -> b c h w', h=H, w=W)
        out = self.project_out(out_feat)

        lap = torch.tensor(
            [[[[0, 1, 0],
               [1,-4, 1],
               [0, 1, 0]]]],
            device=x.device,
            dtype=x.dtype
        )

        lap = lap.repeat(self.dim, 1, 1, 1)  # groups convolution
        hf = F.conv2d(out, lap, stride=1, padding=1, groups=self.dim)
        out = out - self.hf_gate * hf

        return out

class GFB(nn.Module):
    """Gated Fusion Block (GFB)."""
    def __init__(self, dim, gfb_expansion_factor=2.66, bias=False, ffn_expansion_factor=None):
        super(GFB, self).__init__()
        if ffn_expansion_factor is not None:
            gfb_expansion_factor = ffn_expansion_factor
        hidden_features = int(dim * gfb_expansion_factor)
        self.bsconv = BSConvU(dim, hidden_features * 2, kernel_size=3, stride=1, padding=1, bias=bias)

        self.adaptive_fusion = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=1, bias=bias),
            nn.Sigmoid()
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x1, x2 = self.bsconv(x).chunk(2, dim=1)
        x1 = F.mish(x1)
        fusion_weights = self.adaptive_fusion(x1)
        x = fusion_weights * x2
        x = self.project_out(x)
        return x


#(C) Multi-Dimensional Dependency Extraction Stage (MDES)
class MDES(nn.Module):
    """Multi-Dimensional Dependency Extraction Stage (MDES)."""
    def __init__(self, dim, input_resolution=[32,32], drop_path=0.0, split_size=[7, 7], shift_size=[0,0],
                 expand_ratio=4., bias=False, use_ssrb=True, use_icb=True, use_gfb=True):
        super(MDES, self).__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gfb = GFB(dim)

        self.ssrb = SSRB(
            dim,
            input_resolution=input_resolution,
            split_size=split_size,
            shift_size=shift_size,
            expand_ratio=expand_ratio)
        self.icb = ICB(dim, bias, expand_ratio=expand_ratio)

        self.use_ssrb = use_ssrb
        self.use_icb = use_icb
        self.use_gfb = use_gfb

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2)  # b,c,h*w
        x = x.transpose(1, 2)  # b,h*w,c

        shortcut = x  # b,h*w,c
        x = self.norm1(x)  # b,h*w,c

        if self.use_ssrb:
            x = self.ssrb(x, (H, W))  # b,h*w,c

        x = x.view(B, H * W, C)  # b,h*w,c
        x = x.transpose(1, 2).view(B, C, H, W)  # b,c,h,w

        if self.use_icb:
            x = self.icb(x)  # b,c,h,w

        x = x.flatten(2).transpose(1, 2)  # b,h*w,c

        # GFB
        x = self.drop_path(x)  # b,h*w,c
        x = shortcut + x  # b,h*w,c

        if self.use_gfb:
            x = x + self.drop_path(self.gfb(self.norm2(x).transpose(1, 2).view(B, C, H, W)).flatten(2).transpose(1, 2))

        x = x.transpose(1, 2).view(B, C, H, W)  # b,c,h,w
        return x





def dcmarb(dataset):
    model = None
    if dataset == 'chikusei':
        model = DCMArb(inp_channels=128, dim=120, input_resolution=[32,32], depths=[1,1,1],expand_ratio=2.0).cuda()
    elif dataset == 'gf5b':
        model = DCMArb(inp_channels=150, dim=120, input_resolution=[32,32], depths=[1,1,1],expand_ratio=2.0).cuda()
    elif dataset == 'zy1f':
        model = DCMArb(inp_channels=76, dim=120, input_resolution=[32,32], depths=[1,1,1],expand_ratio=2.0).cuda()
    return model


def as2mamba(dataset):
    return dcmarb(dataset)


if __name__ == "__main__":
    model = dcmarb(dataset='chikusei')
    # print(model)
    x = torch.randn(1, 128,32,32).cuda()
    lms = torch.randn(1, 128,128,128).cuda()
    SR = model(x,scale=4).cuda()
    print("Output shape:", SR.shape)
    print('# parameters(M): {:.3f}'.format(sum(param.numel() for param in model.parameters()) / 1e6))

    # from thop import profile
    # Par_FLOP_input_ms = torch.randn(1, 150, 32, 32).cuda()
    # Par_FLOP_input_lms = torch.randn(1, 150, 128, 128).cuda()
    # macs, params_count = profile(model, inputs=(Par_FLOP_input_ms, Par_FLOP_input_lms, 4))
    # print(f"Params(M): {params_count / (1000 ** 2):.3f} | FLOPs(G): {macs / (1000 ** 3):.3f}")
