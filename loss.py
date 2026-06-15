import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HSRLoss(nn.Module):
    def __init__(self, la1=0.3, la2=0.1, la3=0.1, use_l1=True):
        """
        初始化参数
        :param la1: 光谱保真损失权重
        :param la2: 边缘保留损失权重
        :param la3: 梯度一致性损失权重
        :param use_l1: 是否使用L1损失作为空间重建损失，否则使用MSE
        """
        super(HSRLoss, self).__init__()
        self.la1 = la1  # 光谱保真损失权重
        self.la2 = la2  # 边缘保留损失权重
        self.la3 = la3  # 梯度一致性损失权重

        # 空间重建损失
        self.spatial_loss = nn.L1Loss() if use_l1 else nn.MSELoss()

    @staticmethod
    def compute_spectral_fidelity(pred, target, eps=1e-6):
        """
        光谱保真损失 - 使用光谱角映射(SAM)计算光谱相似度
        实现思路与cal_sam函数保持一致
        :param pred: 预测图像 [B, C, H, W]
        :param target: 真实图像 [B, C, H, W]
        :param eps: 数值稳定性小常数
        :return: 光谱保真损失
        """
        # 计算内积
        inner_product = torch.sum(pred * target, dim=1, keepdim=True)

        # 计算向量长度
        true_norm = torch.norm(target, p=2, dim=1, keepdim=True)
        pred_norm = torch.norm(pred, p=2, dim=1, keepdim=True)

        # 计算分母
        divisor = true_norm * pred_norm

        # 处理分母为零的情况
        mask = torch.eq(divisor, 0)
        divisor = divisor + mask.float() * eps

        # 计算余弦值并限制在有效范围内
        cosine = (inner_product / divisor).squeeze(1).clamp(-1 + eps, 1 - eps)

        # 计算光谱角
        sam = torch.acos(cosine)

        # 计算平均SAM并归一化（除以pi）
        return torch.mean(sam) / np.pi

    def compute_edge_preservation(self, pred, target):
        """
        计算边缘保留损失
        使用有限差分近似计算梯度
        """
        # 计算预测图像的梯度
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]  # x方向差分
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]  # y方向差分

        # 为保持维度一致，在差分后添加零填充
        pred_dx = F.pad(pred_dx, (0, 1, 0, 0))
        pred_dy = F.pad(pred_dy, (0, 0, 0, 1))

        # 计算梯度幅值作为边缘强度
        pred_edges = torch.sqrt(pred_dx.pow(2) + pred_dy.pow(2) + 1e-8)

        # 计算目标图像的梯度
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]  # x方向差分
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]  # y方向差分

        # 为保持维度一致，在差分后添加零填充
        target_dx = F.pad(target_dx, (0, 1, 0, 0))
        target_dy = F.pad(target_dy, (0, 0, 0, 1))

        # 计算梯度幅值作为边缘强度
        target_edges = torch.sqrt(target_dx.pow(2) + target_dy.pow(2) + 1e-8)

        # 计算边缘损失
        return F.l1_loss(pred_edges, target_edges)

    def compute_gradient_consistency(self, pred, target):
        """
        计算梯度一致性损失（按照HLoss的梯度计算思路）
        使用有限差分近似计算梯度
        """
        # 计算预测图像的梯度（三个方向）
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]  # x方向差分
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]  # y方向差分
        pred_dc = pred[:, 1:, :, :] - pred[:, :-1, :, :]  # 光谱方向差分

        # 为保持维度一致，在差分后添加零填充
        pred_dx = F.pad(pred_dx, (0, 1, 0, 0))
        pred_dy = F.pad(pred_dy, (0, 0, 0, 1))
        pred_dc = F.pad(pred_dc, (0, 0, 0, 0, 0, 1))

        # 计算目标图像的梯度（三个方向）
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]  # x方向差分
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]  # y方向差分
        target_dc = target[:, 1:, :, :] - target[:, :-1, :, :]  # 光谱方向差分

        # 为保持维度一致，在差分后添加零填充
        target_dx = F.pad(target_dx, (0, 1, 0, 0))
        target_dy = F.pad(target_dy, (0, 0, 0, 1))
        target_dc = F.pad(target_dc, (0, 0, 0, 0, 0, 1))

        # 计算梯度一致性损失（三个方向的L1损失之和）
        grad_loss_x = F.l1_loss(pred_dx, target_dx)
        grad_loss_y = F.l1_loss(pred_dy, target_dy)
        grad_loss_c = F.l1_loss(pred_dc, target_dc)

        return grad_loss_x + grad_loss_y + grad_loss_c

    def forward(self, pred, target):
        """
        前向传播计算总损失
        :param pred: 预测的高光谱图像 [B, C, H, W]
        :param target: 真实的高光谱图像 [B, C, H, W]
        :return: 总损失
        """
        # 1. 空间重建损失
        spatial_loss = self.spatial_loss(pred, target)

        # 2. 光谱保真损失
        spectral_loss = self.compute_spectral_fidelity(pred, target)

        # 3. 边缘保留损失
        edge_loss = self.compute_edge_preservation(pred, target)

        # 4. 梯度一致性损失
        grad_loss = self.compute_gradient_consistency(pred, target)

        # 计算总损失
        total_loss = spatial_loss + \
                     self.la1 * spectral_loss + \
                     self.la2 * edge_loss + \
                     self.la3 * grad_loss

        return total_loss