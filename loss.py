import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HSRLoss(nn.Module):
    def __init__(self, la1=0.3, la2=0.1, la3=0.1, use_l1=True):
        """
        Initialize the hybrid HSI super-resolution loss.

        Args:
            la1: Weight of the SAM spectral fidelity loss.
            la2: Weight of the gradient consistency loss.
            la3: Weight of the edge preservation loss.
            use_l1: If True, use L1 loss as the spatial reconstruction loss;
                    otherwise, use MSE loss.
        """
        super(HSRLoss, self).__init__()
        self.la1 = la1
        self.la2 = la2
        self.la3 = la3

        # Spatial reconstruction loss
        self.spatial_loss = nn.L1Loss() if use_l1 else nn.MSELoss()

    @staticmethod
    def compute_spectral_fidelity(pred, target, eps=1e-6):
        """
        Compute the spectral fidelity loss using Spectral Angle Mapper (SAM).

        This implementation follows the calculation logic of the common SAM metric.

        Args:
            pred: Predicted HSI tensor with shape [B, C, H, W].
            target: Ground-truth HSI tensor with shape [B, C, H, W].
            eps: A small constant for numerical stability.

        Returns:
            Normalized mean SAM loss.
        """
        # Compute the inner product
        inner_product = torch.sum(pred * target, dim=1, keepdim=True)

        # Compute vector norms
        true_norm = torch.norm(target, p=2, dim=1, keepdim=True)
        pred_norm = torch.norm(pred, p=2, dim=1, keepdim=True)

        # Compute the denominator
        divisor = true_norm * pred_norm

        # Avoid division by zero
        mask = torch.eq(divisor, 0)
        divisor = divisor + mask.float() * eps

        # Compute cosine similarity and clamp it to a valid range
        cosine = (inner_product / divisor).squeeze(1).clamp(-1 + eps, 1 - eps)

        # Compute the spectral angle
        sam = torch.acos(cosine)

        # Compute the mean SAM and normalize it by pi
        return torch.mean(sam) / np.pi

    def compute_edge_preservation(self, pred, target):
        """
        Compute the edge preservation loss.

        The loss is calculated by approximating spatial gradients with finite differences.
        """
        # Compute gradients of the predicted image
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]  # Difference along the x direction
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]  # Difference along the y direction

        # Pad the gradient maps to maintain the original tensor shape
        pred_dx = F.pad(pred_dx, (0, 1, 0, 0))
        pred_dy = F.pad(pred_dy, (0, 0, 0, 1))

        # Compute gradient magnitude as edge strength
        pred_edges = torch.sqrt(pred_dx.pow(2) + pred_dy.pow(2) + 1e-8)

        # Compute gradients of the target image
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]  # Difference along the x direction
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]  # Difference along the y direction

        # Pad the gradient maps to maintain the original tensor shape
        target_dx = F.pad(target_dx, (0, 1, 0, 0))
        target_dy = F.pad(target_dy, (0, 0, 0, 1))

        # Compute gradient magnitude as edge strength
        target_edges = torch.sqrt(target_dx.pow(2) + target_dy.pow(2) + 1e-8)

        # Compute the edge preservation loss
        return F.l1_loss(pred_edges, target_edges)

    def compute_gradient_consistency(self, pred, target):
        """
        Compute the gradient consistency loss.

        Finite differences are used to approximate gradients along the spatial and spectral dimensions.
        """
        # Compute gradients of the predicted image along three dimensions
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]  # Difference along the x direction
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]  # Difference along the y direction
        pred_dc = pred[:, 1:, :, :] - pred[:, :-1, :, :]  # Difference along the spectral dimension

        # Pad the gradient maps to maintain the original tensor shape
        pred_dx = F.pad(pred_dx, (0, 1, 0, 0))
        pred_dy = F.pad(pred_dy, (0, 0, 0, 1))
        pred_dc = F.pad(pred_dc, (0, 0, 0, 0, 0, 1))

        # Compute gradients of the target image along three dimensions
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]  # Difference along the x direction
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]  # Difference along the y direction
        target_dc = target[:, 1:, :, :] - target[:, :-1, :, :]  # Difference along the spectral dimension

        # Pad the gradient maps to maintain the original tensor shape
        target_dx = F.pad(target_dx, (0, 1, 0, 0))
        target_dy = F.pad(target_dy, (0, 0, 0, 1))
        target_dc = F.pad(target_dc, (0, 0, 0, 0, 0, 1))

        # Compute the L1 loss for gradients along each dimension
        grad_loss_x = F.l1_loss(pred_dx, target_dx)
        grad_loss_y = F.l1_loss(pred_dy, target_dy)
        grad_loss_c = F.l1_loss(pred_dc, target_dc)

        return grad_loss_x + grad_loss_y + grad_loss_c

    def forward(self, pred, target):
        """
        Compute the total hybrid loss.

        Args:
            pred: Predicted HSI tensor with shape [B, C, H, W].
            target: Ground-truth HSI tensor with shape [B, C, H, W].

        Returns:
            Total loss value.
        """
        # 1. Spatial reconstruction loss
        spatial_loss = self.spatial_loss(pred, target)

        # 2. Spectral fidelity loss
        spectral_loss = self.compute_spectral_fidelity(pred, target)

        # 3. Edge preservation loss
        edge_loss = self.compute_edge_preservation(pred, target)

        # 4. Gradient consistency loss
        grad_loss = self.compute_gradient_consistency(pred, target)

        # Compute the total loss
        total_loss = spatial_loss + \
                     self.la1 * spectral_loss + \
                     self.la2 * grad_loss + \
                     self.la3 * edge_loss

        return total_loss