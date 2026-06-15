"""Evaluation metrics for hyperspectral image super-resolution.

All functions expect images in H x W x C format with float values.
"""

import numpy as np
from scipy.signal import convolve2d
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


EPS = 1e-8


def _as_float32(x):
    return np.asarray(x, dtype=np.float32)


def img_2d_mat(x_true, x_pred):
    """Convert H x W x C images to C x (H*W) matrices."""
    x_true = _as_float32(x_true)
    x_pred = _as_float32(x_pred)
    if x_true.shape != x_pred.shape:
        raise ValueError(f"Shape mismatch: x_true {x_true.shape}, x_pred {x_pred.shape}")
    if x_true.ndim != 3:
        raise ValueError(f"Expected H x W x C arrays, got shape {x_true.shape}")
    return x_true.transpose(2, 0, 1).reshape(x_true.shape[2], -1), x_pred.transpose(2, 0, 1).reshape(x_pred.shape[2], -1)


def compare_ergas(x_true, x_pred, ratio):
    """Calculate ERGAS. Lower is better; the ideal value is 0."""
    x_true, x_pred = img_2d_mat(x_true, x_pred)
    rmse2 = np.mean((x_true - x_pred) ** 2, axis=1)
    mean2 = np.mean(x_true, axis=1) ** 2
    return float((100.0 / ratio) * np.sqrt(np.mean(rmse2 / (mean2 + EPS))))


def compare_sam(x_true, x_pred):
    """Calculate spectral angle mapper (SAM) in degrees."""
    x_true = _as_float32(x_true)
    x_pred = _as_float32(x_pred)
    if x_true.shape != x_pred.shape:
        raise ValueError(f"Shape mismatch: x_true {x_true.shape}, x_pred {x_pred.shape}")

    true_vec = x_true.reshape(-1, x_true.shape[-1])
    pred_vec = x_pred.reshape(-1, x_pred.shape[-1])
    numerator = np.sum(true_vec * pred_vec, axis=1)
    denominator = np.linalg.norm(true_vec, axis=1) * np.linalg.norm(pred_vec, axis=1)
    valid = denominator > EPS
    if not np.any(valid):
        return 0.0
    cos = np.clip(numerator[valid] / (denominator[valid] + EPS), -1.0, 1.0)
    return float(np.mean(np.arccos(cos)) * 180.0 / np.pi)


def compare_corr(x_true, x_pred):
    """Calculate mean per-band cross correlation."""
    x_true, x_pred = img_2d_mat(x_true, x_pred)
    x_true = x_true - np.mean(x_true, axis=1, keepdims=True)
    x_pred = x_pred - np.mean(x_pred, axis=1, keepdims=True)
    numerator = np.sum(x_true * x_pred, axis=1)
    denominator = np.sqrt(np.sum(x_true * x_true, axis=1) * np.sum(x_pred * x_pred, axis=1))
    return float(np.mean(numerator / (denominator + EPS)))


def compare_rmse(x_true, x_pred):
    """Calculate root mean squared error."""
    x_true = _as_float32(x_true)
    x_pred = _as_float32(x_pred)
    return float(np.sqrt(np.mean((x_true - x_pred) ** 2)))


def compare_mpsnr(x_true, x_pred, data_range):
    """Calculate mean PSNR over all spectral bands."""
    x_true = _as_float32(x_true)
    x_pred = _as_float32(x_pred)
    return float(np.mean([compare_psnr(x_true[:, :, k], x_pred[:, :, k], data_range=data_range) for k in range(x_true.shape[2])]))


def compare_mssim(x_true, x_pred, data_range, multidimension=False):
    """Calculate mean SSIM over all spectral bands.

    The `multidimension` argument is kept for backward compatibility and is not
    used because SSIM is computed band by band.
    """
    x_true = _as_float32(x_true)
    x_pred = _as_float32(x_pred)
    return float(np.mean([compare_ssim(x_true[:, :, k], x_pred[:, :, k], data_range=data_range) for k in range(x_true.shape[2])]))


def compare_sid(x_true, x_pred):
    """Calculate spectral information divergence (SID)."""
    x_true = np.maximum(_as_float32(x_true), EPS)
    x_pred = np.maximum(_as_float32(x_pred), EPS)
    sid = x_pred * np.log10(x_pred / x_true) + x_true * np.log10(x_true / x_pred)
    return float(np.mean(np.sum(sid, axis=(0, 1)) / (x_true.shape[0] * x_true.shape[1])))


def compare_appsa(x_true, x_pred):
    """Calculate average pixel-wise spectral angle in radians."""
    x_true = _as_float32(x_true)
    x_pred = _as_float32(x_pred)
    numerator = np.sum(x_true * x_pred, axis=2)
    denominator = np.linalg.norm(x_true, axis=2) * np.linalg.norm(x_pred, axis=2)
    cos = np.clip(numerator / (denominator + EPS), -1.0, 1.0)
    return float(np.mean(np.arccos(cos)))


def compare_mare(x_true, x_pred):
    """Calculate mean absolute relative error."""
    x_true = _as_float32(x_true)
    x_pred = _as_float32(x_pred)
    return float(np.mean(np.abs(x_true - x_pred) / (np.abs(x_true) + 1.0)))


def img_qi(img1, img2, block_size=8):
    """Universal image quality index for one band."""
    n = block_size ** 2
    sum_filter = np.ones((block_size, block_size), dtype=np.float32)
    img1_sq = img1 * img1
    img2_sq = img2 * img2
    img12 = img1 * img2

    img1_sum = convolve2d(img1, np.rot90(sum_filter), mode="valid")
    img2_sum = convolve2d(img2, np.rot90(sum_filter), mode="valid")
    img1_sq_sum = convolve2d(img1_sq, np.rot90(sum_filter), mode="valid")
    img2_sq_sum = convolve2d(img2_sq, np.rot90(sum_filter), mode="valid")
    img12_sum = convolve2d(img12, np.rot90(sum_filter), mode="valid")

    img12_sum_mul = img1_sum * img2_sum
    img12_sq_sum_mul = img1_sum * img1_sum + img2_sum * img2_sum
    numerator = 4 * (n * img12_sum - img12_sum_mul) * img12_sum_mul
    denominator1 = n * (img1_sq_sum + img2_sq_sum) - img12_sq_sum_mul
    denominator = denominator1 * img12_sq_sum_mul

    quality_map = np.ones(denominator.shape, dtype=np.float32)
    idx = (denominator1 == 0) & (img12_sq_sum_mul != 0)
    quality_map[idx] = 2 * img12_sum_mul[idx] / (img12_sq_sum_mul[idx] + EPS)
    idx = denominator != 0
    quality_map[idx] = numerator[idx] / (denominator[idx] + EPS)
    return float(quality_map.mean())


def compare_qave(x_true, x_pred, block_size=8):
    """Calculate average universal image quality index over bands."""
    x_true = _as_float32(x_true)
    x_pred = _as_float32(x_pred)
    return float(np.mean([img_qi(x_true[:, :, k], x_pred[:, :, k], block_size) for k in range(x_true.shape[2])]))
