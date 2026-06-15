"""Dataset loader for arbitrary-scale HSI super-resolution training/testing."""

import os
import random

import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as data
from skimage.transform import resize

import utils


def is_mat_file(filename):
    return filename.lower().endswith(".mat")


class HSArbitraryData(data.Dataset):
    """Load .mat files containing a `gt` array in H x W x C format."""

    def __init__(self, image_dir, augment=False, use_3D=False, lr_size=(64, 64), scale_range=(1.0, 4.0), round_scale=False):
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Data directory not found: {image_dir}")

        self.image_files = sorted(os.path.join(image_dir, f) for f in os.listdir(image_dir) if is_mat_file(f))
        if len(self.image_files) == 0:
            raise RuntimeError(f"No .mat files found in: {image_dir}")

        self.augment = bool(augment)
        self.use_3Dconv = use_3D
        self.lr_size = tuple(lr_size)
        self.scale_range = tuple(scale_range)
        self.round_scale = round_scale
        self.current_scale_factor = None

    def __len__(self):
        return len(self.image_files) * (8 if self.augment else 1)

    @staticmethod
    def spectral_aware_degrade(hr_img, scale_factor):
        lr_shape = (
            max(1, int(hr_img.shape[0] / scale_factor)),
            max(1, int(hr_img.shape[1] / scale_factor)),
            hr_img.shape[2],
        )
        return resize(hr_img, lr_shape[:2], order=3, anti_aliasing=True, preserve_range=True).astype(np.float32)

    def _sample_scale(self):
        scale = random.uniform(*self.scale_range)
        if self.round_scale:
            return float(round(scale))
        return float(round(scale, 1))

    def __getitem__(self, index):
        if self.current_scale_factor is None:
            self.current_scale_factor = self._sample_scale()

        file_index = index % len(self.image_files)
        load_dir = self.image_files[file_index]
        mat_data = sio.loadmat(load_dir)
        if "gt" not in mat_data:
            raise KeyError(f"Missing key 'gt' in {load_dir}. Expected a H x W x C hyperspectral image.")

        hr = np.array(mat_data["gt"], dtype=np.float32)
        if hr.ndim != 3:
            raise ValueError(f"Expected `gt` to be a 3D array [H, W, C], got shape {hr.shape} in {load_dir}.")

        scale_factor = self.current_scale_factor
        hr_height = int(round(self.lr_size[0] * scale_factor))
        hr_width = int(round(self.lr_size[1] * scale_factor))

        if hr.shape[0] < hr_height or hr.shape[1] < hr_width:
            raise ValueError(
                f"HR image size ({hr.shape[0]}, {hr.shape[1]}) is smaller than required crop "
                f"({hr_height}, {hr_width}) for scale={scale_factor}. File: {load_dir}"
            )

        h_start = random.randint(0, hr.shape[0] - hr_height)
        w_start = random.randint(0, hr.shape[1] - hr_width)
        hr_patch = hr[h_start:h_start + hr_height, w_start:w_start + hr_width, :]
        lr_patch = self.spectral_aware_degrade(hr_patch, scale_factor)

        if lr_patch.shape[0] != self.lr_size[0] or lr_patch.shape[1] != self.lr_size[1]:
            lr_patch = resize(lr_patch, self.lr_size, order=3, anti_aliasing=True, preserve_range=True).astype(np.float32)

        if self.augment:
            aug_num = random.randint(0, 7)
            hr_patch = utils.data_augmentation(hr_patch, mode=aug_num)
            lr_patch = utils.data_augmentation(lr_patch, mode=aug_num)

        hr_tensor = torch.from_numpy(hr_patch.copy())
        lr_tensor = torch.from_numpy(lr_patch.copy())

        if self.use_3Dconv:
            hr_tensor = hr_tensor.permute(2, 0, 1).unsqueeze(0)
            lr_tensor = lr_tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            hr_tensor = hr_tensor.permute(2, 0, 1)
            lr_tensor = lr_tensor.permute(2, 0, 1)

        return {"lr": lr_tensor, "hr": hr_tensor, "scale": scale_factor}
