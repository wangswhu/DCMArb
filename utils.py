"""Utility functions for data augmentation and logging."""

import random

import numpy as np


def data_augmentation(label, mode=0):
    """Apply one of eight common flip/rotation augmentations to an image array."""
    if mode == 0:
        return label
    if mode == 1:
        return np.flipud(label)
    if mode == 2:
        return np.rot90(label)
    if mode == 3:
        return np.flipud(np.rot90(label))
    if mode == 4:
        return np.rot90(label, k=2)
    if mode == 5:
        return np.flipud(np.rot90(label, k=2))
    if mode == 6:
        return np.rot90(label, k=3)
    if mode == 7:
        return np.flipud(np.rot90(label, k=3))
    raise ValueError(f"Unsupported augmentation mode: {mode}")


def loadpath(pathlistfile):
    with open(pathlistfile, "r", encoding="utf-8") as fp:
        pathlist = fp.read().splitlines()
    random.shuffle(pathlist)
    return pathlist


class Tee:
    """Duplicate stdout to multiple file-like objects."""

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for file in self.files:
            file.write(obj)
            file.flush()

    def flush(self):
        for file in self.files:
            file.flush()
