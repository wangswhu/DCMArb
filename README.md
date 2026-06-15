# DCMArb

Official PyTorch implementation of **DCMArb: Decoupled-Collaborative Mamba for Arbitrary-Scale Hyperspectral Super-Resolution**.

DCMArb reconstructs a high-resolution hyperspectral image (HSI) from its low-resolution observation at an arbitrary spatial scale. The model combines spatial and spectral Mamba branches with a scale-aware meta-learned upsampler to model long-range dependencies while preserving spatial details and spectral fidelity.

> Paper link, pretrained models, and complete citation information will be added when they become publicly available.

## Highlights

- **Arbitrary-scale HSI super-resolution** with one model across a continuous scale range.
- **Decoupled spatial-spectral modeling** for heterogeneous HSI dependencies.
- **Collaborative feature integration** using spatial relation, inter-band contextualization, and gated fusion modules.
- **Scale-aware meta upsampling** with coordinate encoding, dynamic offsets, and mixture-of-experts routing.
- End-to-end scripts for **training**, **evaluation**, and **result export**.
- Evaluation with **PSNR, SSIM, ERGAS, SAM, cross-correlation, and RMSE**.

## Network Overview

The main components implemented in this repository are:

- **HFIS**: Heterogeneous Feature Integration Stage.
- **MDES**: Multi-Dimensional Dependency Extraction Stage.
- **SSRB**: Shifted-window Spatial Relation Block for spatial dependency modeling.
- **ICB**: Inter-Band Contextualization Block for spectral dependency modeling.
- **GFB**: Gated Fusion Block for adaptive feature interaction.
- **SMUB**: Scale-Aware Meta-Learned Upsampler Block for arbitrary-scale reconstruction.

The complete network is defined in [`model/DCMArb.py`](model/DCMArb.py), while the arbitrary-scale upsampler is implemented in [`model/SMUB.py`](model/SMUB.py).

## Repository Structure

```text
DCMArb_open_source/
|-- mains.py                         # Training, testing, and visualization
|-- loss.py                          # HSI reconstruction loss
|-- metrics.py                       # Quantitative evaluation metrics
|-- utils.py                         # Data augmentation and logging utilities
|-- requirements.txt
|-- model/
|   |-- DCMArb.py                    # DCMArb network and model presets
|   |-- SSRB.py                      # Spatial relation block
|   |-- SMUB.py                      # Arbitrary-scale upsampler
|   `-- common.py                    # Shared layers and operations
`-- mydataset/
    |-- HSArbitrary_int.py           # Training and evaluation dataset
    `-- HSArbitrary_vis.py           # Visualization dataset
```

## Environment

The current implementation is intended for an NVIDIA GPU with CUDA. Install a CUDA-compatible version of PyTorch first by following the [official PyTorch instructions](https://pytorch.org/get-started/locally/).

A typical environment can be created as follows:

```bash
conda create -n dcmarb python=3.10 -y
conda activate dcmarb

# Install the PyTorch build matching your CUDA environment first.
pip install -r requirements.txt

# Required by the current model module for complexity profiling support.
pip install thop
```

If `mamba-ssm` fails to build through `requirements.txt`, install it according to the [official Mamba repository](https://github.com/state-spaces/mamba):

```bash
pip install causal-conv1d --no-build-isolation
pip install mamba-ssm --no-build-isolation
```

### Important CUDA Note

Although `mains.py` exposes a `--cuda` option, the current dataset presets instantiate the model on CUDA inside `model/DCMArb.py`. Therefore, the released model factory currently requires a working CUDA environment.

## Data Preparation

Each sample must be a MATLAB `.mat` file containing:

- Key: `gt`
- Shape: `H x W x C`
- Type: convertible to `float32`
- Recommended value range: `[0, 1]`

The loader performs bicubic spatial degradation online to generate the corresponding low-resolution input. It does not normalize the input values, so the stored data range should match `--data_range`, whose default is `1.0`.

Organize a dataset as follows:

```text
datasets/
`-- chikusei_x4_256/
    |-- train/
    |   |-- 0001.mat
    |   `-- ...
    |-- val/
    |   |-- 0001.mat
    |   `-- ...
    |-- test/
    |   |-- 0001.mat
    |   `-- ...
    `-- vis_512/
        |-- 0001.mat
        `-- ...
```

With the default `--lr_size 32 32` and maximum scale `4`, each HR image must be at least `128 x 128` pixels. If `--lr_size` is changed, both spatial dimensions should remain divisible by `16` for the default shifted-window configuration.

Datasets and pretrained weights are not distributed in this repository.

## Dataset Presets

The model factory currently provides the following spectral-channel presets:

| Preset | Spectral bands |
| --- | ---: |
| `chikusei` | 128 |
| `gf5b` | 150 |
| `zy1f` | 76 |

Select a preset with `--dataset_name`. For another dataset, add its channel configuration to the `dcmarb()` factory in `model/DCMArb.py`.

## Training

Train one model over scales from `2x` to `4x`:

```bash
python mains.py train \
  --data_root ./datasets/chikusei_x4_256 \
  --dataset_name chikusei \
  --scale_range 2 4 \
  --lr_size 32 32 \
  --batch_size 16 \
  --epochs 500 \
  --learning_rate 1e-4 \
  --checkpoint_dir ./checkpoints/chikusei_dcmarb_x2-x4 \
  --gpus 0
```

During training and validation, scales are sampled from the specified range and rounded to integer values. With `--scale_range 2 4`, the sampled training scales are `2x`, `3x`, and `4x`.

The best validation checkpoint is saved as:

```text
<checkpoint_dir>/<dataset_name>_<model_title>_best.pth
```

For the command above, the default path is:

```text
./checkpoints/chikusei_dcmarb_x2-x4/chikusei_dcmarb_best.pth
```

### Resume Training

```bash
python mains.py train \
  --data_root ./datasets/chikusei_x4_256 \
  --dataset_name chikusei \
  --scale_range 2 4 \
  --checkpoint_dir ./checkpoints/chikusei_dcmarb_x2-x4 \
  --checkpoint ./checkpoints/chikusei_dcmarb_x2-x4/chikusei_dcmarb_best.pth \
  --resume \
  --gpus 0
```

The checkpoint restores the model parameters and starting epoch. Optimizer and gradient-scaler states are not stored by the current training script.

## Evaluation

Evaluate at a fixed scale by setting both ends of `--scale_range` to the same value.

### Integer Scale

```bash
python mains.py test \
  --data_root ./datasets/chikusei_x4_256 \
  --dataset_name chikusei \
  --scale_range 4 4 \
  --checkpoint_dir ./checkpoints/chikusei_dcmarb_x2-x4 \
  --checkpoint ./checkpoints/chikusei_dcmarb_x2-x4/chikusei_dcmarb_best.pth \
  --gpus 0
```

### Non-Integer Scale

```bash
python mains.py test \
  --data_root ./datasets/chikusei_x4_256 \
  --dataset_name chikusei \
  --scale_range 2.5 2.5 \
  --checkpoint_dir ./checkpoints/chikusei_dcmarb_x2-x4 \
  --checkpoint ./checkpoints/chikusei_dcmarb_x2-x4/chikusei_dcmarb_best.pth \
  --gpus 0
```

Testing reports the average inference time, excluding the first batch, and the following full-reference metrics:

- PSNR and SSIM: higher is better.
- Cross-correlation: higher is better.
- ERGAS, SAM, and RMSE: lower is better.

For data outside `[0, 1]`, pass the correct peak range through `--data_range`.

## Export Super-Resolved Results

Export predictions as NumPy arrays:

```bash
python mains.py vis \
  --data_root ./datasets/chikusei_x4_256 \
  --dataset_name chikusei \
  --scale_range 4 4 \
  --checkpoint ./checkpoints/chikusei_dcmarb_x2-x4/chikusei_dcmarb_best.pth \
  --vis_save_path ./results/chikusei_x4 \
  --vis_hr_size 512 \
  --gpus 0
```

The generated arrays are saved in:

```text
./results/chikusei_x4/npy/
```

Each `.npy` output uses the `H x W x C` layout.

## Useful Options

```text
--data_root        Dataset root containing train/val/test/vis_512
--train_dir        Override the training directory
--val_dir          Override the validation directory
--test_dir         Override the testing directory
--vis_dir          Override the visualization directory
--dataset_name     Model preset: chikusei, gf5b, or zy1f
--scale_range      Minimum and maximum scale
--lr_size          Low-resolution patch height and width
--checkpoint       Checkpoint used for resume, test, or visualization
--checkpoint_dir   Directory for checkpoints and text logs
--gpus             Visible GPU IDs, for example 0 or 0,1
```

Run the command-line help for the complete option list:

```bash
python mains.py train -h
python mains.py test -h
python mains.py vis -h
```
