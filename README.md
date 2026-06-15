# DCMArb

Official PyTorch implementation of **DCMArb: Decoupled-Collaborative Mamba for Arbitrary-Scale Hyperspectral Super-Resolution**.

DCMArb is an arbitrary-scale hyperspectral image super-resolution network. It uses decoupled spatial-spectral Mamba modeling and a scale-aware meta-learned upsampler to reconstruct high-resolution HSI from low-resolution inputs.

## Main Modules

* **MDES**: Multi-Dimensional Dependency Extraction Stage
* **HFIS**: Heterogeneous Feature Integration Stage
* **SMUB**: Scale-Aware Meta-Learned Upsampler Block

The main network is implemented in `model/DCMArb.py`.

## Environment

Install PyTorch according to your CUDA version, then install the required packages:

```bash
pip install -r requirements.txt
```

If `mamba-ssm` fails to install, try:

```bash
pip install causal-conv1d --no-build-isolation
pip install mamba-ssm --no-build-isolation
```

## Data Format

Dataset structure:

```text
datasets/
└── chikusei/
    ├── train/
    ├── val/
    ├── test/
    └── vis/
```

## Training

```bash
python mains.py train \
  --data_root ./datasets/chikusei \
  --dataset_name chikusei \
  --scale_range 2 4 \
  --lr_size 32 32 \
  --batch_size 8 \
  --epochs 500 \
  --learning_rate 1e-4 \
  --checkpoint_dir ./checkpoints/chikusei_dcmarb_x2-x4 \
  --gpus 0
```

## Testing

Integer scale:

```bash
python mains.py test \
  --data_root ./datasets/chikusei \
  --dataset_name chikusei \
  --scale_range 4 4 \
  --checkpoint ./checkpoints/chikusei_dcmarb_x2-x4/chikusei_dcmarb_best.pth \
  --gpus 0
```

Non-integer scale:

```bash
python mains.py test \
  --data_root ./datasets/chikusei \
  --dataset_name chikusei \
  --scale_range 2.5 2.5 \
  --checkpoint ./checkpoints/chikusei_dcmarb_x2-x4/chikusei_dcmarb_best.pth \
  --gpus 0
```

## Visualization

```bash
python mains.py vis \
  --data_root ./datasets/chikusei \
  --dataset_name chikusei \
  --scale_range 4 4 \
  --checkpoint ./checkpoints/chikusei_dcmarb_x2-x4/chikusei_dcmarb_best.pth \
  --vis_save_path ./results/chikusei_x4 \
  --gpus 0
```

## Citation

The citation information will be updated once the paper is available online.
