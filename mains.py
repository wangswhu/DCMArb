"""Training, testing, and visualization entry point for DCMArb.

Example:
    python mains.py train --data_root ./datasets/chikusei_x4_256 --gpus 0
"""
import argparse
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.amp import autocast
try:
    from torch.amp import GradScaler
except ImportError:  # Backward compatibility with older PyTorch versions.
    from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torch.utils.data import DataLoader

from loss import HSRLoss
from metrics import compare_corr, compare_ergas, compare_mpsnr, compare_mssim, compare_rmse, compare_sam
from model.DCMArb import dcmarb
from mydataset.HSArbitrary_int import HSArbitraryData
from mydataset.HSArbitrary_vis import HSArbitraryDataVis
from utils import Tee


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
print(PROJECT_DIR)
DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(PROJECT_DIR),
    "datasets",
    "chikusei_x4_256",
)
DEFAULT_VIS_OUTPUT_ROOT = os.path.join(".", "results", "chikusei_vis")
DEFAULT_CHECKPOINT_DIR = os.path.join(".", "checkpoints", "chikusei_dcmarb_x2-x4")
LOG_INTERVAL = 50


def build_common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cuda", type=int, default=1, help="Set to 1 for GPU, 0 for CPU.")
    parser.add_argument("--gpus", type=str, default="0", help="Visible GPU IDs, e.g., '0' or '0,1'.")
    parser.add_argument("--dataset_name", type=str, default="chikusei", help="Dataset name used by the model builder.")
    parser.add_argument("--data_range", type=float, default=1.0, help="Data range used in metric calculation.")
    parser.add_argument("--model_title", type=str, default="dcmarb", help="Model title used in logs/checkpoints.")
    parser.add_argument("--lr_size", type=int, nargs=2, default=(32, 32), help="LR patch size, e.g., --lr_size 32 32.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--scale_range", type=float, nargs=2, default=[2, 4], help="Scale range, e.g., --scale_range 2 4.")

    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT, help="Dataset root containing train/val/test/vis_512 folders.")
    parser.add_argument("--train_dir", type=str, default=None, help="Training data directory. Defaults to <data_root>/train.")
    parser.add_argument("--val_dir", type=str, default=None, help="Validation data directory. Defaults to <data_root>/val.")
    parser.add_argument("--test_dir", type=str, default=None, help="Testing data directory. Defaults to <data_root>/test.")
    parser.add_argument("--vis_dir", type=str, default=None, help="Visualization data directory. Defaults to <data_root>/vis_512.")
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR, help="Directory for checkpoints/logs.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for resume/test/vis.")
    parser.add_argument("--resume", action="store_true", help="Resume training from --checkpoint.")
    return parser


def resolve_data_dirs(args: argparse.Namespace) -> None:
    args.train_dir = args.train_dir or os.path.join(args.data_root, "train")
    args.val_dir = args.val_dir or os.path.join(args.data_root, "val")
    args.test_dir = args.test_dir or os.path.join(args.data_root, "test")
    args.vis_dir = args.vis_dir or os.path.join(args.data_root, "vis_512")


def make_collate_fn(dataset):
    def custom_collate(batch):
        lr_tensor = torch.stack([item["lr"] for item in batch])
        hr_tensor = torch.stack([item["hr"] for item in batch])
        scale_batch = [item["scale"] for item in batch]
        dataset.current_scale_factor = None
        output = {"lr": lr_tensor, "hr": hr_tensor, "scale": scale_batch}
        if "file_name" in batch[0]:
            output["file_name"] = [item["file_name"] for item in batch]
        return output
    return custom_collate


def build_model(dataset_name: str):
    return dcmarb(dataset=dataset_name)


def _strip_or_add_module_prefix(state_dict, model_state):
    state_has_module = all(k.startswith("module.") for k in state_dict.keys())
    model_has_module = all(k.startswith("module.") for k in model_state.keys())
    if state_has_module and not model_has_module:
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    if not state_has_module and model_has_module:
        return {f"module.{k}": v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint_if_available(model: torch.nn.Module, checkpoint_path: Optional[str], device: torch.device) -> int:
    if checkpoint_path is None:
        raise ValueError("A checkpoint path is required. Please pass --checkpoint /path/to/model.pth")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"=> Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        start_epoch = int(checkpoint.get("epoch", 0))
    else:
        state_dict = checkpoint
        start_epoch = 0

    if not isinstance(state_dict, dict):
        state_dict = state_dict.state_dict()

    state_dict = _strip_or_add_module_prefix(state_dict, model.state_dict())
    model.load_state_dict(state_dict)
    print(f"=> Loaded checkpoint epoch: {start_epoch}")
    return start_epoch


def main():
    common_parser = build_common_parser()
    main_parser = argparse.ArgumentParser(description="DCMArb for arbitrary-scale HSI super-resolution.")
    subparsers = main_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_parser = subparsers.add_parser("train", help="Train the model.", parents=[common_parser])
    train_parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs.")
    train_parser.add_argument("--seed", type=int, default=10, help="Random seed.")
    train_parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate.")
    train_parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    train_parser.add_argument("--val_interval", type=int, default=10, help="Validate every N epochs.")
    train_parser.add_argument("--save_pth", type=int, default=50, help="Save checkpoint every N epochs.")
    train_parser.add_argument("--train_workers", type=int, default=8, help="DataLoader workers for training.")
    train_parser.add_argument("--val_workers", type=int, default=4, help="DataLoader workers for validation.")

    test_parser = subparsers.add_parser("test", help="Test the model.", parents=[common_parser])
    test_parser.add_argument("--test_workers", type=int, default=4, help="DataLoader workers for testing.")

    vis_parser = subparsers.add_parser("vis", help="Save SR results as .npy files.", parents=[common_parser])
    vis_parser.add_argument("--vis_save_path", type=str, default=DEFAULT_VIS_OUTPUT_ROOT, help="Directory for visualization outputs.")
    vis_parser.add_argument("--vis_workers", type=int, default=4, help="DataLoader workers for visualization.")
    vis_parser.add_argument("--vis_hr_size", type=int, default=512, help="Reference HR size used to derive LR crop size for visualization.")

    args = main_parser.parse_args()
    if args.subcommand is None:
        main_parser.print_help()
        sys.exit(1)

    resolve_data_dirs(args)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    original_stdout = sys.stdout
    log_file = None
    try:
        if args.subcommand in ["train", "test"]:
            current_time = datetime.now().strftime("%m%d_%H%M%S")
            log_name = f"{args.subcommand}_{args.model_title}_{current_time}.txt"
            if args.subcommand == "test":
                log_name = f"test_{args.model_title}_{current_time}_x{args.scale_range[0]}.txt"
            log_path = os.path.join(args.checkpoint_dir, log_name)
            log_file = open(log_path, "w", encoding="utf-8")
            sys.stdout = Tee(sys.stdout, log_file)

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if args.cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Use --cuda 0 to run on CPU.")

        if args.subcommand == "train":
            train(args)
        elif args.subcommand == "test":
            test(args)
        elif args.subcommand == "vis":
            vis(args)
    finally:
        if log_file is not None:
            log_file.close()
        sys.stdout = original_stdout


def get_device(args):
    return torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")


def make_scaler(args):
    try:
        return GradScaler("cuda", enabled=bool(args.cuda))
    except TypeError:
        return GradScaler(enabled=bool(args.cuda))


def train(args: argparse.Namespace):
    device = get_device(args)
    print("Start seed:", args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = bool(args.cuda)

    print("===> Loading datasets")
    train_set = HSArbitraryData(args.train_dir, augment=True, lr_size=args.lr_size, scale_range=args.scale_range, round_scale=True)
    val_set = HSArbitraryData(args.val_dir, augment=False, lr_size=args.lr_size, scale_range=args.scale_range, round_scale=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.train_workers, shuffle=True, collate_fn=make_collate_fn(train_set))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.val_workers, shuffle=False, collate_fn=make_collate_fn(val_set))

    print("===> Building model")
    model = build_model(args.dataset_name)
    print("# parameters: {:.4f}M".format(sum(param.numel() for param in model.parameters()) / 1e6))
    args.model_title = f"{args.dataset_name}_{args.model_title}"

    model = model.to(device)
    if torch.cuda.device_count() > 1 and args.cuda:
        print("===> Using", torch.cuda.device_count(), "GPUs.")
        model = torch.nn.DataParallel(model)

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint_if_available(model, args.checkpoint, device)

    model.train()
    h_loss = HSRLoss().to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = make_scaler(args)

    best_psnr = -np.inf
    best_epoch = start_epoch
    print("===> Start training")

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        adjust_learning_rate(args.learning_rate, optimizer, epoch + 1)
        loss_values = []
        print("Start epoch {}, learning rate = {}".format(epoch + 1, optimizer.param_groups[0]["lr"]))

        for iteration, batch in enumerate(train_loader):
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            scale = batch["scale"][0]

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", dtype=torch.float16, enabled=bool(args.cuda)):
                sr = model(lr, scale)
                loss = h_loss(sr, hr)

            loss_values.append(float(loss.item()))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (iteration + 1) % LOG_INTERVAL == 0:
                print("===> {} GPU{}\tEpoch[{}]({}/{}): Loss: {:.6f}".format(time.ctime(), args.gpus, epoch + 1, iteration + 1, len(train_loader), loss.item()))

        epoch_time = time.time() - start_time
        print("===> {}\tEpoch {} Training Complete: AvgLoss: {:.6f}, training time: {:.2f} seconds".format(time.ctime(), epoch + 1, np.mean(loss_values), epoch_time))

        if (epoch + 1) % args.val_interval == 0 or (epoch + 1) == args.epochs:
            val_indices = validate(args, val_loader, model)
            current_psnr = val_indices["PSNR"]
            if current_psnr > best_psnr:
                best_psnr = current_psnr
                best_epoch = epoch + 1
                save_checkpoint(args, model, best_epoch, is_best=True)
            print(f"Best PSNR: {best_psnr:.4f} (Achieved at Epoch {best_epoch})")

        if (epoch + 1) % args.save_pth == 0:
            save_checkpoint(args, model, epoch + 1)

    save_model_filename = f"{args.model_title}_epoch_{args.epochs}_{time.strftime('%Y%m%d_%H%M%S')}.pth"
    save_model_path = os.path.join(args.checkpoint_dir, save_model_filename)
    torch.save(get_state_dict(model), save_model_path)
    print("\nDone, trained model saved at", save_model_path)


def adjust_learning_rate(start_lr: float, optimizer: torch.optim.Optimizer, epoch: int):
    lr = start_lr * (0.5 ** (epoch // 150))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def validate(args: argparse.Namespace, loader: DataLoader, model: torch.nn.Module) -> Dict[str, float]:
    device = get_device(args)
    model.eval()
    indices: Dict[str, List[float]] = {}

    with torch.no_grad():
        for batch in loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            scale = batch["scale"][0]
            print("validate scale:", scale)
            sr = model(lr, scale)
            y_np = sr.permute(0, 2, 3, 1).cpu().numpy()
            gt_np = hr.permute(0, 2, 3, 1).cpu().numpy()
            for b in range(y_np.shape[0]):
                current_indices = quality_assessment(gt_np[b], y_np[b], data_range=args.data_range, ratio=scale)
                for key, value in current_indices.items():
                    indices.setdefault(key, []).append(value)

    final_indices = {k: float(np.mean(v)) for k, v in indices.items()}
    formatted_metrics = {k: round(v, 4) for k, v in final_indices.items()}
    print("===> {}\tValidation Complete\nMetrics: {}".format(time.ctime(), ", ".join(f"{k}: {v}" for k, v in formatted_metrics.items())))
    model.train()
    return final_indices


def test(args: argparse.Namespace):
    device = get_device(args)
    print("===> Loading testset")
    test_set = HSArbitraryData(args.test_dir, augment=False, lr_size=args.lr_size, scale_range=args.scale_range, round_scale=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.test_workers, shuffle=False, collate_fn=make_collate_fn(test_set))

    print("===> Building model")
    model = build_model(args.dataset_name).to(device)
    load_checkpoint_if_available(model, args.checkpoint, device)
    model.eval()

    print("===> Start testing")
    indices: Dict[str, List[float]] = {}
    total_infer_time = 0.0
    total_img_num = 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            scale = batch["scale"][0]
            print("test scale:", scale)

            infer_start = time.time()
            sr = model(lr, scale)
            batch_infer_time = time.time() - infer_start
            batch_size = lr.shape[0]
            print(f"Test Batch [{i + 1}] - Single Image Inference Time: {batch_infer_time / batch_size:.6f} seconds")
            if i > 0:
                total_infer_time += batch_infer_time
                total_img_num += batch_size

            y_np = sr.permute(0, 2, 3, 1).cpu().numpy()
            gt_np = hr.permute(0, 2, 3, 1).cpu().numpy()
            for b in range(y_np.shape[0]):
                current_indices = quality_assessment(gt_np[b], y_np[b], data_range=args.data_range, ratio=scale)
                for key, value in current_indices.items():
                    indices.setdefault(key, []).append(value)

    final_indices = {k: float(np.mean(v)) for k, v in indices.items()}
    formatted_metrics = {k: round(v, 4) for k, v in final_indices.items()}
    avg_infer_time = total_infer_time / total_img_num if total_img_num > 0 else 0.0
    print(f"\n===> Average Single Image Inference Time (without first batch): {avg_infer_time:.6f} seconds")
    print("===> {}\tTest Complete\nMetrics: {}".format(time.ctime(), ", ".join(f"{k}: {v}" for k, v in formatted_metrics.items())))


def get_state_dict(model):
    return model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()


def save_checkpoint(args: argparse.Namespace, model: torch.nn.Module, epoch: int, is_best: bool = False):
    was_training = model.training
    model.eval()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    filename = f"{args.model_title}_best.pth" if is_best else f"{args.model_title}_epoch_{epoch}.pth"
    path = os.path.join(args.checkpoint_dir, filename)
    torch.save({"epoch": epoch, "model": get_state_dict(model)}, path)
    if was_training:
        model.train()
    print(f"Checkpoint saved to {path}")


def get_vis_size(scale: float, hr_size: int = 512) -> Tuple[int, int]:
    lr = max(1, int(round(hr_size / float(scale))))
    return lr, lr


def vis(args: argparse.Namespace):
    device = get_device(args)
    print("===> Loading visualization data")
    vis_set = HSArbitraryDataVis(args.vis_dir, augment=False, lr_size=get_vis_size(args.scale_range[0], args.vis_hr_size), scale_range=args.scale_range)
    vis_loader = DataLoader(vis_set, batch_size=1, num_workers=args.vis_workers, shuffle=False, collate_fn=make_collate_fn(vis_set))

    save_vis_dir = os.path.join(args.vis_save_path, "npy")
    os.makedirs(save_vis_dir, exist_ok=True)

    print("===> Building model")
    model = build_model(args.dataset_name).to(device)
    load_checkpoint_if_available(model, args.checkpoint, device)
    model.eval()

    with torch.no_grad():
        for batch in vis_loader:
            lr = batch["lr"].to(device)
            scale = batch["scale"][0]
            lr_name = os.path.splitext(batch["file_name"][0])[0]
            print("vis scale:", scale)
            sr = model(lr, scale)
            y_np = sr.cpu().numpy().squeeze(0).transpose(1, 2, 0)
            save_name = f"{lr_name}_{args.model_title}_x{args.scale_range[0]}.npy"
            save_path = os.path.join(save_vis_dir, save_name)
            np.save(save_path, y_np)
            print(f"Saved: {save_path}")


def quality_assessment(x_true, x_pred, data_range, ratio, multi_dimension=False):
    return {
        "PSNR": compare_mpsnr(x_true=x_true, x_pred=x_pred, data_range=data_range),
        "MSSIM": compare_mssim(x_true=x_true, x_pred=x_pred, data_range=data_range, multidimension=multi_dimension),
        "ERGAS": compare_ergas(x_true=x_true, x_pred=x_pred, ratio=ratio),
        "SAM": compare_sam(x_true=x_true, x_pred=x_pred),
        "CrossCorrelation": compare_corr(x_true=x_true, x_pred=x_pred),
        "RMSE": compare_rmse(x_true=x_true, x_pred=x_pred),
    }


if __name__ == "__main__":
    main()
