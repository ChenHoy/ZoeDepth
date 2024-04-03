#! /usr/bin/env python3

import argparse
from pathlib import Path
import random
import time
import os
import ipdb
from glob import glob
from omegaconf import DictConfig

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from tqdm.auto import tqdm

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

import matplotlib
import torch
import open3d as o3d


"""
Single image inference with MiDaS v3.0 Large. 
- Predict depth from a single image and view the result both in 2D and 3D.
- Run on folder of images and save the depth maps in output folder.
"""


EXTENSION_LIST = [".jpg", ".jpeg", ".png"]



def parse_args():
    parser = argparse.ArgumentParser(description="Run single-image depth estimation using Zoedepth.")
    parser.add_argument(
        "--model",
        type=str,
        default="ZoeD_N",
        choices=["ZoeD_N", "ZoeD_K", "ZoeD_NK"],
        help="""Checkpoint path or hub name. 
        Use ZoeD_N for indoor scenes like (NyU Depth), ZoeD_K for outdoor scenes (like Kitti), 
        and ZoeD_NK for generic in the wild data.""",
    )

    parser.add_argument(
        "--input_rgb_dir",
        type=str,
        required=True,
        help="Path to the input image folder.",
    )

    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    # depth map colormap
    parser.add_argument(
        "--depth_cmap",
        type=str,
        default="Spectral",
        help="Colormap used to render depth predictions.",
    )

    return parser.parse_args()


def colorize_depth_maps(depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().clone().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc


def plot_2d(rgb: Image, depth: np.ndarray, cmap: str) -> None:
    import matplotlib.pyplot as plt

    # Colorize manually to appealing colormap
    percentile = 0.03
    min_depth_pct = np.percentile(depth, percentile)
    max_depth_pct = np.percentile(depth, 100 - percentile)
    depth_colored = colorize_depth_maps(
        depth, min_depth_pct, max_depth_pct, cmap=cmap
    ).squeeze()  # [3, H, W], value in (0, 1)
    depth_colored = (depth_colored * 255).astype(np.uint8)

    # Plot the Image, Depth, and Uncertainty side-by-side in a 1x2 grid
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(rgb)
    ax[0].set_title("Image")
    ax[1].imshow(chw2hwc(depth_colored))
    ax[1].set_title("Depth")
    ax[0].axis("off"), ax[1].axis("off")

    plt.show()


def plot_3d(rgb: Image, depth: np.ndarray):
    """Use Open3d to plot the 3D point cloud from the monocular depth and input image."""

    def get_calib_heuristic(ht: int, wd: int) -> np.ndarray:
        """On in-the-wild data we dont have any calibration file.
        Since we optimize this calibration as well, we can start with an initial guess
        using the heuristic from DeepV2D and other papers"""
        cx, cy = wd // 2, ht // 2
        fx, fy = wd * 1.2, wd * 1.2
        return fx, fy, cx, cy

    rgb = np.asarray(rgb)
    depth = np.asarray(depth)
    invalid = filter_prediction(depth).flatten()

    # Get 3D point cloud from depth map
    depth = depth.squeeze()
    h, w = depth.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.flatten()
    y = y.flatten()
    depth = depth.flatten()

    # Convert to 3D points
    fx, fy, cx, cy = get_calib_heuristic(h, w)
    # Unproject
    x3 = (x - cx) * depth / fx
    y3 = (y - cy) * depth / fy
    z3 = depth

    # Convert to Open3D format
    xyz = np.stack([x3, y3, z3], axis=1)
    rgb = np.stack([rgb[:, :, 0].flatten(), rgb[:, :, 1].flatten(), rgb[:, :, 2].flatten()], axis=1)

    xyz = clean_with_nerfbusters(
        torch.Tensor(xyz),
        model_ckpt_path="../nerfbusters/weights/nerfbusters-diffusion-cube-weights.ckpt",
    ).numpy()
    ipdb.set_trace()

    depth = depth[~invalid]
    xyz = xyz[~invalid]
    rgb = rgb[~invalid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)

    # Plot the point cloud
    o3d.visualization.draw_geometries([pcd])


def filter_prediction(depth_pred: np.ndarray, percentile: float = 0.01) -> np.ndarray:
    """Filter out invalid outlier depths according the distribution.
    Most of outliers are likely at near 0 and very far away compared to the actual object depths.
    """
    min_depth_pct = np.percentile(depth_pred, percentile)
    max_depth_pct = np.percentile(depth_pred, 100 - percentile)
    # Return mask for invalid predictions
    return np.logical_or(depth_pred < min_depth_pct, depth_pred > max_depth_pct)


def clean_with_nerfbusters(
    pcl: torch.Tensor,
    model_ckpt_path: str = "weights/nerfbusters-diffusion-cube-weights.ckpt",
) -> torch.Tensor:
    """Use the NeRF busters 3D diffusion model to clean the pointcloud outliers away and get a better reconstruction.
    This assumes that you have setup nerfbusters https://github.com/ethanweber/nerfbusters
    and addedd this to your PythonPath.
    """
    from clean import Nerfbusters

    model = Nerfbusters()
    model.load_model(model_ckpt_path)
    cleaned = model(pcl)
    del model
    return cleaned


def seed_all(seed: int = 0):
    """
    Set random seeds of all components.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    # Random seed
    if args.seed is None:
        seed = int(time.time())
    seed_all(seed)

    # Device
    cuda_avail = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_avail else "cpu")
    print(f"device = {device}")

    # -------------------- Data --------------------
    rgb_filename_list = glob(os.path.join(args.input_rgb_dir, "*"))
    rgb_filename_list = [f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST]
    rgb_filename_list = sorted(rgb_filename_list)
    n_images = len(rgb_filename_list)
    if n_images > 0:
        print(f"Found {n_images} images")
    else:
        raise RuntimeError(f"No image found in '{args.input_rgb_dir}'")

    # -------------------- Model --------------------
    # Triggers fresh download of MiDaS repo
    # torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=False)
    # Using Huggingface Hub
    # model = torch.hub.load("isl-org/ZoeDepth", args.model, pretrained=True)

    # Build and load manually
    if args.model == "ZoeD_N":
        conf = get_config("zoedepth", "infer")
    elif args.model == "ZoeD_K":
        conf = get_config("zoedepth", "infer", config_version="kitti")
    # ZoeD_NK
    else:
        conf = get_config("zoedepth_nk", "infer")
    model = build_model(conf).to(device).eval()

    os.makedirs(args.output_dir, exist_ok=True)
    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        for rgb_path in tqdm(rgb_filename_list, desc=f"Estimating depth", leave=True):
            # Load PIL from rgb_path
            image = Image.open(rgb_path).convert("RGB")
            rgb = np.asarray(image)

            # Inference
            depth_pred = model.infer_pil(image)

            # plot_2d(rgb, depth_pred, cmap=args.depth_cmap)
            # plot_3d(rgb, depth_pred)

            # Save depth map with numpy
            fname = Path(rgb_path).stem + ".npy"
            output_path = os.path.join(args.output_dir, fname)
            np.save(output_path, depth_pred)


if __name__ == "__main__":
    main()
