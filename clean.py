from typing import Any, Mapping, Optional, Dict, List, Tuple
from tqdm import tqdm
import ipdb
from omegaconf import DictConfig
import math

import numpy as np
import torch.nn as nn
import torch

from diffusers import DDIMPipeline, DDIMScheduler, DDPMPipeline, DDPMScheduler
from diffusers.training_utils import EMAModel

from nerfbusters.models.model import get_model
import open3d as o3d


"""
Clean point clouds with NerfBusters model. In their paper they use a 3D diffusion model, which operates on corrupted voxel volumes 
to remove floaters from a NeRf volume. We use this class for inference/test mode only to clean point clouds / meshes / voxel volumes.

The network works on corrupted voxel volumes of the shape [B, C, H, W, D] with values {-1, 1} and C=1.
It additionally gets a scale scalar [0, 1] which measures the percentage of the cube compared to the original scene.
This is put into a sort of positional encoding. During training the scale varies between [0.03, 0.1], meaning we have a 
maximum of 10% scene in each cube. 
voxel volumes during training have ~ resolution 32 x 32 x 32 for H, W, D respectively.
"""

# FIXME this somehow does not work as well as I expected, we do not get good reconstructions from the voxel volumes we provide
# A potential reason for this might be the kind of shapes that ShapeNet uses compared to in-the-wild depth reconstructions, which follow just a vastly different distribution


def normalize_occupancy_grid(occupancy_grid: torch.Tensor) -> torch.Tensor:
    """Normalize the occupancy grid to [-1, 1]"""
    return occupancy_grid * 2.0 - 1.0


def unnormalize_occupancy_grid(occupancy_grid: torch.Tensor) -> torch.Tensor:
    """Unnormalize the occupancy grid to [0, 1]"""
    return (occupancy_grid + 1.0) / 2.0


def normalize_point_cloud(
    point_cloud: torch.Tensor, return_scale_shift: bool = True
) -> torch.Tensor:
    """Normalize the point cloud, so that it fits inside a [1, 1, 1] cube
    that is centered around the origin. Therefore the cube lies in
    the range [-0.5, 0.5] for each dimension and the point cloud is normalized."""

    def center_point_cloud(point_cloud: torch.Tensor) -> torch.Tensor:
        """Center the point cloud around the origin."""
        offsets = point_cloud.mean(dim=1, keepdim=True)
        return point_cloud - offsets, offsets

    def scale_point_cloud(point_cloud: torch.Tensor) -> torch.Tensor:
        """Scale the point cloud to fit inside a [1, 1, 1] cube."""
        # Get the maximum absolute value of the point cloud
        max_abs = point_cloud.abs().max()
        # Divide by the maximum absolute value to get a point cloud in the range [-1, 1]
        return point_cloud / max_abs, max_abs

    point_cloud, offsets = center_point_cloud(point_cloud)
    point_cloud, scale = scale_point_cloud(point_cloud)
    if return_scale_shift:
        return point_cloud, scale, offsets
    else:
        return point_cloud


# This chunks up the volume given a fixed size
def chunk_voxel_volume(
    voxel_volume: torch.Tensor, chunk_size: int = 32
) -> torch.Tensor:
    """Chunk the voxel volume into smaller voxel volume chunks of size [chunk_size, chunk_size, chunk_size]

    args:
    ---
    voxel_volume (torch.Tensor): The voxel volume of shape [B, C, H, W, D] with H = W = D
    chunk_size (int): The size of the chunks in each dimension.

    returns:
    ---
    voxel_chunks (List[torch.Tensor]): The voxel chunks of shape [B, C, chunk_size, chunk_size, chunk_size]
    """
    assert voxel_volume.ndim == 5, "Volume should be of size [B, C, H, W, D]"
    B, C, H, W, D = voxel_volume.shape
    voxel_chunks = []
    # Index the array with the chunk size
    # This has complexity O(N^3) with N = H // chunk_size
    # TODO Improve efficiency with vectorized indexing using torch
    for h in range(0, H, chunk_size):
        for w in range(0, W, chunk_size):
            for d in range(0, D, chunk_size):
                # Get the chunk
                chunk = voxel_volume[
                    ..., h : h + chunk_size, w : w + chunk_size, d : d + chunk_size
                ]
                voxel_chunks.append(chunk)
    return voxel_chunks


def reassemble_voxel_volume(
    volume_chunks: List[torch.Tensor], volume_shape: Tuple[int, int, int, int, int]
) -> torch.Tensor:
    """Reassemble the voxel volume from the chunks.

    args:
    ---
    volume_chunks (List[torch.Tensor]): The voxel chunks of shape [B, C, chunk_size, chunk_size, chunk_size]
    volume_shape (Tuple[int, int, int]): The shape of the original volume, i.e. [H, W, D]

    returns:
    ---
    voxel_volume (torch.Tensor): The voxel volume of shape [B, C, H, W, D]
    """
    assert len(volume_shape) == 5
    B, C, H, W, D = volume_shape
    bb, cc, hh, ww, dd = volume_chunks[0].shape
    voxel_volume = torch.zeros((B, C, H, W, D))
    # Index the array with the chunk size
    # This has complexity O(N^3) with N = H // chunk_size
    # TODO Improve efficiency with vectorized indexing using torch
    idx = 0
    for h in range(0, H, hh):
        for w in range(0, W, ww):
            for d in range(0, D, dd):
                # Get the chunk
                voxel_volume[..., h : h + hh, w : w + ww, d : d + dd] = volume_chunks[
                    idx
                ]
                idx += 1
    return voxel_volume


# TODO this needs additional scale and shift, not only offsets (low priority)
def voxel_volume_to_point_cloud(
    voxel_volume: torch.Tensor, offsets=[0, 0, 0]
) -> torch.Tensor:
    """Convert a voxel volume into a point cloud.

    args:
    ---
    voxel_volume (torch.Tensor): The voxel volume of shape [B, C, H, W, D]
    offsets (List[int]): The offsets for the point cloud in each dimension.

    returns:
    ---
    point_cloud (torch.Tensor): The point cloud of shape [B, 3, N]
    """
    assert len(voxel_volume.shape) == 4
    C, H, W, D = voxel_volume.shape
    points = torch.where(voxel_volume > 0.0)
    # TODO does this work correctly with .T?
    points = torch.stack(points).T.cpu()  # N, 3
    points = points + torch.tensor(offsets).unsqueeze(0).repeat(points.shape[0], 1)

    if points.shape[0] == 0:
        print("WARNING: occupancy is empty. Adding a single voxel.")
        points = torch.tensor([[0, 0, 0]])

    # rescale to unit cube
    points = points.float() / torch.Tensor([H, W, D])[None, :]
    return points


def test_volume(new_volume: torch.Tensor) -> None:
    """Visualize the voxel volume with no respect to the actual metric size of the surfaces.
    Simply take the index coordinates of the occupied voxels and plot them as point cloud.
    """
    cat_indices = np.where(new_volume[0, 0].numpy() > 0.0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack(cat_indices, axis=1))
    o3d.visualization.draw_geometries([pcd])


def point_cloud_to_voxel_volume(
    xyz: torch.Tensor, return_pcl: bool = True, voxel_size: Optional[float] = 0.01
) -> torch.Tensor:
    # Create a point cloud from torch.Tensor with Open3D
    xyz = xyz.cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # Voxelize the point cloud into an occupancy volume
    # This is done based on a fixed voxel size in [m] -> How to find the optimal voxel size for our point cloud?
    if voxel_size is None:
        # This gives you the nearest neighbor distance of each point in the pointcloud
        all_dist = torch.Tensor(pcd.compute_nearest_neighbor_distance())
        # If you choose the minimum distance you will not lose much information
        # You only lose no information at all if you were to choose the minimum distance for each spatial dimension
        min_dist = all_dist.min()
        # If we have double points, we will get a min_dist of 0.0, which will cause an error
        if min_dist < 1e-3:
            min_dist = 1e-3
        voxel_size = min_dist
    voxel_volume = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd, voxel_size=voxel_size
    )
    if return_pcl:
        return voxel_volume, pcd
    else:
        return voxel_volume


class Nerfbusters(nn.Module):
    """
    3D diffusion model for cleaning voxel volumes. This is optimized during training to remove
    floaters from occupancy grids. During inference with e.g. a NeRF model, this can be used to
    clean the 3D rendering by using this as a loss signal for optimizing the Rendering parameters.
    """

    def __init__(
        self,
        noise_scheduler: str = "ddpm",
        num_inference_steps: int = 100,
        beta_start: float = 0.0015,
        beta_end: float = 0.05,
        model_channels: int = 32,
        num_res_blocks: int = 1,
        channel_mult: Tuple[int, int, int] = (1, 2, 4),
        attention_resolutions: List[int] = [4],
        architecture: str = "unet3d",
        condition_on_scale: bool = True,
        guidance_weight: float = 1.0,
    ):
        super().__init__()
        # Convert kwargs into DictConfig
        config = {
            "noise_scheduler": noise_scheduler,
            "num_inference_steps": num_inference_steps,
            "beta_start": beta_start,
            "beta_end": beta_end,
            "model_channels": model_channels,
            "num_res_blocks": num_res_blocks,
            "channel_mult": channel_mult,
            "attention_resolutions": attention_resolutions,
            "architecture": architecture,
            "condition_on_scale": condition_on_scale,
            "guidance_weight": guidance_weight,
        }
        config = DictConfig(config)

        self.model = get_model(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model.convert_to_fp16()

        self.config = config

        if config.noise_scheduler == "ddim":
            num_train_timesteps = config.get("num_train_timesteps", 1000)
            self.num_inference_steps = config.get("num_inference_steps", 50)
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=num_train_timesteps
            )
            self.noise_scheduler.num_inference_steps = num_inference_steps
        elif config.noise_scheduler == "ddpm":
            num_train_timesteps = config.get("num_train_timesteps", 1000)
            self.num_inference_steps = config.get("num_inference_steps", 1000)
            beta_start = config.get("beta_start", 0.0015)
            beta_end = config.get("beta_end", 0.05)
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
            )
            self.noise_scheduler.num_inference_steps = num_inference_steps
        else:
            raise ValueError(f"Unknown noise scheduler: {config.noise_scheduler}")

    def load_model(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.model.to(self.device)
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
            self.device
        )
        print("Loaded diffusion checkpoint from", checkpoint_path)

    def density_to_x(
        self,
        density: torch.Tensor,
        density_x_crossing: float = 0.01,
        density_x_max: float = 500.0,
        activation: str = "sigmoid",
        binary_threshold: float = 0.5,
    ) -> torch.Tensor:
        """Converts density to x for diffusion model."""
        if activation == "sigmoid":
            x = 2 * torch.sigmoid(1000.0 * (density - density_x_crossing)) - 1.0

        elif activation == "clamp":
            x = torch.clamp(density_x_crossing * density - 1, -1, 1)

        elif activation == "sigmoid_complex":
            density_to_x_temperature = -1.0 * math.log(1.0 / 3.0) / density_x_crossing
            x = ((torch.sigmoid(density_to_x_temperature * density)) - 0.5) * 4.0 - 1.0

        elif activation == "binarize":
            x = torch.where(density.detach() < density_x_crossing, -1.0, 1.0)

        elif activation == "rescale_clamp":
            x = torch.clamp(density / density_x_crossing - 1.0, -1.0, 1.0)

        elif activation == "piecewise_linear":
            x_fir = density / density_x_crossing - 1.0
            x_sec = (
                1.0
                / (density_x_max - density_x_crossing)
                * (density - density_x_crossing)
            )
            x = torch.where(density < density_x_crossing, x_fir, x_sec).clamp(-1.0, 1.0)

        elif activation == "piecewise_loglinear":
            x_fir = density / density_x_crossing - 1.0
            x_sec = torch.log(1 + density - density_x_crossing) / torch.log(
                torch.tensor(density_x_max)
            )
            x = torch.where(density < density_x_crossing, x_fir, x_sec).clamp(-1.0, 1.0)

        elif activation == "piecewise_loglinear_threshold":
            x_fir = density / density_x_crossing - 1.0
            x_sec = torch.log(1 + density - density_x_crossing) / torch.log(
                torch.tensor(density_x_max)
            )
            x = torch.where(density < density_x_crossing, x_fir, x_sec).clamp(-1.0, 1.0)
            # Piecewise loglinear threshold might need to be adjusted
            x = torch.where(density < 1e-3, -1.0 * torch.ones_like(x), x)

        elif activation == "piecewise_loglinear_sigmoid":
            temperature = 1500
            x_fir = (
                2 / (1 + torch.exp(-(temperature * (density - density_x_crossing)))) - 1
            )
            x_sec = torch.log(1 + density - density_x_crossing) / torch.log(
                torch.tensor(density_x_max)
            )
            x = torch.where(density < density_x_crossing, x_fir, x_sec).clamp(-1.0, 1.0)

        elif activation == "piecewise_exp":
            x_fir = density / density_x_crossing - 1.0
            x_sec = density_x_max - density_x_crossing
            x_thir = 1.0 / x_sec * (density - density_x_crossing)
            x = torch.where(density < density_x_crossing, x_fir, x_thir).clamp(
                -1.0, 1.0
            )

        elif activation == "batchnorm":
            with torch.no_grad():
                running_mean = density.mean()
                running_var = density.var()

            mu = running_mean
            sigma = running_var.sqrt()
            x = (density - mu) / (sigma + 1e-7)
            x = torch.clamp(x, -1.0, 1.0)

        elif activation == "meannorm":
            x = torch.log(
                density
            )  # - torch.mean(torch.log(density))  # / (density.std() + 1e-7)
            x = torch.clamp(x, -1.0, 1.0)

        # This might not be in {-1, 1} ->  threshold after unnormalizing
        # x = unnormalize_occupancy_grid(x)
        x[x > binary_threshold] = 1.0
        return x.int()

    @torch.no_grad()
    def reverse_process(
        self,
        sample: Optional[torch.Tensor] = None,
        scale: Optional[float] = None,
        bs: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        starting_t: int = 1000,
        min_points: int = 10,
    ) -> torch.Tensor:
        """Run the reverse process of the model, either starting from a sample or from random noise.

        Args:
            sample (torch.Tensor): The samples to denoise.
            bs (int): The number of samples to generate.
        """

        self.noise_scheduler.betas = self.noise_scheduler.betas.to(self.device)
        self.noise_scheduler.alphas = self.noise_scheduler.alphas.to(self.device)
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
            self.device
        )

        # Sanity check sample
        assert (
            sample is not None
        ), "You need to provide a sample to denoise. We will not generate a random one during inference"
        # If a sample volume does not contain many points at all, the model will generate some odd ShapeNet shape
        # It is also a big waste of time to run the model over too much empty space
        test = unnormalize_occupancy_grid(sample)
        if test.sum() < min_points:
            print(
                "Voxel volume contains only {} points, skipping denoising process ...".format(
                    int(test.sum())
                )
            )
            return sample

        if num_inference_steps is None:
            num_inference_steps = self.noise_scheduler.num_inference_steps

        step_size = starting_t // num_inference_steps
        timesteps = torch.arange(starting_t - 1, 0, -step_size).to(self.device)
        # Measure the time with tqdm
        for t in tqdm(timesteps):
            # 1. predict noise model_output
            bs = sample.shape[0]
            t = torch.tensor([t], dtype=torch.long, device=self.device)
            noise_pred = self.model(sample, t, scale=scale).sample
            # 2. compute previous image: x_t -> x_t-1
            sample = self.noise_scheduler.step(noise_pred, t, sample).prev_sample

        return sample

    def convert(self, xyz: torch.Tensor) -> torch.Tensor:
        """Convert a point cloud into an occupancy grid, so that we can remove floaters from it.

        args:
        ---
        xyz (torch.Tensor): The point cloud tensor of shape [B, 3, N]

        returns:
        ---
        voxel_volume (torch.Tensor): The voxel volume of shape [B, C, H, W, D], where H=W=D = resolution of the volume
        """
        voxel_volume, pcl = point_cloud_to_voxel_volume(xyz, voxel_size=0.01)
        voxels = voxel_volume.get_voxels()  # returns list of voxels
        # Now you have the index and color of occupied voxels inside of the grid
        indices = np.stack(list(vx.grid_index for vx in voxels))
        colors = np.stack(list(vx.color for vx in voxels))

        # Create an occupancy grid of shape [B, C, H, W, D] and assign the indices to occupied with value = 1.0
        # NOTE Open3D creates a voxel_volume with an unequal number of bins for each dimension according to the size of the point cloud
        # num_bins = (max_bound - min_bound) / voxel_size
        # They a sparse data structure with a has map so that they can store larger scenes and scale better
        # Open3D uses indexing starting from 0 to the max_idx, i.e. we actually have max_idx + 1 cells
        # TODO if the num_cells grows too big, we need to create the individual chunks on the fly and not the whole array
        non_uniform_voxel_dims = (
            voxel_volume.get_max_bound() - voxel_volume.get_min_bound()
        ) / voxel_volume.voxel_size
        num_cells = int(non_uniform_voxel_dims.max())
        # Get the nearest divisible number by 32
        num_cells = int(np.ceil(num_cells / 32.0) * 32.0)
        assert (
            num_cells < 500
        ), "The number of cells is too big, this will overflow CPU memory"

        # Create empty volume as array
        uniform_volume = torch.zeros(
            (1, 1, num_cells, num_cells, num_cells), dtype=bool
        )
        # Occupy indexed voxel cells from open3d volume
        uniform_volume[
            ..., indices[:, 0] - 1, indices[:, 1] - 1, indices[:, 2] - 1
        ] = 1.0
        return uniform_volume, colors, non_uniform_voxel_dims.astype(int)

    def debug_single_diffusion(
        self, noisy_volume: torch.Tensor, scale: float = 1.0
    ) -> torch.Tensor:
        return self.reverse_process(
            sample=noisy_volume,
            scale=scale,
            num_inference_steps=self.num_inference_steps,
        )

    def forward(self, corrupted_pcl: torch.Tensor) -> torch.Tensor:
        chunk_size = 128  # You get OOM for 24GB GPU with 256
        # PCL -> Occupancy Grid in [0, 1] as binary
        corrupted_voxel_volume, colors, og_bins = self.convert(corrupted_pcl)
        # corrupted_voxel_volume = normalize_occupancy_grid(corrupted_voxel_volume.int())
        # Keep the volume in [0, 1] range
        corrupted_voxel_volume = corrupted_voxel_volume.float()

        # normally we could chunk the volume into smaller volumes with 32 x 32 x 32 size like they use in training
        # OR we just throw the whole thing at it and see what happens
        # -> Immediately OOM, so we need to chunk it up
        chunks = chunk_voxel_volume(corrupted_voxel_volume, chunk_size=chunk_size)

        scale = chunk_size**3 / corrupted_voxel_volume.shape[-1] ** 3
        # NOTE scale is actually just the percentage of the scene, i.e. the cube during training has size [0.01, 0.1] from the original scene
        scale = torch.Tensor([scale]).to(self.device)
        scale = scale.repeat(corrupted_voxel_volume.shape[0])

        # NOTE always use the ddpm scheduler since this was probably used during training
        # DDIM noise somehow creates all these odd artifacts while DDPM does not
        ### Defuse a single chunk to understand the 3D diffusion model and debug
        # test_idx = 13
        # clean_density = self.debug_single_diffusion(
        #     chunks[test_idx].to(self.device), scale
        # ).cpu()
        ### Go over full scene in chunks and reassemble
        cleaned = []
        print("Going over {} sub volumes".format(len(chunks)))
        for chunk in tqdm(chunks):
            decorrupted = self.reverse_process(
                sample=chunk.to(self.device),
                scale=scale,
                num_inference_steps=self.num_inference_steps,
            )
            cleaned.append(decorrupted)
        clean_density = reassemble_voxel_volume(
            cleaned, tuple(corrupted_voxel_volume.shape)
        )

        # After running the Diffusion model, we actually have a density volume like in NeRf, which we need to convert back to an occupancy grid
        clean_volume = self.density_to_x(clean_density, activation="sigmoid")
        # Remove padded dimensions from larger uniform volume and common divisor
        clean_volume = clean_volume[..., : og_bins[0], : og_bins[1], : og_bins[2]]

        ### Visualize the volume before and after the diffusion process
        # test_volume(chunks[test_idx])
        test_volume(corrupted_voxel_volume)
        test_volume(clean_volume)
        ipdb.set_trace()

        # Release memory
        torch.cuda.empty_cache()

        return clean_volume
