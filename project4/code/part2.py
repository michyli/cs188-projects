"""
Part 2: Fit a Neural Radiance Field from Multi-view Images
CS188 Project 4

This script implements the complete NeRF pipeline including:
- Ray generation from camera parameters
- Ray sampling in 3D space
- NeRF network architecture
- Volume rendering
- Training and evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from tqdm import tqdm


# ============================================================================
# Part 2.1: Create Rays from Cameras
# ============================================================================

def transform(c2w, x_c):
    """
    Transform points from camera space to world space.
    
    Args:
        c2w: Camera-to-world transformation matrix
             Shape: (4, 4) or (B, 4, 4) for batched
        x_c: Points in camera space
             Shape: (3,) or (N, 3) or (B, N, 3)
             
    Returns:
        x_w: Points in world space, same shape as x_c
    
    Note:
        The transformation is: x_w = R @ x_c + t
        where c2w = [[R, t], [0, 1]]
    """
    # Handle different input shapes
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).float()
    if isinstance(x_c, np.ndarray):
        x_c = torch.from_numpy(x_c).float()
    
    # Get rotation and translation from c2w
    # c2w shape: (..., 4, 4)
    # R shape: (..., 3, 3)
    # t shape: (..., 3)
    R = c2w[..., :3, :3]
    t = c2w[..., :3, 3]
    
    # Transform: x_w = R @ x_c + t
    # Use einsum for efficient batched matrix multiplication
    # x_c shape: (..., N, 3) or (..., 3)
    if x_c.dim() == 1:
        # Single point: (3,)
        x_w = R @ x_c + t
    elif x_c.dim() == 2:
        # Multiple points: (N, 3)
        x_w = (R @ x_c.T).T + t
    else:
        # Batched points: (B, N, 3) or more dimensions
        # Use einsum: 'bij,b...j->b...i'
        x_w = torch.einsum('...ij,...j->...i', R, x_c) + t
    
    return x_w


def pixel_to_camera(K, uv, s):
    """
    Transform points from pixel coordinates to camera coordinates.
    
    This inverts the projection: s * [u, v, 1]^T = K @ [x_c, y_c, z_c]^T
    So: [x_c, y_c, z_c]^T = s * K^-1 @ [u, v, 1]^T
    
    Args:
        K: Camera intrinsic matrix
           Shape: (3, 3) or (B, 3, 3)
        uv: Pixel coordinates (u, v)
            Shape: (2,) or (N, 2) or (B, N, 2)
        s: Depth values (z_c)
           Shape: scalar, (N,) or (B, N)
           
    Returns:
        x_c: Points in camera space (x_c, y_c, z_c)
             Shape matches input uv with last dim 3
    """
    if isinstance(K, np.ndarray):
        K = torch.from_numpy(K).float()
    if isinstance(uv, np.ndarray):
        uv = torch.from_numpy(uv).float()
    if isinstance(s, (int, float)):
        s = torch.tensor(s).float()
    elif isinstance(s, np.ndarray):
        s = torch.from_numpy(s).float()
    
    # Invert the intrinsic matrix
    K_inv = torch.inverse(K)
    
    # Convert uv to homogeneous coordinates [u, v, 1]
    if uv.dim() == 1:
        # Single point: (2,)
        uv_homo = torch.cat([uv, torch.ones(1)])
        x_c = s * (K_inv @ uv_homo)
    elif uv.dim() == 2:
        # Multiple points: (N, 2)
        ones = torch.ones(uv.shape[0], 1)
        uv_homo = torch.cat([uv, ones], dim=-1)  # (N, 3)
        x_c = s.unsqueeze(-1) * (K_inv @ uv_homo.T).T  # (N, 3)
    else:
        # Batched points: (B, N, 2)
        ones = torch.ones(*uv.shape[:-1], 1)
        uv_homo = torch.cat([uv, ones], dim=-1)  # (B, N, 3)
        # Use einsum for batched multiplication
        x_c = torch.einsum('...ij,...j->...i', K_inv, uv_homo)
        x_c = s.unsqueeze(-1) * x_c
    
    return x_c


def pixel_to_ray(K, c2w, uv):
    """
    Convert pixel coordinates to rays with origin and direction.
    
    Args:
        K: Camera intrinsic matrix, shape (3, 3)
        c2w: Camera-to-world transformation matrix, shape (4, 4)
        uv: Pixel coordinates, shape (2,) or (N, 2)
        
    Returns:
        ray_o: Ray origins in world space, shape (3,) or (N, 3)
        ray_d: Ray directions (normalized) in world space, shape (3,) or (N, 3)
    """
    if isinstance(K, np.ndarray):
        K = torch.from_numpy(K).float()
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).float()
    if isinstance(uv, np.ndarray):
        uv = torch.from_numpy(uv).float()
    
    # Ray origin is the camera position (translation component of c2w)
    ray_o_single = c2w[:3, 3]
    
    # Get point in camera space at depth 1
    s = torch.ones(uv.shape[0] if uv.dim() == 2 else 1)
    x_c = pixel_to_camera(K, uv, s)
    
    # Transform to world space
    x_w = transform(c2w, x_c)
    
    # Ray direction is normalized vector from origin to the point
    if uv.dim() == 1:
        ray_d = x_w - ray_o_single
        ray_o = ray_o_single
    else:
        # Batch case: repeat ray_o for all rays
        ray_d = x_w - ray_o_single.unsqueeze(0)
        ray_o = ray_o_single.unsqueeze(0).expand(uv.shape[0], -1)
    
    # Normalize direction
    ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
    
    return ray_o, ray_d


def get_rays(H, W, K, c2w):
    """
    Generate rays for all pixels in an image.
    
    Args:
        H: Image height
        W: Image width
        K: Camera intrinsic matrix, shape (3, 3)
        c2w: Camera-to-world transformation matrix, shape (4, 4)
        
    Returns:
        rays_o: Ray origins, shape (H, W, 3)
        rays_d: Ray directions, shape (H, W, 3)
    """
    if isinstance(K, np.ndarray):
        K = torch.from_numpy(K).float()
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).float()
    
    # Create pixel coordinate grid
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        indexing='xy'
    )
    
    # Flatten to (H*W, 2)
    uv = torch.stack([i.flatten(), j.flatten()], dim=-1)
    
    # Get rays for all pixels
    rays_o, rays_d = pixel_to_ray(K, c2w, uv)
    
    # Reshape to (H, W, 3)
    rays_o = rays_o.reshape(H, W, 3) if rays_o.dim() == 2 else rays_o.unsqueeze(0).unsqueeze(0).expand(H, W, 3)
    rays_d = rays_d.reshape(H, W, 3)
    
    return rays_o, rays_d


# ============================================================================
# Part 2.2: Sampling
# ============================================================================

def sample_rays_from_images(images, c2ws, K, num_rays, batch_size=None):
    """
    Sample rays from multiple images.
    
    Two sampling strategies:
    1. If batch_size is None: Global sampling across all images
    2. If batch_size is set: Sample batch_size images, then sample rays per image
    
    Args:
        images: Training images, shape (N_images, H, W, 3)
        c2ws: Camera-to-world matrices, shape (N_images, 4, 4)
        K: Camera intrinsic matrix, shape (3, 3)
        num_rays: Total number of rays to sample
        batch_size: Number of images to sample (None for global sampling)
        
    Returns:
        rays_o: Ray origins, shape (num_rays, 3)
        rays_d: Ray directions, shape (num_rays, 3)
        rgb_gt: Ground truth RGB colors, shape (num_rays, 3)
    """
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images).float()
    if isinstance(c2ws, np.ndarray):
        c2ws = torch.from_numpy(c2ws).float()
    if isinstance(K, np.ndarray):
        K = torch.from_numpy(K).float()
    
    N_images, H, W, _ = images.shape
    
    if batch_size is None:
        # Option 1: Global sampling across all images
        # Flatten all pixels from all images
        total_pixels = N_images * H * W
        
        # Sample random pixel indices
        indices = torch.randperm(total_pixels)[:num_rays]
        
        # Convert flat indices to (image_idx, y, x)
        img_indices = indices // (H * W)
        pixel_indices = indices % (H * W)
        y_indices = pixel_indices // W
        x_indices = pixel_indices % W
        
        # Add 0.5 offset for pixel center
        uv = torch.stack([x_indices.float() + 0.5, y_indices.float() + 0.5], dim=-1)
        
        # Get rays for sampled pixels
        rays_o_list = []
        rays_d_list = []
        rgb_gt_list = []
        
        for i in range(num_rays):
            img_idx = img_indices[i]
            ray_o, ray_d = pixel_to_ray(K, c2ws[img_idx], uv[i])
            rays_o_list.append(ray_o)
            rays_d_list.append(ray_d)
            rgb_gt_list.append(images[img_idx, y_indices[i], x_indices[i]])
        
        rays_o = torch.stack(rays_o_list)
        rays_d = torch.stack(rays_d_list)
        rgb_gt = torch.stack(rgb_gt_list)
        
    else:
        # Option 2: Sample images first, then rays per image
        # Sample random images
        img_indices = torch.randint(0, N_images, (batch_size,))
        rays_per_image = num_rays // batch_size
        
        rays_o_list = []
        rays_d_list = []
        rgb_gt_list = []
        
        for img_idx in img_indices:
            # Convert to Python int for proper indexing
            img_idx_int = img_idx.item()
            
            # Sample random pixels from this image
            pixel_indices = torch.randperm(H * W)[:rays_per_image]
            y_indices = pixel_indices // W
            x_indices = pixel_indices % W
            
            # Add 0.5 offset for pixel center
            uv = torch.stack([x_indices.float() + 0.5, y_indices.float() + 0.5], dim=-1)
            
            # Get rays
            ray_o, ray_d = pixel_to_ray(K, c2ws[img_idx_int], uv)
            
            rays_o_list.append(ray_o)
            rays_d_list.append(ray_d)
            rgb_gt_list.append(images[img_idx_int, y_indices, x_indices])
        
        rays_o = torch.cat(rays_o_list, dim=0)
        rays_d = torch.cat(rays_d_list, dim=0)
        rgb_gt = torch.cat(rgb_gt_list, dim=0)
    
    return rays_o, rays_d, rgb_gt


def sample_points_along_rays(rays_o, rays_d, near, far, n_samples, perturb=True):
    """
    Sample points along rays between near and far bounds.
    
    Args:
        rays_o: Ray origins, shape (N_rays, 3)
        rays_d: Ray directions, shape (N_rays, 3)
        near: Near clipping distance (scalar)
        far: Far clipping distance (scalar)
        n_samples: Number of samples per ray
        perturb: Whether to add random perturbation (True for training)
        
    Returns:
        points: 3D sample points, shape (N_rays, n_samples, 3)
        t_vals: Depth values along rays, shape (N_rays, n_samples)
    """
    if isinstance(rays_o, np.ndarray):
        rays_o = torch.from_numpy(rays_o).float()
    if isinstance(rays_d, np.ndarray):
        rays_d = torch.from_numpy(rays_d).float()
    
    N_rays = rays_o.shape[0]
    
    # Create linearly spaced depth values on the same device as input rays
    device = rays_o.device
    t_vals = torch.linspace(near, far, n_samples, device=device)
    t_vals = t_vals.unsqueeze(0).expand(N_rays, n_samples)  # (N_rays, n_samples)
    
    if perturb:
        # Add random perturbation during training
        # Get interval width
        t_width = (far - near) / n_samples
        
        # Add random offset within each interval
        # This ensures we sample different points along the ray each iteration
        t_vals = t_vals + torch.rand_like(t_vals) * t_width
    
    # Compute 3D points: x = r_o + r_d * t
    # rays_o: (N_rays, 3)
    # rays_d: (N_rays, 3)
    # t_vals: (N_rays, n_samples)
    # points: (N_rays, n_samples, 3)
    points = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t_vals.unsqueeze(2)
    
    return points, t_vals


def get_all_rays_and_pixels(images, c2ws, K):
    """
    Generate all rays and corresponding pixel colors for all images.
    Useful for validation/test set evaluation.
    
    Args:
        images: Images, shape (N_images, H, W, 3)
        c2ws: Camera-to-world matrices, shape (N_images, 4, 4)
        K: Camera intrinsic matrix, shape (3, 3)
        
    Returns:
        all_rays_o: All ray origins, shape (N_images, H, W, 3)
        all_rays_d: All ray directions, shape (N_images, H, W, 3)
        all_pixels: All pixel colors, shape (N_images, H, W, 3)
    """
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images).float()
    if isinstance(c2ws, np.ndarray):
        c2ws = torch.from_numpy(c2ws).float()
    if isinstance(K, np.ndarray):
        K = torch.from_numpy(K).float()
    
    N_images, H, W, _ = images.shape
    
    all_rays_o = []
    all_rays_d = []
    
    for i in range(N_images):
        rays_o, rays_d = get_rays(H, W, K, c2ws[i])
        all_rays_o.append(rays_o)
        all_rays_d.append(rays_d)
    
    all_rays_o = torch.stack(all_rays_o)  # (N_images, H, W, 3)
    all_rays_d = torch.stack(all_rays_d)  # (N_images, H, W, 3)
    all_pixels = images  # (N_images, H, W, 3)
    
    return all_rays_o, all_rays_d, all_pixels


# ============================================================================
# Part 2.3: Comprehensive Dataloader
# ============================================================================

class RaysData:
    """
    Comprehensive dataloader for NeRF training.
    Precomputes all rays and pixels from multi-view images for efficient sampling.
    
    This dataloader:
    1. Precomputes all rays (origins and directions) for all pixels in all images
    2. Flattens everything for easy random sampling
    3. Stores UV coordinates for verification
    4. Provides efficient random sampling during training
    """
    
    def __init__(self, images, K, c2ws):
        """
        Initialize the dataloader by precomputing all rays.
        
        Args:
            images: Training images, shape (N_images, H, W, 3), range [0, 1]
            K: Camera intrinsic matrix, shape (3, 3)
            c2ws: Camera-to-world matrices, shape (N_images, 4, 4)
        """
        print("Initializing RaysData loader...")
        print(f"  Images shape: {images.shape}")
        
        # Convert to numpy for easier manipulation
        if isinstance(images, torch.Tensor):
            images = images.numpy()
        if isinstance(K, torch.Tensor):
            K = K.numpy()
        if isinstance(c2ws, torch.Tensor):
            c2ws = c2ws.numpy()
        
        self.images = images
        self.K = K
        self.c2ws = c2ws
        
        N_images, H, W, _ = images.shape
        self.N_images = N_images
        self.H = H
        self.W = W
        
        # Precompute all rays and pixels
        print("  Precomputing all rays...")
        all_rays_o = []
        all_rays_d = []
        all_pixels = []
        all_uvs = []
        
        for img_idx in range(N_images):
            # Generate all rays for this image
            rays_o, rays_d = get_rays(H, W, K, c2ws[img_idx])
            
            # Convert to numpy if needed
            if isinstance(rays_o, torch.Tensor):
                rays_o = rays_o.numpy()
            if isinstance(rays_d, torch.Tensor):
                rays_d = rays_d.numpy()
            
            # Flatten spatial dimensions: (H, W, 3) -> (H*W, 3)
            rays_o_flat = rays_o.reshape(-1, 3)
            rays_d_flat = rays_d.reshape(-1, 3)
            pixels_flat = images[img_idx].reshape(-1, 3)
            
            # Generate UV coordinates (x, y format, NOT y, x)
            # Shape: (H*W, 2) where each row is [x, y]
            u_coords = np.arange(W)
            v_coords = np.arange(H)
            u_grid, v_grid = np.meshgrid(u_coords, v_coords)
            uvs_flat = np.stack([u_grid.flatten(), v_grid.flatten()], axis=-1)  # (H*W, 2)
            
            all_rays_o.append(rays_o_flat)
            all_rays_d.append(rays_d_flat)
            all_pixels.append(pixels_flat)
            all_uvs.append(uvs_flat)
        
        # Concatenate all images: (N_images * H * W, 3)
        self.rays_o = np.concatenate(all_rays_o, axis=0)
        self.rays_d = np.concatenate(all_rays_d, axis=0)
        self.pixels = np.concatenate(all_pixels, axis=0)
        self.uvs = np.concatenate(all_uvs, axis=0).astype(np.int32)
        
        self.total_rays = self.rays_o.shape[0]
        
        print(f"  Total rays: {self.total_rays}")
        print(f"  rays_o shape: {self.rays_o.shape}")
        print(f"  rays_d shape: {self.rays_d.shape}")
        print(f"  pixels shape: {self.pixels.shape}")
        print(f"  uvs shape: {self.uvs.shape}")
        print("RaysData initialization complete!")
    
    def sample_rays(self, num_rays):
        """
        Randomly sample rays for training.
        
        Args:
            num_rays: Number of rays to sample
            
        Returns:
            rays_o: Ray origins, shape (num_rays, 3)
            rays_d: Ray directions, shape (num_rays, 3)
            pixels: Ground truth pixel colors, shape (num_rays, 3)
        """
        # Random sampling without replacement
        indices = np.random.choice(self.total_rays, size=num_rays, replace=False)
        
        rays_o = self.rays_o[indices]
        rays_d = self.rays_d[indices]
        pixels = self.pixels[indices]
        
        return rays_o, rays_d, pixels
    
    def get_rays_from_image(self, img_idx):
        """
        Get all rays from a specific image.
        
        Args:
            img_idx: Image index
            
        Returns:
            rays_o: Ray origins for this image, shape (H*W, 3)
            rays_d: Ray directions for this image, shape (H*W, 3)
            pixels: Pixel colors for this image, shape (H*W, 3)
        """
        start_idx = img_idx * (self.H * self.W)
        end_idx = start_idx + (self.H * self.W)
        
        return self.rays_o[start_idx:end_idx], \
               self.rays_d[start_idx:end_idx], \
               self.pixels[start_idx:end_idx]
    
    def __len__(self):
        """Return total number of rays."""
        return self.total_rays
    
    def __getitem__(self, idx):
        """
        Get ray and pixel by index.
        
        Args:
            idx: Index or array of indices
            
        Returns:
            Dictionary with 'rays_o', 'rays_d', 'pixels'
        """
        return {
            'rays_o': self.rays_o[idx],
            'rays_d': self.rays_d[idx],
            'pixels': self.pixels[idx]
        }


# ============================================================================
# Part 2.4: Neural Radiance Field Network
# ============================================================================

class PositionalEncoding(torch.nn.Module):
    """
    Sinusoidal positional encoding for mapping coordinates to higher dimensions.
    PE(x) = [x, sin(2^0*pi*x), cos(2^0*pi*x), ..., sin(2^(L-1)*pi*x), cos(2^(L-1)*pi*x)]
    """
    
    def __init__(self, L):
        """
        Args:
            L: Maximum frequency level
        """
        super().__init__()
        self.L = L
    
    def forward(self, x):
        """
        Apply positional encoding to input coordinates.
        
        Args:
            x: Input coordinates, shape (..., D) where D is the dimension (2 or 3)
            
        Returns:
            Encoded coordinates, shape (..., D * (2*L + 1))
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Start with the original input
        encoded = [x]
        
        # Apply sinusoidal encoding for each frequency
        for i in range(self.L):
            freq = 2 ** i
            encoded.append(torch.sin(freq * np.pi * x))
            encoded.append(torch.cos(freq * np.pi * x))
        
        # Concatenate all encodings
        return torch.cat(encoded, dim=-1)
    
    def output_dim(self, input_dim):
        """Calculate output dimension after encoding."""
        return input_dim * (2 * self.L + 1)


class NeRF(torch.nn.Module):
    """
    Neural Radiance Field (NeRF) network.
    
    Architecture:
    1. Position encoding with high frequency (L=10)
    2. 8 layers of Linear(256) with ReLU activation
    3. Skip connection at layer 4 (concatenate original PE(x))
    4. Two output heads:
       - Density head: Linear(1) + ReLU
       - Color head: Concat with PE(ray_d) + Linear(256) + Linear(128) + Linear(3) + Sigmoid
    """
    
    def __init__(self, pos_L=10, dir_L=4, hidden_dim=256):
        """
        Args:
            pos_L: Positional encoding frequency for 3D coordinates (default: 10)
            dir_L: Positional encoding frequency for ray directions (default: 4)
            hidden_dim: Hidden layer dimension (default: 256)
        """
        super().__init__()
        
        self.pos_L = pos_L
        self.dir_L = dir_L
        self.hidden_dim = hidden_dim
        
        # Positional encoding for 3D coordinates
        self.pos_encoder = PositionalEncoding(pos_L)
        pos_encoded_dim = self.pos_encoder.output_dim(3)  # 3D input
        
        # Positional encoding for 3D ray directions
        self.dir_encoder = PositionalEncoding(dir_L)
        dir_encoded_dim = self.dir_encoder.output_dim(3)  # 3D input
        
        # First 4 layers (before skip connection)
        self.layers_before_skip = torch.nn.ModuleList([
            torch.nn.Linear(pos_encoded_dim, hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
        ])
        
        # Layer after skip connection (input is hidden_dim + pos_encoded_dim due to concatenation)
        self.layer_after_skip = torch.nn.Linear(hidden_dim + pos_encoded_dim, hidden_dim)
        
        # Remaining layers
        self.layers_after = torch.nn.ModuleList([
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
        ])
        
        # Density head
        self.density_head = torch.nn.Linear(hidden_dim, 1)
        
        # Color head
        # Input: hidden features + encoded ray direction
        self.color_layer1 = torch.nn.Linear(hidden_dim + dir_encoded_dim, hidden_dim)
        self.color_layer2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.color_layer3 = torch.nn.Linear(hidden_dim // 2, 3)
        
        # Activations
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x, ray_d):
        """
        Forward pass through NeRF network.
        
        Args:
            x: 3D world coordinates, shape (..., 3)
            ray_d: Ray directions, shape (..., 3)
            
        Returns:
            density: Volume density, shape (..., 1)
            rgb: RGB color, shape (..., 3)
        """
        # Encode inputs
        x_encoded = self.pos_encoder(x)  # (..., 3*(2*L+1))
        ray_d_encoded = self.dir_encoder(ray_d)  # (..., 3*(2*L'+1))
        
        # Store original encoding for skip connection
        x_input = x_encoded
        
        # First 4 layers
        h = x_encoded
        for layer in self.layers_before_skip:
            h = self.relu(layer(h))
        
        # Skip connection: concatenate with original input
        h = torch.cat([h, x_input], dim=-1)
        
        # Layer after skip
        h = self.relu(self.layer_after_skip(h))
        
        # Remaining layers
        for layer in self.layers_after:
            h = self.relu(layer(h))
        
        # Density head
        density = self.relu(self.density_head(h))  # (..., 1)
        
        # Color head
        # Concatenate features with encoded ray direction
        h_color = torch.cat([h, ray_d_encoded], dim=-1)
        h_color = self.relu(self.color_layer1(h_color))
        h_color = self.relu(self.color_layer2(h_color))
        rgb = self.sigmoid(self.color_layer3(h_color))  # (..., 3)
        
        return density, rgb
    
    def forward_density_only(self, x):
        """
        Forward pass for density only (useful for some applications).
        
        Args:
            x: 3D world coordinates, shape (..., 3)
            
        Returns:
            density: Volume density, shape (..., 1)
        """
        # Encode inputs
        x_encoded = self.pos_encoder(x)
        x_input = x_encoded
        
        # First 4 layers
        h = x_encoded
        for layer in self.layers_before_skip:
            h = self.relu(layer(h))
        
        # Skip connection
        h = torch.cat([h, x_input], dim=-1)
        h = self.relu(self.layer_after_skip(h))
        
        # Remaining layers
        for layer in self.layers_after:
            h = self.relu(layer(h))
        
        # Density head
        density = self.relu(self.density_head(h))
        
        return density


# ============================================================================
# Part 2.5: Volume Rendering
# ============================================================================

def volrend(sigmas, rgbs, step_size):
    """
    Volume rendering using the discrete approximation of the volume rendering equation.
    
    The continuous form:
    C(r) = ∫ T(t)σ(r(t))c(r(t),d)dt, where T(t) = exp(-∫ σ(r(s))ds)
    
    The discrete approximation:
    Ĉ(r) = Σ T_i(1 - exp(-σ_i·δ_i))c_i, where T_i = exp(-Σ σ_j·δ_j)
    
    Args:
        sigmas: Density values, shape (N_rays, N_samples, 1)
        rgbs: RGB colors, shape (N_rays, N_samples, 3)
        step_size: Distance between samples (δ_i), scalar or tensor
        
    Returns:
        rendered_colors: Final rendered colors, shape (N_rays, 3)
    """
    # Compute alpha values: α_i = 1 - exp(-σ_i·δ_i)
    # This represents the probability of ray terminating at sample i
    alphas = 1.0 - torch.exp(-sigmas * step_size)  # (N_rays, N_samples, 1)
    
    # Compute transmittance T_i = exp(-Σ[j=1 to i-1] σ_j·δ_j)
    # T_i represents the probability of ray NOT terminating before sample i
    # Using the approximation: T_i ≈ Π[j=1 to i-1] (1 - α_j)
    
    one_minus_alpha = 1.0 - alphas  # (N_rays, N_samples, 1)
    
    # For transmittance:
    # T_0 = 1 (no attenuation before first sample)
    # T_1 = (1 - α_0)
    # T_2 = (1 - α_0) * (1 - α_1)
    # T_i = Π[j=0 to i-1] (1 - α_j)
    
    # We use cumprod to compute cumulative products
    # Prepend 1 for the first sample, compute cumprod on [:-1] to get products up to i-1
    transmittance = torch.cat([
        torch.ones_like(alphas[:, :1, :]),  # T_0 = 1
        torch.cumprod(one_minus_alpha[:, :-1, :], dim=1)  # T_i for i > 0
    ], dim=1)  # (N_rays, N_samples, 1)
    
    # Compute weights: w_i = T_i * α_i
    # This is the contribution of sample i to the final color
    weights = transmittance * alphas  # (N_rays, N_samples, 1)
    
    # Compute final rendered color: C = Σ w_i * c_i
    rendered_colors = torch.sum(weights * rgbs, dim=1)  # (N_rays, 3)
    
    return rendered_colors


def render_rays(nerf, rays_o, rays_d, near=2.0, far=6.0, n_samples=64, perturb=True):
    """
    Render rays using NeRF network and volume rendering.
    
    Args:
        nerf: NeRF network
        rays_o: Ray origins, shape (N_rays, 3)
        rays_d: Ray directions, shape (N_rays, 3)
        near: Near clipping distance
        far: Far clipping distance
        n_samples: Number of samples per ray
        perturb: Whether to add random perturbation (True for training)
        
    Returns:
        rendered_colors: Rendered RGB colors, shape (N_rays, 3)
    """
    # Sample points along rays
    points, t_vals = sample_points_along_rays(rays_o, rays_d, near, far, n_samples, perturb)
    # points: (N_rays, n_samples, 3)
    
    # Expand ray directions to match points shape
    rays_d_expanded = rays_d.unsqueeze(1).expand(-1, n_samples, -1)  # (N_rays, n_samples, 3)
    
    # Query network for density and color
    sigmas, rgbs = nerf(points, rays_d_expanded)
    # sigmas: (N_rays, n_samples, 1)
    # rgbs: (N_rays, n_samples, 3)
    
    # Compute step size
    step_size = (far - near) / n_samples
    
    # Perform volume rendering
    rendered_colors = volrend(sigmas, rgbs, step_size)
    
    return rendered_colors


# ============================================================================
# Verification and Testing Functions
# ============================================================================

def verify_transform():
    """
    Verify that the transform function works correctly.
    Tests: x == transform(c2w.inv(), transform(c2w, x))
    """
    print("="*60)
    print("Verifying transform function...")
    print("="*60)
    
    # Create a random c2w matrix
    c2w = torch.eye(4)
    c2w[:3, :3] = torch.randn(3, 3)  # Random rotation (not orthonormal, just for testing)
    c2w[:3, 3] = torch.randn(3)  # Random translation
    
    # Create random points
    x_c = torch.randn(10, 3)
    
    # Forward and backward transform
    x_w = transform(c2w, x_c)
    w2c = torch.inverse(c2w)
    x_c_recovered = transform(w2c, x_w)
    
    # Check if they match
    error = torch.abs(x_c - x_c_recovered).max().item()
    print(f"Max error: {error:.6e}")
    
    if error < 1e-5:
        print("[PASS] Transform verification PASSED")
    else:
        print("[FAIL] Transform verification FAILED")
    
    return error < 1e-5


def test_pixel_to_camera():
    """
    Test pixel_to_camera function with known values.
    """
    print("\n" + "="*60)
    print("Testing pixel_to_camera function...")
    print("="*60)
    
    # Create a simple intrinsic matrix
    focal = 100.0
    W, H = 200, 200
    K = torch.tensor([
        [focal, 0, W/2],
        [0, focal, H/2],
        [0, 0, 1]
    ])
    
    # Test center pixel at depth 1
    uv_center = torch.tensor([W/2, H/2])
    s = 1.0
    x_c = pixel_to_camera(K, uv_center, s)
    
    print(f"Center pixel {uv_center.numpy()} at depth {s}:")
    print(f"Camera coordinates: {x_c.numpy()}")
    print(f"Expected: [0, 0, 1]")
    
    # Should be approximately [0, 0, 1]
    expected = torch.tensor([0., 0., 1.])
    error = torch.abs(x_c - expected).max().item()
    
    if error < 1e-5:
        print("[PASS] pixel_to_camera test PASSED")
    else:
        print("[FAIL] pixel_to_camera test FAILED")
    
    return error < 1e-5


def test_pixel_to_ray():
    """
    Test pixel_to_ray function.
    """
    print("\n" + "="*60)
    print("Testing pixel_to_ray function...")
    print("="*60)
    
    # Create intrinsic matrix
    focal = 100.0
    W, H = 200, 200
    K = torch.tensor([
        [focal, 0, W/2],
        [0, focal, H/2],
        [0, 0, 1]
    ])
    
    # Identity c2w (camera at origin, looking down +Z)
    c2w = torch.eye(4)
    
    # Test center pixel
    uv_center = torch.tensor([W/2, H/2])
    ray_o, ray_d = pixel_to_ray(K, c2w, uv_center)
    
    print(f"Center pixel ray:")
    print(f"Origin: {ray_o.numpy()}")
    print(f"Direction: {ray_d.numpy()}")
    print(f"Expected origin: [0, 0, 0]")
    print(f"Expected direction: [0, 0, 1]")
    
    # Ray should point down +Z axis
    expected_o = torch.tensor([0., 0., 0.])
    expected_d = torch.tensor([0., 0., 1.])
    
    error_o = torch.abs(ray_o - expected_o).max().item()
    error_d = torch.abs(ray_d - expected_d).max().item()
    
    if error_o < 1e-5 and error_d < 1e-5:
        print("[PASS] pixel_to_ray test PASSED")
    else:
        print("[FAIL] pixel_to_ray test FAILED")
    
    return error_o < 1e-5 and error_d < 1e-5


def visualize_rays(rays_o, rays_d, num_rays=100, save_path=None):
    """
    Visualize a subset of rays in 3D.
    
    Args:
        rays_o: Ray origins, shape (H, W, 3) or (N, 3)
        rays_d: Ray directions, shape (H, W, 3) or (N, 3)
        num_rays: Number of rays to visualize
        save_path: Path to save the figure
    """
    # Flatten if needed
    if rays_o.dim() == 3:
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
    
    # Sample random rays
    indices = torch.randperm(rays_o.shape[0])[:num_rays]
    rays_o_sample = rays_o[indices].numpy()
    rays_d_sample = rays_d[indices].numpy()
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot rays
    for i in range(num_rays):
        o = rays_o_sample[i]
        d = rays_d_sample[i]
        # Plot ray from origin to origin + direction
        ax.plot([o[0], o[0] + d[0]*0.5],
                [o[1], o[1] + d[1]*0.5],
                [o[2], o[2] + d[2]*0.5],
                'b-', alpha=0.3, linewidth=0.5)
    
    # Plot camera position
    ax.scatter([0], [0], [0], c='r', marker='o', s=100, label='Camera')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Ray Visualization ({num_rays} rays)')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Ray visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main_part2_1():
    """
    Main function to test Part 2.1 implementation.
    """
    print("\n" + "="*60)
    print("PART 2.1: CREATE RAYS FROM CAMERAS")
    print("="*60)
    
    # Run verification tests
    test_results = []
    test_results.append(("Transform", verify_transform()))
    test_results.append(("Pixel to Camera", test_pixel_to_camera()))
    test_results.append(("Pixel to Ray", test_pixel_to_ray()))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, passed in test_results:
        status = "[PASS] PASSED" if passed else "[FAIL] FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(result[1] for result in test_results)
    if all_passed:
        print("\n[PASS] All tests passed!")
    else:
        print("\n[FAIL] Some tests failed. Please check the implementation.")
    
    # Generate and visualize rays for a sample image
    print("\n" + "="*60)
    print("Generating rays for sample image...")
    print("="*60)
    
    H, W = 200, 200
    focal = 100.0
    K = torch.tensor([
        [focal, 0, W/2],
        [0, focal, H/2],
        [0, 0, 1]
    ])
    
    # Camera at origin looking down +Z
    c2w = torch.eye(4)
    
    rays_o, rays_d = get_rays(H, W, K, c2w)
    print(f"Generated rays:")
    print(f"  Origins shape: {rays_o.shape}")
    print(f"  Directions shape: {rays_d.shape}")
    
    # Visualize
    os.makedirs('project4/assets/part_2', exist_ok=True)
    visualize_rays(rays_o, rays_d, num_rays=100, 
                   save_path='project4/assets/part_2/ray_visualization.png')
    
    return all_passed


def test_part2_2():
    """Test Part 2.2: Sampling functions"""
    print("\n" + "="*60)
    print("PART 2.2: SAMPLING")
    print("="*60)
    
    # Create dummy data
    N_images = 3
    H, W = 50, 50
    images = np.random.rand(N_images, H, W, 3)
    c2ws = np.array([np.eye(4) for _ in range(N_images)])
    c2ws[:, :3, 3] = np.random.randn(N_images, 3)  # Random positions
    K = np.array([[100, 0, W/2], [0, 100, H/2], [0, 0, 1]])
    
    print("\nTest 1: Global sampling")
    rays_o, rays_d, rgb_gt = sample_rays_from_images(images, c2ws, K, num_rays=100)
    print(f"  Sampled rays_o shape: {rays_o.shape}")
    print(f"  Sampled rays_d shape: {rays_d.shape}")
    print(f"  Sampled rgb_gt shape: {rgb_gt.shape}")
    assert rays_o.shape == (100, 3), f"Expected (100, 3), got {rays_o.shape}"
    assert rays_d.shape == (100, 3), f"Expected (100, 3), got {rays_d.shape}"
    assert rgb_gt.shape == (100, 3), f"Expected (100, 3), got {rgb_gt.shape}"
    print("  [PASS] Global sampling test passed!")
    
    print("\nTest 2: Batch sampling")
    rays_o, rays_d, rgb_gt = sample_rays_from_images(images, c2ws, K, num_rays=120, batch_size=3)
    print(f"  Sampled rays_o shape: {rays_o.shape}")
    print(f"  Sampled rays_d shape: {rays_d.shape}")
    print(f"  Sampled rgb_gt shape: {rgb_gt.shape}")
    assert rays_o.shape == (120, 3), f"Expected (120, 3), got {rays_o.shape}"
    print("  [PASS] Batch sampling test passed!")
    
    print("\nTest 3: Sample points along rays (no perturbation)")
    rays_o_test = np.array([[0, 0, 0], [1, 1, 1]])
    rays_d_test = np.array([[0, 0, 1], [0, 0, 1]])
    points, t_vals = sample_points_along_rays(rays_o_test, rays_d_test, near=2.0, far=6.0, n_samples=5, perturb=False)
    print(f"  Points shape: {points.shape}")
    print(f"  t_vals shape: {t_vals.shape}")
    print(f"  t_vals for ray 0: {t_vals[0].numpy()}")
    assert points.shape == (2, 5, 3), f"Expected (2, 5, 3), got {points.shape}"
    assert torch.allclose(t_vals[0], torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0])), "t_vals should be linearly spaced"
    print("  [PASS] Points sampling test passed!")
    
    print("\nTest 4: Sample points along rays (with perturbation)")
    points_perturb, t_vals_perturb = sample_points_along_rays(rays_o_test, rays_d_test, near=2.0, far=6.0, n_samples=5, perturb=True)
    print(f"  Points shape: {points_perturb.shape}")
    print(f"  t_vals with perturbation: {t_vals_perturb[0].numpy()}")
    assert not torch.allclose(t_vals[0], t_vals_perturb[0]), "Perturbed t_vals should be different"
    print("  [PASS] Perturbation test passed!")
    
    return True


def test_part2_3():
    """Test Part 2.3: RaysData dataloader"""
    print("\n" + "="*60)
    print("PART 2.3: RAYSDATA DATALOADER")
    print("="*60)
    
    # Create dummy data
    N_images = 3
    H, W = 50, 50
    images = np.random.rand(N_images, H, W, 3)
    c2ws = np.array([np.eye(4) for _ in range(N_images)])
    c2ws[:, :3, 3] = np.random.randn(N_images, 3)
    K = np.array([[100, 0, W/2], [0, 100, H/2], [0, 0, 1]])
    
    print("\nTest 1: RaysData initialization")
    dataset = RaysData(images, K, c2ws)
    expected_total = N_images * H * W
    assert dataset.total_rays == expected_total, f"Expected {expected_total} rays, got {dataset.total_rays}"
    assert dataset.rays_o.shape == (expected_total, 3), f"rays_o shape mismatch"
    assert dataset.rays_d.shape == (expected_total, 3), f"rays_d shape mismatch"
    assert dataset.pixels.shape == (expected_total, 3), f"pixels shape mismatch"
    assert dataset.uvs.shape == (expected_total, 2), f"uvs shape mismatch"
    print("  [PASS] Initialization test passed!")
    
    print("\nTest 2: UV coordinate verification")
    # Check that uvs are in (x, y) format and pixels match
    uvs_start = 0
    uvs_end = min(1000, H * W)
    sample_uvs = dataset.uvs[uvs_start:uvs_end]
    # uvs are (x, y), so we index with [y, x] which is [uvs[:,1], uvs[:,0]]
    expected_pixels = images[0, sample_uvs[:,1], sample_uvs[:,0]]
    actual_pixels = dataset.pixels[uvs_start:uvs_end]
    assert np.allclose(expected_pixels, actual_pixels), "UV indexing mismatch!"
    print(f"  Verified {uvs_end - uvs_start} pixels match UV coordinates")
    print("  [PASS] UV coordinate test passed!")
    
    print("\nTest 3: Sample rays")
    rays_o, rays_d, pixels = dataset.sample_rays(100)
    assert rays_o.shape == (100, 3), f"Expected (100, 3), got {rays_o.shape}"
    assert rays_d.shape == (100, 3), f"Expected (100, 3), got {rays_d.shape}"
    assert pixels.shape == (100, 3), f"Expected (100, 3), got {pixels.shape}"
    print(f"  Sampled 100 rays successfully")
    print("  [PASS] Sample rays test passed!")
    
    print("\nTest 4: Get rays from specific image")
    rays_o_img, rays_d_img, pixels_img = dataset.get_rays_from_image(0)
    assert rays_o_img.shape == (H * W, 3), f"Expected ({H * W}, 3), got {rays_o_img.shape}"
    print(f"  Retrieved all rays from image 0")
    print("  [PASS] Get rays from image test passed!")
    
    print("\nTest 5: Indexing")
    indices = [0, 10, 100]
    data = dataset[indices]
    assert data['rays_o'].shape == (3, 3), "Indexing failed for rays_o"
    assert data['rays_d'].shape == (3, 3), "Indexing failed for rays_d"
    assert data['pixels'].shape == (3, 3), "Indexing failed for pixels"
    print("  [PASS] Indexing test passed!")
    
    return True


def test_part2_4():
    """Test Part 2.4: NeRF Network"""
    print("\n" + "="*60)
    print("PART 2.4: NERF NETWORK")
    print("="*60)
    
    print("\nTest 1: Positional Encoding")
    pe = PositionalEncoding(L=10)
    x = torch.randn(5, 3)  # 5 points in 3D
    x_encoded = pe(x)
    expected_dim = 3 * (2 * 10 + 1)  # 3 * 21 = 63
    print(f"  Input shape: {x.shape}")
    print(f"  Encoded shape: {x_encoded.shape}")
    print(f"  Expected encoded dim: {expected_dim}")
    assert x_encoded.shape == (5, expected_dim), f"Expected {(5, expected_dim)}, got {x_encoded.shape}"
    print("  [PASS] Positional encoding test passed!")
    
    print("\nTest 2: NeRF Network Initialization")
    nerf = NeRF(pos_L=10, dir_L=4, hidden_dim=256)
    print(f"  Position encoding L: {nerf.pos_L}")
    print(f"  Direction encoding L: {nerf.dir_L}")
    print(f"  Hidden dimension: {nerf.hidden_dim}")
    print(f"  Number of parameters: {sum(p.numel() for p in nerf.parameters()):,}")
    print("  [PASS] Network initialization test passed!")
    
    print("\nTest 3: Forward Pass (Single Point)")
    x = torch.randn(3)  # Single 3D point
    ray_d = torch.randn(3)  # Single ray direction
    ray_d = ray_d / torch.norm(ray_d)  # Normalize
    
    density, rgb = nerf(x, ray_d)
    print(f"  Input x shape: {x.shape}")
    print(f"  Input ray_d shape: {ray_d.shape}")
    print(f"  Output density shape: {density.shape}")
    print(f"  Output rgb shape: {rgb.shape}")
    print(f"  Density value: {density.squeeze().item():.6f}")
    print(f"  RGB values: [{rgb[0, 0].item():.4f}, {rgb[0, 1].item():.4f}, {rgb[0, 2].item():.4f}]")
    
    # For single point, positional encoding adds a batch dimension
    assert density.shape == (1, 1), f"Expected (1, 1), got {density.shape}"
    assert rgb.shape == (1, 3), f"Expected (1, 3), got {rgb.shape}"
    assert density.squeeze().item() >= 0, "Density should be non-negative (ReLU)"
    assert torch.all((rgb >= 0) & (rgb <= 1)), "RGB should be in [0, 1] (Sigmoid)"
    print("  [PASS] Single point forward pass test passed!")
    
    print("\nTest 4: Forward Pass (Batch)")
    batch_size = 10
    n_samples = 32
    x_batch = torch.randn(batch_size, n_samples, 3)  # 10 rays, 32 samples each
    ray_d_batch = torch.randn(batch_size, n_samples, 3)
    ray_d_batch = ray_d_batch / torch.norm(ray_d_batch, dim=-1, keepdim=True)
    
    density_batch, rgb_batch = nerf(x_batch, ray_d_batch)
    print(f"  Input x shape: {x_batch.shape}")
    print(f"  Input ray_d shape: {ray_d_batch.shape}")
    print(f"  Output density shape: {density_batch.shape}")
    print(f"  Output rgb shape: {rgb_batch.shape}")
    
    assert density_batch.shape == (batch_size, n_samples, 1), f"Expected {(batch_size, n_samples, 1)}, got {density_batch.shape}"
    assert rgb_batch.shape == (batch_size, n_samples, 3), f"Expected {(batch_size, n_samples, 3)}, got {rgb_batch.shape}"
    assert torch.all(density_batch >= 0), "All densities should be non-negative"
    assert torch.all((rgb_batch >= 0) & (rgb_batch <= 1)), "All RGB values should be in [0, 1]"
    print("  [PASS] Batch forward pass test passed!")
    
    print("\nTest 5: Forward Density Only")
    x = torch.randn(5, 3)
    density_only = nerf.forward_density_only(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Density output shape: {density_only.shape}")
    assert density_only.shape == (5, 1), f"Expected (5, 1), got {density_only.shape}"
    assert torch.all(density_only >= 0), "All densities should be non-negative"
    print("  [PASS] Density only forward pass test passed!")
    
    print("\nTest 6: Gradient Flow")
    nerf.train()
    x = torch.randn(3, requires_grad=True)
    ray_d = torch.randn(3)
    ray_d = ray_d / torch.norm(ray_d)
    
    density, rgb = nerf(x, ray_d)
    loss = density.sum() + rgb.sum()
    loss.backward()
    
    assert x.grad is not None, "Gradients should flow back to input"
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Input gradient norm: {torch.norm(x.grad).item():.6f}")
    print("  [PASS] Gradient flow test passed!")
    
    return True


def test_part2_5():
    """Test Part 2.5: Volume Rendering"""
    print("\n" + "="*60)
    print("PART 2.5: VOLUME RENDERING")
    print("="*60)
    
    print("\nTest 1: Volume Rendering with Provided Test Case")
    # Exact test case from the assignment
    torch.manual_seed(42)
    sigmas = torch.rand((10, 64, 1))
    rgbs = torch.rand((10, 64, 3))
    step_size = (6.0 - 2.0) / 64
    rendered_colors = volrend(sigmas, rgbs, step_size)
    
    correct = torch.tensor([
        [0.5006, 0.3728, 0.4728],
        [0.4322, 0.3559, 0.4134],
        [0.4027, 0.4394, 0.4610],
        [0.4514, 0.3829, 0.4196],
        [0.4002, 0.4599, 0.4103],
        [0.4471, 0.4044, 0.4069],
        [0.4285, 0.4072, 0.3777],
        [0.4152, 0.4190, 0.4361],
        [0.4051, 0.3651, 0.3969],
        [0.3253, 0.3587, 0.4215]
    ])
    
    print(f"  Rendered colors shape: {rendered_colors.shape}")
    print(f"  Expected shape: {correct.shape}")
    print(f"  Sample rendered color: [{rendered_colors[0, 0]:.4f}, {rendered_colors[0, 1]:.4f}, {rendered_colors[0, 2]:.4f}]")
    print(f"  Sample expected color: [{correct[0, 0]:.4f}, {correct[0, 1]:.4f}, {correct[0, 2]:.4f}]")
    print(f"  Max difference: {torch.abs(rendered_colors - correct).max():.6f}")
    
    assert torch.allclose(rendered_colors, correct, rtol=1e-4, atol=1e-4), \
        f"Volume rendering test failed! Max diff: {torch.abs(rendered_colors - correct).max()}"
    print("  [PASS] Volume rendering test PASSED!")
    
    print("\nTest 2: Volume Rendering Shape Test")
    # Test with different shapes
    sigmas = torch.rand((5, 32, 1))
    rgbs = torch.rand((5, 32, 3))
    step_size = 0.1
    rendered = volrend(sigmas, rgbs, step_size)
    
    assert rendered.shape == (5, 3), f"Expected (5, 3), got {rendered.shape}"
    assert torch.all((rendered >= 0) & (rendered <= 1)), "Rendered colors should be in [0, 1]"
    print(f"  Rendered shape: {rendered.shape} [PASS]")
    print("  [PASS] Shape test passed!")
    
    print("\nTest 3: render_rays Function")
    # Create a simple test with NeRF
    nerf = NeRF(pos_L=10, dir_L=4, hidden_dim=256)
    nerf.eval()
    
    # Create random rays
    rays_o = torch.randn(5, 3)
    rays_d = torch.randn(5, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    
    with torch.no_grad():
        rendered = render_rays(nerf, rays_o, rays_d, near=2.0, far=6.0, n_samples=32, perturb=False)
    
    print(f"  Rendered colors shape: {rendered.shape}")
    assert rendered.shape == (5, 3), f"Expected (5, 3), got {rendered.shape}"
    assert torch.all((rendered >= 0) & (rendered <= 1)), "Rendered colors should be in [0, 1]"
    print("  [PASS] render_rays test passed!")
    
    print("\nTest 4: Gradient Flow Through Volume Rendering")
    nerf.train()
    rays_o = torch.randn(3, 3)
    rays_d = torch.randn(3, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    
    rendered = render_rays(nerf, rays_o, rays_d, near=2.0, far=6.0, n_samples=16, perturb=False)
    loss = rendered.sum()
    loss.backward()
    
    # Check that gradients exist for network parameters
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in nerf.parameters())
    assert has_grad, "Gradients should flow through volume rendering to network"
    print(f"  Loss: {loss.item():.6f}")
    print("  [PASS] Gradient flow test passed!")
    
    return True


def visualize_rivian_rays_viser(data_path='rivian.npz', num_rays=100, near=0.02, far=0.5, n_samples=64):
    """
    Visualize Rivian dataset rays and sample points using viser (Part 2.3).
    
    Args:
        data_path: Path to rivian.npz
        num_rays: Number of rays to visualize
        near: Near plane
        far: Far plane
        n_samples: Number of samples per ray
    """
    try:
        import viser
        import time
    except ImportError:
        print("Error: viser not installed!")
        print("Install with: pip install viser")
        return
    
    print("="*60)
    print("Visualizing Rivian Rays and Sample Points (Part 2.3)")
    print("="*60)
    
    # Load dataset
    print(f"\nLoading dataset: {data_path}")
    data = np.load(data_path)
    
    images_train = data['images_train']
    c2ws_train = data['c2ws_train']
    images_val = data['images_val']
    c2ws_val = data['c2ws_val']
    focal = float(data['focal'])
    H, W = data['image_shape'][:2]
    
    # Combine train and val for visualization
    all_images = np.concatenate([images_train, images_val], axis=0)
    all_c2ws = np.concatenate([c2ws_train, c2ws_val], axis=0)
    
    print(f"  Train images: {len(images_train)}")
    print(f"  Val images: {len(images_val)}")
    print(f"  Total cameras: {len(all_images)}")
    print(f"  Image size: {W}x{H}")
    print(f"  Focal length: {focal:.2f}")
    print(f"  Near/Far: {near:.3f} / {far:.3f}")
    
    # Camera intrinsic matrix
    K = torch.tensor([
        [focal, 0, W/2],
        [0, focal, H/2],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    # Sample random rays from random images
    print(f"\nSampling {num_rays} random rays...")
    rays_o_list = []
    rays_d_list = []
    
    position_scale = 0.8  # Scale factor for camera positions
    
    for _ in range(num_rays):
        # Random image
        img_idx = np.random.randint(0, len(all_images))
        # Random pixel
        u = np.random.randint(0, W)
        v = np.random.randint(0, H)
        
        # Get camera pose
        c2w = torch.from_numpy(all_c2ws[img_idx]).float()
        
        # Compute ray using get_rays (get single pixel ray)
        rays_o, rays_d = get_rays(1, 1, K, c2w)  # 1x1 image
        # But we need to get the ray for pixel (u,v)
        # Let's use the direct computation instead
        x = (u - K[0, 2]) / K[0, 0]
        y = (v - K[1, 2]) / K[1, 1]
        
        # Ray direction in camera space (z = 1)
        ray_d_cam = torch.tensor([x, y, 1.0])
        ray_d_cam = ray_d_cam / torch.norm(ray_d_cam)
        
        # Transform to world space
        ray_d_world = (c2w[:3, :3] @ ray_d_cam).numpy()
        ray_o_world = c2w[:3, 3].numpy()
        
        # Apply coordinate transformation (flip Y and Z)
        ray_o_world_4d = np.array([ray_o_world[0], ray_o_world[1], ray_o_world[2], 1.0])
        ray_o_transformed = (np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ], dtype=np.float32) @ ray_o_world_4d)[:3]
        
        ray_d_world_3d = np.array([ray_d_world[0], ray_d_world[1], ray_d_world[2]])
        ray_d_transformed = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ], dtype=np.float32) @ ray_d_world_3d
        
        # Apply position scale
        ray_o_transformed = ray_o_transformed * position_scale
        
        rays_o_list.append(ray_o_transformed)
        rays_d_list.append(ray_d_transformed)
    
    rays_o_sample = torch.from_numpy(np.array(rays_o_list)).float()
    rays_d_sample = torch.from_numpy(np.array(rays_d_list)).float()
    
    # Sample points along rays
    print(f"Sampling {n_samples} points along each ray...")
    points, _ = sample_points_along_rays(
        rays_o_sample, rays_d_sample, near, far, n_samples, perturb=False
    )
    
    print(f"  Points shape: {points.shape}")
    
    # Start viser server
    print("\nStarting viser server...")
    server = viser.ViserServer(share=False, port=8080)
    
    print("\n" + "="*60)
    print("🌐 VISUALIZATION SERVER STARTED!")
    print("="*60)
    print("\nOpen this URL in your web browser:")
    print("   http://localhost:8080")
    print("\n" + "="*60)
    
    # Coordinate system transformation: OpenCV to Graphics convention
    # OpenCV: X right, Y down, Z forward
    # Graphics: X right, Y up, Z backward
    # Flip Y and Z axes
    flip_transform = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ], dtype=np.float32)
    
    # Add cameras
    print(f"\nAdding {len(all_images)} cameras...")
    for i, (image, c2w) in enumerate(zip(all_images, all_c2ws)):
        # Apply coordinate transformation
        c2w_transformed = flip_transform @ c2w
        
        # Scale camera positions to bring them closer together
        c2w_scaled = c2w_transformed.copy()
        c2w_scaled[:3, 3] *= 0.8  # Position scale
        
        server.scene.add_camera_frustum(
            f"/cameras/{i}",
            fov=2 * np.arctan2(H / 2, focal),
            aspect=W / H,
            scale=0.05,  # Larger frustums
            wxyz=viser.transforms.SO3.from_matrix(c2w_scaled[:3, :3]).wxyz,
            position=c2w_scaled[:3, 3],
            image=image
        )
    
    # Add rays
    print(f"Adding {num_rays} rays...")
    rays_o_np = rays_o_sample.numpy()
    rays_d_np = rays_d_sample.numpy()
    for i, (o, d) in enumerate(zip(rays_o_np, rays_d_np)):
        # Ray from origin to far plane
        positions = np.stack((o, o + d * far))
        server.scene.add_spline_catmull_rom(
            f"/rays/{i}", positions=positions, color=(255, 0, 0)
        )
    
    # Add sample points
    print(f"Adding {num_rays * n_samples} sample points...")
    all_points = points.numpy().reshape(-1, 3)
    server.scene.add_point_cloud(
        "/samples",
        colors=np.tile([0, 255, 0], (len(all_points), 1)),  # Green points
        points=all_points,
        point_size=0.003,
    )
    
    # Add coordinate frame
    axes_length = 0.04  # Scaled axes for position_scale=0.8
    server.scene.add_frame("/world", wxyz=(1, 0, 0, 0), position=(0, 0, 0), 
                          show_axes=True, axes_length=axes_length)
    
    print("\n✓ Visualization complete!")
    print(f"\nYou should see:")
    print(f"  - {len(all_images)} camera frustums (27 train + 5 val, click to see images)")
    print(f"  - {num_rays} red rays passing through random pixels")
    print(f"  - {num_rays * n_samples} green sample points along rays")
    print("\nThis shows how rays are cast through pixels and sampled for NeRF!")
    print("\nVisualization controls:")
    print("  - Left click + drag: Rotate view")
    print("  - Right click + drag: Pan view")
    print("  - Scroll: Zoom in/out")
    print("  - Click on camera frustums to see the Rivian images")
    print("\nTip: Take a screenshot for your Part 2.3 deliverable!")
    print("\nPress Ctrl+C to stop the server...")
    
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down viser server...")


if __name__ == "__main__":
    # Test Part 2.1
    success_2_1 = main_part2_1()
    
    # Test Part 2.2
    success_2_2 = test_part2_2()
    
    # Test Part 2.3
    success_2_3 = test_part2_3()
    
    # Test Part 2.4
    success_2_4 = test_part2_4()
    
    # Test Part 2.5
    success_2_5 = test_part2_5()
    
    # Summary
    print("\n" + "="*60)
    print("FINAL TEST SUMMARY")
    print("="*60)
    print(f"Part 2.1 (Create Rays): {'[PASS] PASSED' if success_2_1 else '[FAIL] FAILED'}")
    print(f"Part 2.2 (Sampling): {'[PASS] PASSED' if success_2_2 else '[FAIL] FAILED'}")
    print(f"Part 2.3 (Dataloader): {'[PASS] PASSED' if success_2_3 else '[FAIL] FAILED'}")
    print(f"Part 2.4 (NeRF Network): {'[PASS] PASSED' if success_2_4 else '[FAIL] FAILED'}")
    print(f"Part 2.5 (Volume Rendering): {'[PASS] PASSED' if success_2_5 else '[FAIL] FAILED'}")
    
    if success_2_1 and success_2_2 and success_2_3 and success_2_4 and success_2_5:
        print("\n" + "="*60)
        print("[PASS][PASS][PASS] ALL TESTS PASSED! [PASS][PASS][PASS]")
        print("="*60)
        print("\nReady for NeRF training!")
        print("  - All core components implemented and verified")
        print("  - Next: Train NeRF on lego_200x200.npz dataset")
    else:
        print("\n[FAIL] Some tests failed. Please check the implementation.")

