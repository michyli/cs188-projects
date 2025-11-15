"""
NeRF Training Script for Custom Object Dataset
Project 4 - CS 180

This script trains a Neural Radiance Field on your custom object dataset
created from Part 0 (camera calibration and pose estimation).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import time
import imageio
from code.part2 import (
    NeRF, RaysData, render_rays, sample_points_along_rays,
    pixel_to_ray, get_rays
)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def compute_psnr(mse):
    """Compute PSNR from MSE."""
    return 10 * torch.log10(1.0 / mse)


def load_custom_data(data_path='my_data.npz'):
    """Load the custom object dataset."""
    # Handle relative paths - if file doesn't exist, try in script directory
    if not os.path.exists(data_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, data_path)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            f"Please ensure my_data.npz is in the project4 directory."
        )
    
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    
    images_train = data["images_train"] / 255.0
    c2ws_train = data["c2ws_train"]
    images_val = data["images_val"] / 255.0
    c2ws_val = data["c2ws_val"]
    c2ws_test = data["c2ws_test"]
    focal = float(data["focal"])
    
    print(f"  Training images: {images_train.shape}")
    print(f"  Validation images: {images_val.shape}")
    print(f"  Test cameras: {c2ws_test.shape}")
    print(f"  Focal length: {focal:.2f}")
    
    return images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal


def create_intrinsic_matrix(focal, H, W):
    """Create camera intrinsic matrix."""
    K = np.array([
        [focal, 0, W/2],
        [0, focal, H/2],
        [0, 0, 1]
    ])
    return K


def estimate_scene_bounds(c2ws_train):
    """
    Estimate appropriate near and far bounds based on camera positions.
    
    For real object datasets, we want:
    - near: slightly in front of the closest camera
    - far: well beyond the furthest camera
    """
    camera_positions = c2ws_train[:, :3, 3]
    
    # Compute distances from world origin (where ArUco tag is)
    distances = np.linalg.norm(camera_positions, axis=1)
    
    min_dist = distances.min()
    max_dist = distances.max()
    mean_dist = distances.mean()
    
    print(f"\nCamera distance statistics:")
    print(f"  Min distance: {min_dist:.4f}")
    print(f"  Max distance: {max_dist:.4f}")
    print(f"  Mean distance: {mean_dist:.4f}")
    
    # Conservative estimates:
    # near = halfway to closest camera
    # far = 2x the furthest camera distance
    near = min_dist * 0.5
    far = max_dist * 2.0
    
    # Clamp to reasonable values
    near = max(near, 0.01)  # At least 1cm
    far = max(far, near * 5)  # At least 5x near
    
    print(f"\nRecommended bounds:")
    print(f"  near: {near:.4f}")
    print(f"  far: {far:.4f}")
    
    return near, far


def evaluate_on_validation(nerf, images_val, c2ws_val, K, near=2.0, far=6.0, n_samples=64):
    """Evaluate model on validation set and compute average PSNR."""
    nerf.eval()
    H, W = images_val.shape[1:3]
    psnrs = []
    
    with torch.no_grad():
        for i in range(len(images_val)):
            # Generate all rays for this image
            rays_o, rays_d = get_rays(H, W, torch.from_numpy(K).float(), 
                                     torch.from_numpy(c2ws_val[i]).float())
            rays_o = rays_o.reshape(-1, 3).numpy()
            rays_d = rays_d.reshape(-1, 3).numpy()
            
            # Render in chunks to avoid OOM
            chunk_size = 1024
            rendered_colors = []
            for j in range(0, len(rays_o), chunk_size):
                rays_o_chunk = torch.from_numpy(rays_o[j:j+chunk_size]).float().to(device)
                rays_d_chunk = torch.from_numpy(rays_d[j:j+chunk_size]).float().to(device)
                
                colors = render_rays(nerf, rays_o_chunk, rays_d_chunk, near, far, n_samples, perturb=False)
                rendered_colors.append(colors.cpu())
            
            rendered_colors = torch.cat(rendered_colors, dim=0)
            rendered_image = rendered_colors.reshape(H, W, 3)
            
            # Compute MSE and PSNR
            gt_image = torch.from_numpy(images_val[i]).float()
            mse = torch.mean((rendered_image - gt_image) ** 2)
            psnr = compute_psnr(mse)
            psnrs.append(psnr.item())
    
    nerf.train()
    return np.mean(psnrs)


def render_novel_view(nerf, c2w, K, H, W, near=2.0, far=6.0, n_samples=64):
    """Render a novel view from a given camera pose."""
    nerf.eval()
    
    with torch.no_grad():
        # Generate all rays for this camera
        rays_o, rays_d = get_rays(H, W, torch.from_numpy(K).float(), 
                                 torch.from_numpy(c2w).float())
        rays_o = rays_o.reshape(-1, 3).numpy()
        rays_d = rays_d.reshape(-1, 3).numpy()
        
        # Render in chunks
        chunk_size = 1024
        rendered_colors = []
        for j in range(0, len(rays_o), chunk_size):
            rays_o_chunk = torch.from_numpy(rays_o[j:j+chunk_size]).float().to(device)
            rays_d_chunk = torch.from_numpy(rays_d[j:j+chunk_size]).float().to(device)
            
            colors = render_rays(nerf, rays_o_chunk, rays_d_chunk, near, far, n_samples, perturb=False)
            rendered_colors.append(colors.cpu())
        
        rendered_colors = torch.cat(rendered_colors, dim=0)
        rendered_image = rendered_colors.reshape(H, W, 3).numpy()
    
    nerf.train()
    return rendered_image


def interpolate_poses(c2w_start, c2w_end, num_steps):
    """
    Interpolate between two camera poses using SLERP for rotation and LERP for translation.
    
    Args:
        c2w_start: Starting camera-to-world matrix (4, 4)
        c2w_end: Ending camera-to-world matrix (4, 4)
        num_steps: Number of interpolated poses (including start and end)
        
    Returns:
        interpolated: List of interpolated c2w matrices
    """
    from scipy.spatial.transform import Rotation, Slerp
    
    # Extract rotations and translations
    R_start = c2w_start[:3, :3]
    t_start = c2w_start[:3, 3]
    R_end = c2w_end[:3, :3]
    t_end = c2w_end[:3, 3]
    
    # Convert rotation matrices to scipy Rotation objects
    rot_start = Rotation.from_matrix(R_start)
    rot_end = Rotation.from_matrix(R_end)
    
    # Create SLERP interpolator for rotations
    key_times = [0, 1]
    key_rots = Rotation.concatenate([rot_start, rot_end])
    slerp = Slerp(key_times, key_rots)
    
    # Interpolate
    interpolated = []
    for i in range(num_steps):
        alpha = i / (num_steps - 1) if num_steps > 1 else 0
        
        # Interpolate rotation (SLERP)
        rot_interp = slerp(alpha)
        R_interp = rot_interp.as_matrix()
        
        # Interpolate translation (LERP)
        t_interp = (1 - alpha) * t_start + alpha * t_end
        
        # Construct interpolated c2w
        c2w_interp = np.eye(4)
        c2w_interp[:3, :3] = R_interp
        c2w_interp[:3, 3] = t_interp
        
        interpolated.append(c2w_interp)
    
    return interpolated


def sort_cameras_by_angle(c2ws):
    """
    Sort camera poses by their angular position around the object center.
    This creates a circular path for smooth rotation videos.
    
    Args:
        c2ws: Camera poses (N, 4, 4)
        
    Returns:
        sorted_c2ws: Camera poses sorted by angle
    """
    # Extract camera positions
    positions = c2ws[:, :3, 3]  # (N, 3)
    
    # Compute center (mean position)
    center = positions.mean(axis=0)
    
    # Compute vectors from center to each camera
    vectors = positions - center  # (N, 3)
    
    # Project onto XZ plane (assuming Y is up) and compute angles
    angles = np.arctan2(vectors[:, 2], vectors[:, 0])
    
    # Sort by angle
    sorted_indices = np.argsort(angles)
    
    print(f"  Sorting {len(c2ws)} cameras by angle for circular path")
    print(f"  Angles range: {np.degrees(angles.min()):.1f}° to {np.degrees(angles.max()):.1f}°")
    
    return c2ws[sorted_indices]


def interpolate_camera_path(c2ws, steps_between=8, sort_by_angle=True):
    """
    Create a smooth camera path by interpolating between existing poses.
    
    Args:
        c2ws: Original camera poses (N, 4, 4)
        steps_between: Number of interpolated frames between each pair
        sort_by_angle: If True, sort cameras by angle first for circular motion
        
    Returns:
        smooth_c2ws: Interpolated camera path with more frames
    """
    # Sort cameras by angle for smooth circular motion
    if sort_by_angle:
        c2ws = sort_cameras_by_angle(c2ws)
    
    smooth_path = []
    
    # Interpolate between consecutive camera poses
    for i in range(len(c2ws)):
        c2w_start = c2ws[i]
        c2w_end = c2ws[(i + 1) % len(c2ws)]  # Loop back to first
        
        # Interpolate between this pair (exclude end to avoid duplicates)
        interpolated = interpolate_poses(c2w_start, c2w_end, steps_between + 1)
        smooth_path.extend(interpolated[:-1])  # Exclude last to avoid duplicate
    
    return np.array(smooth_path)


def create_novel_view_video(nerf, c2ws_test, K, H, W, output_path='novel_views.mp4', fps=30, near=0.02, far=0.5, n_samples=64, interpolate=False, steps_between=8):
    """
    Create a video of novel views using test camera poses.
    
    Args:
        interpolate: If True, interpolate between test poses for smoother video
        steps_between: Number of frames to interpolate between each pair of test poses
    """
    print("Rendering novel views for video...")
    
    # Optionally interpolate for smoother motion
    if interpolate:
        print(f"Interpolating camera path: {len(c2ws_test)} poses → {len(c2ws_test) * steps_between} poses")
        c2ws_render = interpolate_camera_path(c2ws_test, steps_between)
        print(f"  Smooth path has {len(c2ws_render)} frames")
    else:
        c2ws_render = c2ws_test
    
    frames = []
    
    for i, c2w in enumerate(tqdm(c2ws_render, desc="Rendering frames")):
        frame = render_novel_view(nerf, c2w, K, H, W, near, far, n_samples)
        frame = np.clip(frame, 0, 1)  # Ensure valid range
        frame = (frame * 255).astype(np.uint8)
        frames.append(frame)
    
    # Save as video
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"  Video saved to {output_path}")
    
    # Also save as GIF for easy viewing
    gif_path = output_path.replace('.mp4', '.gif')
    imageio.mimsave(gif_path, frames[::2], fps=fps//2)  # Every other frame for smaller size
    print(f"  GIF saved to {gif_path}")


def train_object_nerf(data_path='my_data.npz', 
                      output_dir='object_nerf_output',
                      num_iterations=2000,
                      batch_size=4096,
                      learning_rate=5e-4,
                      n_samples=64,
                      near=None,
                      far=None,
                      val_every=100,
                      save_every=200,
                      pos_L=10,
                      dir_L=4,
                      hidden_dim=256):
    """
    Train NeRF on your custom object dataset.
    
    Args:
        data_path: Path to dataset .npz file
        output_dir: Directory to save outputs
        num_iterations: Number of training iterations
        batch_size: Number of rays per gradient step
        learning_rate: Learning rate for Adam optimizer
        n_samples: Number of samples per ray
        near: Near clipping distance (auto-estimated if None)
        far: Far clipping distance (auto-estimated if None)
        val_every: Evaluate on validation set every N iterations
        save_every: Save visualizations every N iterations
        pos_L: Positional encoding frequency for coordinates
        dir_L: Positional encoding frequency for directions
        hidden_dim: Hidden layer dimension
    """
    # Create output directory
    if not os.path.isabs(output_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'progression'), exist_ok=True)
    
    # Load data
    images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal = load_custom_data(data_path)
    N_train, H, W, _ = images_train.shape
    K = create_intrinsic_matrix(focal, H, W)
    
    # Auto-estimate near/far bounds if not provided
    if near is None or far is None:
        near_est, far_est = estimate_scene_bounds(c2ws_train)
        if near is None:
            near = near_est
        if far is None:
            far = far_est
    
    print(f"\nTraining Configuration:")
    print(f"  Iterations: {num_iterations}")
    print(f"  Batch size: {batch_size} rays")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Samples per ray: {n_samples}")
    print(f"  Near/Far: {near:.4f}/{far:.4f}")
    print(f"  Position encoding L: {pos_L}")
    print(f"  Direction encoding L: {dir_L}")
    print(f"  Hidden dimension: {hidden_dim}")
    
    # Create dataloader
    print("\nInitializing dataloader...")
    dataset = RaysData(images_train, K, c2ws_train)
    
    # Create NeRF model
    print("Initializing NeRF model...")
    nerf = NeRF(pos_L=pos_L, dir_L=dir_L, hidden_dim=hidden_dim).to(device)
    print(f"  Model has {sum(p.numel() for p in nerf.parameters()):,} parameters")
    
    # Create optimizer
    optimizer = optim.Adam(nerf.parameters(), lr=learning_rate)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training metrics
    train_losses = []
    val_psnrs = []
    iterations = []
    
    # Training loop
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}\n")
    
    nerf.train()
    start_time = time.time()
    
    for iteration in range(1, num_iterations + 1):
        # Sample rays
        rays_o, rays_d, pixels_gt = dataset.sample_rays(batch_size)
        rays_o = torch.from_numpy(rays_o).float().to(device)
        rays_d = torch.from_numpy(rays_d).float().to(device)
        pixels_gt = torch.from_numpy(pixels_gt).float().to(device)
        
        # Forward pass
        rendered_colors = render_rays(nerf, rays_o, rays_d, near, far, n_samples, perturb=True)
        
        # Compute loss
        loss = criterion(rendered_colors, pixels_gt)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record training loss
        train_losses.append(loss.item())
        
        # Validation and logging
        if iteration % val_every == 0 or iteration == 1:
            avg_psnr = evaluate_on_validation(nerf, images_val, c2ws_val, K, near, far, n_samples)
            val_psnrs.append(avg_psnr)
            iterations.append(iteration)
            
            elapsed = time.time() - start_time
            print(f"Iter {iteration:4d} | Loss: {loss.item():.6f} | Val PSNR: {avg_psnr:.2f} dB | Time: {elapsed:.1f}s")
        
        # Save visualizations
        if iteration % save_every == 0 or iteration == 1:
            # Render a validation image
            rendered_img = render_novel_view(nerf, c2ws_val[0], K, H, W, near, far, n_samples)
            
            # Save progression image
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(images_val[0])
            axes[0].set_title('Ground Truth')
            axes[0].axis('off')
            axes[1].imshow(np.clip(rendered_img, 0, 1))
            axes[1].set_title(f'Rendered (Iter {iteration})')
            axes[1].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'progression', f'iter_{iteration:04d}.png'), 
                       dpi=100, bbox_inches='tight')
            plt.close()
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    
    final_psnr = evaluate_on_validation(nerf, images_val, c2ws_val, K, near, far, n_samples)
    print(f"Final Validation PSNR: {final_psnr:.2f} dB")
    
    # Save PSNR curve
    print("\nSaving PSNR curve...")
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, val_psnrs, 'b-', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title('Validation PSNR over Training', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'psnr_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {os.path.join(output_dir, 'psnr_curve.png')}")
    
    # Save training curve
    print("Saving training loss curve...")
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, 'r-', alpha=0.7, linewidth=1)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Training Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {os.path.join(output_dir, 'training_loss.png')}")
    
    # Render all validation images
    print("\nRendering all validation images...")
    for i in range(len(images_val)):
        rendered_img = render_novel_view(nerf, c2ws_val[i], K, H, W, near, far, n_samples)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(images_val[i])
        axes[0].set_title(f'Ground Truth (Val {i+1})')
        axes[0].axis('off')
        axes[1].imshow(np.clip(rendered_img, 0, 1))
        axes[1].set_title(f'Rendered (Val {i+1})')
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'val_image_{i+1}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    print(f"  Saved {len(images_val)} validation comparisons")
    
    # Create novel view video
    print("\nCreating novel view video...")
    video_path = os.path.join(output_dir, 'novel_views.mp4')
    
    # Use interpolation to create smooth video (7 poses → 56 frames with 8 steps between each)
    create_novel_view_video(
        nerf, c2ws_test, K, H, W, 
        output_path=video_path, 
        fps=30, 
        near=near, 
        far=far, 
        n_samples=n_samples,
        interpolate=True,
        steps_between=8  # 7 original poses × 8 = 56 smooth frames
    )
    
    # Save model
    print("\nSaving model...")
    model_path = os.path.join(output_dir, 'nerf_model.pth')
    torch.save({
        'iteration': num_iterations,
        'model_state_dict': nerf.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_psnr': final_psnr,
        'config': {
            'pos_L': pos_L,
            'dir_L': dir_L,
            'hidden_dim': hidden_dim,
            'near': near,
            'far': far,
            'n_samples': n_samples,
        }
    }, model_path)
    print(f"  Model saved to {model_path}")
    
    print(f"\n{'='*60}")
    print("All outputs saved successfully!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"  - PSNR curve: psnr_curve.png")
    print(f"  - Training loss: training_loss.png")
    print(f"  - Validation comparisons: val_image_*.png")
    print(f"  - Training progression: progression/iter_*.png")
    print(f"  - Novel view video: novel_views.mp4")
    print(f"  - Novel view GIF: novel_views.gif")
    print(f"  - Model checkpoint: nerf_model.pth")
    
    return nerf, val_psnrs, train_losses


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train NeRF on custom object dataset')
    parser.add_argument('--data', type=str, default='my_data.npz', help='Path to dataset')
    parser.add_argument('--output', type=str, default='object_nerf_output', help='Output directory')
    parser.add_argument('--iterations', type=int, default=2000, help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, default=4096, help='Rays per gradient step')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--n_samples', type=int, default=64, help='Samples per ray')
    parser.add_argument('--near', type=float, default=None, help='Near plane (auto-estimated if not provided)')
    parser.add_argument('--far', type=float, default=None, help='Far plane (auto-estimated if not provided)')
    parser.add_argument('--val_every', type=int, default=100, help='Validation frequency')
    parser.add_argument('--save_every', type=int, default=200, help='Save frequency')
    parser.add_argument('--pos_L', type=int, default=10, help='Position encoding frequency')
    parser.add_argument('--dir_L', type=int, default=4, help='Direction encoding frequency')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension')
    
    args = parser.parse_args()
    
    # Train NeRF
    train_object_nerf(
        data_path=args.data,
        output_dir=args.output,
        num_iterations=args.iterations,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        n_samples=args.n_samples,
        near=args.near,
        far=args.far,
        val_every=args.val_every,
        save_every=args.save_every,
        pos_L=args.pos_L,
        dir_L=args.dir_L,
        hidden_dim=args.hidden_dim
    )

