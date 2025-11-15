"""
NeRF Training Script for Lego Dataset
Project 4 - CS 180

This script trains a Neural Radiance Field on the lego_200x200.npz dataset
and generates all required visualizations and outputs.
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


def load_lego_data(data_path='lego_200x200.npz'):
    """Load the lego dataset."""
    # Handle relative paths - if file doesn't exist, try in script directory
    if not os.path.exists(data_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, data_path)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            f"Please ensure lego_200x200.npz is in the project4 directory."
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


def visualize_rays_and_samples(images_train, c2ws_train, K, rays_o_sample, rays_d_sample, 
                               near=2.0, far=6.0, n_samples=64, save_path='rays_viz.png'):
    """Visualize cameras, rays, and sample points in 3D."""
    import viser
    
    print("Creating 3D visualization of cameras, rays, and samples...")
    H, W = images_train.shape[1:3]
    
    # Sample points along the rays
    points, _ = sample_points_along_rays(
        torch.from_numpy(rays_o_sample).float(),
        torch.from_numpy(rays_d_sample).float(),
        near, far, n_samples, perturb=False
    )
    points = points.numpy()
    
    # Start viser server
    server = viser.ViserServer(share=False, port=8080)
    
    # Add cameras
    for i, (image, c2w) in enumerate(zip(images_train, c2ws_train)):
        server.scene.add_camera_frustum(
            f"/cameras/{i}",
            fov=2 * np.arctan2(H / 2, K[0, 0]),
            aspect=W / H,
            scale=0.15,
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
            image=image
        )
    
    # Add rays
    for i, (o, d) in enumerate(zip(rays_o_sample, rays_d_sample)):
        positions = np.stack((o, o + d * 6.0))
        server.scene.add_spline_catmull_rom(
            f"/rays/{i}", positions=positions, color=(255, 0, 0)
        )
    
    # Add sample points
    server.scene.add_point_cloud(
        "/samples",
        colors=np.zeros_like(points).reshape(-1, 3),
        points=points.reshape(-1, 3),
        point_size=0.02,
    )
    
    print(f"  Viser visualization running at http://localhost:8080")
    print(f"  Please take a screenshot and save it to {save_path}")
    print("  Press Enter after saving screenshot to continue...")
    input()


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


def create_novel_view_video(nerf, c2ws_test, K, H, W, output_path='novel_views.mp4', fps=30):
    """Create a video of novel views using test camera poses."""
    print("Rendering novel views for video...")
    frames = []
    
    for i, c2w in enumerate(tqdm(c2ws_test, desc="Rendering frames")):
        frame = render_novel_view(nerf, c2w, K, H, W)
        frame = (frame * 255).astype(np.uint8)
        frames.append(frame)
    
    # Save as video
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"  Video saved to {output_path}")
    
    # Also save as GIF for easy viewing
    gif_path = output_path.replace('.mp4', '.gif')
    imageio.mimsave(gif_path, frames[::2], fps=fps//2)  # Every other frame for smaller size
    print(f"  GIF saved to {gif_path}")


def train_nerf(data_path='lego_200x200.npz', 
               output_dir='nerf_output',
               num_iterations=1000,
               batch_size=10000,
               learning_rate=5e-4,
               n_samples=64,
               near=2.0,
               far=6.0,
               val_every=50,
               save_every=100):
    """
    Train NeRF on the lego dataset.
    
    Args:
        data_path: Path to dataset .npz file
        output_dir: Directory to save outputs
        num_iterations: Number of training iterations
        batch_size: Number of rays per gradient step
        learning_rate: Learning rate for Adam optimizer
        n_samples: Number of samples per ray
        near: Near clipping distance
        far: Far clipping distance
        val_every: Evaluate on validation set every N iterations
        save_every: Save visualizations every N iterations
    """
    # Create output directory (relative to script location if not absolute)
    if not os.path.isabs(output_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'progression'), exist_ok=True)
    
    # Load data
    images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal = load_lego_data(data_path)
    N_train, H, W, _ = images_train.shape
    K = create_intrinsic_matrix(focal, H, W)
    
    print(f"\nTraining Configuration:")
    print(f"  Iterations: {num_iterations}")
    print(f"  Batch size: {batch_size} rays")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Samples per ray: {n_samples}")
    print(f"  Near/Far: {near}/{far}")
    
    # Create dataloader
    print("\nInitializing dataloader...")
    dataset = RaysData(images_train, K, c2ws_train)
    
    # Create NeRF model
    print("Initializing NeRF model...")
    nerf = NeRF(pos_L=10, dir_L=4, hidden_dim=256).to(device)
    print(f"  Model has {sum(p.numel() for p in nerf.parameters()):,} parameters")
    
    # Create optimizer
    optimizer = optim.Adam(nerf.parameters(), lr=learning_rate)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training metrics
    train_losses = []
    val_psnrs = []
    iterations = []
    
    # Visualize rays and samples (disabled by default to not block training)
    # To enable: pass visualize_rays=True to train_nerf()
    print("\nSkipping viser visualization (can be enabled with --visualize flag)")
    print("  You can manually run viser visualization later if needed")
    
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
            axes[1].imshow(rendered_img)
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
    
    if final_psnr >= 23.0:
        print("[PASS] Achieved target PSNR of 23+ dB!")
    else:
        print(f"[INFO] Current PSNR: {final_psnr:.2f} dB (Target: 23+ dB)")
    
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
        axes[1].imshow(rendered_img)
        axes[1].set_title(f'Rendered (Val {i+1})')
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'val_image_{i+1}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    print(f"  Saved {len(images_val)} validation comparisons")
    
    # Create novel view video
    print("\nCreating novel view video...")
    video_path = os.path.join(output_dir, 'novel_views.mp4')
    create_novel_view_video(nerf, c2ws_test, K, H, W, output_path=video_path, fps=30)
    
    # Save model
    print("\nSaving model...")
    model_path = os.path.join(output_dir, 'nerf_model.pth')
    torch.save({
        'iteration': num_iterations,
        'model_state_dict': nerf.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_psnr': final_psnr,
        'config': {
            'pos_L': 10,
            'dir_L': 4,
            'hidden_dim': 256,
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
    
    parser = argparse.ArgumentParser(description='Train NeRF on Lego dataset')
    parser.add_argument('--data', type=str, default='lego_200x200.npz', help='Path to dataset')
    parser.add_argument('--output', type=str, default='nerf_output', help='Output directory')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, default=10000, help='Rays per gradient step')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--n_samples', type=int, default=64, help='Samples per ray')
    parser.add_argument('--val_every', type=int, default=50, help='Validation frequency')
    parser.add_argument('--save_every', type=int, default=100, help='Save frequency')
    
    args = parser.parse_args()
    
    # Train NeRF
    train_nerf(
        data_path=args.data,
        output_dir=args.output,
        num_iterations=args.iterations,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        n_samples=args.n_samples,
        val_every=args.val_every,
        save_every=args.save_every
    )

