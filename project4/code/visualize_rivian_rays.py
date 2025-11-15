"""
Visualize rays for Rivian dataset using viser (Part 2.3).
Shows cameras, rays, and sample points in 3D.
"""

import numpy as np
import torch
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from code.part2 import sample_points_along_rays


def visualize_rivian_rays(data_path='rivian.npz', num_rays=100, near=0.02, far=0.5, n_samples=64):
    """
    Visualize Rivian dataset rays and sample points using viser.
    
    Args:
        data_path: Path to rivian.npz
        num_rays: Number of rays to visualize
        near: Near plane
        far: Far plane
        n_samples: Number of samples per ray
    """
    try:
        import viser
    except ImportError:
        print("Error: viser not installed!")
        print("Install with: pip install viser")
        return
    
    print("="*60)
    print("Visualizing Rivian Rays and Sample Points")
    print("="*60)
    
    # Load dataset
    print(f"\nLoading dataset: {data_path}")
    data = np.load(data_path)
    
    images_train = data['images_train']
    c2ws_train = data['c2ws_train']
    focal = float(data['focal'])
    H, W = data['image_shape'][:2]
    
    print(f"  Train images: {len(images_train)}")
    print(f"  Image size: {W}x{H}")
    print(f"  Focal length: {focal:.2f}")
    print(f"  Near/Far: {near:.3f} / {far:.3f}")
    
    # Camera intrinsic matrix
    K = np.array([
        [focal, 0, W/2],
        [0, focal, H/2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Sample random rays from random images
    print(f"\nSampling {num_rays} random rays...")
    rays_o_list = []
    rays_d_list = []
    
    for _ in range(num_rays):
        # Random image
        img_idx = np.random.randint(0, len(images_train))
        # Random pixel
        u = np.random.randint(0, W)
        v = np.random.randint(0, H)
        
        # Get camera pose
        c2w = c2ws_train[img_idx]
        
        # Compute ray
        # Pixel coordinates (u, v) to camera coordinates
        x = (u - K[0, 2]) / K[0, 0]
        y = (v - K[1, 2]) / K[1, 1]
        
        # Ray direction in camera space (z = 1)
        ray_d_cam = np.array([x, y, 1.0])
        ray_d_cam = ray_d_cam / np.linalg.norm(ray_d_cam)
        
        # Transform to world space
        ray_d_world = (c2w[:3, :3] @ ray_d_cam).astype(np.float32)
        ray_o_world = c2w[:3, 3].astype(np.float32)
        
        rays_o_list.append(ray_o_world)
        rays_d_list.append(ray_d_world)
    
    rays_o_sample = np.array(rays_o_list)
    rays_d_sample = np.array(rays_d_list)
    
    # Sample points along rays
    print(f"Sampling {n_samples} points along each ray...")
    points, _ = sample_points_along_rays(
        torch.from_numpy(rays_o_sample).float(),
        torch.from_numpy(rays_d_sample).float(),
        near, far, n_samples, perturb=False
    )
    points = points.numpy()  # (num_rays, n_samples, 3)
    
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
    
    # Add cameras
    print(f"\nAdding {len(images_train)} cameras...")
    for i, (image, c2w) in enumerate(zip(images_train, c2ws_train)):
        server.scene.add_camera_frustum(
            f"/cameras/{i}",
            fov=2 * np.arctan2(H / 2, K[0, 0]),
            aspect=W / H,
            scale=0.01,  # Small frustums for Rivian scale
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
            image=image
        )
    
    # Add rays
    print(f"Adding {num_rays} rays...")
    for i, (o, d) in enumerate(zip(rays_o_sample, rays_d_sample)):
        # Ray from origin to far plane
        positions = np.stack((o, o + d * far))
        server.scene.add_spline_catmull_rom(
            f"/rays/{i}", positions=positions, color=(255, 0, 0)
        )
    
    # Add sample points
    print(f"Adding {num_rays * n_samples} sample points...")
    all_points = points.reshape(-1, 3)
    server.scene.add_point_cloud(
        "/samples",
        colors=np.tile([0, 255, 0], (len(all_points), 1)),  # Green points
        points=all_points,
        point_size=0.003,
    )
    
    # Add coordinate frame
    server.scene.add_frame("/world", wxyz=(1, 0, 0, 0), position=(0, 0, 0), 
                          show_axes=True, axes_length=0.05)
    
    print("\n✓ Visualization complete!")
    print(f"\nYou should see:")
    print(f"  - {len(images_train)} camera frustums (click to see images)")
    print(f"  - {num_rays} red rays passing through random pixels")
    print(f"  - {num_rays * n_samples} green sample points along rays")
    print("\nThis shows how rays are cast through pixels and sampled!")
    print("\nVisualization controls:")
    print("  - Left click + drag: Rotate view")
    print("  - Right click + drag: Pan view")
    print("  - Scroll: Zoom in/out")
    print("\nTip: Take a screenshot for your report!")
    print("\nPress Ctrl+C to stop the server...")
    
    try:
        import time
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down viser server...")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize Rivian rays with viser')
    parser.add_argument('--data', type=str, default='rivian.npz',
                       help='Path to dataset file')
    parser.add_argument('--num_rays', type=int, default=100,
                       help='Number of rays to visualize (default: 100)')
    parser.add_argument('--n_samples', type=int, default=64,
                       help='Samples per ray (default: 64)')
    
    args = parser.parse_args()
    
    visualize_rivian_rays(
        data_path=args.data,
        num_rays=args.num_rays,
        n_samples=args.n_samples
    )

