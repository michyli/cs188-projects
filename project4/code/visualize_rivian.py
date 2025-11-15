"""
Visualize Rivian camera poses using viser (Part 0.3 for multi-tag setup).
"""

import numpy as np
import cv2
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))


def visualize_rivian_poses(data_path='rivian.npz', frustum_scale=0.005, position_scale=1.0):
    """
    Visualize Rivian camera poses using viser.
    
    Args:
        data_path: Path to rivian.npz dataset
        frustum_scale: Scale of camera frustums (smaller = smaller frustums)
        position_scale: Scale to apply to camera positions (larger = more spread out)
    """
    try:
        import viser
        import time
    except ImportError:
        print("\nError: viser not installed!")
        print("Install with: pip install viser")
        return
    
    print("="*60)
    print("Visualizing Rivian Camera Poses")
    print("="*60)
    
    # Load dataset
    print(f"\nLoading dataset: {data_path}")
    data = np.load(data_path)
    
    # Get all camera poses (train + val)
    c2ws_train = data['c2ws_train']
    c2ws_val = data['c2ws_val']
    c2ws_test = data['c2ws_test']
    
    images_train = data['images_train']
    images_val = data['images_val']
    
    focal = float(data['focal'])
    H, W = data['image_shape'][:2]
    
    print(f"  Train: {len(c2ws_train)} poses with images")
    print(f"  Val: {len(c2ws_val)} poses with images")
    print(f"  Test: {len(c2ws_test)} poses (no images)")
    print(f"  Image size: {W}x{H}")
    print(f"  Focal length: {focal:.2f}")
    
    # Camera intrinsic matrix
    K = np.array([
        [focal, 0, W/2],
        [0, focal, H/2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    print("\nStarting viser server...")
    server = viser.ViserServer(share=False)
    
    print("\n" + "="*60)
    print("🌐 VISUALIZATION SERVER STARTED!")
    print("="*60)
    print("\nOpen this URL in your web browser:")
    print("   http://localhost:8080")
    print("\n" + "="*60)
    print(f"\nVisualization settings:")
    print(f"  - Frustum scale: {frustum_scale}")
    print(f"  - Position scale: {position_scale}")
    print("="*60)
    
    # Coordinate system transformation: OpenCV to Graphics convention
    # Flip Y and Z axes for better visualization
    flip_transform = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ])
    
    # Add training camera frustums
    for i, (c2w_opencv, img) in enumerate(zip(c2ws_train, images_train)):
        # Apply coordinate system transformation
        c2w = flip_transform @ c2w_opencv
        
        # Scale the camera positions
        c2w_scaled = c2w.copy()
        c2w_scaled[:3, 3] *= position_scale
        
        # Calculate field of view
        fov = 2 * np.arctan2(H / 2, K[0, 0])
        aspect = W / H
        
        # Add camera frustum (image is already RGB from dataset)
        server.scene.add_camera_frustum(
            f"/train/camera_{i}",
            fov=fov,
            aspect=aspect,
            scale=frustum_scale,
            wxyz=viser.transforms.SO3.from_matrix(c2w_scaled[:3, :3]).wxyz,
            position=c2w_scaled[:3, 3],
            image=img
        )
        
        print(f"Added training camera {i+1}/{len(c2ws_train)}")
    
    # Add validation camera frustums
    for i, (c2w_opencv, img) in enumerate(zip(c2ws_val, images_val)):
        # Apply coordinate system transformation
        c2w = flip_transform @ c2w_opencv
        
        # Scale the camera positions
        c2w_scaled = c2w.copy()
        c2w_scaled[:3, 3] *= position_scale
        
        # Calculate field of view
        fov = 2 * np.arctan2(H / 2, K[0, 0])
        aspect = W / H
        
        # Add camera frustum
        server.scene.add_camera_frustum(
            f"/val/camera_{i}",
            fov=fov,
            aspect=aspect,
            scale=frustum_scale,
            wxyz=viser.transforms.SO3.from_matrix(c2w_scaled[:3, :3]).wxyz,
            position=c2w_scaled[:3, 3],
            image=img
        )
        
        print(f"Added validation camera {i+1}/{len(c2ws_val)}")
    
    # Add test camera frustums (no images, just frustums)
    for i, c2w_opencv in enumerate(c2ws_test):
        # Apply coordinate system transformation
        c2w = flip_transform @ c2w_opencv
        
        # Scale the camera positions
        c2w_scaled = c2w.copy()
        c2w_scaled[:3, 3] *= position_scale
        
        # Calculate field of view
        fov = 2 * np.arctan2(H / 2, K[0, 0])
        aspect = W / H
        
        # Add camera frustum (wireframe only, no image)
        server.scene.add_camera_frustum(
            f"/test/camera_{i}",
            fov=fov,
            aspect=aspect,
            scale=frustum_scale * 0.8,  # Slightly smaller for test cameras
            wxyz=viser.transforms.SO3.from_matrix(c2w_scaled[:3, :3]).wxyz,
            position=c2w_scaled[:3, 3],
        )
        
        print(f"Added test camera {i+1}/{len(c2ws_test)}")
    
    # Add coordinate frame at origin (where the ArUco tag board is)
    axes_length = 0.05 * position_scale
    server.scene.add_frame("/world", wxyz=(1, 0, 0, 0), position=(0, 0, 0), 
                          show_axes=True, axes_length=axes_length)
    
    print(f"\n✓ Visualization complete!")
    print(f"  Added {len(c2ws_train)} training cameras (with images)")
    print(f"  Added {len(c2ws_val)} validation cameras (with images)")
    print(f"  Added {len(c2ws_test)} test cameras (wireframe only)")
    print("\nVisualization controls:")
    print("  - Left click + drag: Rotate view")
    print("  - Right click + drag: Pan view")
    print("  - Scroll: Zoom in/out")
    print("  - Click on camera frustums to see the image")
    print("\nTip: Take a screenshot for your report!")
    print("\nPress Ctrl+C to stop the server...")
    
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down viser server...")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize Rivian camera poses with viser')
    parser.add_argument('--data', type=str, default='rivian.npz',
                       help='Path to dataset file')
    parser.add_argument('--frustum_scale', type=float, default=0.005,
                       help='Scale of camera frustums (default: 0.005)')
    parser.add_argument('--position_scale', type=float, default=1.0,
                       help='Scale of camera positions (default: 1.0)')
    
    args = parser.parse_args()
    
    visualize_rivian_poses(
        data_path=args.data,
        frustum_scale=args.frustum_scale,
        position_scale=args.position_scale
    )

