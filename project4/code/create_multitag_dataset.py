"""
Create NeRF dataset from images with multiple ArUco tags.

This script handles scenes with 6 ArUco tags (IDs 0-5), where not all tags
may be visible in every image. It establishes a consistent world coordinate
system and computes camera poses robustly.

Approach:
1. Detect all tags in all images
2. Establish world coordinate system from tag positions
3. For each image, compute camera pose using visible tags
"""

import cv2
import numpy as np
import glob
import os
from tqdm import tqdm
from collections import defaultdict


def load_calibration(calibration_file='calibration_results.npz'):
    """Load camera calibration results."""
    # Try as-is first
    if os.path.exists(calibration_file):
        data = np.load(calibration_file)
        return data['camera_matrix'], data['dist_coeffs']
    
    # Try relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    alt_path = os.path.join(script_dir, calibration_file)
    if os.path.exists(alt_path):
        print(f"Using calibration file: {alt_path}")
        data = np.load(alt_path)
        return data['camera_matrix'], data['dist_coeffs']
    
    raise FileNotFoundError(
        f"Calibration file not found!\n"
        f"  Tried: {calibration_file}\n"
        f"  Tried: {alt_path}"
    )


def detect_tags_in_image(img_rgb, aruco_dict, aruco_params):
    """
    Detect ArUco tags in an image.
    
    Returns:
        dict: {tag_id: corners} for all detected tags
    """
    corners, ids, _ = cv2.aruco.detectMarkers(img_rgb, aruco_dict, parameters=aruco_params)
    
    if ids is None:
        return {}
    
    # Convert to dictionary
    tag_detections = {}
    for i, tag_id in enumerate(ids.flatten()):
        tag_detections[tag_id] = corners[i].reshape(-1, 2).astype(np.float32)
    
    return tag_detections


def build_tag_world_positions(tag_size=0.02, grid_spacing=0.10, horizontal_spacing=None, vertical_spacing=None):
    """
    Define the 3D positions of the 6 ArUco tags in world coordinates.
    
    Layout: 2x3 grid (2 columns, 3 rows)
    
        0   1
        2   3
        4   5
    
    Args:
        tag_size: Size of each tag in meters (default: 2cm)
        grid_spacing: Uniform spacing if horizontal/vertical not specified (default: 10cm)
        horizontal_spacing: Center-to-center spacing in X direction (overrides grid_spacing)
        vertical_spacing: Center-to-center spacing in Y direction (overrides grid_spacing)
        
    Returns:
        dict: {tag_id: 4 corner positions in 3D}
    """
    # Use specific spacings if provided, otherwise use uniform grid_spacing
    h_spacing = horizontal_spacing if horizontal_spacing is not None else grid_spacing
    v_spacing = vertical_spacing if vertical_spacing is not None else grid_spacing
    
    # Define tag centers in world space
    # 2x3 grid layout with tag 0 at origin
    # X-axis: left-right, Y-axis: down, Z-axis: out toward camera
    tag_centers = {
        0: np.array([0.0, 0.0, 0.0]),                    # Top-left
        1: np.array([h_spacing, 0.0, 0.0]),              # Top-right
        2: np.array([0.0, v_spacing, 0.0]),              # Middle-left
        3: np.array([h_spacing, v_spacing, 0.0]),        # Middle-right
        4: np.array([0.0, 2*v_spacing, 0.0]),            # Bottom-left
        5: np.array([h_spacing, 2*v_spacing, 0.0]),      # Bottom-right
    }
    
    # Define 4 corners for each tag (relative to tag center)
    # Assuming tags lie in XY plane, facing +Z
    corner_offsets = np.array([
        [-tag_size/2, -tag_size/2, 0],
        [tag_size/2, -tag_size/2, 0],
        [tag_size/2, tag_size/2, 0],
        [-tag_size/2, tag_size/2, 0]
    ], dtype=np.float32)
    
    # Compute 4 corners for each tag
    tag_world_positions = {}
    for tag_id, center in tag_centers.items():
        corners = center + corner_offsets
        tag_world_positions[tag_id] = corners
    
    return tag_world_positions


def estimate_camera_pose(tag_detections_2d, tag_world_positions, camera_matrix, dist_coeffs):
    """
    Estimate camera pose from detected tags using solvePnP.
    
    Args:
        tag_detections_2d: {tag_id: 2D corners} for visible tags
        tag_world_positions: {tag_id: 3D corners} for all tags
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        
    Returns:
        c2w: Camera-to-world 4x4 matrix, or None if estimation failed
    """
    if len(tag_detections_2d) == 0:
        return None
    
    # Collect all corresponding 3D-2D points
    object_points = []
    image_points = []
    
    for tag_id, corners_2d in tag_detections_2d.items():
        if tag_id not in tag_world_positions:
            continue  # Skip unknown tags
        
        corners_3d = tag_world_positions[tag_id]
        object_points.append(corners_3d)
        image_points.append(corners_2d)
    
    if len(object_points) == 0:
        return None
    
    # Concatenate all points
    object_points = np.vstack(object_points).astype(np.float32)
    image_points = np.vstack(image_points).astype(np.float32)
    
    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return None
    
    # Convert to c2w matrix
    R, _ = cv2.Rodrigues(rvec)
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = tvec.flatten()
    c2w = np.linalg.inv(w2c)
    
    return c2w


def process_images_multitag(image_folder, camera_matrix, dist_coeffs, 
                           tag_size=0.02, grid_spacing=0.10,
                           horizontal_spacing=None, vertical_spacing=None,
                           min_tags_per_image=2):
    """
    Process all images with multiple ArUco tags.
    
    Args:
        image_folder: Path to folder with images
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        tag_size: Size of each tag in meters
        grid_spacing: Uniform spacing between tags in meters
        horizontal_spacing: Horizontal (X) center-to-center spacing
        vertical_spacing: Vertical (Y) center-to-center spacing
        min_tags_per_image: Minimum number of tags required per image
        
    Returns:
        list: [{'image': img_rgb, 'c2w': c2w, 'name': filename, 'num_tags': n}]
    """
    print(f"\n{'='*60}")
    print("Processing Multi-Tag Images")
    print(f"{'='*60}")
    
    # Setup ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    
    # Define world tag positions
    tag_world_positions = build_tag_world_positions(tag_size, grid_spacing, 
                                                     horizontal_spacing, vertical_spacing)
    print(f"World coordinate system defined with {len(tag_world_positions)} tags")
    print(f"  Tag size: {tag_size*100:.1f} cm")
    if horizontal_spacing is not None or vertical_spacing is not None:
        h_sp = horizontal_spacing if horizontal_spacing else grid_spacing
        v_sp = vertical_spacing if vertical_spacing else grid_spacing
        print(f"  Horizontal spacing: {h_sp*100:.1f} cm")
        print(f"  Vertical spacing: {v_sp*100:.1f} cm")
    else:
        print(f"  Grid spacing: {grid_spacing*100:.1f} cm")
    
    # Find all images
    image_paths = sorted(
        glob.glob(os.path.join(image_folder, '*.jpg')) + 
        glob.glob(os.path.join(image_folder, '*.png')) +
        glob.glob(os.path.join(image_folder, '*.jpeg'))
    )
    
    print(f"\nFound {len(image_paths)} images in {image_folder}")
    print(f"Minimum tags per image: {min_tags_per_image}")
    
    # Process each image
    poses = []
    tag_counts = defaultdict(int)
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect tags
        tag_detections = detect_tags_in_image(img_rgb, aruco_dict, aruco_params)
        
        num_tags = len(tag_detections)
        if num_tags == 0:
            print(f"Warning: No tags in {os.path.basename(img_path)}")
            continue
        
        if num_tags < min_tags_per_image:
            print(f"Warning: Only {num_tags} tag(s) in {os.path.basename(img_path)} (min: {min_tags_per_image})")
            continue
        
        # Estimate camera pose
        c2w = estimate_camera_pose(tag_detections, tag_world_positions, 
                                   camera_matrix, dist_coeffs)
        
        if c2w is None:
            print(f"Warning: Pose estimation failed for {os.path.basename(img_path)}")
            continue
        
        # Store result
        poses.append({
            'image': img_rgb,
            'c2w': c2w,
            'name': os.path.basename(img_path),
            'num_tags': num_tags,
            'tag_ids': list(tag_detections.keys())
        })
        
        # Track which tags were seen
        for tag_id in tag_detections.keys():
            tag_counts[tag_id] += 1
    
    print(f"\nSuccessfully processed {len(poses)} images")
    print(f"\nTag visibility statistics:")
    for tag_id in sorted(tag_counts.keys()):
        count = tag_counts[tag_id]
        percentage = 100 * count / len(poses)
        print(f"  Tag {tag_id}: {count}/{len(poses)} images ({percentage:.1f}%)")
    
    if len(poses) == 0:
        raise ValueError("No valid poses found! Check tag detection and calibration.")
    
    return poses


def create_dataset(image_folder='assets/rivian',
                  calibration_file='calibration_results.npz',
                  output_file='rivian_data.npz',
                  target_width=400,
                  tag_size=0.02,
                  grid_spacing=0.10,
                  horizontal_spacing=None,
                  vertical_spacing=None,
                  min_tags_per_image=2,
                  train_ratio=0.7,
                  val_ratio=0.15):
    """
    Create NeRF dataset with multi-tag setup.
    """
    print("="*60)
    print("Creating Multi-Tag NeRF Dataset")
    print("="*60)
    print(f"Image folder: {image_folder}")
    print(f"Target width: {target_width}px")
    print(f"Output: {output_file}")
    print()
    
    # Load calibration
    K_orig, dist_coeffs = load_calibration(calibration_file)
    print(f"Camera intrinsics:")
    print(K_orig)
    print(f"Focal: fx={K_orig[0,0]:.2f}, fy={K_orig[1,1]:.2f}")
    
    # Process images
    poses = process_images_multitag(
        image_folder, K_orig, dist_coeffs,
        tag_size=tag_size,
        grid_spacing=grid_spacing,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
        min_tags_per_image=min_tags_per_image
    )
    
    # Get original size
    h_orig, w_orig = poses[0]['image'].shape[:2]
    print(f"\nOriginal image size: {w_orig}x{h_orig}")
    
    # Calculate resize
    scale = target_width / w_orig
    new_w = target_width
    new_h = int(h_orig * scale)
    
    # Make divisible by 8
    new_w = (new_w // 8) * 8
    new_h = (new_h // 8) * 8
    
    # Actual scales
    actual_scale_w = new_w / w_orig
    actual_scale_h = new_h / h_orig
    
    print(f"Target size: {new_w}x{new_h}")
    print(f"Scale: w={actual_scale_w:.4f}, h={actual_scale_h:.4f}")
    
    # Adjust camera matrix
    K_new = K_orig.copy()
    K_new[0, 0] *= actual_scale_w  # fx
    K_new[1, 1] *= actual_scale_h  # fy
    K_new[0, 2] *= actual_scale_w  # cx
    K_new[1, 2] *= actual_scale_h  # cy
    
    print(f"\nAdjusted camera matrix:")
    print(K_new)
    print(f"Focal: fx={K_new[0,0]:.2f}, fy={K_new[1,1]:.2f}")
    print(f"Focal/width ratio: {K_new[0,0]/new_w:.4f} (should be ~0.5-1.5)")
    
    # Resize all images
    print(f"\nResizing {len(poses)} images...")
    images = []
    c2ws = []
    
    for pose in tqdm(poses, desc="Resizing"):
        img = pose['image']
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        images.append(resized)
        c2ws.append(pose['c2w'])
    
    images = np.array(images, dtype=np.uint8)
    c2ws = np.array(c2ws, dtype=np.float32)
    
    print(f"\nProcessed:")
    print(f"  Images: {images.shape}")
    print(f"  c2ws: {c2ws.shape}")
    
    # Split dataset
    N = len(images)
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    images_train = images[train_idx]
    c2ws_train = c2ws[train_idx]
    images_val = images[val_idx]
    c2ws_val = c2ws[val_idx]
    c2ws_test = c2ws[test_idx]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(images_train)} images")
    print(f"  Val: {len(images_val)} images")
    print(f"  Test: {len(c2ws_test)} poses")
    
    # Save
    focal = K_new[0, 0]
    
    print(f"\nSaving to {output_file}...")
    np.savez(
        output_file,
        images_train=images_train,
        c2ws_train=c2ws_train,
        images_val=images_val,
        c2ws_val=c2ws_val,
        c2ws_test=c2ws_test,
        focal=focal,
        camera_matrix=K_new,
        image_shape=np.array([new_h, new_w, 3])
    )
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n✓ Dataset saved!")
    print(f"  File: {output_file}")
    print(f"  Size: {file_size_mb:.1f} MB")
    print(f"  Images: {new_w}x{new_h}")
    print(f"  Focal: {focal:.2f}")
    
    print(f"\n{'='*60}")
    print("Done! Ready for NeRF training.")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create NeRF dataset from multi-tag images')
    parser.add_argument('--image_folder', type=str, default='assets/rivian',
                       help='Folder with object images')
    parser.add_argument('--calibration', type=str, default='calibration_results.npz',
                       help='Calibration file')
    parser.add_argument('--output', type=str, default='rivian_data.npz',
                       help='Output dataset file')
    parser.add_argument('--width', type=int, default=400,
                       help='Target image width')
    parser.add_argument('--tag_size', type=float, default=0.02,
                       help='ArUco tag size in meters (default: 0.02 = 2cm)')
    parser.add_argument('--grid_spacing', type=float, default=0.10,
                       help='Uniform spacing between tags in meters (default: 0.10 = 10cm)')
    parser.add_argument('--horizontal_spacing', type=float, default=None,
                       help='Horizontal (X) center-to-center spacing in meters (overrides grid_spacing)')
    parser.add_argument('--vertical_spacing', type=float, default=None,
                       help='Vertical (Y) center-to-center spacing in meters (overrides grid_spacing)')
    parser.add_argument('--min_tags', type=int, default=2,
                       help='Minimum tags per image (default: 2)')
    
    args = parser.parse_args()
    
    create_dataset(
        image_folder=args.image_folder,
        calibration_file=args.calibration,
        output_file=args.output,
        target_width=args.width,
        tag_size=args.tag_size,
        grid_spacing=args.grid_spacing,
        horizontal_spacing=args.horizontal_spacing,
        vertical_spacing=args.vertical_spacing,
        min_tags_per_image=args.min_tags
    )

