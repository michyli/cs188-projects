"""
Create NeRF dataset from object images with ArUco tag
Simple approach: resize images and adjust intrinsics proportionally
"""

import cv2
import numpy as np
import glob
import os
from tqdm import tqdm


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
        f"  Tried: {alt_path}\n"
        f"  Please specify with --calibration flag"
    )


def estimate_poses(image_folder, camera_matrix, dist_coeffs, tag_size=0.02):
    """Estimate camera poses from images with ArUco tags."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    
    image_paths = sorted(glob.glob(os.path.join(image_folder, '*.jpg')) + 
                        glob.glob(os.path.join(image_folder, '*.png')) +
                        glob.glob(os.path.join(image_folder, '*.jpeg')))
    
    print(f"\nFound {len(image_paths)} images in {image_folder}")
    
    obj_points = np.array([
        [0, 0, 0],
        [tag_size, 0, 0],
        [tag_size, tag_size, 0],
        [0, tag_size, 0]
    ], dtype=np.float32)
    
    poses = []
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect ArUco
        corners, ids, _ = cv2.aruco.detectMarkers(img_rgb, aruco_dict, parameters=aruco_params)
        
        if ids is None:
            print(f"Warning: No tag in {os.path.basename(img_path)}")
            continue
        
        img_points = corners[0].reshape(-1, 2).astype(np.float32)
        
        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
        
        if not success:
            print(f"Warning: PnP failed for {os.path.basename(img_path)}")
            continue
        
        # Get c2w matrix
        R, _ = cv2.Rodrigues(rvec)
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = tvec.flatten()
        c2w = np.linalg.inv(w2c)
        
        poses.append({
            'image': img_rgb,
            'c2w': c2w,
            'name': os.path.basename(img_path)
        })
    
    print(f"Successfully processed {len(poses)} images")
    return poses


def create_dataset(image_folder='assets/object_tag_new',
                  calibration_file='calibration_results.npz',
                  output_file='my_data.npz',
                  target_width=400,
                  tag_size=0.02,
                  train_ratio=0.7,
                  val_ratio=0.15):
    """
    Create NeRF dataset with resized images.
    """
    print("="*60)
    print("Creating NeRF Dataset")
    print("="*60)
    print(f"Image folder: {image_folder}")
    print(f"Target width: {target_width}px")
    print()
    
    # Load calibration
    K_orig, dist_coeffs = load_calibration(calibration_file)
    print(f"Camera matrix:")
    print(K_orig)
    print(f"Focal: fx={K_orig[0,0]:.2f}, fy={K_orig[1,1]:.2f}")
    
    # Estimate poses
    poses = estimate_poses(image_folder, K_orig, dist_coeffs, tag_size)
    
    if len(poses) == 0:
        raise ValueError("No valid poses found!")
    
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
    
    parser = argparse.ArgumentParser(description='Create NeRF dataset from object images')
    parser.add_argument('--image_folder', type=str, default='assets/object_tag_new',
                       help='Folder with object images')
    parser.add_argument('--calibration', type=str, default='calibration_results.npz',
                       help='Calibration file')
    parser.add_argument('--output', type=str, default='my_data.npz',
                       help='Output dataset file')
    parser.add_argument('--width', type=int, default=400,
                       help='Target image width')
    parser.add_argument('--tag_size', type=float, default=0.02,
                       help='ArUco tag size in meters (default: 0.02 = 2cm)')
    
    args = parser.parse_args()
    
    create_dataset(
        image_folder=args.image_folder,
        calibration_file=args.calibration,
        output_file=args.output,
        target_width=args.width,
        tag_size=args.tag_size
    )

