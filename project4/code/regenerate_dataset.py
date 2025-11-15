"""
Regenerate my_data.npz with Downsized Images
Project 4 - CS 180

This script regenerates the dataset with smaller images to reduce file size.
It loads the original images, resizes them, adjusts camera intrinsics, and saves.
"""

import cv2
import numpy as np
import glob
import os
from tqdm import tqdm


def load_calibration_results(calibration_file='calibration_results.npz'):
    """Load camera calibration results."""
    if not os.path.exists(calibration_file):
        raise FileNotFoundError(f"Calibration file not found: {calibration_file}")
    
    data = np.load(calibration_file)
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']
    
    print(f"Loaded calibration from: {calibration_file}")
    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Distortion coefficients: {dist_coeffs.flatten()}")
    
    return camera_matrix, dist_coeffs


def estimate_poses_from_images(image_folder, camera_matrix, dist_coeffs, tag_size=0.02):
    """Estimate camera poses from images with ArUco tags."""
    # Create ArUco dictionary and detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    
    # Get image paths
    image_paths = glob.glob(os.path.join(image_folder, '*.*'))
    image_paths = [p for p in image_paths if p.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_paths.sort()
    
    print(f"\nFound {len(image_paths)} images in {image_folder}")
    
    # Define ArUco tag corners in world coordinates
    obj_points = np.array([
        [0, 0, 0],
        [tag_size, 0, 0],
        [tag_size, tag_size, 0],
        [0, tag_size, 0]
    ], dtype=np.float32)
    
    poses = []
    
    for img_path in tqdm(image_paths, desc="Estimating poses"):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect ArUco markers
        corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)
        
        if ids is None or len(ids) == 0:
            print(f"Warning: No tags detected in {os.path.basename(img_path)}")
            continue
        
        # Use first detected tag
        img_points = corners[0].reshape(-1, 2).astype(np.float32)
        
        # Solve PnP to get camera pose
        success, rvec, tvec = cv2.solvePnP(
            obj_points, img_points, camera_matrix, dist_coeffs
        )
        
        if not success:
            print(f"Warning: PnP failed for {os.path.basename(img_path)}")
            continue
        
        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Create world-to-camera matrix
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = tvec.flatten()
        
        # Invert to get camera-to-world
        c2w = np.linalg.inv(w2c)
        
        poses.append({
            'image': img,
            'c2w': c2w,
            'name': os.path.basename(img_path)
        })
    
    print(f"Successfully processed {len(poses)} images with valid poses")
    return poses


def resize_image_and_adjust_intrinsics(image, K, target_size=400):
    """
    Resize image and adjust camera intrinsics accordingly.
    
    Args:
        image: Input image (H, W, 3)
        K: Camera intrinsic matrix (3, 3)
        target_size: Target size for the smaller dimension
        
    Returns:
        resized_image: Resized image
        K_new: Adjusted intrinsic matrix
        actual_scale: The actual scale factor used
    """
    h, w = image.shape[:2]
    
    # Calculate scale factor (maintain aspect ratio)
    scale = target_size / min(h, w)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Make dimensions divisible by 8
    new_w = (new_w // 8) * 8
    new_h = (new_h // 8) * 8
    
    # Calculate actual scale factors after rounding
    actual_scale_w = new_w / w
    actual_scale_h = new_h / h
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Adjust intrinsics with actual scales
    K_new = K.copy()
    K_new[0, 0] *= actual_scale_w  # fx (width dimension)
    K_new[1, 1] *= actual_scale_h  # fy (height dimension)
    K_new[0, 2] *= actual_scale_w  # cx (width dimension)
    K_new[1, 2] *= actual_scale_h  # cy (height dimension)
    
    return resized, K_new, actual_scale_w


def regenerate_dataset_with_resize(image_folder='assets/object_tag',
                                   calibration_file='calibration_results.npz',
                                   output_file='my_data.npz',
                                   target_size=400,
                                   tag_size=0.02,
                                   train_ratio=0.7,
                                   val_ratio=0.15,
                                   test_ratio=0.15,
                                   alpha=0):
    """
    Regenerate dataset with resized images.
    
    Args:
        image_folder: Folder with object images
        calibration_file: Path to calibration results
        output_file: Output .npz file
        target_size: Target size for smaller dimension (pixels)
        tag_size: Physical size of ArUco tag in meters
        train_ratio: Fraction of images for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for test
        alpha: Alpha for getOptimalNewCameraMatrix (0=max crop, 1=keep all)
    """
    print("="*60)
    print("Regenerating Dataset with Resized Images")
    print("="*60)
    print(f"Target image size: {target_size}px (smaller dimension)")
    print()
    
    # Load calibration
    camera_matrix, dist_coeffs = load_calibration_results(calibration_file)
    
    # Estimate poses
    poses = estimate_poses_from_images(image_folder, camera_matrix, dist_coeffs, tag_size)
    
    if len(poses) == 0:
        raise ValueError("No valid poses found!")
    
    # Get original image size
    first_img = poses[0]['image']
    h, w = first_img.shape[:2]
    print(f"\nOriginal image size: {w}x{h}")
    
    # Compute optimal new camera matrix for undistortion
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=alpha
    )
    
    x, y, w_roi, h_roi = roi
    print(f"ROI after undistortion: x={x}, y={y}, width={w_roi}, height={h_roi}")
    
    # Make dimensions consistent
    w_roi = (w_roi // 8) * 8
    h_roi = (h_roi // 8) * 8
    
    # Update principal point for crop
    new_camera_matrix_corrected = new_camera_matrix.copy()
    new_camera_matrix_corrected[0, 2] -= x
    new_camera_matrix_corrected[1, 2] -= y
    
    print(f"\nUndistorting and resizing {len(poses)} images...")
    processed_images = []
    c2w_matrices = []
    
    # Process first image to get final dimensions
    test_img = poses[0]['image']
    test_undistorted = cv2.undistort(test_img, camera_matrix, dist_coeffs, None, new_camera_matrix)
    test_cropped = test_undistorted[y:y+h_roi, x:x+w_roi]
    test_resized, K_resized, actual_scale = resize_image_and_adjust_intrinsics(
        test_cropped, new_camera_matrix_corrected, target_size
    )
    
    final_h, final_w = test_resized.shape[:2]
    print(f"Final image size after resize: {final_w}x{final_h}")
    print(f"Resize scale factor: {actual_scale:.4f}")
    print(f"\nOriginal camera matrix (after crop):")
    print(new_camera_matrix_corrected)
    print(f"\nAdjusted camera matrix (after resize):")
    print(K_resized)
    
    focal = K_resized[0, 0]
    focal_y = K_resized[1, 1]
    print(f"\nFinal focal length: fx={focal:.2f}, fy={focal_y:.2f}")
    print(f"Focal/width ratio: {focal/final_w:.4f} (should be ~0.5-1.5 for normal cameras)")
    
    # Process all images
    for pose in tqdm(poses, desc="Processing images"):
        img = pose['image']
        
        # Undistort
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
        
        # Crop
        cropped = undistorted[y:y+h_roi, x:x+w_roi]
        
        # Resize
        resized, _, _ = resize_image_and_adjust_intrinsics(cropped, new_camera_matrix_corrected, target_size)
        
        # Ensure consistent dimensions
        if resized.shape[:2] != (final_h, final_w):
            resized = cv2.resize(resized, (final_w, final_h), interpolation=cv2.INTER_AREA)
        
        processed_images.append(resized)
        c2w_matrices.append(pose['c2w'])
    
    # Convert to arrays
    images = np.array(processed_images, dtype=np.uint8)
    c2ws = np.array(c2w_matrices, dtype=np.float32)
    
    print(f"\nProcessed images shape: {images.shape}")
    print(f"c2w matrices shape: {c2ws.shape}")
    
    # Calculate file size
    estimated_size_mb = (images.nbytes + c2ws.nbytes) / (1024 * 1024)
    print(f"Estimated dataset size: {estimated_size_mb:.1f} MB")
    
    # Split into train/val/test
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
    print(f"  Training: {len(images_train)} images")
    print(f"  Validation: {len(images_val)} images")
    print(f"  Test: {len(c2ws_test)} poses")
    
    # Save dataset
    print(f"\nSaving dataset to: {output_file}")
    np.savez(
        output_file,
        images_train=images_train,
        c2ws_train=c2ws_train,
        images_val=images_val,
        c2ws_val=c2ws_val,
        c2ws_test=c2ws_test,
        focal=focal,
        camera_matrix=K_resized,
        image_shape=np.array([final_h, final_w, 3])
    )
    
    # Verify saved file size
    actual_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n✓ Dataset saved successfully!")
    print(f"  File size: {actual_size_mb:.1f} MB")
    print(f"  Image size: {final_w}x{final_h}")
    print(f"  Focal length: {focal:.2f}")
    
    print(f"\n{'='*60}")
    print("Regeneration Complete!")
    print(f"{'='*60}")
    
    return images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Regenerate dataset with resized images')
    parser.add_argument('--image_folder', type=str, default='assets/object_tag',
                       help='Folder containing object images')
    parser.add_argument('--calibration', type=str, default='calibration_results.npz',
                       help='Calibration file')
    parser.add_argument('--output', type=str, default='my_data.npz',
                       help='Output dataset file')
    parser.add_argument('--target_size', type=int, default=400,
                       help='Target size for smaller dimension (pixels)')
    parser.add_argument('--tag_size', type=float, default=0.02,
                       help='Physical size of ArUco tag in meters')
    
    args = parser.parse_args()
    
    regenerate_dataset_with_resize(
        image_folder=args.image_folder,
        calibration_file=args.calibration,
        output_file=args.output,
        target_size=args.target_size,
        tag_size=args.tag_size
    )

