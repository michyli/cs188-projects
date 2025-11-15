"""
Camera Calibration using ArUco Markers
Project 4A - CS188

This script performs camera calibration by:
1. Detecting ArUco markers in calibration images
2. Extracting corner coordinates
3. Computing camera intrinsics and distortion coefficients using cv2.calibrateCamera()
"""

import cv2
import numpy as np
import glob
import os


def detect_aruco_tags(image, aruco_dict, aruco_params):
    """
    Detect ArUco markers in an image.
    
    Args:
        image: Input image (grayscale or color)
        aruco_dict: ArUco dictionary
        aruco_params: ArUco detector parameters
        
    Returns:
        corners: List of detected corner coordinates
        ids: Array of detected marker IDs
    """
    corners, ids, rejected = cv2.aruco.detectMarkers(
        image, aruco_dict, parameters=aruco_params
    )
    return corners, ids


def get_aruco_world_coordinates(tag_size=0.02):
    """
    Define 3D world coordinates for ArUco tag corners.
    
    Args:
        tag_size: Physical size of the ArUco tag in meters (default: 0.02m = 2cm)
        
    Returns:
        numpy array of shape (4, 3) with 3D coordinates of the 4 corners
        Corners are ordered: top-left, top-right, bottom-right, bottom-left
    """
    obj_points = np.array([
        [0, 0, 0],              # Top-left
        [tag_size, 0, 0],       # Top-right
        [tag_size, tag_size, 0], # Bottom-right
        [0, tag_size, 0]        # Bottom-left
    ], dtype=np.float32)
    
    return obj_points


def calibrate_camera(image_folder, tag_size=0.02, visualize=True):
    """
    Perform camera calibration using ArUco markers.
    
    Args:
        image_folder: Path to folder containing calibration images
        tag_size: Physical size of ArUco tags in meters
        visualize: Whether to visualize detected markers
        
    Returns:
        ret: RMS re-projection error
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        rvecs: Rotation vectors for each image
        tvecs: Translation vectors for each image
    """
    # Create ArUco dictionary and detector parameters (4x4 tags)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    
    # Get list of calibration images
    image_paths = glob.glob(os.path.join(image_folder, '*.*'))
    image_paths = [p for p in image_paths if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {image_folder}")
    
    print(f"Found {len(image_paths)} calibration images")
    
    # Prepare object points for ArUco tags
    obj_points_template = get_aruco_world_coordinates(tag_size)
    
    # Arrays to store object points and image points from all images
    all_obj_points = []  # 3D points in real world space
    all_img_points = []  # 2D points in image plane
    
    valid_images = 0
    total_tags_detected = 0
    
    # Process each calibration image
    for i, image_path in enumerate(image_paths):
        print(f"\nProcessing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"  Warning: Could not read image {image_path}")
            continue
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        corners, ids = detect_aruco_tags(gray, aruco_dict, aruco_params)
        
        # Check if any markers were detected
        if ids is not None and len(ids) > 0:
            print(f"  Detected {len(ids)} ArUco marker(s)")
            total_tags_detected += len(ids)
            
            # Collect all object points and image points for THIS IMAGE
            img_obj_points = []
            img_img_points = []
            
            # Process each detected marker
            for j, marker_id in enumerate(ids):
                # Extract corner coordinates for this marker
                # corners[j] has shape (1, 4, 2), we need shape (4, 2)
                marker_corners = corners[j].reshape(4, 2)
                
                # Collect points for this image (all tags combined)
                img_obj_points.append(obj_points_template)
                img_img_points.append(marker_corners.astype(np.float32))
            
            # Combine all tags from this image into single arrays
            # This is the correct format for cv2.calibrateCamera()
            img_obj_points = np.vstack(img_obj_points)  # Shape: (N*4, 3) where N is number of tags
            img_img_points = np.vstack(img_img_points)  # Shape: (N*4, 2)
            
            # Add the combined points for this image
            all_obj_points.append(img_obj_points)
            all_img_points.append(img_img_points)
            
            valid_images += 1
            
            # Visualize detected markers (optional)
            if visualize:
                vis_image = image.copy()
                cv2.aruco.drawDetectedMarkers(vis_image, corners, ids)
                
                # Resize for display if image is too large
                height, width = vis_image.shape[:2]
                if width > 1200:
                    scale = 1200 / width
                    new_width = 1200
                    new_height = int(height * scale)
                    vis_image = cv2.resize(vis_image, (new_width, new_height))
                
                cv2.imshow('Detected ArUco Markers', vis_image)
                cv2.waitKey(500)  # Display for 500ms
        else:
            print(f"  Warning: No ArUco markers detected in this image")
    
    if visualize:
        cv2.destroyAllWindows()
    
    # Check if we have enough data for calibration
    if len(all_obj_points) == 0:
        raise ValueError("No ArUco markers detected in any images. Cannot perform calibration.")
    
    print(f"\n{'='*60}")
    print(f"Calibration Summary:")
    print(f"  Total images processed: {len(image_paths)}")
    print(f"  Images with detected tags: {valid_images}")
    print(f"  Total tags detected: {total_tags_detected}")
    print(f"  Total images for calibration: {len(all_obj_points)}")
    print(f"  Total points per image (first): {len(all_obj_points[0])}")
    print(f"{'='*60}\n")
    
    # Get image size from the first valid image
    first_image = cv2.imread(image_paths[0])
    image_size = (first_image.shape[1], first_image.shape[0])  # (width, height)
    print(f"Image size for calibration: {image_size[0]}x{image_size[1]}")
    
    # Perform camera calibration
    print("Running camera calibration (this may take a few seconds)...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        all_obj_points,
        all_img_points,
        image_size,
        None,  # Initial camera matrix (None = auto-initialize)
        None   # Initial distortion coefficients (None = auto-initialize)
    )
    
    print(f"Calibration complete!")
    print(f"\nRMS Re-projection Error: {ret:.4f} pixels")
    
    return ret, camera_matrix, dist_coeffs, rvecs, tvecs


def print_calibration_results(ret, camera_matrix, dist_coeffs):
    """
    Print the calibration results in a readable format.
    
    Args:
        ret: RMS re-projection error
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients
    """
    print("\n" + "="*60)
    print("CAMERA CALIBRATION RESULTS")
    print("="*60)
    
    print(f"\nRMS Re-projection Error: {ret:.6f} pixels")
    print("(Lower is better - typically < 1.0 is good)")
    
    print("\nCamera Intrinsic Matrix (K):")
    print(camera_matrix)
    
    # Extract focal lengths and principal point
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    print(f"\nFocal Lengths:")
    print(f"  fx = {fx:.2f} pixels")
    print(f"  fy = {fy:.2f} pixels")
    
    print(f"\nPrincipal Point (optical center):")
    print(f"  cx = {cx:.2f} pixels")
    print(f"  cy = {cy:.2f} pixels")
    
    print("\nDistortion Coefficients:")
    print(dist_coeffs.ravel())
    print(f"  k1 = {dist_coeffs[0, 0]:.6f}")
    print(f"  k2 = {dist_coeffs[0, 1]:.6f}")
    print(f"  p1 = {dist_coeffs[0, 2]:.6f}")
    print(f"  p2 = {dist_coeffs[0, 3]:.6f}")
    print(f"  k3 = {dist_coeffs[0, 4]:.6f}")
    
    print("="*60 + "\n")


def save_calibration_results(camera_matrix, dist_coeffs, output_file='calibration_results.npz'):
    """
    Save calibration results to a file for later use.
    
    Args:
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        output_file: Output file path
    """
    np.savez(output_file, 
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs)
    print(f"Calibration results saved to: {output_file}")


def load_calibration_results(calibration_file='calibration_results.npz'):
    """
    Load previously saved calibration results.
    
    Args:
        calibration_file: Path to the calibration file
        
    Returns:
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients
    """
    if not os.path.exists(calibration_file):
        raise FileNotFoundError(f"Calibration file '{calibration_file}' not found. Run calibration first.")
    
    data = np.load(calibration_file)
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']
    
    print(f"Loaded calibration from: {calibration_file}")
    return camera_matrix, dist_coeffs


def estimate_camera_poses(image_folder, camera_matrix, dist_coeffs, tag_size=0.02, visualize=False):
    """
    Estimate camera pose for each image using PnP.
    
    Args:
        image_folder: Path to folder containing object images with ArUco tag
        camera_matrix: 3x3 camera intrinsic matrix from calibration
        dist_coeffs: Distortion coefficients from calibration
        tag_size: Physical size of ArUco tag in meters
        visualize: Whether to visualize detected markers
        
    Returns:
        poses: List of dictionaries containing pose information for each image
               Each dict has: 'image_path', 'rvec', 'tvec', 'R', 'c2w', 'image'
    """
    # Create ArUco dictionary and detector parameters (4x4 tags)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    
    # Get list of images
    image_paths = glob.glob(os.path.join(image_folder, '*.*'))
    image_paths = [p for p in image_paths if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {image_folder}")
    
    print(f"\nFound {len(image_paths)} object images")
    
    # Define 3D object points for ArUco tag corners (world coordinates)
    obj_points = get_aruco_world_coordinates(tag_size)
    
    # Store pose information for each image
    poses = []
    successful_poses = 0
    
    print("\nEstimating camera poses using PnP...")
    print("="*60)
    
    # Process each image
    for i, image_path in enumerate(image_paths):
        print(f"\nProcessing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"  Warning: Could not read image {image_path}")
            continue
        
        H, W = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        corners, ids = detect_aruco_tags(gray, aruco_dict, aruco_params)
        
        # Check if any markers were detected
        if ids is not None and len(ids) > 0:
            print(f"  Detected {len(ids)} ArUco marker(s)")
            
            # Use the first detected tag (assuming single tag setup)
            marker_corners = corners[0].reshape(4, 2)
            
            # Solve PnP to get camera pose
            success, rvec, tvec = cv2.solvePnP(
                obj_points,           # 3D object points
                marker_corners,       # 2D image points
                camera_matrix,        # Camera intrinsic matrix
                dist_coeffs,          # Distortion coefficients
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                print(f"  ✓ Pose estimation successful")
                
                # Convert rotation vector to rotation matrix
                R, _ = cv2.Rodrigues(rvec)
                
                # solvePnP returns the transformation from world to camera
                # tvec is the camera position in the tag's coordinate system
                # R rotates world points to camera frame
                
                # Create world-to-camera transformation
                w2c = np.eye(4)
                w2c[:3, :3] = R
                w2c[:3, 3] = tvec.ravel()
                
                # Invert to get camera-to-world transformation
                # c2w tells us where the camera is in world coordinates
                c2w = np.linalg.inv(w2c)
                
                # Alternative computation (should be the same):
                # Camera position in world = -R^T @ tvec
                camera_pos_world = -R.T @ tvec.ravel()
                
                # Store pose information
                pose_info = {
                    'image_path': image_path,
                    'image_name': os.path.basename(image_path),
                    'rvec': rvec,
                    'tvec': tvec,
                    'R': R,
                    'w2c': w2c,
                    'c2w': c2w,
                    'image': image,
                    'width': W,
                    'height': H
                }
                poses.append(pose_info)
                successful_poses += 1
                
                # Print pose information for debugging
                print(f"  Translation (tvec): {tvec.ravel()}")
                print(f"  Camera position (c2w method): {c2w[:3, 3]}")
                print(f"  Camera position (direct calc): {camera_pos_world}")
                print(f"  Distance from origin: {np.linalg.norm(camera_pos_world):.3f}m")
                
                # Visualize detected markers (optional)
                if visualize:
                    vis_image = image.copy()
                    cv2.aruco.drawDetectedMarkers(vis_image, corners, ids)
                    
                    # Draw axis for visualization
                    cv2.drawFrameAxes(vis_image, camera_matrix, dist_coeffs, rvec, tvec, tag_size * 0.5)
                    
                    # Resize for display if too large
                    if W > 1200:
                        scale = 1200 / W
                        new_width = 1200
                        new_height = int(H * scale)
                        vis_image = cv2.resize(vis_image, (new_width, new_height))
                    
                    cv2.imshow('Camera Pose Estimation', vis_image)
                    cv2.waitKey(500)
            else:
                print(f"  ✗ Pose estimation failed")
        else:
            print(f"  Warning: No ArUco markers detected in this image")
    
    if visualize:
        cv2.destroyAllWindows()
    
    print(f"\n{'='*60}")
    print(f"Pose Estimation Summary:")
    print(f"  Total images processed: {len(image_paths)}")
    print(f"  Successful pose estimations: {successful_poses}")
    print(f"  Failed/Skipped: {len(image_paths) - successful_poses}")
    print(f"{'='*60}\n")
    
    if len(poses) == 0:
        raise ValueError("No successful pose estimations. Cannot proceed.")
    
    return poses


def visualize_camera_poses(poses, camera_matrix, frustum_scale=0.005, position_scale=1.0):
    """
    Visualize camera poses using viser.
    
    Args:
        poses: List of pose dictionaries from estimate_camera_poses()
        camera_matrix: 3x3 camera intrinsic matrix
        frustum_scale: Scale of camera frustums (smaller = smaller frustums, default: 0.005)
        position_scale: Scale to apply to camera positions (larger = more spread out, default: 1.0)
    """
    try:
        import viser
        import time
    except ImportError:
        print("\nWarning: viser not installed. Install with: pip install viser")
        print("Skipping visualization.")
        return
    
    print("\nStarting 3D visualization with viser...")
    print("Opening visualization server (this may take a moment)...")
    
    # Note: share=False for local visualization (avoids encoding issues on Windows)
    server = viser.ViserServer(share=False)
    
    print("\n" + "="*60)
    print("VISUALIZATION SERVER STARTED!")
    print("="*60)
    print("\n🌐 Open this URL in your web browser:")
    print("   http://localhost:8080")
    print("\n   (Check terminal output above for the exact port if different)")
    print(f"\nVisualization settings:")
    print(f"  - Frustum scale: {frustum_scale} (smaller = tinier cameras)")
    print(f"  - Position scale: {position_scale} (larger = more spread out)")
    print("="*60)
    
    # Coordinate system transformation: OpenCV to Graphics convention
    # OpenCV: X right, Y down, Z forward
    # Graphics: X right, Y up, Z backward (or forward depending on convention)
    # Flip Y and Z axes
    flip_transform = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ])
    
    # Add camera frustums for each pose
    for i, pose in enumerate(poses):
        c2w_opencv = pose['c2w']
        img = pose['image']
        H = pose['height']
        W = pose['width']
        K = camera_matrix
        
        # Apply coordinate system transformation
        c2w = flip_transform @ c2w_opencv
        
        # Scale the camera positions to spread them out more
        c2w_scaled = c2w.copy()
        c2w_scaled[:3, 3] *= position_scale
        
        # Calculate field of view
        fov = 2 * np.arctan2(H / 2, K[0, 0])
        aspect = W / H
        
        # Convert image from BGR to RGB for visualization
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Add camera frustum to the scene
        server.scene.add_camera_frustum(
            f"/cameras/{i}",  # Unique name for this camera
            fov=fov,  # Field of view
            aspect=aspect,  # Aspect ratio
            scale=frustum_scale,  # Scale of the camera frustum (smaller = tinier)
            wxyz=viser.transforms.SO3.from_matrix(c2w_scaled[:3, :3]).wxyz,  # Orientation in quaternion format
            position=c2w_scaled[:3, 3],  # Position of the camera (scaled)
            image=img_rgb  # Image to visualize
        )
        
        print(f"Added camera {i+1}/{len(poses)}: {pose['image_name']}")
    
    # Add coordinate frame at origin (where the ArUco tag is)
    # Scale the axes to match the scene
    axes_length = 0.05 * position_scale
    server.scene.add_frame("/world", wxyz=(1, 0, 0, 0), position=(0, 0, 0), show_axes=True, axes_length=axes_length)
    
    print(f"\nVisualization complete! Added {len(poses)} camera frustums.")
    print("\nVisualization controls:")
    print("  - Left click + drag: Rotate view")
    print("  - Right click + drag: Pan view")
    print("  - Scroll: Zoom in/out")
    print("  - Click on camera frustums to see the image")
    print("\nTip: If cameras are too close together, increase position_scale")
    print("     If frustums are too large, decrease frustum_scale")
    print("\nPress Ctrl+C to exit visualization.")
    
    # Keep the server running
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nShutting down visualization...")


def main():
    """Main function to run camera calibration."""
    # Configuration
    IMAGE_FOLDER = 'project4/assets/tag'  # Folder containing calibration images
    TAG_SIZE = 0.02  # ArUco tag size in meters (20mm = 0.02m)
    VISUALIZE = False  # Set to False to disable visualization
    SAVE_RESULTS = True
    
    # Check if image folder exists
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Error: Image folder '{IMAGE_FOLDER}' not found!")
        print(f"Please create the folder and add calibration images.")
        return
    
    try:
        # Run calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(
            IMAGE_FOLDER, 
            tag_size=TAG_SIZE,
            visualize=VISUALIZE
        )
        
        # Print results
        print_calibration_results(ret, camera_matrix, dist_coeffs)
        
        # Save results
        if SAVE_RESULTS:
            save_calibration_results(camera_matrix, dist_coeffs, 'calibration_results.npz')
            
        print("\nCalibration successful!")
        print("\nYou can now use these parameters for:")
        print("  - Undistorting images")
        print("  - 3D reconstruction")
        print("  - Augmented reality applications")
        
    except Exception as e:
        print(f"\nError during calibration: {str(e)}")
        import traceback
        traceback.print_exc()


def undistort_and_create_dataset(poses, camera_matrix, dist_coeffs, output_file='my_data.npz', 
                                  train_ratio=0.7, val_ratio=0.15, alpha=0):
    """
    Undistort images and create a dataset for NeRF training.
    
    Args:
        poses: List of pose dictionaries from estimate_camera_poses()
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        output_file: Output .npz file path
        train_ratio: Ratio of images for training (default: 0.7)
        val_ratio: Ratio of images for validation (default: 0.15)
        alpha: Cropping parameter for getOptimalNewCameraMatrix (0=max crop, 1=no crop)
    
    Returns:
        Dictionary containing the dataset
    """
    print("\n" + "="*60)
    print("UNDISTORTING IMAGES AND CREATING DATASET")
    print("="*60)
    
    if len(poses) == 0:
        raise ValueError("No poses provided. Cannot create dataset.")
    
    # Get image dimensions from first pose
    first_img = poses[0]['image']
    h, w = first_img.shape[:2]
    
    print(f"\nOriginal image size: {w}x{h}")
    
    # Compute optimal new camera matrix to handle black boundaries
    # alpha=0: max crop (no black pixels), alpha=1: keep all source pixels
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=alpha
    )
    
    x, y, w_roi, h_roi = roi
    print(f"ROI after undistortion: x={x}, y={y}, width={w_roi}, height={h_roi}")
    
    # Ensure dimensions are consistent by using slightly smaller ROI if needed
    # This prevents edge cases where different images might have slightly different valid regions
    w_roi = (w_roi // 8) * 8  # Make width divisible by 8 for compatibility
    h_roi = (h_roi // 8) * 8  # Make height divisible by 8 for compatibility
    
    print(f"Adjusted ROI for consistency: width={w_roi}, height={h_roi}")
    
    # Update principal point to account for crop offset
    new_camera_matrix_corrected = new_camera_matrix.copy()
    new_camera_matrix_corrected[0, 2] -= x  # cx
    new_camera_matrix_corrected[1, 2] -= y  # cy
    
    print(f"\nOriginal camera matrix:")
    print(camera_matrix)
    print(f"\nNew camera matrix (after undistortion and crop):")
    print(new_camera_matrix_corrected)
    
    # Extract focal length (assuming fx ≈ fy)
    focal = new_camera_matrix_corrected[0, 0]
    focal_y = new_camera_matrix_corrected[1, 1]
    print(f"\nFocal length: fx={focal:.2f}, fy={focal_y:.2f}")
    
    if abs(focal - focal_y) > 1.0:
        print(f"Warning: fx and fy differ by {abs(focal - focal_y):.2f} pixels. Using fx.")
    
    # Undistort all images
    print(f"\nUndistorting {len(poses)} images...")
    undistorted_images = []
    c2w_matrices = []
    image_names = []
    
    for i, pose in enumerate(poses):
        img = pose['image']
        
        # Undistort image
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
        
        # Crop to valid region of interest
        undistorted_cropped = undistorted[y:y+h_roi, x:x+w_roi]
        
        # Ensure exact dimensions (in case of rounding issues)
        if undistorted_cropped.shape[:2] != (h_roi, w_roi):
            undistorted_cropped = cv2.resize(undistorted_cropped, (w_roi, h_roi), interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB
        undistorted_rgb = cv2.cvtColor(undistorted_cropped, cv2.COLOR_BGR2RGB)
        
        # Verify shape consistency
        if i == 0:
            expected_shape = undistorted_rgb.shape
            print(f"  Target shape for all images: {expected_shape}")
        elif undistorted_rgb.shape != expected_shape:
            print(f"  Warning: Image {i+1} shape mismatch. Resizing to match.")
            undistorted_rgb = cv2.resize(undistorted_rgb, (expected_shape[1], expected_shape[0]), interpolation=cv2.INTER_LINEAR)
        
        undistorted_images.append(undistorted_rgb)
        c2w_matrices.append(pose['c2w'])
        image_names.append(pose['image_name'])
        
        if (i + 1) % 10 == 0 or (i + 1) == len(poses):
            print(f"  Processed {i + 1}/{len(poses)} images")
    
    # Convert to numpy arrays
    print(f"\n  Converting to numpy arrays...")
    undistorted_images = np.array(undistorted_images, dtype=np.uint8)  # Keep in 0-255 range
    c2w_matrices = np.array(c2w_matrices, dtype=np.float32)
    
    print(f"\nUndistorted images shape: {undistorted_images.shape}")
    print(f"C2W matrices shape: {c2w_matrices.shape}")
    
    # Split into train/val/test sets
    n_images = len(poses)
    n_train = int(n_images * train_ratio)
    n_val = int(n_images * val_ratio)
    n_test = n_images - n_train - n_val
    
    print(f"\nSplitting dataset:")
    print(f"  Training: {n_train} images ({train_ratio*100:.0f}%)")
    print(f"  Validation: {n_val} images ({val_ratio*100:.0f}%)")
    print(f"  Test: {n_test} images ({(1-train_ratio-val_ratio)*100:.0f}%)")
    
    # Shuffle indices for random split
    indices = np.random.permutation(n_images)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    
    # Create splits
    images_train = undistorted_images[train_indices]
    c2ws_train = c2w_matrices[train_indices]
    
    images_val = undistorted_images[val_indices]
    c2ws_val = c2w_matrices[val_indices]
    
    # For test, we only need camera poses (for novel view rendering)
    c2ws_test = c2w_matrices[test_indices]
    
    print(f"\nFinal dataset shapes:")
    print(f"  images_train: {images_train.shape}")
    print(f"  c2ws_train: {c2ws_train.shape}")
    print(f"  images_val: {images_val.shape}")
    print(f"  c2ws_val: {c2ws_val.shape}")
    print(f"  c2ws_test: {c2ws_test.shape}")
    print(f"  focal: {focal}")
    
    # Save dataset
    print(f"\nSaving dataset to: {output_file}")
    np.savez(
        output_file,
        images_train=images_train,    # (N_train, H, W, 3) in 0-255 range
        c2ws_train=c2ws_train,        # (N_train, 4, 4)
        images_val=images_val,        # (N_val, H, W, 3) in 0-255 range
        c2ws_val=c2ws_val,            # (N_val, 4, 4)
        c2ws_test=c2ws_test,          # (N_test, 4, 4)
        focal=focal,                  # float
        # Store additional metadata for reference
        camera_matrix=new_camera_matrix_corrected,
        image_shape=np.array([h_roi, w_roi, 3])
    )
    
    print(f"\n✓ Dataset saved successfully!")
    print(f"\nDataset can now be used for NeRF training.")
    print(f"Image size: {w_roi}x{h_roi}")
    print(f"Focal length: {focal:.2f}")
    
    # Return dataset dictionary
    dataset = {
        'images_train': images_train,
        'c2ws_train': c2ws_train,
        'images_val': images_val,
        'c2ws_val': c2ws_val,
        'c2ws_test': c2ws_test,
        'focal': focal,
        'camera_matrix': new_camera_matrix_corrected,
        'image_shape': (h_roi, w_roi, 3)
    }
    
    return dataset


def main_pose_estimation():
    """Main function to run camera pose estimation."""
    # Configuration
    CALIBRATION_FILE = 'calibration_results.npz'
    OBJECT_FOLDER = 'assets/rivian'  # Folder with Rivian images
    TAG_SIZE = 0.056  # ArUco tag size in meters (56mm = 0.056m)
    VISUALIZE_DETECTION = False  # Set to True to see detected markers
    VISUALIZE_3D = True  # Set to True to visualize camera poses in 3D
    
    # Visualization settings
    FRUSTUM_SCALE = 0.05  # Smaller = tinier camera frustums (try 0.002-0.01)
    POSITION_SCALE = 0.8   # Larger = more spread out (try 1.0-5.0)
    
    try:
        # Load calibration results
        print("Loading calibration results...")
        camera_matrix, dist_coeffs = load_calibration_results(CALIBRATION_FILE)
        
        # Check if object folder exists
        if not os.path.exists(OBJECT_FOLDER):
            print(f"\nError: Object folder '{OBJECT_FOLDER}' not found!")
            print(f"Please create the folder and add images with the object and ArUco tag.")
            return
        
        # Estimate camera poses
        poses = estimate_camera_poses(
            OBJECT_FOLDER,
            camera_matrix,
            dist_coeffs,
            tag_size=TAG_SIZE,
            visualize=VISUALIZE_DETECTION
        )
        
        print(f"\nSuccessfully estimated poses for {len(poses)} images!")
        
        # Visualize poses in 3D
        if VISUALIZE_3D:
            visualize_camera_poses(
                poses, 
                camera_matrix,
                frustum_scale=FRUSTUM_SCALE,
                position_scale=POSITION_SCALE
            )
        
    except FileNotFoundError as e:
        print(f"\nError: {str(e)}")
        print("Please run calibration first (Part 0.1) before pose estimation.")
    except Exception as e:
        print(f"\nError during pose estimation: {str(e)}")
        import traceback
        traceback.print_exc()


def main_create_dataset():
    """Main function to create NeRF dataset from poses."""
    # Configuration
    CALIBRATION_FILE = 'calibration_results.npz'
    OBJECT_FOLDER = 'project4/assets/object_tag'  # Folder with object images
    TAG_SIZE = 0.02  # ArUco tag size in meters (20mm = 0.02m)
    OUTPUT_FILE = 'my_data.npz'  # Output dataset file
    TRAIN_RATIO = 0.7  # 70% for training
    VAL_RATIO = 0.15   # 15% for validation (15% left for test)
    ALPHA = 0  # Crop parameter: 0=max crop (no black borders), 1=no crop
    
    try:
        # Load calibration results
        print("Loading calibration results...")
        camera_matrix, dist_coeffs = load_calibration_results(CALIBRATION_FILE)
        
        # Check if object folder exists
        if not os.path.exists(OBJECT_FOLDER):
            print(f"\nError: Object folder '{OBJECT_FOLDER}' not found!")
            print(f"Please create the folder and add images with the object and ArUco tag.")
            return
        
        # Estimate camera poses
        print("\nEstimating camera poses...")
        poses = estimate_camera_poses(
            OBJECT_FOLDER,
            camera_matrix,
            dist_coeffs,
            tag_size=TAG_SIZE,
            visualize=False  # Don't visualize during dataset creation
        )
        
        print(f"\nSuccessfully estimated poses for {len(poses)} images!")
        
        # Undistort images and create dataset
        dataset = undistort_and_create_dataset(
            poses,
            camera_matrix,
            dist_coeffs,
            output_file=OUTPUT_FILE,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            alpha=ALPHA
        )
        
        print(f"\n{'='*60}")
        print("DATASET CREATION COMPLETE!")
        print(f"{'='*60}")
        print(f"\nDataset saved to: {OUTPUT_FILE}")
        print(f"\nYou can now use this dataset for NeRF training!")
        
    except FileNotFoundError as e:
        print(f"\nError: {str(e)}")
        print("Please run calibration first (Part 0.1) before creating dataset.")
    except Exception as e:
        print(f"\nError during dataset creation: {str(e)}")
        import traceback
        traceback.print_exc()


def main_visualize_rays():
    """Visualize rays and sample points for Rivian dataset using viser."""
    from code.part2 import visualize_rivian_rays_viser
    
    print("="*60)
    print("Visualizing Rivian Rays (Part 2.3)")
    print("="*60)
    
    # Configuration
    DATA_FILE = 'rivian.npz'
    NUM_RAYS = 100  # Number of rays to visualize
    NEAR = 0.02     # Near plane (from Rivian training)
    FAR = 0.5       # Far plane (from Rivian training)
    N_SAMPLES = 64  # Samples per ray
    
    if not os.path.exists(DATA_FILE):
        print(f"\nError: Dataset '{DATA_FILE}' not found!")
        print("Please create the Rivian dataset first.")
        return
    
    # Call the visualization function from part2.py
    visualize_rivian_rays_viser(
        data_path=DATA_FILE,
        num_rays=NUM_RAYS,
        near=NEAR,
        far=FAR,
        n_samples=N_SAMPLES
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "pose":
            # Run pose estimation with visualization
            main_pose_estimation()
        elif command == "dataset":
            # Create NeRF dataset
            main_create_dataset()
        elif command == "rays":
            # Visualize rays and sample points (Part 2.3)
            main_visualize_rays()
        else:
            print(f"Unknown command: {command}")
            print("\nUsage:")
            print("  python code.py            - Run camera calibration")
            print("  python code.py pose       - Estimate camera poses and visualize")
            print("  python code.py dataset    - Create NeRF dataset (.npz file)")
            print("  python code.py rays       - Visualize rays and sample points (Part 2.3)")
    else:
        # Run calibration
        main()

