import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from scipy.ndimage import map_coordinates
import os

from harris import (
    get_harris_corners,
    anms,
    extract_feature_descriptors,
    match_features,
    ransac_homography,
    compute_homography
)


def warp_image(img, H, output_shape):
    """
    Warp image using homography with bilinear interpolation.
    
    Parameters:
    - img: Input image
    - H: 3x3 homography matrix
    - output_shape: (height, width) of output image
    
    Returns:
    - warped: Warped image
    """
    h, w = output_shape
    
    # Create coordinate grid for output image
    y, x = np.mgrid[0:h, 0:w]
    coords = np.stack([x.ravel(), y.ravel(), np.ones(h*w)])
    
    # Apply inverse homography
    H_inv = np.linalg.inv(H)
    src_coords = H_inv @ coords
    
    # Convert from homogeneous coordinates
    src_coords = src_coords[:2] / src_coords[2]
    
    # Warp each channel
    if len(img.shape) == 3:
        warped = np.zeros((h, w, img.shape[2]), dtype=img.dtype)
        for c in range(img.shape[2]):
            warped[:, :, c] = map_coordinates(
                img[:, :, c],
                [src_coords[1], src_coords[0]],
                order=1,
                mode='constant',
                cval=0
            ).reshape(h, w)
    else:
        warped = map_coordinates(
            img,
            [src_coords[1], src_coords[0]],
            order=1,
            mode='constant',
            cval=0
        ).reshape(h, w)
    
    return warped


def compute_canvas_size(images, homographies):
    """
    Compute canvas size needed to fit all warped images.
    
    Parameters:
    - images: List of images
    - homographies: List of homography matrices (identity for reference image)
    
    Returns:
    - canvas_shape: (height, width)
    - offset: (y_offset, x_offset) to apply to all images
    """
    # Get corners of all images in the panorama coordinate system
    all_corners = []
    
    for img, H in zip(images, homographies):
        h, w = img.shape[:2]
        corners = np.array([
            [0, 0, 1],
            [w, 0, 1],
            [w, h, 1],
            [0, h, 1]
        ]).T
        
        transformed = H @ corners
        transformed = transformed[:2] / transformed[2]
        all_corners.append(transformed.T)
    
    all_corners = np.vstack(all_corners)
    
    # Find bounding box
    min_x, min_y = np.floor(all_corners.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(all_corners.max(axis=0)).astype(int)
    
    canvas_width = max_x - min_x
    canvas_height = max_y - min_y
    
    # Offset to shift all coordinates to positive
    offset = (-min_y, -min_x)
    
    return (canvas_height, canvas_width), offset


def create_mosaic(images, homographies, canvas_shape, offset):
    """
    Create mosaic by warping and blending images.
    
    Parameters:
    - images: List of images
    - homographies: List of homography matrices
    - canvas_shape: (height, width) of canvas
    - offset: (y_offset, x_offset)
    
    Returns:
    - mosaic: Blended mosaic image
    """
    h, w = canvas_shape
    
    # Create translation matrix for offset
    T_offset = np.array([
        [1, 0, offset[1]],
        [0, 1, offset[0]],
        [0, 0, 1]
    ])
    
    # Warp all images
    warped_images = []
    masks = []
    
    for img, H in zip(images, homographies):
        # Apply offset to homography
        H_offset = T_offset @ H
        
        # Warp image
        warped = warp_image(img, H_offset, canvas_shape)
        warped_images.append(warped)
        
        # Create mask for valid pixels
        if len(img.shape) == 3:
            mask = (warped.sum(axis=2) > 0).astype(float)
        else:
            mask = (warped > 0).astype(float)
        masks.append(mask)
    
    # Blend using simple averaging with feathering
    mosaic = np.zeros((h, w, 3), dtype=float)
    weight_sum = np.zeros((h, w), dtype=float)
    
    for warped, mask in zip(warped_images, masks):
        if len(warped.shape) == 2:
            warped = np.stack([warped, warped, warped], axis=2)
        
        # Distance-based feathering
        from scipy.ndimage import distance_transform_edt
        weight = distance_transform_edt(mask)
        weight = np.minimum(weight, 50) / 50.0  # Cap at 50 pixels
        
        mosaic += warped * weight[:, :, np.newaxis]
        weight_sum += weight
    
    # Normalize
    weight_sum = np.maximum(weight_sum, 1e-10)
    mosaic = mosaic / weight_sum[:, :, np.newaxis]
    
    # Convert to uint8
    mosaic = np.clip(mosaic, 0, 255).astype(np.uint8)
    
    return mosaic


def process_image_for_mosaic(image_path):
    """
    Process a single image: detect corners, extract descriptors.
    
    Returns: img, img_gray, descriptors, corners
    """
    img = imread(image_path)
    img_gray = rgb2gray(img)
    
    h, corners_harris = get_harris_corners(
        img_gray,
        edge_discard=20,
        min_distance=5,
        threshold_rel=0.01
    )
    
    corners_anms = anms(corners_harris, h, num_points=500, c_robust=0.9)
    descriptors, valid_corners = extract_feature_descriptors(
        img_gray,
        corners_anms,
        window_size=40,
        descriptor_size=8,
        sample_spacing=5
    )
    
    print(f"Processed {os.path.basename(image_path)}: {descriptors.shape[0]} descriptors")
    
    return img, img_gray, descriptors, valid_corners


def create_automatic_mosaic(image_paths, output_path):
    """
    Create automatic mosaic from a list of images.
    
    Parameters:
    - image_paths: List of image file paths (e.g., [left, middle, right])
    - output_path: Path to save output mosaic
    """
    print(f"\n{'='*60}")
    print(f"Creating automatic mosaic from {len(image_paths)} images")
    print(f"{'='*60}\n")
    
    # Process all images
    images = []
    descriptors_list = []
    corners_list = []
    
    for img_path in image_paths:
        img, img_gray, desc, corners = process_image_for_mosaic(img_path)
        images.append(img)
        descriptors_list.append(desc)
        corners_list.append(corners)
    
    # Use middle image as reference (identity homography)
    ref_idx = len(images) // 2
    homographies = [None] * len(images)
    homographies[ref_idx] = np.eye(3)
    
    print(f"\nUsing image {ref_idx} ({os.path.basename(image_paths[ref_idx])}) as reference")
    
    # Match and compute homographies for images before reference
    for i in range(ref_idx - 1, -1, -1):
        print(f"\n--- Matching {os.path.basename(image_paths[i])} -> {os.path.basename(image_paths[i+1])} (ref) ---")
        
        # Match i to i+1 (so H transforms from i to i+1)
        matches, _ = match_features(
            descriptors_list[i],
            descriptors_list[i+1],
            corners_list[i],
            corners_list[i+1],
            ratio_threshold=0.7
        )
        
        if len(matches) < 4:
            print(f"Warning: Not enough matches ({len(matches)})")
            continue
        
        # Compute homography using RANSAC (H transforms from image i to image i+1)
        H, inliers = ransac_homography(
            matches,
            corners_list[i],
            corners_list[i+1],
            num_iterations=1000,
            threshold=5.0
        )
        
        # Accumulate homography (to transform from image i to reference)
        homographies[i] = homographies[i+1] @ H
    
    # Match and compute homographies for images after reference
    for i in range(ref_idx + 1, len(images)):
        print(f"\n--- Matching {os.path.basename(image_paths[i])} -> {os.path.basename(image_paths[i-1])} (ref) ---")
        
        # Match i to i-1 (so H transforms from i to i-1)
        matches, _ = match_features(
            descriptors_list[i],
            descriptors_list[i-1],
            corners_list[i],
            corners_list[i-1],
            ratio_threshold=0.7
        )
        
        if len(matches) < 4:
            print(f"Warning: Not enough matches ({len(matches)})")
            continue
        
        # Compute homography using RANSAC (H transforms from image i to image i-1)
        H, inliers = ransac_homography(
            matches,
            corners_list[i],
            corners_list[i-1],
            num_iterations=1000,
            threshold=5.0
        )
        
        # Accumulate homography (to transform from image i to reference)
        homographies[i] = homographies[i-1] @ H
    
    # Compute canvas size
    print(f"\n--- Computing canvas size ---")
    canvas_shape, offset = compute_canvas_size(images, homographies)
    print(f"Canvas size: {canvas_shape}, Offset: {offset}")
    
    # Create mosaic
    print(f"\n--- Creating mosaic ---")
    mosaic = create_mosaic(images, homographies, canvas_shape, offset)
    
    # Save result
    plt.figure(figsize=(20, 10))
    plt.imshow(mosaic)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved mosaic: {output_path}")
    plt.close()
    
    return mosaic


def main():
    output_dir = 'assets'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create Parker Street panorama
    print("\n" + "="*60)
    print("Creating Parker Street Panorama")
    print("="*60)
    
    parker_paths = [
        'assets/ParkerL.jpg',
        'assets/ParkerM.jpg',
        'assets/ParkerR.jpg'
    ]
    
    mosaic_parker = create_automatic_mosaic(
        parker_paths,
        os.path.join(output_dir, 'auto_mosaic_parker.jpg')
    )
    
    # Create Room panorama
    print("\n" + "="*60)
    print("Creating Room Panorama")
    print("="*60)
    
    room_paths = [
        'assets/RoomL.jpg',
        'assets/RoomM.jpg',
        'assets/RoomR.jpg'
    ]
    
    mosaic_room = create_automatic_mosaic(
        room_paths,
        os.path.join(output_dir, 'auto_mosaic_room.jpg')
    )
    
    print("\n" + "="*60)
    print("DONE! All mosaics created successfully")
    print("="*60)


if __name__ == '__main__':
    main()

