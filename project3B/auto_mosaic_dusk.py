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
    """Warp image using homography with bilinear interpolation."""
    h, w = output_shape
    y, x = np.mgrid[0:h, 0:w]
    coords = np.stack([x.ravel(), y.ravel(), np.ones(h*w)])
    
    H_inv = np.linalg.inv(H)
    src_coords = H_inv @ coords
    src_coords = src_coords[:2] / src_coords[2]
    
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
    """Compute canvas size needed to fit all warped images."""
    all_corners = []
    
    for img, H in zip(images, homographies):
        if H is None:
            continue
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
    
    min_x, min_y = np.floor(all_corners.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(all_corners.max(axis=0)).astype(int)
    
    canvas_width = max_x - min_x
    canvas_height = max_y - min_y
    offset = (-min_y, -min_x)
    
    return (canvas_height, canvas_width), offset


def create_mosaic(images, homographies, canvas_shape, offset):
    """Create mosaic by warping and blending images."""
    h, w = canvas_shape
    
    T_offset = np.array([
        [1, 0, offset[1]],
        [0, 1, offset[0]],
        [0, 0, 1]
    ])
    
    warped_images = []
    masks = []
    
    for img, H in zip(images, homographies):
        if H is None:
            continue
            
        H_offset = T_offset @ H
        warped = warp_image(img, H_offset, canvas_shape)
        warped_images.append(warped)
        
        if len(img.shape) == 3:
            mask = (warped.sum(axis=2) > 0).astype(float)
        else:
            mask = (warped > 0).astype(float)
        masks.append(mask)
    
    mosaic = np.zeros((h, w, 3), dtype=float)
    weight_sum = np.zeros((h, w), dtype=float)
    
    from scipy.ndimage import distance_transform_edt
    
    for warped, mask in zip(warped_images, masks):
        if len(warped.shape) == 2:
            warped = np.stack([warped, warped, warped], axis=2)
        
        weight = distance_transform_edt(mask)
        weight = np.minimum(weight, 50) / 50.0
        
        mosaic += warped * weight[:, :, np.newaxis]
        weight_sum += weight
    
    weight_sum = np.maximum(weight_sum, 1e-10)
    mosaic = mosaic / weight_sum[:, :, np.newaxis]
    mosaic = np.clip(mosaic, 0, 255).astype(np.uint8)
    
    return mosaic


def process_image_for_mosaic(image_path):
    """Process a single image: detect corners, extract descriptors."""
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


def match_and_compute_homography(desc1, desc2, corners1, corners2, name1, name2):
    """Match features and compute homography with RANSAC."""
    print(f"\n--- Matching {name1} -> {name2} ---")
    
    matches, _ = match_features(desc1, desc2, corners1, corners2, ratio_threshold=0.7)
    
    if len(matches) < 4:
        print(f"Warning: Not enough matches ({len(matches)})")
        return None, 0, 0
    
    H, inliers = ransac_homography(matches, corners1, corners2, num_iterations=1000, threshold=5.0)
    
    return H, len(matches), np.sum(inliers)


def create_dusk_mosaic(output_path):
    """
    Create mosaic from 5 Dusk images arranged as:
        TL      TR
      BL  BM  BR
    Using BM (bottom middle) as reference.
    """
    print(f"\n{'='*60}")
    print(f"Creating Dusk 5-Image Panorama")
    print(f"{'='*60}\n")
    
    # Image paths in order: BL, BM, BR, TL, TR
    image_paths = {
        'BL': 'assets/duskBL.jpg',
        'BM': 'assets/duskBM.jpg',
        'BR': 'assets/duskBR.jpg',
        'TL': 'assets/duskTL.jpg',
        'TR': 'assets/duskTR.jpg'
    }
    
    # Process all images
    data = {}
    for name, path in image_paths.items():
        img, img_gray, desc, corners = process_image_for_mosaic(path)
        data[name] = {'img': img, 'desc': desc, 'corners': corners}
    
    # Use BM (bottom middle) as reference
    homographies = {
        'BM': np.eye(3),
        'BL': None,
        'BR': None,
        'TL': None,
        'TR': None
    }
    
    print(f"\nUsing DuskBM (bottom middle) as reference")
    
    # Match BL -> BM
    H_BL, m_BL, i_BL = match_and_compute_homography(
        data['BL']['desc'], data['BM']['desc'],
        data['BL']['corners'], data['BM']['corners'],
        'DuskBL', 'DuskBM'
    )
    if H_BL is not None:
        homographies['BL'] = homographies['BM'] @ H_BL
    
    # Match BR -> BM
    H_BR, m_BR, i_BR = match_and_compute_homography(
        data['BR']['desc'], data['BM']['desc'],
        data['BR']['corners'], data['BM']['corners'],
        'DuskBR', 'DuskBM'
    )
    if H_BR is not None:
        homographies['BR'] = homographies['BM'] @ H_BR
    
    # Match TL -> BL (if BL successful)
    if homographies['BL'] is not None:
        H_TL_BL, m_TL_BL, i_TL_BL = match_and_compute_homography(
            data['TL']['desc'], data['BL']['desc'],
            data['TL']['corners'], data['BL']['corners'],
            'DuskTL', 'DuskBL'
        )
        if H_TL_BL is not None:
            homographies['TL'] = homographies['BL'] @ H_TL_BL
    
    # Match TR -> BR (if BR successful)
    if homographies['BR'] is not None:
        H_TR_BR, m_TR_BR, i_TR_BR = match_and_compute_homography(
            data['TR']['desc'], data['BR']['desc'],
            data['TR']['corners'], data['BR']['corners'],
            'DuskTR', 'DuskBR'
        )
        if H_TR_BR is not None:
            homographies['TR'] = homographies['BR'] @ H_TR_BR
    
    # Also try TL -> TM (via BM) as alternative
    # This creates more connections for robustness
    
    # Collect valid images and homographies
    valid_images = []
    valid_homographies = []
    for name in ['BL', 'BM', 'BR', 'TL', 'TR']:
        if homographies[name] is not None:
            valid_images.append(data[name]['img'])
            valid_homographies.append(homographies[name])
    
    print(f"\n--- Computing canvas size ---")
    canvas_shape, offset = compute_canvas_size(valid_images, valid_homographies)
    print(f"Canvas size: {canvas_shape}, Offset: {offset}")
    
    print(f"\n--- Creating mosaic ---")
    mosaic = create_mosaic(valid_images, valid_homographies, canvas_shape, offset)
    
    plt.figure(figsize=(20, 15))
    plt.imshow(mosaic)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved mosaic: {output_path}")
    plt.close()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Mosaic Creation Summary:")
    print(f"  Total images: 5")
    print(f"  Successfully aligned: {len(valid_images)}")
    print(f"  Canvas size: {canvas_shape}")
    if H_BL is not None:
        print(f"  DuskBL→DuskBM: {i_BL}/{m_BL} inliers ({i_BL/m_BL*100:.1f}%)")
    if H_BR is not None:
        print(f"  DuskBR→DuskBM: {i_BR}/{m_BR} inliers ({i_BR/m_BR*100:.1f}%)")
    if H_TL_BL is not None:
        print(f"  DuskTL→DuskBL: {i_TL_BL}/{m_TL_BL} inliers ({i_TL_BL/m_TL_BL*100:.1f}%)")
    if H_TR_BR is not None:
        print(f"  DuskTR→DuskBR: {i_TR_BR}/{m_TR_BR} inliers ({i_TR_BR/m_TR_BR*100:.1f}%)")
    print(f"{'='*60}\n")
    
    return mosaic


def main():
    output_dir = 'assets'
    os.makedirs(output_dir, exist_ok=True)
    
    mosaic_dusk = create_dusk_mosaic(
        os.path.join(output_dir, 'auto_mosaic_dusk.jpg')
    )
    
    print("\n" + "="*60)
    print("DONE! Dusk mosaic created successfully")
    print("="*60)


if __name__ == '__main__':
    main()

