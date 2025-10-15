import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
import os

# Import functions from harris.py
from harris import (
    get_harris_corners,
    anms,
    extract_feature_descriptors,
    match_features,
    visualize_matches
)


def process_image_pair(image1_path, image2_path, output_dir='assets', ratio_threshold=0.5):
    """
    Process a pair of images: detect corners, extract descriptors, and match features.
    
    Parameters:
    - image1_path: Path to first image
    - image2_path: Path to second image
    - output_dir: Directory to save outputs
    - ratio_threshold: Threshold for Lowe's ratio test
    """
    print(f"\n{'='*60}")
    print(f"Processing image pair:")
    print(f"  Image 1: {image1_path}")
    print(f"  Image 2: {image2_path}")
    print(f"{'='*60}\n")
    
    # Load images
    img1 = imread(image1_path)
    img2 = imread(image2_path)
    img1_gray = rgb2gray(img1)
    img2_gray = rgb2gray(img2)
    
    print(f"Image 1 shape: {img1.shape}")
    print(f"Image 2 shape: {img2.shape}")
    
    # Process Image 1
    print(f"\n--- Processing Image 1 ---")
    h1, corners1_harris = get_harris_corners(
        img1_gray,
        edge_discard=20,
        min_distance=5,
        threshold_rel=0.01
    )
    print(f"Harris corners in image 1: {corners1_harris.shape[1]}")
    
    corners1_anms = anms(corners1_harris, h1, num_points=500, c_robust=0.9)
    print(f"ANMS corners in image 1: {corners1_anms.shape[1]}")
    
    descriptors1, valid_corners1 = extract_feature_descriptors(
        img1_gray,
        corners1_anms,
        window_size=40,
        descriptor_size=8,
        sample_spacing=5
    )
    print(f"Descriptors extracted from image 1: {descriptors1.shape[0]}")
    
    # Process Image 2
    print(f"\n--- Processing Image 2 ---")
    h2, corners2_harris = get_harris_corners(
        img2_gray,
        edge_discard=20,
        min_distance=5,
        threshold_rel=0.01
    )
    print(f"Harris corners in image 2: {corners2_harris.shape[1]}")
    
    corners2_anms = anms(corners2_harris, h2, num_points=500, c_robust=0.9)
    print(f"ANMS corners in image 2: {corners2_anms.shape[1]}")
    
    descriptors2, valid_corners2 = extract_feature_descriptors(
        img2_gray,
        corners2_anms,
        window_size=40,
        descriptor_size=8,
        sample_spacing=5
    )
    print(f"Descriptors extracted from image 2: {descriptors2.shape[0]}")
    
    # Match features
    print(f"\n--- Matching Features ---")
    matches, match_distances = match_features(
        descriptors1,
        descriptors2,
        valid_corners1,
        valid_corners2,
        ratio_threshold=ratio_threshold
    )
    
    print(f"\nMatch statistics:")
    print(f"  Total matches: {len(matches)}")
    print(f"  Match rate: {len(matches) / descriptors1.shape[0] * 100:.1f}% of features in image 1")
    if len(match_distances) > 0:
        print(f"  Average matching distance: {np.mean(match_distances):.4f}")
        print(f"  Min/Max matching distance: {np.min(match_distances):.4f} / {np.max(match_distances):.4f}")
    
    # Visualize matches
    print(f"\n--- Visualizing Matches ---")
    base_name = f"{os.path.basename(image1_path).split('.')[0]}_{os.path.basename(image2_path).split('.')[0]}"
    output_path = os.path.join(output_dir, f'matches_{base_name}.jpg')
    
    visualize_matches(
        img1,
        img2,
        valid_corners1,
        valid_corners2,
        matches,
        output_path=output_path
    )
    
    print(f"\n{'='*60}")
    print(f"Completed processing image pair")
    print(f"{'='*60}\n")
    
    return matches, descriptors1, descriptors2, valid_corners1, valid_corners2


def main():
    # Create output directory
    output_dir = 'assets'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process ParkerM and ParkerR
    print("\n" + "="*60)
    print("FEATURE MATCHING: ParkerM.jpg <-> ParkerR.jpg")
    print("="*60)
    
    matches, desc1, desc2, corners1, corners2 = process_image_pair(
        'assets/ParkerM.jpg',
        'assets/ParkerR.jpg',
        output_dir=output_dir,
        ratio_threshold=0.7  # Based on Figure 6b - more lenient for more matches
    )
    
    print("\n" + "="*60)
    print("DONE! All results saved to assets/")
    print("="*60)


if __name__ == '__main__':
    main()

