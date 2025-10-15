
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import corner_harris, peak_local_max
from skimage.io import imread
from skimage.color import rgb2gray
import os


def get_harris_corners(im, edge_discard=20, min_distance=5, threshold_rel=0.01):
    """
    This function takes a b&w image and an optional amount to discard
    on the edge (default is 5 pixels), and finds all harris corners
    in the image. Harris corners near the edge are discarded and the
    coordinates of the remaining corners are returned. A 2d array (h)
    containing the h value of every pixel is also returned.

    h is the same shape as the original image, im.
    coords is 2 x n (ys, xs).
    
    Parameters:
    - min_distance: Minimum number of pixels separating corners (helps reduce corner count)
    - threshold_rel: Minimum intensity threshold for corners (relative to max)
    """

    assert edge_discard >= 20

    # find harris corners
    h = corner_harris(im, method='eps', sigma=1)
    # Adjust these parameters to reduce the number of corners before ANMS
    coords = peak_local_max(h, min_distance=min_distance, threshold_rel=threshold_rel)

    # discard points on edge
    edge = edge_discard  # pixels
    mask = (coords[:, 0] > edge) & \
           (coords[:, 0] < im.shape[0] - edge) & \
           (coords[:, 1] > edge) & \
           (coords[:, 1] < im.shape[1] - edge)
    coords = coords[mask].T
    return h, coords


def dist2(x, c):
    """
    dist2  Calculates squared distance between two sets of points.

    Description
    D = DIST2(X, C) takes two matrices of vectors and calculates the
    squared Euclidean distance between them.  Both matrices must be of
    the same column dimension.  If X has M rows and N columns, and C has
    L rows and N columns, then the result has M rows and L columns.  The
    I, Jth entry is the  squared distance from the Ith row of X to the
    Jth row of C.

    Adapted from code by Christopher M Bishop and Ian T Nabney.
    """

    ndata, dimx = x.shape
    ncenters, dimc = c.shape
    assert dimx == dimc, 'Data dimension does not match dimension of centers'

    return (np.ones((ncenters, 1)) * np.sum((x**2).T, axis=0)).T + \
            np.ones((   ndata, 1)) * np.sum((c**2).T, axis=0)    - \
            2 * np.inner(x, c)


def anms(corners, harris_values, num_points=500, c_robust=0.9):
    """
    Adaptive Non-Maximal Suppression
    
    Parameters:
    - corners: 2 x n array of corner coordinates (ys, xs)
    - harris_values: 2D array of harris corner response values
    - num_points: Number of corners to select
    - c_robust: Robustness constant (default 0.9)
    
    Returns:
    - selected_corners: 2 x num_points array of selected corners
    """
    # Get harris values for each corner
    corner_strengths = harris_values[corners[0, :], corners[1, :]]
    
    # Convert corners to n x 2 for distance calculation
    corners_transposed = corners.T  # n x 2 (y, x)
    
    n = corners_transposed.shape[0]
    
    # Calculate radius for each point
    radii = np.inf * np.ones(n)
    
    print(f"Computing suppression radii for {n} corners...")
    
    # For each corner i
    for i in range(n):
        # Find all corners j where f(xi) < c_robust * f(xj)
        # This is equivalent to: f(xj) > f(xi) / c_robust
        # Meaning: neighbor must be significantly stronger (> 1.11x for c_robust=0.9)
        stronger_mask = corner_strengths > corner_strengths[i] / c_robust
        
        if np.any(stronger_mask):
            # Get corners that are stronger
            stronger_corners = corners_transposed[stronger_mask]
            
            # Compute distances to all stronger corners
            distances = np.sqrt(np.sum((stronger_corners - corners_transposed[i])**2, axis=1))
            
            # Minimum distance to any stronger corner
            radii[i] = np.min(distances)
    
    # Select top num_points corners with largest radii
    if n > num_points:
        # Get indices of corners with largest radii
        selected_indices = np.argsort(radii)[-num_points:]
    else:
        selected_indices = np.arange(n)
    
    selected_corners = corners[:, selected_indices]
    
    return selected_corners


def match_features(descriptors1, descriptors2, corners1, corners2, ratio_threshold=0.5):
    """
    Match features between two images using Lowe's ratio test.
    
    Parameters:
    - descriptors1: n1 x 64 array of descriptors from image 1
    - descriptors2: n2 x 64 array of descriptors from image 2
    - corners1: 2 x n1 array of corner locations in image 1
    - corners2: 2 x n2 array of corner locations in image 2
    - ratio_threshold: Threshold for Lowe's ratio test (default 0.5)
    
    Returns:
    - matches: List of tuples (idx1, idx2) where idx1 is index in image1 and idx2 in image2
    - match_distances: List of matching distances (e1-NN)
    """
    matches = []
    match_distances = []
    
    print(f"Matching {descriptors1.shape[0]} features from image 1 with {descriptors2.shape[0]} features from image 2...")
    
    # For each feature in image 1, find matches in image 2
    for i in range(descriptors1.shape[0]):
        # Compute Euclidean distances to all features in image 2
        distances = np.sqrt(np.sum((descriptors2 - descriptors1[i])**2, axis=1))
        
        # Sort to get nearest neighbors
        sorted_indices = np.argsort(distances)
        
        if len(sorted_indices) < 2:
            continue
        
        # Get 1-NN and 2-NN distances
        e1_nn = distances[sorted_indices[0]]
        e2_nn = distances[sorted_indices[1]]
        
        # Lowe's ratio test
        if e2_nn > 0 and (e1_nn / e2_nn) < ratio_threshold:
            matches.append((i, sorted_indices[0]))
            match_distances.append(e1_nn)
    
    print(f"Found {len(matches)} matches using ratio threshold {ratio_threshold}")
    
    return matches, match_distances


def visualize_matches(img1, img2, corners1, corners2, matches, output_path='assets/feature_matches.jpg'):
    """
    Visualize feature matches between two images.
    
    Parameters:
    - img1, img2: The two images
    - corners1, corners2: Corner locations in both images
    - matches: List of tuples (idx1, idx2)
    - output_path: Path to save visualization
    """
    # Create side-by-side image
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    w = w1 + w2
    
    # Create output image
    if len(img1.shape) == 3:
        vis_img = np.zeros((h, w, 3), dtype=np.uint8)
        vis_img[:h1, :w1] = img1
        vis_img[:h2, w1:w1+w2] = img2
    else:
        vis_img = np.zeros((h, w), dtype=np.uint8)
        vis_img[:h1, :w1] = (img1 * 255).astype(np.uint8)
        vis_img[:h2, w1:w1+w2] = (img2 * 255).astype(np.uint8)
        vis_img = np.stack([vis_img, vis_img, vis_img], axis=2)
    
    # Draw matches
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.imshow(vis_img)
    
    # Draw lines for each match
    np.random.seed(42)  # For consistent colors
    for idx1, idx2 in matches:
        y1, x1 = corners1[0, idx1], corners1[1, idx1]
        y2, x2 = corners2[0, idx2], corners2[1, idx2]
        
        # Offset x2 by width of first image
        x2_offset = x2 + w1
        
        # Random color for each match
        color = plt.cm.hsv(np.random.rand())
        
        # Draw line - thicker and more visible
        ax.plot([x1, x2_offset], [y1, y2], 'r-', linewidth=2.0, alpha=0.7)
        
        # Draw keypoints - larger
        ax.plot(x1, y1, 'go', markersize=6)
        ax.plot(x2_offset, y2, 'go', markersize=6)
    
    ax.set_title(f'Feature Matches ({len(matches)} matches)', fontsize=16)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Also create a version with fewer matches for clarity
    if len(matches) > 50:
        # Select subset of matches evenly spaced
        subset_indices = np.linspace(0, len(matches) - 1, 50, dtype=int)
        subset_matches = [matches[i] for i in subset_indices]
        
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.imshow(vis_img)
        
        for idx, (idx1, idx2) in enumerate(subset_matches):
            y1, x1 = corners1[0, idx1], corners1[1, idx1]
            y2, x2 = corners2[0, idx2], corners2[1, idx2]
            
            x2_offset = x2 + w1
            
            # Use distinct colors
            color = plt.cm.tab20(idx % 20)
            
            ax.plot([x1, x2_offset], [y1, y2], color=color, linewidth=2.5, alpha=0.8)
            ax.plot(x1, y1, 'ro', markersize=7)
            ax.plot(x2_offset, y2, 'ro', markersize=7)
        
        ax.set_title(f'Feature Matches (50 selected matches shown for clarity)', fontsize=16)
        ax.axis('off')
        
        plt.tight_layout()
        output_path_subset = output_path.replace('.jpg', '_subset.jpg')
        plt.savefig(output_path_subset, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path_subset}")
        plt.close()


def extract_feature_descriptors(image, corners, window_size=40, descriptor_size=8, sample_spacing=5):
    """
    Extract feature descriptors from corners.
    
    Parameters:
    - image: Grayscale image
    - corners: 2 x n array of corner coordinates (ys, xs)
    - window_size: Size of the window to sample from (default 40x40)
    - descriptor_size: Size of the descriptor (default 8x8)
    - sample_spacing: Spacing between samples (default 5 pixels)
    
    Returns:
    - descriptors: n x (descriptor_size^2) array of feature descriptors
    - valid_corners: 2 x m array of corners with valid descriptors
    """
    n_corners = corners.shape[1]
    descriptor_dim = descriptor_size * descriptor_size
    descriptors = []
    valid_corners_list = []
    
    # Calculate the half-size of the window we need
    half_window = (descriptor_size - 1) * sample_spacing // 2
    
    for i in range(n_corners):
        y, x = corners[0, i], corners[1, i]
        
        # Check if the window is fully within the image
        if (y - half_window < 0 or y + half_window >= image.shape[0] or
            x - half_window < 0 or x + half_window >= image.shape[1]):
            continue
        
        # Sample 8x8 descriptor from 40x40 window with spacing of 5 pixels
        descriptor = np.zeros((descriptor_size, descriptor_size))
        
        for dy in range(descriptor_size):
            for dx in range(descriptor_size):
                # Calculate sample position relative to center
                sample_y = int(y + (dy - descriptor_size // 2 + 0.5) * sample_spacing)
                sample_x = int(x + (dx - descriptor_size // 2 + 0.5) * sample_spacing)
                
                # Get pixel value
                descriptor[dy, dx] = image[sample_y, sample_x]
        
        # Bias/gain normalization: mean=0, std=1
        mean_val = np.mean(descriptor)
        std_val = np.std(descriptor)
        
        if std_val > 1e-6:  # Avoid division by zero
            descriptor = (descriptor - mean_val) / std_val
        else:
            descriptor = descriptor - mean_val
        
        # Flatten to 64-dimensional vector
        descriptors.append(descriptor.flatten())
        valid_corners_list.append([y, x])
    
    if len(descriptors) == 0:
        return np.array([]), corners[:, :0]
    
    descriptors = np.array(descriptors)
    valid_corners = np.array(valid_corners_list).T
    
    return descriptors, valid_corners


def visualize_feature_descriptors(image, corners, descriptors, num_features=16, output_path='assets/feature_descriptors.jpg'):
    """
    Visualize several extracted feature descriptors.
    
    Parameters:
    - image: Original image
    - corners: 2 x n array of corner coordinates
    - descriptors: n x 64 array of feature descriptors
    - num_features: Number of features to visualize
    - output_path: Path to save the visualization
    """
    n_features = min(num_features, descriptors.shape[0])
    
    # Create a grid layout
    grid_size = int(np.ceil(np.sqrt(n_features)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle(f'Feature Descriptors (8×8 normalized patches, sampled from 40×40 windows)', fontsize=14)
    
    # Flatten axes array for easier indexing
    if grid_size == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Select evenly spaced features
    indices = np.linspace(0, descriptors.shape[0] - 1, n_features, dtype=int)
    
    for idx, ax in enumerate(axes):
        if idx < n_features:
            # Get descriptor and reshape to 8x8
            descriptor = descriptors[indices[idx]].reshape(8, 8)
            y, x = corners[0, indices[idx]], corners[1, indices[idx]]
            
            # Display the descriptor
            im = ax.imshow(descriptor, cmap='gray', interpolation='nearest')
            ax.set_title(f'Corner ({int(x)}, {int(y)})', fontsize=8)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Also create a visualization showing the patches on the original image
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Feature Descriptors: Original 40×40 Windows and 8×8 Descriptors', fontsize=14)
    
    # Show first 4 features
    n_show = min(4, n_features)
    indices_show = indices[:n_show]
    
    for idx in range(n_show):
        y, x = int(corners[0, indices_show[idx]]), int(corners[1, indices_show[idx]])
        
        # Extract and show 40x40 window
        window_size = 40
        half_window = window_size // 2
        y_start = max(0, y - half_window)
        y_end = min(image.shape[0], y + half_window)
        x_start = max(0, x - half_window)
        x_end = min(image.shape[1], x + half_window)
        
        window = image[y_start:y_end, x_start:x_end]
        
        # Top row: 40x40 windows
        axes[0, idx].imshow(window, cmap='gray')
        axes[0, idx].set_title(f'40×40 Window at ({x}, {y})', fontsize=10)
        axes[0, idx].axis('off')
        
        # Mark sample points
        for dy in range(8):
            for dx in range(8):
                sample_y_rel = (dy - 3.5) * 5
                sample_x_rel = (dx - 3.5) * 5
                if (0 <= y - y_start + sample_y_rel < window.shape[0] and
                    0 <= x - x_start + sample_x_rel < window.shape[1]):
                    axes[0, idx].plot(x - x_start + sample_x_rel, y - y_start + sample_y_rel, 
                                     'r.', markersize=3)
        
        # Bottom row: 8x8 descriptors
        descriptor = descriptors[indices_show[idx]].reshape(8, 8)
        axes[1, idx].imshow(descriptor, cmap='gray', interpolation='nearest')
        axes[1, idx].set_title(f'8×8 Descriptor (normalized)', fontsize=10)
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    output_path_detailed = output_path.replace('.jpg', '_detailed.jpg')
    plt.savefig(output_path_detailed, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path_detailed}")
    plt.close()


def compute_homography(src_pts, dst_pts):
    """
    Compute homography matrix from point correspondences.
    
    Parameters:
    - src_pts: n x 2 array of source points
    - dst_pts: n x 2 array of destination points
    
    Returns:
    - H: 3x3 homography matrix
    """
    n = src_pts.shape[0]
    
    # Build matrix A for the linear system Ah = 0
    A = []
    for i in range(n):
        x, y = src_pts[i]
        xp, yp = dst_pts[i]
        A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])
    
    A = np.array(A)
    
    # Solve using SVD
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    
    # Reshape to 3x3 matrix
    H = h.reshape(3, 3)
    
    # Normalize so that H[2,2] = 1
    H = H / H[2, 2]
    
    return H


def ransac_homography(matches, corners1, corners2, num_iterations=1000, threshold=5.0):
    """
    Compute robust homography using 4-point RANSAC.
    
    Parameters:
    - matches: List of tuples (idx1, idx2) of matched features
    - corners1: 2 x n1 array of corners in image 1
    - corners2: 2 x n2 array of corners in image 2
    - num_iterations: Number of RANSAC iterations
    - threshold: Inlier threshold in pixels
    
    Returns:
    - best_H: 3x3 homography matrix with most inliers
    - inliers: Boolean array indicating inlier matches
    """
    if len(matches) < 4:
        print("Error: Need at least 4 matches for RANSAC")
        return None, np.array([])
    
    # Extract matched point pairs
    src_pts = []
    dst_pts = []
    for idx1, idx2 in matches:
        # corners are (y, x), convert to (x, y)
        src_pts.append([corners1[1, idx1], corners1[0, idx1]])
        dst_pts.append([corners2[1, idx2], corners2[0, idx2]])
    
    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)
    
    best_H = None
    best_inliers = np.zeros(len(matches), dtype=bool)
    best_inlier_count = 0
    
    print(f"Running RANSAC with {num_iterations} iterations...")
    
    for iteration in range(num_iterations):
        # Randomly select 4 points
        indices = np.random.choice(len(matches), 4, replace=False)
        
        # Compute homography from these 4 points
        try:
            H = compute_homography(src_pts[indices], dst_pts[indices])
        except:
            continue
        
        # Transform all source points using this homography
        src_pts_homogeneous = np.hstack([src_pts, np.ones((len(src_pts), 1))])
        transformed_pts = (H @ src_pts_homogeneous.T).T
        
        # Convert from homogeneous coordinates
        transformed_pts = transformed_pts[:, :2] / transformed_pts[:, 2:3]
        
        # Compute distances to destination points
        distances = np.sqrt(np.sum((transformed_pts - dst_pts)**2, axis=1))
        
        # Count inliers
        inliers = distances < threshold
        inlier_count = np.sum(inliers)
        
        # Update best model if this is better
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inliers = inliers
            best_H = H
    
    print(f"RANSAC completed: {best_inlier_count}/{len(matches)} inliers ({best_inlier_count/len(matches)*100:.1f}%)")
    
    # Recompute homography using all inliers
    if best_inlier_count >= 4:
        inlier_src = src_pts[best_inliers]
        inlier_dst = dst_pts[best_inliers]
        best_H = compute_homography(inlier_src, inlier_dst)
        print(f"Refined homography using {best_inlier_count} inliers")
    
    return best_H, best_inliers


def visualize_corners(image, corners, title, output_path):
    """
    Visualize corners overlaid on image
    
    Parameters:
    - image: Original image (grayscale or RGB)
    - corners: 2 x n array of corner coordinates (ys, xs)
    - title: Title for the plot
    - output_path: Path to save the figure
    """
    plt.figure(figsize=(12, 10))
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    
    # Plot corners
    plt.scatter(corners[1, :], corners[0, :], c='red', marker='x', s=20, linewidths=1)
    plt.title(f'{title} ({corners.shape[1]} corners)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    # Load image
    image_path = 'assets/ParkerM.jpg'
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    print(f"Loading image: {image_path}")
    img = imread(image_path)
    img_gray = rgb2gray(img)
    
    print(f"Image shape: {img.shape}")
    
    # Create output directory
    output_dir = 'assets'
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Harris Corner Detection
    print("\n=== Step 1: Harris Corner Detection ===")
    h, corners_harris = get_harris_corners(
        img_gray, 
        edge_discard=20, 
        min_distance=5,  # Increase this to reduce corner count
        threshold_rel=0.01  # Increase this to reduce corner count
    )
    
    print(f"Number of Harris corners detected: {corners_harris.shape[1]}")
    
    # Visualize Harris corners
    visualize_corners(
        img, 
        corners_harris, 
        'Harris Corners', 
        os.path.join(output_dir, 'harris_corners.jpg')
    )
    
    # Step 2: Adaptive Non-Maximal Suppression (ANMS)
    print("\n=== Step 2: ANMS ===")
    num_anms_points = 500  # Number of corners to select
    
    if corners_harris.shape[1] > 0:
        corners_anms = anms(
            corners_harris, 
            h, 
            num_points=num_anms_points,
            c_robust=0.9
        )
        
        print(f"Number of corners after ANMS: {corners_anms.shape[1]}")
        
        # Visualize ANMS corners
        visualize_corners(
            img, 
            corners_anms, 
            'ANMS Selected Corners', 
            os.path.join(output_dir, 'anms_corners.jpg')
        )
        
        # Step 3: Feature Descriptor Extraction
        print("\n=== Step 3: Feature Descriptor Extraction ===")
        descriptors, valid_corners = extract_feature_descriptors(
            img_gray,
            corners_anms,
            window_size=40,
            descriptor_size=8,
            sample_spacing=5
        )
        
        print(f"Number of valid descriptors extracted: {descriptors.shape[0]}")
        print(f"Descriptor dimensions: {descriptors.shape[1]} (8x8 = 64)")
        print(f"Descriptor statistics - Mean: {np.mean(descriptors):.4f}, Std: {np.std(descriptors):.4f}")
        
        # Visualize feature descriptors
        visualize_feature_descriptors(
            img_gray,
            valid_corners,
            descriptors,
            num_features=16,
            output_path=os.path.join(output_dir, 'feature_descriptors.jpg')
        )
        
    else:
        print("No corners detected!")
    
    print("\n=== Done! ===")
    print(f"Results saved in {output_dir}/")


if __name__ == '__main__':
    main()

