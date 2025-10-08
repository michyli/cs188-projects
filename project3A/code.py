import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
import os

def computeH(im1_pts, im2_pts):
    n = im1_pts.shape[0]
    # Build 2n×9 system Ah=0 from point pairs
    A = np.zeros((2 * n, 9))
    
    for i in range(n):
        x, y = im1_pts[i]
        x_prime, y_prime = im2_pts[i]
        
        A[2*i] = [-x, -y, -1, 0, 0, 0, x*x_prime, y*x_prime, x_prime]
        A[2*i + 1] = [0, 0, 0, -x, -y, -1, x*y_prime, y*y_prime, y_prime]

    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H = h.reshape(3, 3)
    H = H / H[2, 2]
    return H


def warpImageNearestNeighbor(im, H):
    h, w = im.shape[:2]
    has_color = len(im.shape) == 3
    
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]]).T
    warped_corners = H @ corners
    warped_corners = warped_corners[:2, :] / warped_corners[2, :]
    
    min_x, min_y = np.floor(warped_corners.min(axis=1)).astype(int)
    max_x, max_y = np.ceil(warped_corners.max(axis=1)).astype(int)
    
    out_w = max_x - min_x
    out_h = max_y - min_y
    
    T = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
    H_combined = T @ H
    
    H_inv = np.linalg.inv(H_combined)
    if has_color:
        warped_im = np.zeros((out_h, out_w, 4), dtype=im.dtype)
    else:
        warped_im = np.zeros((out_h, out_w, 2), dtype=im.dtype)
    
    y_coords, x_coords = np.mgrid[0:out_h, 0:out_w]
    coords = np.stack([x_coords.ravel(), y_coords.ravel(), np.ones(out_h * out_w)])
    
    src_coords = H_inv @ coords
    src_coords = src_coords[:2, :] / src_coords[2, :]
    
    src_x = src_coords[0, :].reshape(out_h, out_w)
    src_y = src_coords[1, :].reshape(out_h, out_w)
    
    src_x_nn = np.round(src_x).astype(int)
    src_y_nn = np.round(src_y).astype(int)
    
    valid = (src_x_nn >= 0) & (src_x_nn < w) & (src_y_nn >= 0) & (src_y_nn < h)
    
    if has_color:
        warped_im[valid, :3] = im[src_y_nn[valid], src_x_nn[valid], :]
        warped_im[valid, 3] = 255  # Set alpha for valid pixels
    else:
        warped_im[valid, 0] = im[src_y_nn[valid], src_x_nn[valid]]
        warped_im[valid, 1] = 255  # Alpha channel
    
    return warped_im


def warpImageBilinear(im, H):
    h, w = im.shape[:2]
    has_color = len(im.shape) == 3
    
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]]).T
    warped_corners = H @ corners
    warped_corners = warped_corners[:2, :] / warped_corners[2, :]
    
    min_x, min_y = np.floor(warped_corners.min(axis=1)).astype(int)
    max_x, max_y = np.ceil(warped_corners.max(axis=1)).astype(int)
    
    out_w = max_x - min_x
    out_h = max_y - min_y
    
    T = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
    H_combined = T @ H
    H_inv = np.linalg.inv(H_combined)
    
    if has_color:
        warped_im = np.zeros((out_h, out_w, 4), dtype=np.float32)
    else:
        warped_im = np.zeros((out_h, out_w, 2), dtype=np.float32)
    
    y_coords, x_coords = np.mgrid[0:out_h, 0:out_w]
    coords = np.stack([x_coords.ravel(), y_coords.ravel(), np.ones(out_h * out_w)])
    
    src_coords = H_inv @ coords
    src_coords = src_coords[:2, :] / src_coords[2, :]
    
    src_x = src_coords[0, :].reshape(out_h, out_w)
    src_y = src_coords[1, :].reshape(out_h, out_w)
    
    x0 = np.floor(src_x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(src_y).astype(int)
    y1 = y0 + 1
    
    wx = src_x - x0
    wy = src_y - y0
    
    valid = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)
    
    if has_color:
        im_float = im.astype(np.float32)
        for c in range(3):
            warped_im[valid, c] = (
                im_float[y0[valid], x0[valid], c] * (1 - wx[valid]) * (1 - wy[valid]) +
                im_float[y0[valid], x1[valid], c] * wx[valid] * (1 - wy[valid]) +
                im_float[y1[valid], x0[valid], c] * (1 - wx[valid]) * wy[valid] +
                im_float[y1[valid], x1[valid], c] * wx[valid] * wy[valid]
            )
        warped_im[valid, 3] = 255
    else:
        im_float = im.astype(np.float32)
        warped_im[valid, 0] = (
            im_float[y0[valid], x0[valid]] * (1 - wx[valid]) * (1 - wy[valid]) +
            im_float[y0[valid], x1[valid]] * wx[valid] * (1 - wy[valid]) +
            im_float[y1[valid], x0[valid]] * (1 - wx[valid]) * wy[valid] +
            im_float[y1[valid], x1[valid]] * wx[valid] * wy[valid]
        )
        warped_im[valid, 1] = 255
    
    warped_im = warped_im.astype(np.uint8)
    return warped_im


def visualize_correspondences(im1_path, im2_path, im1_pts, im2_pts, output_path, im1_label='Image 1', im2_label='Image 2'):
    im1 = cv2.imread(im1_path)
    im2 = cv2.imread(im2_path)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    ax1.imshow(im1)
    ax1.set_title(im1_label, fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    ax2.imshow(im2)
    ax2.set_title(im2_label, fontsize=16, fontweight='bold')
    ax2.axis('off')
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(im1_pts)))
    
    for i, (pt1, pt2, color) in enumerate(zip(im1_pts, im2_pts, colors)):
        ax1.plot(pt1[0], pt1[1], 'o', color=color, markersize=10, markeredgecolor='white', markeredgewidth=2)
        ax2.plot(pt2[0], pt2[1], 'o', color=color, markersize=10, markeredgecolor='white', markeredgewidth=2)
        ax1.text(pt1[0], pt1[1] - 20, str(i+1), color='white', fontsize=12, fontweight='bold',
                ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
        ax2.text(pt2[0], pt2[1] - 20, str(i+1), color='white', fontsize=12, fontweight='bold',
                ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved correspondence visualization to {output_path}")


def print_homography_matrix(H, pair_name):
    print(f"\n{'='*60}")
    print(f"Homography Matrix for {pair_name}")
    print(f"{'='*60}")
    print("H = ")
    for row in H:
        print(f"  [{row[0]:10.6f}  {row[1]:10.6f}  {row[2]:10.6f}]")
    print(f"{'='*60}\n")


def print_linear_system(im1_pts, im2_pts, pair_name):
    n = im1_pts.shape[0]
    A = np.zeros((2 * n, 9))
    
    for i in range(n):
        x, y = im1_pts[i]
        x_prime, y_prime = im2_pts[i]
        A[2*i] = [-x, -y, -1, 0, 0, 0, x*x_prime, y*x_prime, x_prime]
        A[2*i + 1] = [0, 0, 0, -x, -y, -1, x*y_prime, y*y_prime, y_prime]
    
    print(f"\n{'='*80}")
    print(f"Linear System of Equations (Ah = 0) for {pair_name}")
    print(f"{'='*80}")
    print(f"Number of correspondences: {n}")
    print(f"System size: {A.shape[0]} equations × {A.shape[1]} unknowns")
    print(f"\nMatrix A (first 6 rows shown):")
    print(A[:6])
    print("...")
    print(f"{'='*80}\n")


def main():
    # Create output directory for results
    output_dir = "assets"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process both pairs of images
    pairs = [
        ("ParkerM_ParkerL.json", "assets/ParkerM.jpg", "assets/ParkerL.jpg", "ParkerM to ParkerL", False),
        ("ParkerM_ParkerR.json", "assets/ParkerM.jpg", "assets/ParkerR.jpg", "ParkerM to ParkerR", False)
    ]
    
    results = []
    
    for json_file, im1_path, im2_path, pair_name, swap_display in pairs:
        print(f"\n{'#'*80}")
        print(f"Processing {pair_name}")
        print(f"{'#'*80}")
        
        # Load correspondence points from JSON
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        im1_pts = np.array(data['im1Points'])
        im2_pts = np.array(data['im2Points'])
        
        print(f"\nLoaded {len(im1_pts)} point correspondences")
        
        # Print the linear system
        print_linear_system(im1_pts, im2_pts, pair_name)
        
        # Compute homography
        H = computeH(im1_pts, im2_pts)
        
        # Print the homography matrix
        print_homography_matrix(H, pair_name)
        
        # Visualize correspondences (swap display if needed)
        output_name = json_file.replace('.json', '_correspondences.png')
        output_path = os.path.join(output_dir, output_name)
        if swap_display:
            visualize_correspondences(im2_path, im1_path, im2_pts, im1_pts, output_path)
        else:
            # Keep the semantic labels per pair
            label_left = 'ParkerM (Middle)' if 'ParkerM.jpg' in im1_path else os.path.basename(im1_path).replace('.jpg','')
            label_right = 'ParkerL (Leftmost)' if 'ParkerL.jpg' in im2_path else ('ParkerR (Rightmost)' if 'ParkerR.jpg' in im2_path else os.path.basename(im2_path).replace('.jpg',''))
            visualize_correspondences(im1_path, im2_path, im1_pts, im2_pts, output_path,
                                       im1_label=label_left, im2_label=label_right)
        
        # Verify the homography by checking reprojection error
        # Convert points to homogeneous coordinates
        im1_pts_h = np.column_stack([im1_pts, np.ones(len(im1_pts))])
        
        # Apply homography
        im2_pts_pred = (H @ im1_pts_h.T).T
        im2_pts_pred = im2_pts_pred[:, :2] / im2_pts_pred[:, 2:3]
        
        # Compute error
        errors = np.linalg.norm(im2_pts - im2_pts_pred, axis=1)
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        print(f"Reprojection Error Statistics:")
        print(f"  Mean error: {mean_error:.4f} pixels")
        print(f"  Max error:  {max_error:.4f} pixels")
        print(f"  All errors: {errors}")
        
        results.append({
            'pair_name': pair_name,
            'json_file': json_file,
            'num_points': len(im1_pts),
            'homography': H,
            'mean_error': mean_error,
            'max_error': max_error,
            'correspondence_image': output_name
        })
    
    # Save homography matrices to file
    with open('homography_results.txt', 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"{'='*80}\n")
            f.write(f"{result['pair_name']}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Number of correspondences: {result['num_points']}\n\n")
            f.write("Homography Matrix H:\n")
            for row in result['homography']:
                f.write(f"  [{row[0]:12.8f}  {row[1]:12.8f}  {row[2]:12.8f}]\n")
            f.write(f"\nReprojection Error:\n")
            f.write(f"  Mean: {result['mean_error']:.4f} pixels\n")
            f.write(f"  Max:  {result['max_error']:.4f} pixels\n\n")
    
    print("\n" + "="*80)
    print("All results saved!")
    print("  - Correspondence visualizations: assets/")
    print("  - Homography matrices: homography_results.txt")
    print("="*80)


def create_mosaic(images, homographies, reference_idx=1):
    """
    Create a mosaic by warping and blending multiple images.
    
    Args:
        images: list of images (RGB)
        homographies: list of homographies to warp each image to reference frame
                      homographies[i] warps images[i] to reference frame
        reference_idx: index of reference image (will not be warped)
    
    Returns:
        mosaic: blended mosaic image
    """
    # Warp all images (including reference) and collect them
    warped_images = []
    
    # Compute overall bounding box
    min_x, min_y = 0, 0
    max_x, max_y = images[reference_idx].shape[1], images[reference_idx].shape[0]
    
    for i, (im, H) in enumerate(zip(images, homographies)):
        if i == reference_idx:
            continue
        
        h, w = im.shape[:2]
        corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]]).T
        warped_corners = H @ corners
        warped_corners = warped_corners[:2, :] / warped_corners[2, :]
        
        min_x = min(min_x, warped_corners[0, :].min())
        min_y = min(min_y, warped_corners[1, :].min())
        max_x = max(max_x, warped_corners[0, :].max())
        max_y = max(max_y, warped_corners[1, :].max())
    
    min_x, min_y = int(np.floor(min_x)), int(np.floor(min_y))
    max_x, max_y = int(np.ceil(max_x)), int(np.ceil(max_y))
    
    # Create translation to shift everything into positive coordinates
    T = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
    
    out_w = max_x - min_x
    out_h = max_y - min_y
    
    print(f"Mosaic size: {out_w} x {out_h}")
    
    # Helper warp directly into the final canvas coordinates without re-bboxing
    def warp_to_canvas_bilinear(image_rgb: np.ndarray, H_canvas: np.ndarray, canvas_h: int, canvas_w: int) -> np.ndarray:
        has_color = len(image_rgb.shape) == 3
        h, w = image_rgb.shape[:2]
        # Prepare output RGBA
        canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
        # Inverse homography (maps canvas -> source image)
        H_inv = np.linalg.inv(H_canvas)
        # Meshgrid over canvas
        y_coords, x_coords = np.mgrid[0:canvas_h, 0:canvas_w]
        ones = np.ones_like(x_coords)
        dst = np.stack([x_coords.ravel(), y_coords.ravel(), ones.ravel()])
        src = H_inv @ dst
        src_x = (src[0, :] / src[2, :]).reshape(canvas_h, canvas_w)
        src_y = (src[1, :] / src[2, :]).reshape(canvas_h, canvas_w)
        # Bilinear sampling
        x0 = np.floor(src_x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(src_y).astype(int)
        y1 = y0 + 1
        wx = src_x - x0
        wy = src_y - y0
        valid = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)
        if np.any(valid):
            src_float = image_rgb.astype(np.float32)
            for c in range(3):
                sampled = (
                    src_float[y0[valid], x0[valid], c] * (1 - wx[valid]) * (1 - wy[valid]) +
                    src_float[y0[valid], x1[valid], c] * wx[valid] * (1 - wy[valid]) +
                    src_float[y1[valid], x0[valid], c] * (1 - wx[valid]) * wy[valid] +
                    src_float[y1[valid], x1[valid], c] * wx[valid] * wy[valid]
                )
                canvas[..., c][valid] = sampled.astype(np.uint8)
            canvas[..., 3][valid] = 255
        return canvas

    # Warp all images into the shared canvas using T for global translation
    for i, (im, H) in enumerate(zip(images, homographies)):
        if i == reference_idx:
            # Reference should also receive the global translation T
            H_canvas = T @ np.eye(3)
            warped = warp_to_canvas_bilinear(im, H_canvas, out_h, out_w)
        else:
            H_canvas = T @ H
            warped = warp_to_canvas_bilinear(im, H_canvas, out_h, out_w)
        # Apply distance alpha falloff based on valid mask
        warped = apply_distance_alpha_to_warped(warped)
        warped_images.append(warped)
    
    # Blend all images
    mosaic = blend_images(warped_images)
    
    return mosaic


def create_distance_alpha(h, w, falloff=0.1):
    """
    Create an alpha mask that falls off from center to edges.
    
    Args:
        h: height
        w: width
        falloff: fraction of dimension to use for falloff (0 to 1)
    
    Returns:
        alpha: alpha mask (h x w)
    """
    from scipy.ndimage import distance_transform_edt
    
    # Create a mask of the image region
    mask = np.ones((h, w), dtype=np.uint8)
    
    # Compute distance from edge
    dist = distance_transform_edt(mask)
    
    # Normalize to [0, 1]
    max_dist = dist.max()
    if max_dist > 0:
        alpha = dist / max_dist
    else:
        alpha = np.ones_like(dist)
    
    # Convert to uint8
    alpha = (alpha * 255).astype(np.uint8)
    
    return alpha


def apply_distance_alpha_to_warped(warped_image):
    """
    Apply distance-based alpha falloff to a warped image.
    Modifies the existing alpha channel.
    
    Args:
        warped_image: warped image with alpha channel (H x W x 4)
    
    Returns:
        warped_image: image with updated alpha
    """
    from scipy.ndimage import distance_transform_edt
    
    # Get existing alpha (binary mask of valid pixels)
    valid_mask = (warped_image[:, :, 3] > 0).astype(np.uint8)
    
    if valid_mask.sum() == 0:
        return warped_image
    
    # Compute distance from invalid pixels
    dist = distance_transform_edt(valid_mask)
    
    # Normalize
    max_dist = dist.max()
    if max_dist > 0:
        alpha_float = dist / max_dist
    else:
        alpha_float = valid_mask.astype(float)
    
    # Combine with existing alpha
    warped_image[:, :, 3] = (alpha_float * 255).astype(np.uint8) * (valid_mask > 0)
    
    return warped_image


def blend_images(images):
    """
    Blend multiple images with alpha channels using weighted averaging.
    
    Args:
        images: list of images with alpha channel (H x W x 4)
    
    Returns:
        blended: blended image (H x W x 3)
    """
    h, w = images[0].shape[:2]
    
    # Accumulate weighted sum
    sum_rgb = np.zeros((h, w, 3), dtype=np.float32)
    sum_alpha = np.zeros((h, w), dtype=np.float32)
    
    for img in images:
        alpha = img[:, :, 3].astype(np.float32) / 255.0
        # Expand alpha to 3D for broadcasting
        alpha_3d = alpha[:, :, np.newaxis]
        sum_rgb += img[:, :, :3].astype(np.float32) * alpha_3d
        sum_alpha += alpha
    
    # Avoid division by zero
    sum_alpha = np.maximum(sum_alpha, 1e-6)
    
    # Compute weighted average
    sum_alpha_3d = sum_alpha[:, :, np.newaxis]
    blended = (sum_rgb / sum_alpha_3d).astype(np.uint8)
    
    return blended


def test_rectification():
    """
    Test rectification on images with rectangular objects.
    This demonstrates that the warping functions work correctly.
    """
    print("\n" + "#"*80)
    print("Testing Rectification (Part A.3)")
    print("#"*80)
    
    output_dir = "assets"
    
    # Load the light switch image
    test_image_path = "assets/light_switch.jpg"
    if not os.path.exists(test_image_path):
        print(f"Test image {test_image_path} not found. Skipping rectification test.")
        return
    
    im = cv2.imread(test_image_path)
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    # Load correspondence points from JSON
    # These define the corners of the light switch plate
    with open('Light_Light.json', 'r') as f:
        data = json.load(f)
    
    # Input order is TL, TR, BL, BR per user; map accordingly
    im1_pts_raw = np.array(data['im1Points'])
    im1_pts = im1_pts_raw
    
    # Calculate dimensions to preserve aspect ratio
    # Width: top edge
    width_input = np.linalg.norm(im1_pts[1] - im1_pts[0])
    # Height: left edge  
    height_input = np.linalg.norm(im1_pts[3] - im1_pts[0])
    
    # Create output rectangle maintaining aspect ratio
    # Scale to reasonable output size (e.g., width=300)
    scale = 300 / width_input
    width = 300
    height = int(height_input * scale)
    
    # Define target rectangle
    im2_pts = np.array([
        [0, 0],           # top-left
        [width, 0],       # top-right
        [width, height],  # bottom-right
        [0, height]       # bottom-left
    ])
    
    print(f"\nRectifying light switch plate:")
    print(f"  Input quadrilateral corners: {im1_pts.tolist()}")
    print(f"  Output rectangle size: {width}x{height}")
    
    # Compute homography
    H = computeH(im1_pts, im2_pts)
    
    print("\nRectification Homography:")
    print(H)
    
    # Warp using both methods
    print("\nWarping with Nearest Neighbor...")
    warped_nn = warpImageNearestNeighbor(im_rgb, H)
    
    print("Warping with Bilinear...")
    warped_bil = warpImageBilinear(im_rgb, H)
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original with marked points
    axes[0].imshow(im_rgb)
    axes[0].plot([im1_pts[0, 0], im1_pts[1, 0], im1_pts[2, 0], im1_pts[3, 0], im1_pts[0, 0]],
                 [im1_pts[0, 1], im1_pts[1, 1], im1_pts[2, 1], im1_pts[3, 1], im1_pts[0, 1]],
                 'r-', linewidth=3)
    axes[0].scatter(im1_pts[:, 0], im1_pts[:, 1], c='red', s=150, zorder=5, edgecolors='white', linewidths=2)
    # Label corners
    labels = ['TL', 'TR', 'BR', 'BL']
    for i, (pt, label) in enumerate(zip(im1_pts, labels)):
        axes[0].text(pt[0], pt[1]-15, label, color='white', fontsize=12, fontweight='bold',
                    ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8))
    axes[0].set_title('Original: Light Switch (Distorted)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Nearest Neighbor result
    axes[1].imshow(warped_nn)
    axes[1].set_title('Rectified: Nearest Neighbor', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Bilinear result
    axes[2].imshow(warped_bil)
    axes[2].set_title('Rectified: Bilinear Interpolation', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'rectification_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved rectification comparison to {output_path}")
    
    # Save individual warped images for closer inspection
    cv2.imwrite(os.path.join(output_dir, 'rectified_nn.png'), 
                cv2.cvtColor(warped_nn[:,:,:3], cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, 'rectified_bilinear.png'), 
                cv2.cvtColor(warped_bil[:,:,:3], cv2.COLOR_RGB2BGR))
    
    # Create a zoomed comparison to show quality difference
    # Take a crop from the center
    h, w = warped_nn.shape[:2]
    crop_size = 200
    y_start = h // 2 - crop_size // 2
    x_start = w // 2 - crop_size // 2
    
    if y_start >= 0 and x_start >= 0 and y_start + crop_size < h and x_start + crop_size < w:
        crop_nn = warped_nn[y_start:y_start+crop_size, x_start:x_start+crop_size]
        crop_bil = warped_bil[y_start:y_start+crop_size, x_start:x_start+crop_size]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(crop_nn)
        ax1.set_title('Nearest Neighbor (Zoomed)', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        ax2.imshow(crop_bil)
        ax2.set_title('Bilinear (Zoomed)', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'interpolation_comparison_zoomed.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved zoomed comparison to {output_path}")
    
    print("\n" + "="*80)
    print("Rectification test completed!")
    print("="*80)


def test_rectification_window():
    """
    Additional rectification test using Window.jpg with points from
    Window_Window.json.
    """
    print("\n" + "#"*80)
    print("Testing Rectification (Window)")
    print("#"*80)
    output_dir = "assets"
    image_path = os.path.join(output_dir, "Window.jpg")
    json_path = "Window_Window.json"
    if not (os.path.exists(image_path) and os.path.exists(json_path)):
        print("Window assets not found. Skipping window rectification test.")
        return
    im = cv2.imread(image_path)
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    with open(json_path, 'r') as f:
        data = json.load(f)
    im1_pts = np.array(data['im1Points'])
    # Keep aspect ratio based on top edge and left edge
    width_input = np.linalg.norm(im1_pts[1] - im1_pts[0])
    height_input = np.linalg.norm(im1_pts[3] - im1_pts[0])
    width = 400
    scale = width / max(width_input, 1e-6)
    height = int(height_input * scale)
    # Target rectangle must match the same corner ordering (TL, TR, BL, BR)
    im2_pts = np.array([[0, 0], [width, 0], [0, height], [width, height]])
    H = computeH(im1_pts, im2_pts)
    warped_nn = warpImageNearestNeighbor(im_rgb, H)
    warped_bil = warpImageBilinear(im_rgb, H)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(im_rgb)
    axes[0].plot([im1_pts[0, 0], im1_pts[1, 0], im1_pts[2, 0], im1_pts[3, 0], im1_pts[0, 0]],
                 [im1_pts[0, 1], im1_pts[1, 1], im1_pts[2, 1], im1_pts[3, 1], im1_pts[0, 1]],
                 'r-', linewidth=3)
    axes[0].scatter(im1_pts[:, 0], im1_pts[:, 1], c='red', s=150, zorder=5, edgecolors='white', linewidths=2)
    axes[0].set_title('Original: Window (Distorted)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    axes[1].imshow(warped_nn)
    axes[1].set_title('Rectified: Nearest Neighbor', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    axes[2].imshow(warped_bil)
    axes[2].set_title('Rectified: Bilinear', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'rectification_window_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved window rectification comparison to {out_path}")

def create_parker_mosaics():
    """
    Create mosaics from the Parker images (Part A.4).
    """
    print("\n" + "#"*80)
    print("Creating Image Mosaics (Part A.4)")
    print("#"*80)
    
    output_dir = "assets"
    
    # Load images
    im_L = cv2.imread("assets/ParkerL.jpg")
    im_M = cv2.imread("assets/ParkerM.jpg")
    im_R = cv2.imread("assets/ParkerR.jpg")
    
    im_L_rgb = cv2.cvtColor(im_L, cv2.COLOR_BGR2RGB)
    im_M_rgb = cv2.cvtColor(im_M, cv2.COLOR_BGR2RGB)
    im_R_rgb = cv2.cvtColor(im_R, cv2.COLOR_BGR2RGB)
    
    # Load correspondences and compute homographies
    with open("ParkerM_ParkerL.json", 'r') as f:
        data_ML = json.load(f)
    with open("ParkerM_ParkerR.json", 'r') as f:
        data_MR = json.load(f)
    
    # Compute homographies from M to L and M to R
    # im1Points are from ParkerM, im2Points are from ParkerL/R
    # So H maps points from M to L/R
    H_M_to_L = computeH(np.array(data_ML['im1Points']), np.array(data_ML['im2Points']))
    H_M_to_R = computeH(np.array(data_MR['im1Points']), np.array(data_MR['im2Points']))
    
    # Also compute inverse homographies (L to M, R to M)
    # Important: Normalize after inversion to ensure H[2,2] = 1!
    H_L_to_M = np.linalg.inv(H_M_to_L)
    H_L_to_M = H_L_to_M / H_L_to_M[2, 2]  # Normalize
    
    H_R_to_M = np.linalg.inv(H_M_to_R)
    H_R_to_M = H_R_to_M / H_R_to_M[2, 2]  # Normalize
    
    print("\n" + "="*80)
    print("Mosaic 1: Left + Middle (L and M)")
    print("="*80)
    
    # Mosaic 1: L + M (use M as reference)
    images_LM = [im_L_rgb, im_M_rgb]
    homographies_LM = [H_L_to_M, np.eye(3)]  # L warped to M, M is identity
    mosaic_LM = create_mosaic(images_LM, homographies_LM, reference_idx=1)
    
    output_path = os.path.join(output_dir, 'mosaic_L_M.jpg')
    cv2.imwrite(output_path, cv2.cvtColor(mosaic_LM, cv2.COLOR_RGB2BGR))
    print(f"Saved mosaic to {output_path}")
    
    print("\n" + "="*80)
    print("Mosaic 2: Middle + Right (M and R)")
    print("="*80)
    
    # Mosaic 2: M + R (use M as reference)
    images_MR = [im_M_rgb, im_R_rgb]
    homographies_MR = [np.eye(3), H_R_to_M]  # M is identity, R warped to M
    mosaic_MR = create_mosaic(images_MR, homographies_MR, reference_idx=0)
    
    output_path = os.path.join(output_dir, 'mosaic_M_R.jpg')
    cv2.imwrite(output_path, cv2.cvtColor(mosaic_MR, cv2.COLOR_RGB2BGR))
    print(f"Saved mosaic to {output_path}")
    
    print("\n" + "="*80)
    print("Mosaic 3: Left + Middle + Right (Full Panorama)")
    print("="*80)
    
    # Mosaic 3: L + M + R (use M as reference)
    images_LMR = [im_L_rgb, im_M_rgb, im_R_rgb]
    homographies_LMR = [H_L_to_M, np.eye(3), H_R_to_M]  # All warped to M
    mosaic_LMR = create_mosaic(images_LMR, homographies_LMR, reference_idx=1)
    
    output_path = os.path.join(output_dir, 'mosaic_L_M_R.jpg')
    cv2.imwrite(output_path, cv2.cvtColor(mosaic_LMR, cv2.COLOR_RGB2BGR))
    print(f"Saved mosaic to {output_path}")
    
    # Create a comparison figure showing all three source images
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(im_L_rgb)
    axes[0].set_title('Source: ParkerL (Left)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(im_M_rgb)
    axes[1].set_title('Source: ParkerM (Middle)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(im_R_rgb)
    axes[2].set_title('Source: ParkerR (Right)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'mosaic_source_images.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved source images comparison to {output_path}")
    
    # Create comparison showing mosaics
    fig, axes = plt.subplots(3, 1, figsize=(18, 16))
    
    axes[0].imshow(mosaic_LM)
    axes[0].set_title('Mosaic 1: Left + Middle', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(mosaic_MR)
    axes[1].set_title('Mosaic 2: Middle + Right', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(mosaic_LMR)
    axes[2].set_title('Mosaic 3: Full Panorama (Left + Middle + Right)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'mosaics_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved mosaics comparison to {output_path}")
    
    print("\n" + "="*80)
    print("Mosaic creation completed!")
    print("="*80)


def create_room_mosaics():
    """
    Create mosaics from the Room images using the same pipeline (Part A.4).
    RoomM is used as the reference frame.
    """
    print("\n" + "#"*80)
    print("Creating Room Image Mosaics (Part A.4)")
    print("#"*80)
    
    output_dir = "assets"
    
    # Load images
    im_L = cv2.imread("assets/RoomL.jpg")
    im_M = cv2.imread("assets/RoomM.jpg")
    im_R = cv2.imread("assets/RoomR.jpg")
    
    if im_L is None or im_M is None or im_R is None:
        print("Room images not found. Skipping Room mosaics.")
        return
    
    im_L_rgb = cv2.cvtColor(im_L, cv2.COLOR_BGR2RGB)
    im_M_rgb = cv2.cvtColor(im_M, cv2.COLOR_BGR2RGB)
    im_R_rgb = cv2.cvtColor(im_R, cv2.COLOR_BGR2RGB)
    
    # Load correspondences
    with open("RoomL_RoomM.json", 'r') as f:
        data_LM = json.load(f)
    with open("RoomM_RoomR.json", 'r') as f:
        data_MR = json.load(f)
    
    # Compute homographies
    # RoomL -> RoomM
    H_L_to_M = computeH(np.array(data_LM['im1Points']), np.array(data_LM['im2Points']))
    # RoomM -> RoomR (we need RoomR -> RoomM)
    H_M_to_R = computeH(np.array(data_MR['im1Points']), np.array(data_MR['im2Points']))
    H_R_to_M = np.linalg.inv(H_M_to_R)
    H_R_to_M = H_R_to_M / H_R_to_M[2, 2]
    
    print("\n" + "="*80)
    print("Room Mosaic 1: Left + Middle (L and M)")
    print("="*80)
    images_LM = [im_L_rgb, im_M_rgb]
    homographies_LM = [H_L_to_M, np.eye(3)]
    mosaic_LM = create_mosaic(images_LM, homographies_LM, reference_idx=1)
    cv2.imwrite(os.path.join(output_dir, 'room_mosaic_L_M.jpg'), cv2.cvtColor(mosaic_LM, cv2.COLOR_RGB2BGR))
    
    print("\n" + "="*80)
    print("Room Mosaic 2: Middle + Right (M and R)")
    print("="*80)
    images_MR = [im_M_rgb, im_R_rgb]
    homographies_MR = [np.eye(3), H_R_to_M]
    mosaic_MR = create_mosaic(images_MR, homographies_MR, reference_idx=0)
    cv2.imwrite(os.path.join(output_dir, 'room_mosaic_M_R.jpg'), cv2.cvtColor(mosaic_MR, cv2.COLOR_RGB2BGR))
    
    print("\n" + "="*80)
    print("Room Mosaic 3: Left + Middle + Right (Full Panorama)")
    print("="*80)
    images_LMR = [im_L_rgb, im_M_rgb, im_R_rgb]
    homographies_LMR = [H_L_to_M, np.eye(3), H_R_to_M]
    mosaic_LMR = create_mosaic(images_LMR, homographies_LMR, reference_idx=1)
    cv2.imwrite(os.path.join(output_dir, 'room_mosaic_L_M_R.jpg'), cv2.cvtColor(mosaic_LMR, cv2.COLOR_RGB2BGR))
    
    # Source comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, img, title in zip(
        axes,
        [im_L_rgb, im_M_rgb, im_R_rgb],
        ['RoomL (Left)', 'RoomM (Middle)', 'RoomR (Right)']
    ):
        ax.imshow(img)
        ax.set_title(f'Source: {title}', fontsize=14, fontweight='bold')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'room_mosaic_source_images.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Mosaics comparison
    fig, axes = plt.subplots(3, 1, figsize=(18, 16))
    axes[0].imshow(mosaic_LM); axes[0].set_title('Room: Left + Middle', fontsize=14, fontweight='bold'); axes[0].axis('off')
    axes[1].imshow(mosaic_MR); axes[1].set_title('Room: Middle + Right', fontsize=14, fontweight='bold'); axes[1].axis('off')
    axes[2].imshow(mosaic_LMR); axes[2].set_title('Room: Full Panorama (L + M + R)', fontsize=14, fontweight='bold'); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'room_mosaics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*80)
    print("Room mosaic creation completed!")
    print("="*80)


def create_room_correspondences():
    """
    Generate correspondence visualizations and print homography matrices for
    the Room image set (A.2 display support).
    """
    print("\n" + "#"*80)
    print("Room Correspondence Visualizations (Part A.2)")
    print("#"*80)
    
    pairs = [
        ("RoomL_RoomM.json", "assets/RoomL.jpg", "assets/RoomM.jpg", "RoomL to RoomM"),
        ("RoomM_RoomR.json", "assets/RoomM.jpg", "assets/RoomR.jpg", "RoomM to RoomR")
    ]
    for json_file, im1_path, im2_path, title in pairs:
        if not (os.path.exists(json_file) and os.path.exists(im1_path) and os.path.exists(im2_path)):
            print(f"Skipping {title}: files not found")
            continue
        with open(json_file, 'r') as f:
            data = json.load(f)
        im1_pts = np.array(data['im1Points'])
        im2_pts = np.array(data['im2Points'])
        H = computeH(im1_pts, im2_pts)
        print("\n============================================================")
        print(f"Homography Matrix for {title}")
        print("============================================================")
        for row in H:
            print(f"  [{row[0]:10.6f}  {row[1]:10.6f}  {row[2]:10.6f}]")
        out_name = json_file.replace('.json', '_correspondences.png')
        visualize_correspondences(im1_path, im2_path, im1_pts, im2_pts, os.path.join('assets', out_name),
                                  im1_label=os.path.basename(im1_path).replace('.jpg',''),
                                  im2_label=os.path.basename(im2_path).replace('.jpg',''))


if __name__ == "__main__":
    # Run Part A.2 - Homography recovery
    main()
    
    # Run Part A.3 - Rectification test
    test_rectification()
    
    # Run Part A.4 - Create mosaics
    create_parker_mosaics()

