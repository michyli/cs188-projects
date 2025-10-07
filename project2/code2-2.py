import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import skimage.transform as sktr
from PIL import Image
import math

def load_image(image_path):
    """Load an image and normalize to [0,1] range."""
    try:
        img = plt.imread(image_path)
        if img.dtype == np.uint8:
            img = img.astype(np.float64) / 255.0
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def create_gaussian_filter_2d(size, sigma):
    """Create a 2D Gaussian filter."""
    kernel = np.zeros((size, size))
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # Normalize the kernel
    kernel = kernel / np.sum(kernel)
    return kernel

def low_pass_filter(image, sigma):
    """Apply low-pass Gaussian filter to image."""
    if len(image.shape) == 3:  # Color image
        filtered = np.zeros_like(image)
        for channel in range(image.shape[2]):
            filtered[:, :, channel] = gaussian_filter(image[:, :, channel], sigma=sigma)
        return filtered
    else:  # Grayscale image
        return gaussian_filter(image, sigma=sigma)

def high_pass_filter(image, sigma):
    """Apply high-pass filter (original - gaussian) to image."""
    low_freq = low_pass_filter(image, sigma)
    return image - low_freq

def hybrid_image(image_far, image_close, sigma):
    """
    Create hybrid image by combining low frequencies from image_far and high frequencies from image_close.
    
    Args:
        image_far: Image to be viewed from far (will be low-pass filtered)
        image_close: Image to be viewed from close (will be high-pass filtered) 
        sigma: Standard deviation for both filters
    
    Returns:
        Hybrid image
    """
    # Get low frequencies from image_far (for distant viewing)
    low_freq = low_pass_filter(image_far, sigma)
    
    # Get high frequencies from image_close (for close viewing)
    high_freq = high_pass_filter(image_close, sigma)
    
    # Combine them: image_hybrid = image_far_low_passed + image_close_high_passed
    hybrid = low_freq + high_freq
    
    # Clip to valid range
    hybrid = np.clip(hybrid, 0, 1)
    
    return hybrid

def align_images_manual(im1, im2, rotation_angle=0, translation_x=0, translation_y=0, scale_factor=1.0):
    """
    Manually align images with tunable parameters.
    
    Args:
        im1: First image
        im2: Second image to be transformed
        rotation_angle: Rotation angle in degrees
        translation_x: Translation in x direction (pixels)
        translation_y: Translation in y direction (pixels) 
        scale_factor: Scale factor for im2
    
    Returns:
        Aligned images
    """
    # Apply transformations to im2
    im2_transformed = im2.copy()
    
    # Scale
    if scale_factor != 1.0:
        im2_transformed = sktr.rescale(im2_transformed, scale_factor, channel_axis=2)
    
    # Rotate
    if rotation_angle != 0:
        im2_transformed = sktr.rotate(im2_transformed, rotation_angle)
    
    # Translate
    if translation_x != 0 or translation_y != 0:
        # Create translation matrix
        tform = sktr.AffineTransform(translation=[translation_x, translation_y])
        im2_transformed = sktr.warp(im2_transformed, tform.inverse)
    
    # Make images same size
    h1, w1 = im1.shape[:2]
    h2, w2 = im2_transformed.shape[:2]
    
    # Crop or pad to match sizes
    min_h, min_w = min(h1, h2), min(w1, w2)
    
    # Center crop both images
    start_h1, start_w1 = (h1 - min_h) // 2, (w1 - min_w) // 2
    start_h2, start_w2 = (h2 - min_h) // 2, (w2 - min_w) // 2
    
    im1_aligned = im1[start_h1:start_h1+min_h, start_w1:start_w1+min_w]
    im2_aligned = im2_transformed[start_h2:start_h2+min_h, start_w2:start_w2+min_w]
    
    return im1_aligned, im2_aligned

def show_frequency_analysis(im1, im2, hybrid, im1_filtered, im2_filtered):
    """Show Fourier transform analysis of the images."""
    
    def get_fft_magnitude(image):
        """Get log magnitude of FFT for an image."""
        if len(image.shape) == 3:
            # Convert to grayscale for FFT analysis
            gray = np.dot(image[...,:3], [0.299, 0.587, 0.114])
        else:
            gray = image
        
        fft = np.fft.fft2(gray)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.log(np.abs(fft_shifted) + 1e-10)  # Add small value to avoid log(0)
        return magnitude
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Frequency Analysis of Hybrid Image Creation', fontsize=16)
    
    # Original images
    axes[0, 0].imshow(get_fft_magnitude(im1), cmap='gray')
    axes[0, 0].set_title('FFT: Original Image 1 (High Freq Source)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(get_fft_magnitude(im2), cmap='gray')
    axes[0, 1].set_title('FFT: Original Image 2 (Low Freq Source)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(get_fft_magnitude(hybrid), cmap='gray')
    axes[0, 2].set_title('FFT: Hybrid Image')
    axes[0, 2].axis('off')
    
    # Filtered images
    axes[1, 0].imshow(get_fft_magnitude(im1_filtered), cmap='gray')
    axes[1, 0].set_title('FFT: High-pass Filtered Image 1')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(get_fft_magnitude(im2_filtered), cmap='gray')
    axes[1, 1].set_title('FFT: Low-pass Filtered Image 2')
    axes[1, 1].axis('off')
    
    # Show hybrid in spatial domain
    axes[1, 2].imshow(hybrid)
    axes[1, 2].set_title('Hybrid Image (Spatial Domain)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('assets/hybrid_frequency_analysis.png', dpi=150, bbox_inches='tight')
    print("Frequency analysis saved as 'assets/hybrid_frequency_analysis.png'")
    return fig


def create_hybrid_pair(image_far_path, image_close_path, pair_name, rotation=0, trans_x=0, trans_y=0, scale=1.0, sigma=10):
    """Create hybrid image for a specific pair with given parameters."""
    print(f"\n{'='*50}")
    print(f"Creating Hybrid Image: {pair_name}")
    print(f"{'='*50}")
    
    # Load images
    print("Loading images...")
    image_far = load_image(image_far_path)   # will be low-passed
    image_close = load_image(image_close_path)  # will be high-passed
    
    if image_far is None or image_close is None:
        print(f"Error loading images for {pair_name}")
        return None
    
    print(f"Image far shape: {image_far.shape}")
    print(f"Image close shape: {image_close.shape}")
    
    # Align images
    print(f"Alignment parameters: rotation={rotation}°, translation=({trans_x}, {trans_y}), scale={scale}")
    image_far_aligned, image_close_aligned = align_images_manual(
        image_far, image_close, 
        rotation_angle=rotation,
        translation_x=trans_x, 
        translation_y=trans_y,
        scale_factor=scale
    )
    
    print(f"Aligned shapes: {image_far_aligned.shape}, {image_close_aligned.shape}")
    
    # Apply filters
    print(f"Applying filters with sigma = {sigma}")
    image_far_low = low_pass_filter(image_far_aligned, sigma)
    image_close_high = high_pass_filter(image_close_aligned, sigma)
    
    # Create hybrid
    hybrid = image_far_low + image_close_high
    hybrid = np.clip(hybrid, 0, 1)
    
    print("Hybrid image created successfully!")
    
    # Create visualization (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Hybrid Image: {pair_name}', fontsize=16)
    
    # Top row: original images and hybrid
    axes[0, 0].imshow(image_far_aligned)
    axes[0, 0].set_title('Original (Far View)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(image_close_aligned)
    axes[0, 1].set_title('Original (Close View)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(hybrid)
    axes[0, 2].set_title('Hybrid Image')
    axes[0, 2].axis('off')
    
    # Bottom row: filtered components and hybrid again
    axes[1, 0].imshow(image_far_low)
    axes[1, 0].set_title('Low-pass Filtered')
    axes[1, 0].axis('off')
    
    # Normalize high-pass for better visualization
    image_close_high_norm = image_close_high - np.min(image_close_high)
    if np.max(image_close_high_norm) > 0:
        image_close_high_norm = image_close_high_norm / np.max(image_close_high_norm)
    
    axes[1, 1].imshow(image_close_high_norm)
    axes[1, 1].set_title('High-pass Filtered')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(hybrid)
    axes[1, 2].set_title('Final Result')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save with appropriate filename
    filename = f"assets/hybrid_{pair_name.lower().replace(' ', '_').replace('+', '_')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    
    return fig, hybrid

def main():
    print("Hybrid Images Implementation - Multiple Pairs")
    print("=" * 50)
    
    # Create hybrid images for all three pairs
    pairs = [
        {
            'name': 'Derek + Nutmeg',
            'far_path': 'assets/DerekPicture.jpg',
            'close_path': 'assets/nutmeg.jpg',
            'rotation': 35,
            'trans_x': 136,
            'trans_y': -30,
            'scale': 1.0,
            'sigma': 10
        },
        {
            'name': 'Man2 + Tiger',
            'far_path': 'assets/man2.png',
            'close_path': 'assets/tiger.png',
            'rotation': 0,
            'trans_x': 0,
            'trans_y': -40,
            'scale': 0.7,
            'sigma': 10
        },
        {
            'name': 'Superman + Lion',
            'far_path': 'assets/superman.png',
            'close_path': 'assets/lion.png',
            'rotation': 0,
            'trans_x': 0,
            'trans_y': 0,
            'scale': 1.0,
            'sigma': 10
        }
    ]
    
    # Create hybrid images for each pair
    for pair in pairs:
        result = create_hybrid_pair(
            pair['far_path'], pair['close_path'], pair['name'],
            rotation=pair['rotation'], trans_x=pair['trans_x'], 
            trans_y=pair['trans_y'], scale=pair['scale'], 
            sigma=pair['sigma']
        )
        
        if result is not None:
            fig, hybrid = result
            
            # For the first pair (Derek + Nutmeg), also create frequency analysis
            if pair['name'] == 'Derek + Nutmeg':
                print("\nPerforming frequency analysis for Derek + Nutmeg...")
                # Reload and process for frequency analysis
                derek_img = load_image(pair['far_path'])
                nutmeg_img = load_image(pair['close_path'])
                derek_aligned, nutmeg_aligned = align_images_manual(
                    derek_img, nutmeg_img, 
                    rotation_angle=pair['rotation'],
                    translation_x=pair['trans_x'], 
                    translation_y=pair['trans_y'],
                    scale_factor=pair['scale']
                )
                derek_low = low_pass_filter(derek_aligned, pair['sigma'])
                nutmeg_high = high_pass_filter(nutmeg_aligned, pair['sigma'])
                show_frequency_analysis(derek_aligned, nutmeg_aligned, hybrid, derek_low, nutmeg_high)
    
    plt.show()
    
    print("\n" + "="*50)
    print("All hybrid images created successfully!")
    print("Generated files:")
    print("- hybrid_derek_nutmeg.png")
    print("- hybrid_man2_panda.png") 
    print("- hybrid_superman_lion.png")
    print("- hybrid_frequency_analysis.png (for Derek + Nutmeg)")
    print("\nVIEWING INSTRUCTIONS:")
    print("- View from CLOSE: See the 'close' image details")
    print("- View from FAR: See the 'far' image overall shape")
    print("="*50)

if __name__ == "__main__":
    main()
