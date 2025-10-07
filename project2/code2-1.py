import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib.pyplot as plt

def create_gaussian_kernel(size, sigma):
    """Create a 2D Gaussian kernel."""
    kernel = np.zeros((size, size))
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # Normalize the kernel
    kernel = kernel / np.sum(kernel)
    return kernel

def create_identity_filter(size):
    """Create unity impulse (identity) filter - all zeros except 1 at center."""
    identity = np.zeros((size, size))
    center = size // 2
    identity[center, center] = 1.0
    return identity

def create_unsharp_mask_filter(size, sigma, alpha):
    """
    Create unsharp mask filter based on the formula:
    sharpening_filter = (1 + alpha) * identity - alpha * gaussian
    """
    gaussian = create_gaussian_kernel(size, sigma)
    identity = create_identity_filter(size)
    
    # Unsharp mask filter
    unsharp_filter = (1 + alpha) * identity - alpha * gaussian
    
    return unsharp_filter, gaussian, identity

def load_color_image(image_path):
    """Load a color image and convert to numpy array."""
    try:
        img = Image.open(image_path)
        # Keep as RGB for colored images
        img_array = np.array(img, dtype=np.float64)
        print(f"Successfully loaded color image: {image_path} with shape: {img_array.shape}")
        return img_array
    except FileNotFoundError:
        print(f"Image {image_path} not found.")
        return None

def load_grayscale_image(image_path):
    """Load an image and convert to grayscale."""
    try:
        img = Image.open(image_path).convert('L')
        img_array = np.array(img, dtype=np.float64)
        print(f"Successfully loaded grayscale image: {image_path} with shape: {img_array.shape}")
        return img_array
    except FileNotFoundError:
        print(f"Image {image_path} not found.")
        return None

def apply_unsharp_mask_color(image, unsharp_filter):
    """Apply unsharp mask filter to a color image."""
    if len(image.shape) == 3:  # Color image
        # Apply filter to each channel separately
        sharpened = np.zeros_like(image)
        for channel in range(image.shape[2]):
            sharpened[:, :, channel] = convolve2d(image[:, :, channel], unsharp_filter, mode='same', boundary='symm')
        return sharpened
    else:  # Grayscale image
        return convolve2d(image, unsharp_filter, mode='same', boundary='symm')

def blur_image(image, sigma):
    """Blur an image using Gaussian filter."""
    if len(image.shape) == 3:  # Color image
        blurred = np.zeros_like(image)
        for channel in range(image.shape[2]):
            blurred[:, :, channel] = gaussian_filter(image[:, :, channel], sigma=sigma)
        return blurred
    else:  # Grayscale image
        return gaussian_filter(image, sigma=sigma)

def main():
    print("Unsharp Masking Implementation")
    print("=" * 40)
    
    # Parameters for unsharp masking
    kernel_size = 15  # Size of the Gaussian kernel
    sigma = 2.0       # Standard deviation for Gaussian
    alpha = 1.5       # Sharpening strength
    
    print(f"Parameters: kernel_size={kernel_size}, sigma={sigma}, alpha={alpha}")
    
    # Create unsharp mask filter
    unsharp_filter, gaussian_filter_kernel, identity_filter = create_unsharp_mask_filter(kernel_size, sigma, alpha)
    
    print(f"Created filters:")
    print(f"- Gaussian filter shape: {gaussian_filter_kernel.shape}")
    print(f"- Identity filter shape: {identity_filter.shape}")
    print(f"- Unsharp mask filter shape: {unsharp_filter.shape}")
    
    # Load and process the palace image (colored)
    print("\n1. Processing palace.png (colored image)")
    palace_image = load_color_image('assets/palace.png')
    
    if palace_image is not None:
        # Apply unsharp masking to the colored palace image
        palace_sharpened = apply_unsharp_mask_color(palace_image, unsharp_filter)
        
        # Clip values to valid range
        palace_sharpened = np.clip(palace_sharpened, 0, 255)
        
        print(f"Palace image sharpened. Shape: {palace_sharpened.shape}")
        
        # Save the sharpened palace image
        palace_sharpened_pil = Image.fromarray(palace_sharpened.astype(np.uint8))
        palace_sharpened_pil.save('assets/palace_sharpened.png')
        print("Saved sharpened palace image as 'assets/palace_sharpened.png'")
    
    # Experiment: Take a sharp image, blur it, then try to sharpen it
    print("\n2. Blur and Sharpen Experiment")
    
    # Load a sharp image (using scenary as it should be sharp)
    sharp_image = load_color_image('assets/scenary.png')
    
    if sharp_image is None:
        # Create a synthetic sharp image if scenary is not available
        print("Creating synthetic sharp test image...")
        h, w = 256, 256
        x = np.linspace(-2, 2, w)
        y = np.linspace(-2, 2, h)
        X, Y = np.meshgrid(x, y)
        
        # Create a sharp pattern with high frequency components
        sharp_image = np.zeros((h, w))
        # Add checkerboard pattern
        sharp_image += ((X > 0) & (Y > 0)) * 100
        sharp_image += ((X < 0) & (Y < 0)) * 100
        sharp_image += ((X > 0) & (Y < 0)) * 200
        sharp_image += ((X < 0) & (Y > 0)) * 200
        
        # Add some fine details
        sharp_image += np.sin(X * 10) * np.cos(Y * 10) * 50
        sharp_image = np.clip(sharp_image, 0, 255)
    
    # Blur the sharp image
    blur_sigma = 3.0
    blurred_image = blur_image(sharp_image, blur_sigma)
    print(f"Blurred image with sigma={blur_sigma}")
    
    # Try to sharpen the blurred image
    sharpened_image = apply_unsharp_mask_color(blurred_image, unsharp_filter)
    sharpened_image = np.clip(sharpened_image, 0, 255)
    print("Applied unsharp masking to blurred image")
    
    # Create visualization
    create_visualization(palace_image, palace_sharpened if palace_image is not None else None,
                        sharp_image, blurred_image, sharpened_image,
                        gaussian_filter_kernel, identity_filter, unsharp_filter)
    
    print("\nObservations from blur-sharpen experiment:")
    print("1. The original sharp image has crisp edges and fine details")
    print("2. After blurring, the image loses high-frequency components (edges become soft)")
    print("3. Unsharp masking partially recovers sharpness but cannot fully restore original detail")
    print("4. The sharpened image may show some artifacts or oversharpening effects")
    print("5. Information lost during blurring cannot be perfectly recovered")

def create_visualization(palace_original, palace_sharpened, sharp_original, blurred, sharpened, 
                        gaussian_kernel, identity_kernel, unsharp_kernel):
    """Create visualization of unsharp masking results."""
    
    # Create figure with subplots - simplified layout
    fig = plt.figure(figsize=(15, 10))
    
    # Row 1: Palace images (if available)
    if palace_original is not None and palace_sharpened is not None:
        plt.subplot(2, 3, 1)
        if len(palace_original.shape) == 3:
            plt.imshow(palace_original.astype(np.uint8))
        else:
            plt.imshow(palace_original, cmap='gray')
        plt.title('Original Palace Image')
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        if len(palace_sharpened.shape) == 3:
            plt.imshow(palace_sharpened.astype(np.uint8))
        else:
            plt.imshow(palace_sharpened, cmap='gray')
        plt.title('Sharpened Palace Image')
        plt.axis('off')
    
    # Row 2: Blur-Sharpen experiment
    plt.subplot(2, 3, 4)
    if len(sharp_original.shape) == 3:
        plt.imshow(sharp_original.astype(np.uint8))
    else:
        plt.imshow(sharp_original, cmap='gray')
    plt.title('Original Sharp Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    if len(blurred.shape) == 3:
        plt.imshow(blurred.astype(np.uint8))
    else:
        plt.imshow(blurred, cmap='gray')
    plt.title('Blurred Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    if len(sharpened.shape) == 3:
        plt.imshow(sharpened.astype(np.uint8))
    else:
        plt.imshow(sharpened, cmap='gray')
    plt.title('Sharpened (Recovered)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('assets/unsharp_mask_results.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'assets/unsharp_mask_results.png'")
    
    # Create a separate comparison for the blur-sharpen experiment
    fig2, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    if len(sharp_original.shape) == 3:
        axes[0].imshow(sharp_original.astype(np.uint8))
    else:
        axes[0].imshow(sharp_original, cmap='gray')
    axes[0].set_title('Original Sharp')
    axes[0].axis('off')
    
    if len(blurred.shape) == 3:
        axes[1].imshow(blurred.astype(np.uint8))
    else:
        axes[1].imshow(blurred, cmap='gray')
    axes[1].set_title('Blurred')
    axes[1].axis('off')
    
    if len(sharpened.shape) == 3:
        axes[2].imshow(sharpened.astype(np.uint8))
    else:
        axes[2].imshow(sharpened, cmap='gray')
    axes[2].set_title('Sharpened (Recovered)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('assets/blur_sharpen_experiment.png', dpi=150, bbox_inches='tight')
    print("Blur-sharpen experiment saved as 'assets/blur_sharpen_experiment.png'")
    
    plt.show()

if __name__ == "__main__":
    main()
