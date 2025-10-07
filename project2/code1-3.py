import numpy as np
from scipy.signal import convolve2d
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def load_cameraman_image():
    """Load the cameraman image and convert to grayscale."""
    try:
        img = Image.open('assets/cameraman.png').convert('L')
        img_array = np.array(img, dtype=np.float64)
        print(f"Successfully loaded cameraman image with shape: {img_array.shape}")
        return img_array
    except FileNotFoundError:
        print("Cameraman image not found. Creating a test image instead.")
        # Create a test image if the actual image is not available
        h, w = 256, 256
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        X, Y = np.meshgrid(x, y)
        
        # Create a cameraman-like pattern
        # Head (circle)
        head_center_x, head_center_y = 0.3, 0.4
        head_dist = np.sqrt((X - head_center_x)**2 + (Y - head_center_y)**2)
        head = (head_dist < 0.15) * 200
        
        # Camera body (rectangle)
        camera_mask = (X > 0.5) & (X < 0.8) & (Y > 0.3) & (Y < 0.7)
        camera = camera_mask * 150
        
        # Background
        background = np.sin(X * 10) * 20 + np.cos(Y * 8) * 15 + 100
        
        # Combine
        image = head + camera + background
        image = np.clip(image, 0, 255)
        
        return image.astype(np.float64)

def create_gaussian_kernel(size, sigma):
    """Create a 2D Gaussian kernel using cv2.getGaussianKernel."""
    # Create 1D Gaussian kernel
    gaussian_1d = cv2.getGaussianKernel(size, sigma)
    
    # Create 2D Gaussian kernel using outer product
    gaussian_2d = np.outer(gaussian_1d, gaussian_1d)
    
    return gaussian_2d

def finite_difference_operators():
    """Create finite difference operators Dx and Dy."""
    Dx = np.array([[1, 0, -1]])  # Horizontal gradient
    Dy = np.array([[1], [0], [-1]])  # Vertical gradient
    return Dx, Dy

def create_dog_filters(gaussian_kernel):
    """Create Derivative of Gaussian (DoG) filters."""
    Dx, Dy = finite_difference_operators()
    
    # Convolve Gaussian with finite difference operators
    dog_x = convolve2d(gaussian_kernel, np.flip(Dx), mode='same')
    dog_y = convolve2d(gaussian_kernel, np.flip(Dy), mode='same')
    
    return dog_x, dog_y

def compute_gradient_magnitude(grad_x, grad_y):
    """Compute gradient magnitude from x and y gradients."""
    return np.sqrt(grad_x**2 + grad_y**2)

def binarize_gradient_magnitude(gradient_magnitude, threshold):
    """Binarize gradient magnitude image using threshold."""
    return (gradient_magnitude > threshold).astype(np.uint8) * 255

def find_optimal_threshold(gradient_magnitude):
    """Find optimal threshold for binarization."""
    # Calculate a reasonable threshold based on the gradient magnitude range
    min_val = np.min(gradient_magnitude)
    max_val = np.max(gradient_magnitude)
    
    # Use a threshold that's a small percentage of the range
    optimal_threshold = min_val + (max_val - min_val) * 0.1
    print(f"Gradient magnitude range: {min_val:.2f} to {max_val:.2f}")
    print(f"Using threshold: {optimal_threshold:.2f}")
    
    return optimal_threshold

def main():
    """Main function for task 1.3: Derivative of Gaussian (DoG) Filter."""
    print("Task 1.3: Derivative of Gaussian (DoG) Filter")
    print("=" * 50)
    
    # Load cameraman image
    image = load_cameraman_image()
    
    # Parameters for Gaussian kernel
    kernel_size = 15
    sigma = 2.0
    
    print(f"Creating Gaussian kernel with size {kernel_size}x{kernel_size} and sigma {sigma}")
    
    # Create Gaussian kernel
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma)
    print(f"Gaussian kernel shape: {gaussian_kernel.shape}")
    print(f"Gaussian kernel sum: {np.sum(gaussian_kernel):.4f}")
    
    # Method 1: Blur first, then compute derivatives
    print("\nMethod 1: Blur first, then compute derivatives...")
    blurred_image = convolve2d(image, gaussian_kernel, mode='same')
    Dx, Dy = finite_difference_operators()
    grad_x_blurred = convolve2d(blurred_image, np.flip(Dx), mode='same')
    grad_y_blurred = convolve2d(blurred_image, np.flip(Dy), mode='same')
    gradient_magnitude_blurred = compute_gradient_magnitude(grad_x_blurred, grad_y_blurred)
    
    # Method 2: Use Derivative of Gaussian (DoG) filters
    print("Method 2: Using Derivative of Gaussian (DoG) filters...")
    dog_x, dog_y = create_dog_filters(gaussian_kernel)
    print(f"DoG X filter shape: {dog_x.shape}")
    print(f"DoG Y filter shape: {dog_y.shape}")
    
    # Apply DoG filters directly to original image
    grad_x_dog = convolve2d(image, dog_x, mode='same')
    grad_y_dog = convolve2d(image, dog_y, mode='same')
    gradient_magnitude_dog = compute_gradient_magnitude(grad_x_dog, grad_y_dog)
    
    # Use the DoG results for final output
    print("Using DoG results for final output...")
    gradient_magnitude = gradient_magnitude_dog
    grad_x_blurred = grad_x_dog
    grad_y_blurred = grad_y_dog
    
    # Binarize with appropriate threshold
    print("Binarizing with appropriate threshold...")
    threshold = find_optimal_threshold(gradient_magnitude)
    edge_image = binarize_gradient_magnitude(gradient_magnitude, threshold)
    
    # Generate visualization directly without saving intermediate files
    print("\nGenerating visualization...")
    visualize_dog_results_direct(image, blurred_image, grad_x_blurred, grad_y_blurred, 
                                gradient_magnitude, edge_image)
    
    # Display statistics
    print("\nImage Statistics:")
    print(f"Original - min: {np.min(image):.2f}, max: {np.max(image):.2f}, mean: {np.mean(image):.2f}")
    print(f"Blurred - min: {np.min(blurred_image):.2f}, max: {np.max(blurred_image):.2f}, mean: {np.mean(blurred_image):.2f}")
    print(f"Gradient X (DoG) - min: {np.min(grad_x_blurred):.2f}, max: {np.max(grad_x_blurred):.2f}, mean: {np.mean(grad_x_blurred):.2f}")
    print(f"Gradient Y (DoG) - min: {np.min(grad_y_blurred):.2f}, max: {np.max(grad_y_blurred):.2f}, mean: {np.mean(grad_y_blurred):.2f}")
    print(f"Gradient Magnitude (DoG) - min: {np.min(gradient_magnitude):.2f}, max: {np.max(gradient_magnitude):.2f}, mean: {np.mean(gradient_magnitude):.2f}")
    print(f"Edge Image - min: {np.min(edge_image):.2f}, max: {np.max(edge_image):.2f}, mean: {np.mean(edge_image):.2f}")
    print(f"Threshold used: {threshold:.2f}")
    
    # Compare the two methods
    print(f"\nComparison between methods:")
    print(f"Difference in gradient magnitude: {np.max(np.abs(gradient_magnitude_blurred - gradient_magnitude_dog)):.2e}")
    
    print("\nProcessing completed successfully!")

def visualize_dog_results_direct(original, blurred, grad_x_blurred, grad_y_blurred, 
                                gradient_magnitude, edge_image):
    """Visualize the DoG filter results directly from computed arrays."""
    # Create visualization with 5 images
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Derivative of Gaussian (DoG) Filter Results', fontsize=16)
    
    # 1. Cameraman smoothed with Gaussian
    axes[0, 0].imshow(blurred, cmap='gray')
    axes[0, 0].set_title('1. Cameraman Smoothed with Gaussian')
    axes[0, 0].axis('off')
    
    # 2. Cameraman smoothed dx (using DoG)
    axes[0, 1].imshow(grad_x_blurred, cmap='gray')
    axes[0, 1].set_title('2. Cameraman Smoothed Dx (DoG)')
    axes[0, 1].axis('off')
    
    # 3. Cameraman smoothed dy (using DoG)
    axes[0, 2].imshow(grad_y_blurred, cmap='gray')
    axes[0, 2].set_title('3. Cameraman Smoothed Dy (DoG)')
    axes[0, 2].axis('off')
    
    # 4. Gradient magnitude (using DoG)
    axes[1, 0].imshow(gradient_magnitude, cmap='gray')
    axes[1, 0].set_title('4. Gradient Magnitude (DoG)')
    axes[1, 0].axis('off')
    
    # 5. Edge with appropriate threshold
    axes[1, 1].imshow(edge_image, cmap='gray')
    axes[1, 1].set_title('5. Edge Image (Appropriate Threshold)')
    axes[1, 1].axis('off')
    
    # Hide the last subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('assets/part1-3.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'assets/part1-3.png'")
    plt.show()

if __name__ == "__main__":
    main()
