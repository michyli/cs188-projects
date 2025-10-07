import numpy as np
from scipy.signal import convolve2d
from PIL import Image
import matplotlib.pyplot as plt

def convolution_four_loops(image, kernel, padding=0):
    # Add zero padding if specified
    if padding > 0:
        padded_image = np.pad(image, padding, mode='constant', constant_values=0)
    else:
        padded_image = image
    
    # Get dimensions
    img_h, img_w = padded_image.shape
    kernel_h, kernel_w = kernel.shape
    
    # Calculate output dimensions
    out_h = img_h - kernel_h + 1
    out_w = img_w - kernel_w + 1
    
    # Initialize output array
    output = np.zeros((out_h, out_w))
    
    # Four nested loops for convolution
    for i in range(out_h):
        for j in range(out_w):
            for k in range(kernel_h):
                for l in range(kernel_w):
                    output[i, j] += padded_image[i + k, j + l] * kernel[k, l]
    
    return output

def convolution_two_loops(image, kernel, padding=0):
    # Add zero padding if specified
    if padding > 0:
        padded_image = np.pad(image, padding, mode='constant', constant_values=0)
    else:
        padded_image = image
    
    # Get dimensions
    img_h, img_w = padded_image.shape
    kernel_h, kernel_w = kernel.shape
    
    # Calculate output dimensions
    out_h = img_h - kernel_h + 1
    out_w = img_w - kernel_w + 1
    
    # Initialize output array
    output = np.zeros((out_h, out_w))
    
    # Two nested loops for convolution
    for i in range(out_h):
        for j in range(out_w):
            # Extract the patch and perform element-wise multiplication
            patch = padded_image[i:i+kernel_h, j:j+kernel_w]
            output[i, j] = np.sum(patch * kernel)
    
    return output

def convolution_with_padding(image, kernel, padding=0):
  # Add zero padding
    if padding > 0:
        padded_image = np.pad(image, padding, mode='constant', constant_values=0)
    else:
        padded_image = image
    
    # Perform convolution
    result = convolution_two_loops(padded_image, kernel)
    
    return result

def create_box_filter(size):
    # Create a box filter (averaging filter) of given size.
    return np.ones((size, size)) / (size * size)

def finite_difference_operators():
    # Create finite difference operators Dx and Dy.
    Dx = np.array([[1, 0, -1]])  # Horizontal gradient
    Dy = np.array([[1], [0], [-1]])  # Vertical gradient
    
    return Dx, Dy

def load_and_process_image(image_path):
    # Load an image and convert to grayscale using PIL.
    try:
        img = Image.open(image_path).convert('L')
        img_array = np.array(img, dtype=np.float64)
        print(f"Successfully loaded image: {image_path}")
        return img_array
    except FileNotFoundError:
        print(f"Image {image_path} not found. Creating a test image instead.")
        # Create a test image if the actual image is not available
        h, w = 400, 400
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        X, Y = np.meshgrid(x, y)
        
        # Create a portrait-like pattern
        face_center_x, face_center_y = 0.5, 0.4
        face_dist = np.sqrt((X - face_center_x)**2 + (Y - face_center_y)**2)
        face = np.exp(-face_dist * 3) * 200
        
        background = (X + Y) * 50 + 30
        image = face + background
        image += np.sin(X * 20) * 10
        image += np.cos(Y * 15) * 8
        
        noise = np.random.normal(0, 15, (h, w))
        image += noise
        image = np.clip(image, 0, 255)
        
        return image.astype(np.float64)

def main():
    # Main function to demonstrate convolution implementations.
    print("Convolution Implementation Demo")
    print("=" * 40)
    
    # Load the portrait image
    image = load_and_process_image('assets/square_portrait.jpg')
    print(f"Loaded image with shape: {image.shape}")
    
    # Define finite difference operators
    Dx, Dy = finite_difference_operators()
    print(f"Dx shape: {Dx.shape}, Dy shape: {Dy.shape}")
    
    # Calculate padding for 9x9 box filter (half of kernel size)
    box_filter = create_box_filter(9)
    padding = 4  # Half of kernel size for 9x9 box filter
    
    print(f"\nGenerating 7 images as requested:")
    print("1. Original grayscale image")
    print("2. 9x9 box filter with padding (2 loops)")
    print("3. Dx gradient (2 loops)")
    print("4. Dy gradient (2 loops)")
    print("5. 9x9 box filter (scipy)")
    print("6. Dx gradient (scipy)")
    print("7. Dy gradient (scipy)")
    
    # 1. Original grayscale image
    original = image.copy()
    
    # 2. 9x9 box filter with padding using 2 loops
    box_2loops_padded = convolution_two_loops(image, box_filter, padding)
    
    # 3. Dx using 2 loops with padding
    grad_x_2loops = convolution_two_loops(image, Dx, padding)
    
    # 4. Dy using 2 loops with padding
    grad_y_2loops = convolution_two_loops(image, Dy, padding)
    
    # 5. Box filter using scipy
    box_scipy = convolve2d(image, box_filter, mode='same')
    
    # 6. Dx using scipy
    grad_x_scipy = convolve2d(image, np.flip(Dx), mode='same')
    
    # 7. Dy using scipy
    grad_y_scipy = convolve2d(image, np.flip(Dy), mode='same')
    
    # Save all results
    print("\nSaving results...")
    np.save('original.npy', original)
    np.save('box_2loops_padded.npy', box_2loops_padded)
    np.save('grad_x_2loops.npy', grad_x_2loops)
    np.save('grad_y_2loops.npy', grad_y_2loops)
    np.save('box_scipy.npy', box_scipy)
    np.save('grad_x_scipy.npy', grad_x_scipy)
    np.save('grad_y_scipy.npy', grad_y_scipy)
    
    print("All 7 images saved successfully!")
    
    # Display statistics
    print("\nImage Statistics:")
    print(f"Original - min: {np.min(original):.2f}, max: {np.max(original):.2f}, mean: {np.mean(original):.2f}")
    print(f"Box 2loops padded - min: {np.min(box_2loops_padded):.2f}, max: {np.max(box_2loops_padded):.2f}, mean: {np.mean(box_2loops_padded):.2f}")
    print(f"Grad X 2loops - min: {np.min(grad_x_2loops):.2f}, max: {np.max(grad_x_2loops):.2f}, mean: {np.mean(grad_x_2loops):.2f}")
    print(f"Grad Y 2loops - min: {np.min(grad_y_2loops):.2f}, max: {np.max(grad_y_2loops):.2f}, mean: {np.mean(grad_y_2loops):.2f}")
    print(f"Box scipy - min: {np.min(box_scipy):.2f}, max: {np.max(box_scipy):.2f}, mean: {np.mean(box_scipy):.2f}")
    print(f"Grad X scipy - min: {np.min(grad_x_scipy):.2f}, max: {np.max(grad_x_scipy):.2f}, mean: {np.mean(grad_x_scipy):.2f}")
    print(f"Grad Y scipy - min: {np.min(grad_y_scipy):.2f}, max: {np.max(grad_y_scipy):.2f}, mean: {np.mean(grad_y_scipy):.2f}")
    
    # Compare results (handle different sizes)
    print("\nComparison (2loops vs scipy):")
    print(f"Box filter difference: {np.max(np.abs(box_2loops_padded - box_scipy)):.2e}")
    
    # For gradients, compare only the overlapping region
    if grad_x_2loops.shape == grad_x_scipy.shape:
        print(f"Grad X difference: {np.max(np.abs(grad_x_2loops - grad_x_scipy)):.2e}")
    else:
        print(f"Grad X shapes differ: {grad_x_2loops.shape} vs {grad_x_scipy.shape}")
    
    if grad_y_2loops.shape == grad_y_scipy.shape:
        print(f"Grad Y difference: {np.max(np.abs(grad_y_2loops - grad_y_scipy)):.2e}")
    else:
        print(f"Grad Y shapes differ: {grad_y_2loops.shape} vs {grad_y_scipy.shape}")

def visualize_convolution_results():
    """
    Visualize all 7 convolution results
    """
    # Load all saved results
    try:
        original = np.load('original.npy')
        box_2loops_padded = np.load('box_2loops_padded.npy')
        grad_x_2loops = np.load('grad_x_2loops.npy')
        grad_y_2loops = np.load('grad_y_2loops.npy')
        box_scipy = np.load('box_scipy.npy')
        grad_x_scipy = np.load('grad_x_scipy.npy')
        grad_y_scipy = np.load('grad_y_scipy.npy')
        print("Successfully loaded all 7 images for visualization.")
    except FileNotFoundError:
        print("Results not found. Please run main() first.")
        return
    
    # Create the visualization with 7 images
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Convolution Results', fontsize=16)
    
    # 1. Original grayscale image
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('1. Original Grayscale')
    axes[0, 0].axis('off')
    
    # 2. 9x9 box filter with padding (2 loops)
    axes[0, 1].imshow(box_2loops_padded, cmap='gray')
    axes[0, 1].set_title('2. Box Filter (2 loops + padding)')
    axes[0, 1].axis('off')
    
    # 3. Dx using 2 loops
    axes[0, 2].imshow(grad_x_2loops, cmap='gray')
    axes[0, 2].set_title('3. Dx Gradient (2 loops)')
    axes[0, 2].axis('off')
    
    # 4. Dy using 2 loops
    axes[0, 3].imshow(grad_y_2loops, cmap='gray')
    axes[0, 3].set_title('4. Dy Gradient (2 loops)')
    axes[0, 3].axis('off')
    
    # 5. Box filter using scipy
    axes[1, 0].imshow(box_scipy, cmap='gray')
    axes[1, 0].set_title('5. Box Filter (scipy)')
    axes[1, 0].axis('off')
    
    # 6. Dx using scipy
    axes[1, 1].imshow(grad_x_scipy, cmap='gray')
    axes[1, 1].set_title('6. Dx Gradient (scipy)')
    axes[1, 1].axis('off')
    
    # 7. Dy using scipy
    axes[1, 2].imshow(grad_y_scipy, cmap='gray')
    axes[1, 2].set_title('7. Dy Gradient (scipy)')
    axes[1, 2].axis('off')
    
    # Hide the last subplot
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('assets/part1-1.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'assets/part1-1.png'")
    plt.show()

if __name__ == "__main__":
    main()
    print("\n" + "="*50)
    print("Generating visualization...")
    visualize_convolution_results()
