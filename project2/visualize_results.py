import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def visualize_convolution_results():
    """
    Visualize all 7 convolution results from code.py
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
        print("Results not found. Please run code.py first.")
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
    plt.savefig('convolution_results_7images.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print("\nConvolution Results Statistics:")
    print("=" * 50)
    print(f"Original shape: {original.shape}")
    print(f"Box 2loops padded shape: {box_2loops_padded.shape}")
    print(f"Grad X 2loops shape: {grad_x_2loops.shape}")
    print(f"Grad Y 2loops shape: {grad_y_2loops.shape}")
    print(f"Box scipy shape: {box_scipy.shape}")
    print(f"Grad X scipy shape: {grad_x_scipy.shape}")
    print(f"Grad Y scipy shape: {grad_y_scipy.shape}")
    
    print(f"\nImage Statistics:")
    print(f"Original - min: {np.min(original):.2f}, max: {np.max(original):.2f}, mean: {np.mean(original):.2f}")
    print(f"Box 2loops - min: {np.min(box_2loops_padded):.2f}, max: {np.max(box_2loops_padded):.2f}, mean: {np.mean(box_2loops_padded):.2f}")
    print(f"Grad X 2loops - min: {np.min(grad_x_2loops):.2f}, max: {np.max(grad_x_2loops):.2f}, mean: {np.mean(grad_x_2loops):.2f}")
    print(f"Grad Y 2loops - min: {np.min(grad_y_2loops):.2f}, max: {np.max(grad_y_2loops):.2f}, mean: {np.mean(grad_y_2loops):.2f}")
    print(f"Box scipy - min: {np.min(box_scipy):.2f}, max: {np.max(box_scipy):.2f}, mean: {np.mean(box_scipy):.2f}")
    print(f"Grad X scipy - min: {np.min(grad_x_scipy):.2f}, max: {np.max(grad_x_scipy):.2f}, mean: {np.mean(grad_x_scipy):.2f}")
    print(f"Grad Y scipy - min: {np.min(grad_y_scipy):.2f}, max: {np.max(grad_y_scipy):.2f}, mean: {np.mean(grad_y_scipy):.2f}")
    
    # Compare 2loops vs scipy
    print(f"\nComparison (2loops vs scipy):")
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

if __name__ == "__main__":
    visualize_convolution_results()