import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import skimage.draw as skdr

def create_gaussian_kernel(size, sigma):
    """Create a 2D Gaussian kernel."""
    kernel = np.zeros((size, size))
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    return kernel / np.sum(kernel)

def apply_gaussian_filter(image, kernel):
    """Apply Gaussian filter to an image using convolution."""
    if len(image.shape) == 3:
        # Color image - apply to each channel
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:, :, c] = cv2.filter2D(image[:, :, c], -1, kernel)
        return result
    else:
        # Grayscale image
        return cv2.filter2D(image, -1, kernel)

def create_gaussian_stack(image, num_levels=20, sigma=3.0):
    """
    Create a Gaussian stack by applying Gaussian filter at each level
    without downsampling. Each level is more blurred than the previous.
    
    Args:
        image: Input image (grayscale or color)
        num_levels: Number of levels in the stack
        sigma: Standard deviation for Gaussian kernel
    
    Returns:
        List of images forming the Gaussian stack
    """
    stack = [image.astype(np.float32)]
    
    # Create Gaussian kernel with fixed width of 19
    kernel_size = 18
    kernel = create_gaussian_kernel(kernel_size, sigma)
    
    current_image = image.astype(np.float32)
    
    for level in range(1, num_levels):
        # Apply Gaussian filter to get next level
        current_image = apply_gaussian_filter(current_image, kernel)
        stack.append(current_image.copy())
    
    return stack

def create_laplacian_stack(gaussian_stack):
    """
    Create a Laplacian stack from a Gaussian stack.
    Laplacian[i] = Gaussian[i] - Gaussian[i+1]
    
    Args:
        gaussian_stack: List of images from Gaussian stack
    
    Returns:
        List of images forming the Laplacian stack (one level less than Gaussian)
    """
    laplacian_stack = []
    
    for i in range(len(gaussian_stack) - 1):
        laplacian = gaussian_stack[i] - gaussian_stack[i + 1]
        laplacian_stack.append(laplacian)
    
    return laplacian_stack

def create_left_right_mask(shape, window_size=50):
    """
    Create a left-right blending mask.
    Left side gets value 1, right side gets value 0,
    with smooth transition in the middle.
    
    Args:
        shape: (height, width) of the mask
        window_size: Size of transition window
    
    Returns:
        2D mask array
    """
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=np.float32)
    
    center = width // 2
    window_start = center - window_size // 2
    window_end = center + window_size // 2
    
    # Left side = 1
    mask[:, :window_start] = 1.0
    
    # Transition zone
    if window_end > window_start:
        transition = np.linspace(1, 0, window_end - window_start)
        mask[:, window_start:window_end] = transition
    
    # Right side = 0 (already initialized)
    
    return mask

def create_top_bottom_mask(shape, window_size=50):
    """
    Create a top-bottom blending mask.
    Top gets value 1, bottom gets value 0,
    with smooth transition in the middle.
    
    Args:
        shape: (height, width) of the mask
        window_size: Size of transition window
    
    Returns:
        2D mask array
    """
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=np.float32)
    
    center = height // 2
    window_start = center - window_size // 2
    window_end = center + window_size // 2
    
    # Top = 1
    mask[:window_start, :] = 1.0
    
    # Transition zone
    if window_end > window_start:
        transition = np.linspace(1, 0, window_end - window_start)
        mask[window_start:window_end, :] = transition[:, None]
    
    # Bottom = 0 (already initialized)
    
    return mask

def create_irregular_mask(shape, num_points=10):
    """
    Create an irregular polygon mask.
    
    Args:
        shape: (height, width) of the mask
        num_points: Number of points for the polygon
    
    Returns:
        2D mask array
    """
    height, width = shape[:2]
    
    # Create a simple circular polygon as example
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 4
    
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    polygon_points = []
    
    for angle in angles:
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        polygon_points.append([y, x])  # Note: skimage expects (row, col)
    
    polygon_points = np.array(polygon_points)
    mask = skdr.polygon2mask((height, width), polygon_points).astype(np.float32)
    return mask

def create_complex_irregular_mask(shape, center_offset_y=0.15):
    """
    Create a larger, smoother irregular polygon mask.
    
    Args:
        shape: (height, width) of the mask
        center_offset_y: Offset from center as fraction of height (positive = lower)
    
    Returns:
        2D mask array
    """
    height, width = shape[:2]
    
    # Position center lower than middle
    center_x = width // 2
    center_y = int(height // 2 + center_offset_y * height)
    
    # Create a larger base radius for bigger shape
    base_radius = min(width, height) // 3
    
    # Use fewer points for smoother, less spiky shape
    num_points = 8
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    
    polygon_points = []
    
    for i, angle in enumerate(angles):
        # Add gentle radius variations for smooth polygon
        radius_variation = 1.0
        
        # Create gentle variations
        radius_variation += 0.3 * np.sin(angle * 2) + 0.2 * np.cos(angle * 3)
        
        # Ensure variation stays reasonable (between 0.7 and 1.4)
        radius_variation = np.clip(radius_variation, 0.7, 1.4)
        
        current_radius = base_radius * radius_variation
        
        x = center_x + current_radius * np.cos(angle)
        y = center_y + current_radius * np.sin(angle)
        
        # Ensure points stay within image bounds
        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)
        
        polygon_points.append([y, x])  # Note: skimage expects (row, col)
    
    polygon_points = np.array(polygon_points)
    mask = skdr.polygon2mask((height, width), polygon_points).astype(np.float32)
    return mask

def create_irregular_vertical_mask(shape, wave_amplitude=50, wave_frequency=3):
    """
    Create an irregular vertical line mask (like left-right but with wavy boundary).
    
    Args:
        shape: (height, width) of the mask
        wave_amplitude: Maximum deviation from center line
        wave_frequency: Number of waves along the height
    
    Returns:
        2D mask array
    """
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=np.float32)
    
    center_x = width // 2
    
    # Create wavy boundary line
    y_coords = np.arange(height)
    
    # Create sinusoidal wave with some variation
    wave1 = wave_amplitude * np.sin(2 * np.pi * wave_frequency * y_coords / height)
    wave2 = (wave_amplitude * 0.3) * np.sin(2 * np.pi * wave_frequency * 2.5 * y_coords / height)
    
    # Combine waves for more irregular pattern
    boundary_x = center_x + wave1 + wave2
    
    # Ensure boundary stays within image bounds
    boundary_x = np.clip(boundary_x, 0, width - 1).astype(int)
    
    # Create smooth transition around the boundary
    transition_width = 30
    
    for y in range(height):
        boundary_pos = boundary_x[y]
        
        # Left side = 1, right side = 0
        mask[y, :boundary_pos - transition_width//2] = 1.0
        
        # Smooth transition zone
        start_transition = max(0, boundary_pos - transition_width//2)
        end_transition = min(width, boundary_pos + transition_width//2)
        
        if end_transition > start_transition:
            transition_length = end_transition - start_transition
            transition_values = np.linspace(1, 0, transition_length)
            mask[y, start_transition:end_transition] = transition_values
    
    return mask

def blend_images_multiband(image1, image2, mask, num_levels=20, sigma=3.0):
    """
    Perform multi-band blending using Gaussian and Laplacian stacks.
    
    Args:
        image1: First image
        image2: Second image  
        mask: Blending mask (1 for image1, 0 for image2)
        num_levels: Number of levels in the stacks
        sigma: Gaussian kernel standard deviation
    
    Returns:
        Blended image, and intermediate results for visualization
    """
    # Ensure images are float32
    img1 = image1.astype(np.float32) / 255.0
    img2 = image2.astype(np.float32) / 255.0
    
    # Create Gaussian stacks for both images
    gaussian_stack1 = create_gaussian_stack(img1, num_levels, sigma)
    gaussian_stack2 = create_gaussian_stack(img2, num_levels, sigma)
    
    # Create Laplacian stacks
    laplacian_stack1 = create_laplacian_stack(gaussian_stack1)
    laplacian_stack2 = create_laplacian_stack(gaussian_stack2)
    
    # Create Gaussian stack for the mask
    mask_stack = create_gaussian_stack(mask, num_levels, sigma)
    
    # Blend at each level of the Laplacian stacks
    blended_laplacian = []
    
    for i in range(len(laplacian_stack1)):
        # Get the mask for this level
        level_mask = mask_stack[i]
        
        # Expand mask to match image dimensions if needed
        if len(img1.shape) == 3:
            level_mask = np.expand_dims(level_mask, axis=2)
        
        # Blend the Laplacian levels
        blended_level = (level_mask * laplacian_stack1[i] + 
                        (1 - level_mask) * laplacian_stack2[i])
        blended_laplacian.append(blended_level)
    
    # Blend the final Gaussian level (lowest frequencies)
    final_mask = mask_stack[-1]
    if len(img1.shape) == 3:
        final_mask = np.expand_dims(final_mask, axis=2)
    
    blended_gaussian_final = (final_mask * gaussian_stack1[-1] + 
                             (1 - final_mask) * gaussian_stack2[-1])
    
    # Reconstruct the final image
    result = blended_gaussian_final.copy()
    for level in reversed(blended_laplacian):
        result += level
    
    # Clip values to valid range
    result = np.clip(result, 0, 1)
    
    return result, {
        'gaussian_stack1': gaussian_stack1,
        'gaussian_stack2': gaussian_stack2,
        'laplacian_stack1': laplacian_stack1,
        'laplacian_stack2': laplacian_stack2,
        'blended_laplacian': blended_laplacian,
        'mask_stack': mask_stack,
        'blended_gaussian_final': blended_gaussian_final
    }

def normalize_for_display(image):
    """Normalize image for display purposes."""
    img_norm = image.copy()
    img_norm = img_norm - np.min(img_norm)
    img_max = np.max(img_norm)
    if img_max > 0:
        img_norm = img_norm / img_max
    return img_norm

def visualize_blending_process(image1, image2, mask, blend_result, intermediate_results, 
                              title="Multi-band Blending Process"):
    """
    Visualize the multi-band blending process showing different frequency bands.
    """
    laplacian_stack1 = intermediate_results['laplacian_stack1']
    laplacian_stack2 = intermediate_results['laplacian_stack2']
    blended_laplacian = intermediate_results['blended_laplacian']
    mask_stack = intermediate_results['mask_stack']
    
    # Show 3 levels + final result
    num_show_levels = min(3, len(laplacian_stack1))
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    fig.suptitle(title, fontsize=16)
    
    # Column titles
    axes[0, 0].set_title('Image 1 (Apple)', fontsize=14)
    axes[0, 1].set_title('Image 2 (Orange)', fontsize=14)
    axes[0, 2].set_title('Blended Result', fontsize=14)
    
    # Show Laplacian levels
    for level in range(num_show_levels):
        row = level
        
        # Normalize Laplacian images for display
        lap1_norm = normalize_for_display(laplacian_stack1[level])
        lap2_norm = normalize_for_display(laplacian_stack2[level])
        blend_norm = normalize_for_display(blended_laplacian[level])
        
        # Apply mask for visualization
        mask_vis = mask_stack[level]
        if len(lap1_norm.shape) == 3:
            mask_vis = np.expand_dims(mask_vis, axis=2)
        
        masked_lap1 = lap1_norm * mask_vis
        masked_lap2 = lap2_norm * (1 - mask_vis)
        
        axes[row, 0].imshow(masked_lap1, cmap='gray' if len(lap1_norm.shape) == 2 else None)
        axes[row, 0].set_ylabel(f'Level {level}', fontsize=12)
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(masked_lap2, cmap='gray' if len(lap2_norm.shape) == 2 else None)
        axes[row, 1].axis('off')
        
        axes[row, 2].imshow(blend_norm, cmap='gray' if len(blend_norm.shape) == 2 else None)
        axes[row, 2].axis('off')
    
    # Show original images and final result
    axes[3, 0].imshow(image1.astype(np.uint8))
    axes[3, 0].set_ylabel('Original', fontsize=12)
    axes[3, 0].axis('off')
    
    axes[3, 1].imshow(image2.astype(np.uint8))
    axes[3, 1].axis('off')
    
    axes[3, 2].imshow((blend_result * 255).astype(np.uint8))
    axes[3, 2].axis('off')
    
    plt.tight_layout()
    return fig

def demonstrate_stacks(image, title="Stack Demonstration"):
    """
    Demonstrate the creation of Gaussian and Laplacian stacks.
    """
    print(f"\nCreating {title}...")
    
    # Create Gaussian stack
    gaussian_stack = create_gaussian_stack(image.astype(np.float32) / 255.0)
    print(f"Gaussian stack created with {len(gaussian_stack)} levels")
    
    # Create Laplacian stack
    laplacian_stack = create_laplacian_stack(gaussian_stack)
    print(f"Laplacian stack created with {len(laplacian_stack)} levels")
    
    # Show a few levels
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f'{title} - Gaussian and Laplacian Stacks', fontsize=16)
    
    # Show first 5 Gaussian levels
    for i in range(5):
        axes[0, i].imshow((gaussian_stack[i * 4] * 255).astype(np.uint8))
        axes[0, i].set_title(f'Gaussian Level {i * 4}')
        axes[0, i].axis('off')
    
    # Show first 5 Laplacian levels (normalized for display)
    for i in range(5):
        lap_norm = normalize_for_display(laplacian_stack[i * 4])
        axes[1, i].imshow(lap_norm, cmap='gray' if len(lap_norm.shape) == 2 else None)
        axes[1, i].set_title(f'Laplacian Level {i * 4}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    return fig

def visualize_figure_342_style(image1, image2, mask, blend_result, intermediate_results, 
                               title="Figure 3.42 Style Visualization"):
    """
    Create a 4x3 visualization similar to Figure 3.42 in Szelski.
    Rows 1-3: High, medium, low frequency Laplacian levels (0, 2, 4)
    Row 4: Original masked images and final blend
    """
    laplacian_stack1 = intermediate_results['laplacian_stack1']
    laplacian_stack2 = intermediate_results['laplacian_stack2']
    blended_laplacian = intermediate_results['blended_laplacian']
    mask_stack = intermediate_results['mask_stack']
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    fig.suptitle(title, fontsize=16)
    
    # Column titles
    axes[0, 0].set_title('Apple (Image 1)', fontsize=14)
    axes[0, 1].set_title('Orange (Image 2)', fontsize=14)
    axes[0, 2].set_title('Blended Result', fontsize=14)
    
    # Levels to show: 0 (high), 2 (medium), 4 (low)
    levels_to_show = [0, 2, 4]
    frequency_labels = ['High Frequency', 'Medium Frequency', 'Low Frequency']
    
    for i, (level, freq_label) in enumerate(zip(levels_to_show, frequency_labels)):
        if level < len(laplacian_stack1):
            # Get Laplacian levels
            lap1 = laplacian_stack1[level]
            lap2 = laplacian_stack2[level]
            lap_blend = blended_laplacian[level]
            
            # Get corresponding mask
            level_mask = mask_stack[level]
            if len(lap1.shape) == 3:
                level_mask = np.expand_dims(level_mask, axis=2)
            
            # Apply mask weighting to show contributions
            weighted_lap1 = lap1 * level_mask
            weighted_lap2 = lap2 * (1 - level_mask)
            
            # Normalize for display
            weighted_lap1_norm = normalize_for_display(weighted_lap1)
            weighted_lap2_norm = normalize_for_display(weighted_lap2)
            lap_blend_norm = normalize_for_display(lap_blend)
            
            # Display weighted contributions
            axes[i, 0].imshow(weighted_lap1_norm, cmap='gray' if len(weighted_lap1_norm.shape) == 2 else None)
            axes[i, 0].set_ylabel(freq_label, fontsize=12)
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(weighted_lap2_norm, cmap='gray' if len(weighted_lap2_norm.shape) == 2 else None)
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(lap_blend_norm, cmap='gray' if len(lap_blend_norm.shape) == 2 else None)
            axes[i, 2].axis('off')
    
    # Bottom row: Original images with mask applied and final result
    # Apply mask to original images for visualization
    mask_2d = mask_stack[0]  # Use finest level mask for original images
    if len(image1.shape) == 3:
        mask_3d = np.expand_dims(mask_2d, axis=2)
    else:
        mask_3d = mask_2d
    
    # Convert images to float for masking
    img1_float = image1.astype(np.float32) / 255.0
    img2_float = image2.astype(np.float32) / 255.0
    
    # Apply half masking for visualization
    masked_img1 = img1_float * mask_3d
    masked_img2 = img2_float * (1 - mask_3d)
    
    axes[3, 0].imshow(masked_img1)
    axes[3, 0].set_ylabel('Original Masked', fontsize=12)
    axes[3, 0].axis('off')
    
    axes[3, 1].imshow(masked_img2)
    axes[3, 1].axis('off')
    
    axes[3, 2].imshow((blend_result * 255).astype(np.uint8))
    axes[3, 2].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    """Main function to demonstrate apple-orange blending."""
    print("=== Multi-band Image Blending Implementation ===")
    print("Parameters:")
    print("- Gaussian Kernel width: 19")
    print("- Gaussian Kernel sigma: 3.0")
    print("- Gaussian stack levels: 20")
    print("- Laplacian stack levels: 19")
    print("- Polygon points for irregular mask: 10")
    print()
    
    print("Loading images...")
    
    # Load apple and orange images
    apple = cv2.imread('assets/apple.jpeg')
    orange = cv2.imread('assets/orange.jpeg')
    
    if apple is None or orange is None:
        print("Error: Could not load images. Make sure apple.jpeg and orange.jpeg are in assets/ folder")
        return
    
    # Convert BGR to RGB
    apple = cv2.cvtColor(apple, cv2.COLOR_BGR2RGB)
    orange = cv2.cvtColor(orange, cv2.COLOR_BGR2RGB)
    
    print(f"Apple shape: {apple.shape}")
    print(f"Orange shape: {orange.shape}")
    
    # Ensure same size
    if apple.shape != orange.shape:
        orange = cv2.resize(orange, (apple.shape[1], apple.shape[0]))
    
    # Demonstrate stack creation
    stack_fig = demonstrate_stacks(apple, "Apple Stack Demonstration")
    stack_fig.savefig('assets/gaussian_laplacian_stacks.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n1. Creating left-right blend...")
    
    # Create left-right mask
    lr_mask = create_left_right_mask(apple.shape, window_size=60)
    
    # Perform multi-band blending
    lr_result, lr_intermediate = blend_images_multiband(
        apple, orange, lr_mask, num_levels=20, sigma=3.0
    )
    
    # Visualize the process
    fig1 = visualize_blending_process(
        apple, orange, lr_mask, lr_result, lr_intermediate,
        "Left-Right Multi-band Blending (Apple-Orange)"
    )
    fig1.savefig('assets/left_right_blending_process.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Create Figure 3.42 style visualization
    fig1_342 = visualize_figure_342_style(
        apple, orange, lr_mask, lr_result, lr_intermediate,
        "Apple-Orange Blending: Figure 3.42 Style"
    )
    fig1_342.savefig('assets/apple_orange_figure_342_style.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n2. Creating irregular blend...")
    
    # Create irregular mask (circular for demonstration)
    irregular_mask = create_irregular_mask(apple.shape, num_points=10)
    
    # Perform multi-band blending
    irreg_result, irreg_intermediate = blend_images_multiband(
        apple, orange, irregular_mask, num_levels=20, sigma=3.0
    )
    
    # Visualize the process
    fig2 = visualize_blending_process(
        apple, orange, irregular_mask, irreg_result, irreg_intermediate,
        "Irregular Multi-band Blending (Apple-Orange)"
    )
    fig2.savefig('assets/irregular_blending_process.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save results
    print("\nSaving results...")
    cv2.imwrite('assets/oraple_left_right.jpg', 
                cv2.cvtColor((lr_result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite('assets/oraple_irregular.jpg', 
                cv2.cvtColor((irreg_result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    print("Results saved as 'oraple_left_right.jpg' and 'oraple_irregular.jpg'")
    
    # Blend day and night scenes
    print("\n3. Creating day-night scene blend...")
    
    # Load day and night scene images
    day_scene = cv2.imread('assets/day_scene.png')
    night_scene = cv2.imread('assets/night_scene.png')
    
    if day_scene is not None and night_scene is not None:
        # Convert BGR to RGB
        day_scene = cv2.cvtColor(day_scene, cv2.COLOR_BGR2RGB)
        night_scene = cv2.cvtColor(night_scene, cv2.COLOR_BGR2RGB)
        
        print(f"Day scene shape: {day_scene.shape}")
        print(f"Night scene shape: {night_scene.shape}")
        
        # Ensure same size
        if day_scene.shape != night_scene.shape:
            night_scene = cv2.resize(night_scene, (day_scene.shape[1], day_scene.shape[0]))
        
        # Create complex irregular mask (positioned lower than center)
        complex_mask = create_complex_irregular_mask(day_scene.shape, center_offset_y=0.15)
        
        # Perform multi-band blending
        day_night_result, day_night_intermediate = blend_images_multiband(
            day_scene, night_scene, complex_mask, num_levels=20, sigma=3.0
        )
        
        # Visualize the process
        fig3 = visualize_blending_process(
            day_scene, night_scene, complex_mask, day_night_result, day_night_intermediate,
            "Day-Night Scene Multi-band Blending"
        )
        fig3.savefig('assets/day_night_blending_process.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Save result
        cv2.imwrite('assets/day_night_blend.jpg', 
                    cv2.cvtColor((day_night_result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        
        print("Day-night blend saved as 'day_night_blend.jpg'")
    else:
        print("Day or night scene images not found. Skipping day-night blending.")
    
    # Blend bridge day and night scenes
    print("\n4. Creating bridge day-night blend with irregular vertical line...")
    
    # Load bridge day and night images
    bridge_day = cv2.imread('assets/bridge_day.png')
    bridge_night = cv2.imread('assets/bridge_night.png')
    
    if bridge_day is not None and bridge_night is not None:
        # Convert BGR to RGB
        bridge_day = cv2.cvtColor(bridge_day, cv2.COLOR_BGR2RGB)
        bridge_night = cv2.cvtColor(bridge_night, cv2.COLOR_BGR2RGB)
        
        print(f"Bridge day shape: {bridge_day.shape}")
        print(f"Bridge night shape: {bridge_night.shape}")
        
        # Ensure same size
        if bridge_day.shape != bridge_night.shape:
            bridge_night = cv2.resize(bridge_night, (bridge_day.shape[1], bridge_day.shape[0]))
        
        # Create irregular vertical mask
        irregular_vertical_mask = create_irregular_vertical_mask(bridge_day.shape, wave_amplitude=60, wave_frequency=2.5)
        
        # Perform multi-band blending
        bridge_result, bridge_intermediate = blend_images_multiband(
            bridge_day, bridge_night, irregular_vertical_mask, num_levels=20, sigma=3.0
        )
        
        # Visualize the process
        fig4 = visualize_blending_process(
            bridge_day, bridge_night, irregular_vertical_mask, bridge_result, bridge_intermediate,
            "Bridge Day-Night Irregular Vertical Blending"
        )
        fig4.savefig('assets/bridge_blending_process.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Save result
        cv2.imwrite('assets/bridge_day_night_blend.jpg', 
                    cv2.cvtColor((bridge_result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        
        print("Bridge day-night blend saved as 'bridge_day_night_blend.jpg'")
    else:
        print("Bridge day or night images not found. Skipping bridge blending.")
    
    # Show comprehensive comparison
    if day_scene is not None and night_scene is not None and bridge_day is not None and bridge_night is not None:
        # Show all four blending results
        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        fig.suptitle('Multi-band Blending Results', fontsize=16)
        
        # Apple-Orange Left-right blending
        axes[0, 0].imshow(apple)
        axes[0, 0].set_title('Apple')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(orange)
        axes[0, 1].set_title('Orange')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow((lr_result * 255).astype(np.uint8))
        axes[0, 2].set_title('Left-Right Blend')
        axes[0, 2].axis('off')
        
        # Apple-Orange Irregular blending
        axes[1, 0].imshow(apple)
        axes[1, 0].set_title('Apple')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(orange)
        axes[1, 1].set_title('Orange')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow((irreg_result * 255).astype(np.uint8))
        axes[1, 2].set_title('Irregular Blend')
        axes[1, 2].axis('off')
        
        # Day-Night scene blending
        axes[2, 0].imshow(day_scene)
        axes[2, 0].set_title('Day Scene')
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(night_scene)
        axes[2, 1].set_title('Night Scene')
        axes[2, 1].axis('off')
        
        axes[2, 2].imshow((day_night_result * 255).astype(np.uint8))
        axes[2, 2].set_title('Day-Night Polygon Blend')
        axes[2, 2].axis('off')
        
        # Bridge Day-Night blending
        axes[3, 0].imshow(bridge_day)
        axes[3, 0].set_title('Bridge Day')
        axes[3, 0].axis('off')
        
        axes[3, 1].imshow(bridge_night)
        axes[3, 1].set_title('Bridge Night')
        axes[3, 1].axis('off')
        
        axes[3, 2].imshow((bridge_result * 255).astype(np.uint8))
        axes[3, 2].set_title('Bridge Wavy Blend')
        axes[3, 2].axis('off')
        
        plt.tight_layout()
        fig.savefig('assets/all_blending_results.png', dpi=150, bbox_inches='tight')
        plt.show()
    elif day_scene is not None and night_scene is not None:
        # Show three blending results (no bridge)
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle('Multi-band Blending Results', fontsize=16)
        
        # Apple-Orange Left-right blending
        axes[0, 0].imshow(apple)
        axes[0, 0].set_title('Apple')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(orange)
        axes[0, 1].set_title('Orange')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow((lr_result * 255).astype(np.uint8))
        axes[0, 2].set_title('Left-Right Blend')
        axes[0, 2].axis('off')
        
        # Apple-Orange Irregular blending
        axes[1, 0].imshow(apple)
        axes[1, 0].set_title('Apple')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(orange)
        axes[1, 1].set_title('Orange')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow((irreg_result * 255).astype(np.uint8))
        axes[1, 2].set_title('Irregular Blend')
        axes[1, 2].axis('off')
        
        # Day-Night scene blending
        axes[2, 0].imshow(day_scene)
        axes[2, 0].set_title('Day Scene')
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(night_scene)
        axes[2, 1].set_title('Night Scene')
        axes[2, 1].axis('off')
        
        axes[2, 2].imshow((day_night_result * 255).astype(np.uint8))
        axes[2, 2].set_title('Day-Night Blend')
        axes[2, 2].axis('off')
        
        plt.tight_layout()
        fig.savefig('assets/all_blending_results.png', dpi=150, bbox_inches='tight')
        plt.show()
    else:
        # Show only apple-orange results
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Apple-Orange Blending Results', fontsize=16)
        
        # Left-right blending
        axes[0, 0].imshow(apple)
        axes[0, 0].set_title('Apple')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(orange)
        axes[0, 1].set_title('Orange')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow((lr_result * 255).astype(np.uint8))
        axes[0, 2].set_title('Left-Right Blend')
        axes[0, 2].axis('off')
        
        # Irregular blending
        axes[1, 0].imshow(apple)
        axes[1, 0].set_title('Apple')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(orange)
        axes[1, 1].set_title('Orange')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow((irreg_result * 255).astype(np.uint8))
        axes[1, 2].set_title('Irregular Blend')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        fig.savefig('assets/apple_orange_blending_results.png', dpi=150, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    main()