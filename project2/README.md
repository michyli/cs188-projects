# Project 2: Filters, Frequencies, and Multi-band Blending

This project implements fundamental image processing techniques including convolution, edge detection, frequency domain filtering, and advanced multi-band image blending algorithms.

## Project Structure

### Core Files

- `index.html` - Main project webpage showcasing all implementations and results
- `style.css` - Styling for the project webpage
- `code2-3.py` - Multi-band blending implementation with Gaussian and Laplacian stacks

### Implementation Files

- `code.py` - Basic convolution implementations
- `code1-2.py` - Edge detection using finite difference operators
- `code1-3.py` - Derivative of Gaussian (DoG) filters for improved edge detection
- `code2-1.py` - Unsharp masking for image sharpening
- `code2-2.py` - Hybrid image creation using frequency domain techniques

### Assets Directory

Contains all input images, generated visualizations, and results:

#### Input Images

- `apple.jpeg`, `orange.jpeg` - Classic apple-orange blending images
- `day_scene.png`, `night_scene.png` - Day/night scene blending
- `bridge_day.png`, `bridge_night.png` - Bridge scene blending
- `cameraman.png` - Standard test image for edge detection
- Various other test images for hybrid image creation

#### Generated Visualizations

- `apple_orange_figure_342_style.png` - Recreation of Figure 3.42 from Szelski
- `gaussian_laplacian_stacks.png` - Frequency decomposition demonstration
- `*_blending_process.png` - Multi-band blending process visualizations
- `all_blending_results.png` - Comprehensive results comparison
- `hybrid_*.png` - Hybrid image results and frequency analysis
- `part1-*.png` - Edge detection and convolution results

## Key Implementations

### Part 1: Filters and Edges

1. **Convolution** - Custom numpy-only implementations (4-loop and 2-loop versions)
2. **Edge Detection** - Finite difference operators with gradient magnitude computation
3. **DoG Filters** - Derivative of Gaussian for noise-robust edge detection

### Part 2: Applications

1. **Unsharp Masking** - Image sharpening through high-frequency enhancement
2. **Hybrid Images** - Frequency domain blending for distance-dependent perception
3. **Multi-band Blending** - Seamless image fusion using Gaussian and Laplacian stacks