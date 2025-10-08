## Project 3A – Code Guide (code.py)

This file implements the core algorithms for homography estimation, image warping, correspondence visualization, rectification, and panorama mosaicing used throughout Part A. The most important functions are:

### Homography

- `computeH(im1_pts, im2_pts)`: Builds the 2n×9 system from point pairs and solves with SVD (last singular vector), then normalizes so H[2,2]=1.

### Warping (Inverse Mapping)

- `warpImageNearestNeighbor(im, H)`: Fast inverse-warp with nearest-neighbor sampling. Produces RGBA (or grayscale+alpha) with valid pixels flagged in alpha.
- `warpImageBilinear(im, H)`: Inverse-warp with bilinear sampling (vectorized). Returns same RGBA layout.

### Visualization / Reporting

- `visualize_correspondences(...)`: Side-by-side plot of two images with indexed, colored correspondences; saved to assets.
- `print_linear_system(...)`, `print_homography_matrix(...)`: Console helpers for A.2 write-ups.

### Rectification (A.3)

- `test_rectification()`: Rectifies a light-switch plate using four clicked corners from `Light_Light.json`; outputs side-by-side comparison (NN vs Bilinear).
- `test_rectification_window()`: Rectifies a window using `Window_Window.json` (corner order TL, TR, BL, BR); outputs a comparison figure.

### Mosaics (A.4)

- `create_mosaic(images, homographies, reference_idx)`: Core blender. Computes a shared canvas, inverse-warps each image (including the reference via global translation) into the canvas to avoid left-edge bias, then blends with distance-based alpha feathering.
- `create_parker_mosaics()`, `create_room_mosaics()`: Build panoramas for the Parker and Room sets (Left+Middle, Middle+Right, Full LMR). Middle view is the reference.
- `create_room_correspondences()`: Generates Room correspondence figures and logs matrices for A.2.

### Run

```bash
cd project3A
python code.py
```

This computes Parker homographies/figures, runs rectification, and builds mosaics. Extra helpers like `create_room_mosaics()` and `create_room_correspondences()` can be invoked similarly from a Python entry.
