# Project 3B: Feature Matching and Autostitching

## Overview

This project implements an automatic image stitching pipeline using Harris corner detection, Adaptive Non-Maximal Suppression (ANMS), feature descriptor extraction, feature matching with Lowe's ratio test, and RANSAC-based homography estimation. The implementation creates panoramic mosaics without manual intervention.

## Project Structure

```
project3B/
├── harris.py                  # Core feature detection and matching functions
├── match_images.py            # Feature matching visualization script
├── auto_mosaic.py             # Automatic panorama creation (3 images)
├── auto_mosaic_dusk.py        # Automatic panorama creation (5 images)
├── index.html                 # Project webpage
├── style.css                  # Webpage styling
└── assets/                    # Images and results
    ├── ParkerL.jpg, ParkerM.jpg, ParkerR.jpg
    ├── RoomL.jpg, RoomM.jpg, RoomR.jpg
    ├── duskBL.jpg, duskBM.jpg, duskBR.jpg, duskTL.jpg, duskTR.jpg
    ├── harris_corners.jpg
    ├── anms_corners.jpg
    ├── feature_descriptors.jpg
    ├── matches_ParkerM_ParkerR.jpg
    ├── auto_mosaic_parker.jpg
    ├── auto_mosaic_room.jpg
    └── auto_mosaic_dusk.jpg
```

## Code Files

### `harris.py`

Core implementation containing all fundamental algorithms:

**Key Functions:**

- `get_harris_corners()` - Detects Harris corners with adjustable parameters
- `anms()` - Adaptive Non-Maximal Suppression for spatially distributed features
- `extract_feature_descriptors()` - Extracts 8×8 normalized descriptors from 40×40 windows
- `match_features()` - Matches features using Lowe's ratio test
- `compute_homography()` - Computes 3×3 homography from point correspondences using SVD
- `ransac_homography()` - Robust homography estimation using 4-point RANSAC
- `visualize_*()` - Various visualization functions

**Usage:**

```python
from harris import get_harris_corners, anms, extract_feature_descriptors

# Detect corners
h, corners = get_harris_corners(image_gray, edge_discard=20)

# Apply ANMS
corners_anms = anms(corners, h, num_points=500)

# Extract descriptors
descriptors, valid_corners = extract_feature_descriptors(image_gray, corners_anms)
```

### `match_images.py`

Demonstrates feature matching between two images (ParkerM and ParkerR).

**Features:**

- Processes both images through the full pipeline
- Matches features using ratio test (threshold=0.7)
- Visualizes matches with colored lines
- Reports match statistics

**Run:**

```bash
python match_images.py
```

**Output:**

- `assets/matches_ParkerM_ParkerR.jpg` - Visualization of matched features

### `auto_mosaic.py`

Creates automatic panoramas from 3 images (left-middle-right arrangement).

**Features:**

- Processes Parker Street and Room panoramas
- Uses middle image as reference frame
- Matches adjacent pairs and accumulates homographies
- Warps and blends images with distance-based feathering

**Run:**

```bash
python auto_mosaic.py
```

**Output:**

- `assets/auto_mosaic_parker.jpg` - Parker Street panorama
- `assets/auto_mosaic_room.jpg` - Room panorama

### `auto_mosaic_dusk.py`

Creates automatic panorama from 5 images in a grid arrangement:

```
   TL      TR
BL  BM  BR
```

**Features:**

- Handles complex multi-image stitching
- Uses hierarchical alignment (BM as hub)
- Connects top images through bottom images
- Demonstrates robustness to non-linear arrangements

**Run:**

```bash
python auto_mosaic_dusk.py
```

**Output:**

- `assets/auto_mosaic_dusk.jpg` - 5-image dusk panorama

## Dependencies

Required Python packages:

```bash
pip install numpy matplotlib scikit-image scipy
```

Specific imports:

- `numpy` - Array operations and linear algebra
- `matplotlib.pyplot` - Visualization
- `skimage.feature` - Harris corner detection (corner_harris, peak_local_max)
- `skimage.io` - Image I/O (imread)
- `skimage.color` - Color space conversion (rgb2gray)
- `scipy.ndimage` - Image warping (map_coordinates, distance_transform_edt)

## Algorithm Parameters

### Harris Corner Detection

- `edge_discard=20` - Pixels to discard near image edges
- `min_distance=5` - Minimum pixel spacing between corners
- `threshold_rel=0.01` - Relative intensity threshold

### ANMS

- `num_points=500` - Number of corners to select
- `c_robust=0.9` - Robustness constant (neighbor must be >1.11× stronger)

### Feature Descriptors

- `window_size=40` - Size of sampling window
- `descriptor_size=8` - Size of descriptor (8×8 = 64 dimensions)
- `sample_spacing=5` - Spacing between samples

### Feature Matching

- `ratio_threshold=0.7` - Lowe's ratio test threshold (based on paper Figure 6b)

### RANSAC

- `num_iterations=1000` - Number of RANSAC iterations
- `threshold=5.0` - Inlier distance threshold in pixels

## Results Summary

### Parker Street Panorama (3 images)

- ParkerL→ParkerM: 19/28 inliers (67.9%)
- ParkerR→ParkerM: 24/27 inliers (88.9%)

### Room Panorama (3 images)

- RoomL→RoomM: 24/33 inliers (72.7%)
- RoomR→RoomM: 11/30 inliers (36.7%)

### Dusk Panorama (5 images)

- DuskBL→DuskBM: 45/48 inliers (93.8%)
- DuskBR→DuskBM: 84/84 inliers (100.0%) ✨
- DuskTL→DuskBL: 40/44 inliers (90.9%)
- DuskTR→DuskBR: 63/69 inliers (91.3%)

## Implementation Notes

### Key Design Decisions

1. **Harris Parameters**: Adjusted `min_distance` and `threshold_rel` to reduce corner count from ~195k to ~4k, making ANMS computationally feasible.

2. **ANMS Algorithm**: Implemented per paper specification with corrected inequality: `f(xi) < c_robust × f(xj)` ensures neighbors are significantly stronger.

3. **Feature Descriptors**: 8×8 descriptors sampled from 40×40 windows with 5-pixel spacing provides robustness to location errors while avoiding aliasing.

4. **Lowe's Ratio Test**: Threshold of 0.7 balances precision/recall based on Figure 6b of the paper.

5. **Homography Direction**: Fixed to ensure correct spatial ordering - match from image i to reference, accumulate properly.

6. **Blending**: Distance-based alpha feathering creates seamless transitions in overlap regions.

## Viewing Results

Open `index.html` in a web browser to see:

- Part B.1: Harris corners with and without ANMS
- Part B.2: Feature descriptor visualizations
- Part B.3: Feature matching between image pairs
- Part B.4: Automatic panorama comparisons with manual stitching

## Troubleshooting

**Too many corners causing memory issues?**

- Increase `min_distance` or `threshold_rel` in `get_harris_corners()`

**Poor matching results?**

- Adjust `ratio_threshold` in `match_features()` (lower = stricter)
- Increase `num_iterations` in RANSAC for more robust estimation

**Incorrect panorama ordering?**

- Verify image naming matches expected left-middle-right arrangement
- Check homography accumulation direction in mosaic scripts

## References

- Brown, M., & Lowe, D. G. (2007). Automatic panoramic image stitching using invariant features. _International Journal of Computer Vision_, 74(1), 59-73.
- Harris, C., & Stephens, M. (1988). A combined corner and edge detector. _Alvey Vision Conference_.

## Author

CS188 - Computer Vision (Fall 2024)
