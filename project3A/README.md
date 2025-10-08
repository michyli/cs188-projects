# Project 3A - Part A.2: Recover Homographies

## Implementation Summary

### Files Created/Modified:

1. **code.py** - Complete implementation with:

   - `computeH()` function
   - Correspondence visualization
   - System of equations display
   - Reprojection error analysis

2. **index.html** - Professional presentation showing:

   - Point correspondences between image pairs
   - Linear system matrices
   - Recovered homography matrices
   - Error analysis

3. **style.css** - Enhanced styling for:
   - Code blocks and equations
   - Result images
   - Error sections

### Key Implementation Details:

#### 1. computeH(im1_pts, im2_pts) Function

The function computes the homography matrix H using:

- **Input**: Two sets of n corresponding points (n ≥ 4)
- **Method**: Least-squares solution via SVD
- **Output**: 3×3 homography matrix (normalized)

For each correspondence (x, y) → (x', y'), we construct two equations:

```
[-x, -y, -1,  0,  0,  0, x·x', y·x', x']   [h1]
[ 0,  0,  0, -x, -y, -1, x·y', y·y', y'] × [h2] = 0
                                            [...]
```

With n correspondences, we get a 2n×9 system Ah = 0.
The solution is the last row of V^T from SVD(A).

#### 2. Results for ParkerM → ParkerL:

- **Correspondences**: 11 points
- **Mean error**: 3.98 pixels
- **Max error**: 6.01 pixels

Homography Matrix:

```
H = [  0.5130    0.0232  553.2171 ]
    [ -0.2569    0.8047  108.5757 ]
    [ -0.0003   -0.0000    1.0000 ]
```

#### 3. Results for ParkerM → ParkerR:

- **Correspondences**: 12 points
- **Mean error**: 2.96 pixels
- **Max error**: 9.15 pixels

Homography Matrix:

```
H = [  1.4993   -0.0812  -699.5494 ]
    [  0.2920    1.2568  -224.3394 ]
    [  0.0004   -0.0001    1.0000 ]
```

### Verification:

The low reprojection errors (mean < 4 pixels) demonstrate that:

1. The homographies accurately capture the transformations
2. The point correspondences were selected carefully
3. The SVD-based least-squares solution is robust

### Generated Assets:

- `assets/ParkerM_ParkerL_correspondences.png` - Visualized point matches
- `assets/ParkerM_ParkerR_correspondences.png` - Visualized point matches
- `homography_results.txt` - Detailed numerical results

## Part A.3: Warp the Images

### Implementation Summary

Two image warping functions with different interpolation methods:

#### 1. warpImageNearestNeighbor(im, H)

- Uses **inverse warping** to avoid holes in output
- Rounds coordinates to nearest pixel
- Fast but produces pixelated results
- Time complexity: O(output_pixels)

#### 2. warpImageBilinear(im, H)

- Uses **inverse warping** with bilinear interpolation
- Computes weighted average of 4 neighboring pixels
- Produces smooth, high-quality results
- Slightly slower but much better visual quality

### Key Implementation Details:

1. **Bounding Box Calculation**: Transform input corners to find output size
2. **Inverse Warping**: Map each output pixel back to input using H^(-1)
3. **Interpolation**: Sample input at mapped coordinates
4. **Alpha Channel**: Handle out-of-bounds pixels gracefully

### Rectification Results:

- Successfully tested on quadrilateral-to-rectangle transformation
- Bilinear interpolation shows significantly smoother results in zoomed comparison
- Nearest neighbor is ~2x faster but produces visible pixelation

### Trade-offs Analysis:

**Nearest Neighbor:**

- ✓ Fast, simple implementation
- ✗ Lower quality, visible pixelation

**Bilinear:**

- ✓ Better quality, anti-aliasing
- ✗ ~2x slower computation

**Recommendation**: Use bilinear for final results, nearest neighbor for preview/debugging

### Generated Assets (Part A.3):

- `assets/rectification_comparison.png` - Side-by-side comparison
- `assets/interpolation_comparison_zoomed.png` - Zoomed quality comparison
- `assets/rectified_nn.png` - Nearest neighbor result
- `assets/rectified_bilinear.png` - Bilinear result

---

## How to Run:

```bash
cd project3A
python code.py
```

This will:

1. Compute homographies for ParkerM→ParkerL and ParkerM→ParkerR (Part A.2)
2. Run rectification test with both interpolation methods (Part A.3)
3. Generate all comparison images and save to `assets/`

Then open `index.html` in a browser to view all results.
