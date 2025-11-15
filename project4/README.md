# Project 4: Camera Calibration and Neural Radiance Fields (NeRF)

## Overview

This project implements a complete pipeline for 3D reconstruction using Neural Radiance Fields (NeRF), from camera calibration with ArUco markers to training a neural network that can synthesize novel views of objects.

## Project Structure

```
project4/
├── README.md                      # This file
├── index.html                     # Project report and visualizations
├── style.css                      # Stylesheet for HTML report
├── code/                          # All Python source code
│   ├── code.py                    # Camera calibration & pose estimation
│   ├── part1.py                   # 2D neural field implementation
│   ├── part2.py                   # NeRF components (Parts 2.1-2.5)
│   ├── train_nerf.py              # NeRF training script (Lego dataset)
│   ├── object_nerf.py             # NeRF training for custom objects
│   ├── create_dataset.py          # Dataset creation (single ArUco tag)
│   ├── create_multitag_dataset.py # Dataset creation (multi-tag setup)
│   ├── regenerate_dataset.py      # Dataset utilities
│   └── visualize_rivian_rays.py   # Ray visualization utilities
├── *.npz                          # Dataset files (calibration, Lego, custom objects)
├── assets/                        # Images and screenshots
├── nerf_output/                   # Lego NeRF training results
├── object_nerf_results/           # Custom object NeRF results (old)
└── rivian_nerf_output/            # Rivian R1T NeRF results (current)
```

---

## Part 0: Camera Calibration and Dataset Creation

### Quick Start

**0.1 Camera Calibration:**

```bash
cd project4
python code/code.py
```

**0.2-0.3 Camera Pose Estimation & Visualization:**

```bash
python code/code.py pose
```

**0.4 Create Dataset:**

```bash
python code/code.py dataset
```

### Multi-Tag Dataset (Rivian R1T)

For improved robustness with 6 ArUco tags (2×3 grid):

```bash
python code/create_multitag_dataset.py \
    --image_folder assets/rivian \
    --output rivian.npz \
    --tag_size 0.056 \
    --horizontal_spacing 0.084 \
    --vertical_spacing 0.071 \
    --min_tags 2
```

**Output:**

- `calibration_results.npz` - Camera intrinsics and distortion coefficients
- `my_data.npz` or `rivian.npz` - NeRF-ready dataset with camera poses

---

## Part 1: 2D Neural Field

Train a neural network to fit a 2D image:

```bash
python code/part1.py
```

**Output:** Training progression images and PSNR curves in `assets/part_1_img/`

---

## Part 2: Neural Radiance Fields (NeRF)

### Implementation Components

All NeRF components are implemented in `code/part2.py`:

- **Part 2.1:** Create Rays from Cameras (`get_rays`, `pixel_to_ray`)
- **Part 2.2:** Sampling Points Along Rays (`sample_points_along_rays`)
- **Part 2.3:** Dataloader (`RaysData` class for efficient ray sampling)
- **Part 2.4:** NeRF Network Architecture (`NeRF` and `PositionalEncoding` classes)
- **Part 2.5:** Volume Rendering (`volrend`, `render_rays`)

### Testing

Verify all components:

```bash
python code/part2.py
```

Expected output:

```
Part 2.1 (Create Rays): [PASS] PASSED
Part 2.2 (Sampling): [PASS] PASSED
Part 2.3 (Dataloader): [PASS] PASSED
Part 2.4 (NeRF Network): [PASS] PASSED
Part 2.5 (Volume Rendering): [PASS] PASSED

[PASS][PASS][PASS] ALL TESTS PASSED! [PASS][PASS][PASS]
```

---

## Part 2.5: Training NeRF on Lego Dataset

### Quick Start

```bash
python code/train_nerf.py
```

### Command Line Options

```bash
python code/train_nerf.py --help
```

**Key Arguments:**

- `--data PATH` - Dataset path (default: `lego_200x200.npz`)
- `--output DIR` - Output directory (default: `nerf_output`)
- `--iterations N` - Training iterations (default: 1000)
- `--batch_size N` - Rays per gradient step (default: 10000)
- `--lr FLOAT` - Learning rate (default: 5e-4)
- `--n_samples N` - Samples per ray (default: 64)

**Example: Extended training**

```bash
python code/train_nerf.py --iterations 2000 --batch_size 20000 --output nerf_long_training
```

### Training Output

The script creates a `nerf_output/` directory with:

1. **PSNR Curve** (`psnr_curve.png`) - Validation PSNR over training
2. **Training Loss** (`training_loss.png`) - MSE loss progression
3. **Validation Comparisons** (`val_image_*.png`) - Ground truth vs rendered
4. **Training Progression** (`progression/iter_*.png`) - Images at different iterations
5. **Novel View Video** (`novel_views.mp4`) - 360° spherical rendering
6. **Novel View GIF** (`novel_views.gif`) - Animated version
7. **Model Checkpoint** (`nerf_model.pth`) - Trained model weights

### Expected Performance

- **Target:** 23+ PSNR on validation set
- **Training Time:**
  - CPU: ~2-3 hours for 1000 iterations
  - GPU (CUDA): ~10-15 minutes for 1000 iterations
- **Memory:** ~2-4 GB RAM, ~1-2 GB VRAM (GPU)

---

## Part 2.6: Training NeRF on Custom Objects

### Rivian R1T with Multi-Tag Calibration

Train NeRF on custom Rivian R1T dataset:

```bash
python code/object_nerf.py \
    --data rivian.npz \
    --output rivian_nerf_output \
    --iterations 2000 \
    --batch_size 8192 \
    --lr 5e-4
```

**Features:**

- **Multi-tag calibration** (6 ArUco tags in 2×3 grid)
- **Robust pose estimation** (handles partial occlusions)
- **Automatic near/far estimation** (0.02m / 0.5m for Rivian)
- **Smooth 360° video generation** (SLERP/LERP interpolation with circular sorting)

**Output:**

- Training progression images
- PSNR and loss curves
- Validation comparisons
- Novel view video with smooth circular motion

---

## Visualization Tools

### Camera Pose Visualization (Part 0.3)

View camera frustums and poses in 3D:

```bash
python code/code.py pose
```

Opens viser server at `http://localhost:8080` showing:

- Camera frustums with captured images
- World coordinate frame

### Ray Visualization (Part 2.3)

Visualize rays and sample points:

```bash
python code/code.py rays
```

Opens viser server showing:

- 32 camera frustums (27 train + 5 val)
- 100 random rays (red lines)
- 6,400 sample points (green dots, 64 per ray)

**Custom options:**

```bash
python -c "from code.part2 import visualize_rivian_rays_viser; visualize_rivian_rays_viser('rivian.npz', num_rays=200)"
```

---

## Technical Details

### NeRF Architecture

- **Network:** 8-layer MLP with skip connection at layer 4
- **Parameters:** ~1.2 million
- **Position Encoding:** L=10 (63 dimensions) - high frequency for fine details
- **Direction Encoding:** L=4 (27 dimensions) - low frequency for view-dependence
- **Hidden Dimension:** 256
- **Activation:** ReLU (hidden), ReLU (density), Sigmoid (RGB)

### Training Configuration

- **Optimizer:** Adam
- **Learning Rate:** 5e-4
- **Batch Size:** 10,000 rays/iteration (Lego), 8,192 rays (custom objects)
- **Samples per Ray:** 64
- **Loss Function:** MSE
- **Evaluation Metric:** PSNR

### Dataset Statistics

**Lego Dataset:**

- Training: 100 images (200×200×3)
- Validation: 10 images
- Test: 60 camera poses
- Total rays: 4,000,000

**Rivian Dataset:**

- Training: 27 images (400×528×3)
- Validation: 5 images
- Test: 7 camera poses
- Calibration: 6-tag ArUco board (2×3 grid)

---

## Dependencies

Required packages:

```bash
pip install numpy torch matplotlib imageio tqdm opencv-python viser scipy
```

**For GPU training (recommended):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Troubleshooting

### Out of Memory Error

- Reduce `--batch_size` (try 5000 or 2000)
- Reduce `--n_samples` (try 32)

### Training Too Slow

- Ensure GPU is detected: check output for "cuda" device
- Install PyTorch with CUDA support
- Reduce validation frequency: `--val_every 100`

### PSNR Not Reaching Target

- Train longer: `--iterations 2000` or `--iterations 3000`
- Try different learning rate: `--lr 1e-3` or `--lr 1e-4`
- Increase samples: `--n_samples 128`

### Viser Visualization Doesn't Show

- Just press Enter to skip if not needed
- Check browser at `http://localhost:8080`
- Ensure viser is installed: `pip install viser`

---

## Loading Trained Models

```python
import torch
from code.part2 import NeRF

# Load checkpoint
checkpoint = torch.load('nerf_output/nerf_model.pth', weights_only=False)

# Create model with saved config
config = checkpoint['config']
nerf = NeRF(pos_L=config['pos_L'], dir_L=config['dir_L'],
           hidden_dim=config['hidden_dim'])

# Load weights
nerf.load_state_dict(checkpoint['model_state_dict'])
nerf.eval()

print(f"Loaded model from iteration {checkpoint['iteration']}")
print(f"Final PSNR: {checkpoint['final_psnr']:.2f} dB")
```

---

## Project Report

All results, visualizations, and explanations are compiled in `index.html`. View it by opening the file in a web browser.

**Report includes:**

- Part 0: Camera calibration and pose estimation
- Part 1: 2D neural field fitting
- Part 2.1-2.5: NeRF implementation details
- Part 2.5: Lego NeRF training results
- Part 2.6: Custom object (Rivian R1T) NeRF results

---

## Key Features

✅ **Complete Pipeline:** From camera calibration to novel view synthesis  
✅ **Multi-Tag Support:** Robust 6-tag ArUco calibration  
✅ **Smooth Videos:** SLERP/LERP interpolation with circular sorting  
✅ **Interactive Visualization:** Real-time 3D viewing with viser  
✅ **Automatic GPU Acceleration:** CUDA support when available  
✅ **Comprehensive Testing:** All components verified  
✅ **Production Ready:** Error handling, progress bars, checkpoints

---

## Achievements

- ✅ **Lego NeRF:** 23+ PSNR achieved on validation set
- ✅ **Custom Object NeRF:** Photorealistic novel views of Rivian R1T
- ✅ **Robust Calibration:** Multi-tag setup handles partial occlusions
- ✅ **Smooth Video Generation:** Circular camera motion with proper interpolation

---

## Author

Michael (CS 188 Project 4)

---

## Acknowledgments

- CS 188 staff for project specifications and starter code
- Original NeRF paper: Mildenhall et al., ECCV 2020
- OpenCV ArUco module for marker detection
- Viser for 3D visualization

---

**Status:** ✅ All components complete and tested. Ready for submission!
