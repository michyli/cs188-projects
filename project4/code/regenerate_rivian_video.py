"""
Regenerate Rivian novel view video with slower playback speed.
"""

import torch
import numpy as np
import os
import sys

# Import from object_nerf
sys.path.insert(0, os.path.dirname(__file__))
from code.object_nerf import NeRF, create_novel_view_video


def regenerate_slow_video(
    model_path='rivian_nerf_output/nerf_model.pth',
    data_path='rivian.npz',
    output_path='rivian_nerf_output/novel_views.mp4',
    fps=15,  # Slower FPS (was 30)
    steps_between=8
):
    """Regenerate the video with slower playback."""
    
    print("="*60)
    print("Regenerating Rivian Video (Slower Playback)")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print(f"FPS: {fps} (slower playback)")
    print()
    
    # Load dataset
    print("Loading dataset...")
    data = np.load(data_path)
    c2ws_test = data['c2ws_test']
    focal = float(data['focal'])
    
    if 'image_shape' in data:
        H, W = data['image_shape'][:2]
    elif 'images_train' in data:
        H, W = data['images_train'].shape[1:3]
    else:
        raise ValueError("Cannot determine image dimensions")
    
    print(f"  Test poses: {len(c2ws_test)}")
    print(f"  Image size: {W}x{H}")
    print(f"  Focal: {focal:.2f}")
    
    # Camera intrinsic matrix
    K = np.array([
        [focal, 0, W/2],
        [0, focal, H/2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Load model
    print("\nLoading trained model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Infer architecture from checkpoint
    first_layer_weight = checkpoint['model_state_dict']['layers_before_skip.0.weight']
    input_dim_pos = first_layer_weight.shape[1]
    hidden_dim = first_layer_weight.shape[0]
    L_pos = (input_dim_pos - 3) // 6
    
    color_layer_weight = checkpoint['model_state_dict']['color_layer1.weight']
    input_dim_dir = color_layer_weight.shape[1] - hidden_dim
    L_dir = (input_dim_dir - 3) // 6
    
    near = checkpoint.get('near', 0.02)
    far = checkpoint.get('far', 0.5)
    n_samples = checkpoint.get('n_samples', 64)
    
    print(f"  Architecture: L_pos={L_pos}, L_dir={L_dir}, hidden={hidden_dim}")
    print(f"  Rendering: near={near:.3f}, far={far:.3f}, samples={n_samples}")
    
    # Create and load model
    nerf = NeRF(pos_L=L_pos, dir_L=L_dir, hidden_dim=hidden_dim).to(device)
    nerf.load_state_dict(checkpoint['model_state_dict'])
    nerf.eval()
    
    print(f"  ✓ Model loaded (iteration {checkpoint['iteration']})")
    
    # Create video
    print(f"\nCreating video at {fps} FPS...")
    print(f"  Original poses: {len(c2ws_test)}")
    print(f"  Interpolated frames: {len(c2ws_test) * steps_between}")
    print(f"  Duration: {len(c2ws_test) * steps_between / fps:.1f}s")
    
    create_novel_view_video(
        nerf, c2ws_test, K, H, W,
        output_path=output_path,
        fps=fps,
        near=near,
        far=far,
        n_samples=n_samples,
        interpolate=True,
        steps_between=steps_between
    )
    
    print(f"\n✓ Slower video saved: {output_path}")
    print(f"  FPS: {fps} (plays at {fps/30:.1%} of original speed)")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Regenerate video with slower playback')
    parser.add_argument('--fps', type=int, default=15,
                       help='Frames per second (lower = slower, default: 15)')
    parser.add_argument('--steps', type=int, default=8,
                       help='Interpolation steps between poses (default: 8)')
    
    args = parser.parse_args()
    
    regenerate_slow_video(fps=args.fps, steps_between=args.steps)

