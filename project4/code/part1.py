"""
Part 1: Fit a Neural Field to a 2D Image
CS188 Project 4

This script implements a neural field that can represent a 2D image using:
- Multilayer Perceptron (MLP) with Sinusoidal Positional Encoding
- Random pixel sampling for training
- MSE loss and PSNR metric
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm


class PositionalEncoding:
    """
    Sinusoidal Positional Encoding for input coordinates.
    
    PE(x) = {x, sin(2^0*pi*x), cos(2^0*pi*x), ..., sin(2^(L-1)*pi*x), cos(2^(L-1)*pi*x)}
    """
    def __init__(self, max_freq_log2, include_input=True):
        """
        Args:
            max_freq_log2: L in the paper, highest frequency level
            include_input: Whether to include original input in the encoding
        """
        self.max_freq_log2 = max_freq_log2
        self.include_input = include_input
        self.num_freqs = max_freq_log2
        
    def get_output_dim(self, input_dim):
        """Calculate output dimension after positional encoding."""
        # For each frequency: sin and cos
        # Plus original input if include_input=True
        output_dim = input_dim * self.num_freqs * 2
        if self.include_input:
            output_dim += input_dim
        return output_dim
    
    def encode(self, x):
        """
        Apply positional encoding to input.
        
        Args:
            x: Input tensor of shape (N, input_dim)
            
        Returns:
            Encoded tensor of shape (N, output_dim)
        """
        out = []
        
        if self.include_input:
            out.append(x)
        
        # Apply sinusoidal functions at different frequencies
        for i in range(self.num_freqs):
            freq = 2.0 ** i
            out.append(torch.sin(freq * np.pi * x))
            out.append(torch.cos(freq * np.pi * x))
        
        return torch.cat(out, dim=-1)


class NeuralField2D(nn.Module):
    """
    MLP-based Neural Field for 2D image representation.
    
    Architecture: Input -> PE -> Linear -> ReLU -> ... -> Linear -> Sigmoid -> RGB
    """
    def __init__(self, max_freq_log2=10, hidden_dim=256, num_layers=4):
        """
        Args:
            max_freq_log2: Maximum frequency for positional encoding (L)
            hidden_dim: Width of hidden layers
            num_layers: Number of hidden layers
        """
        super(NeuralField2D, self).__init__()
        
        self.pe = PositionalEncoding(max_freq_log2, include_input=True)
        
        # Input dimension after positional encoding
        input_dim = self.pe.get_output_dim(2)  # 2D coordinates
        
        # Build MLP layers
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer (RGB colors)
        layers.append(nn.Linear(hidden_dim, 3))
        layers.append(nn.Sigmoid())  # Constrain output to [0, 1]
        
        self.network = nn.Sequential(*layers)
        
        self.max_freq_log2 = max_freq_log2
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    def forward(self, coords):
        """
        Forward pass through the neural field.
        
        Args:
            coords: Normalized coordinates (N, 2) in range [0, 1]
            
        Returns:
            RGB colors (N, 3) in range [0, 1]
        """
        # Apply positional encoding
        encoded = self.pe.encode(coords)
        
        # Pass through MLP
        rgb = self.network(encoded)
        
        return rgb


class PixelDataLoader:
    """
    DataLoader that randomly samples pixels from an image for training.
    """
    def __init__(self, image_path, batch_size=10000, device='cuda'):
        """
        Args:
            image_path: Path to the image file
            batch_size: Number of pixels to sample per batch
            device: Device to put tensors on
        """
        self.batch_size = batch_size
        self.device = device
        
        # Load image
        img = Image.open(image_path).convert('RGB')
        self.img_array = np.array(img)  # (H, W, 3)
        
        self.height, self.width = self.img_array.shape[:2]
        
        print(f"Loaded image: {image_path}")
        print(f"Image size: {self.width}x{self.height}")
        print(f"Total pixels: {self.height * self.width}")
        
        # Create coordinate grid
        y_coords, x_coords = np.meshgrid(
            np.arange(self.height), 
            np.arange(self.width), 
            indexing='ij'
        )
        
        # Flatten and normalize coordinates to [0, 1]
        self.all_coords = np.stack([
            x_coords.flatten() / self.width,
            y_coords.flatten() / self.height
        ], axis=-1).astype(np.float32)  # (H*W, 2)
        
        # Flatten and normalize colors to [0, 1]
        self.all_colors = self.img_array.reshape(-1, 3).astype(np.float32) / 255.0  # (H*W, 3)
        
        self.num_pixels = len(self.all_coords)
        
    def sample_batch(self):
        """
        Sample a random batch of pixels.
        
        Returns:
            coords: (batch_size, 2) normalized coordinates
            colors: (batch_size, 3) normalized RGB colors
        """
        # Random sampling
        indices = np.random.choice(self.num_pixels, self.batch_size, replace=False)
        
        coords = torch.from_numpy(self.all_coords[indices]).to(self.device)
        colors = torch.from_numpy(self.all_colors[indices]).to(self.device)
        
        return coords, colors
    
    def get_all_data(self):
        """
        Get all pixels (for evaluation).
        
        Returns:
            coords: (H*W, 2) normalized coordinates
            colors: (H*W, 3) normalized RGB colors
        """
        coords = torch.from_numpy(self.all_coords).to(self.device)
        colors = torch.from_numpy(self.all_colors).to(self.device)
        return coords, colors


def compute_psnr(mse):
    """
    Compute Peak Signal-to-Noise Ratio from MSE.
    
    PSNR = 10 * log10(1 / MSE)
    
    Args:
        mse: Mean squared error (assumes normalized images [0, 1])
        
    Returns:
        PSNR value in dB
    """
    return 10.0 * torch.log10(1.0 / mse)


def reconstruct_image(model, dataloader, device='cuda'):
    """
    Reconstruct the full image using the trained model.
    
    Args:
        model: Trained neural field model
        dataloader: PixelDataLoader instance
        device: Device to use
        
    Returns:
        Reconstructed image as numpy array (H, W, 3) in [0, 255]
    """
    model.eval()
    
    with torch.no_grad():
        # Get all coordinates
        all_coords, _ = dataloader.get_all_data()
        
        # Reconstruct in batches to avoid memory issues
        batch_size = 100000
        num_batches = (len(all_coords) + batch_size - 1) // batch_size
        
        all_colors = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(all_coords))
            coords_batch = all_coords[start_idx:end_idx]
            
            colors_batch = model(coords_batch)
            all_colors.append(colors_batch)
        
        all_colors = torch.cat(all_colors, dim=0)
        
        # Reshape to image
        img = all_colors.reshape(dataloader.height, dataloader.width, 3)
        img = (img.cpu().numpy() * 255).astype(np.uint8)
    
    return img


def train_neural_field(image_path, max_freq_log2=10, hidden_dim=256, num_layers=4,
                       learning_rate=1e-2, num_iterations=3000, batch_size=10000,
                       log_interval=100, save_dir='project4/assets/part_1_img',
                       experiment_name='default'):
    """
    Train a neural field to fit a 2D image.
    
    Args:
        image_path: Path to input image
        max_freq_log2: Maximum frequency for PE (L)
        hidden_dim: Width of hidden layers
        num_layers: Number of hidden layers
        learning_rate: Learning rate for Adam optimizer
        num_iterations: Number of training iterations
        batch_size: Number of pixels per batch
        log_interval: How often to log/save results
        save_dir: Directory to save results
        experiment_name: Name for this experiment
        
    Returns:
        Dictionary with training history and model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    exp_dir = os.path.join(save_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Initialize dataloader
    dataloader = PixelDataLoader(image_path, batch_size=batch_size, device=device)
    
    # Initialize model
    model = NeuralField2D(
        max_freq_log2=max_freq_log2,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)
    
    print(f"\nModel Architecture:")
    print(f"  Max frequency (L): {max_freq_log2}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'iterations': [],
        'train_loss': [],
        'train_psnr': [],
        'images': []
    }
    
    # Training loop
    print(f"\nStarting training for {num_iterations} iterations...")
    model.train()
    
    for iteration in tqdm(range(num_iterations)):
        # Sample batch
        coords, colors = dataloader.sample_batch()
        
        # Forward pass
        pred_colors = model(coords)
        
        # Compute loss
        loss = criterion(pred_colors, colors)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log progress
        if (iteration + 1) % log_interval == 0 or iteration == 0:
            psnr = compute_psnr(loss)
            
            history['iterations'].append(iteration + 1)
            history['train_loss'].append(loss.item())
            history['train_psnr'].append(psnr.item())
            
            print(f"\nIteration {iteration + 1}/{num_iterations}")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  PSNR: {psnr.item():.2f} dB")
            
            # Reconstruct and save image
            recon_img = reconstruct_image(model, dataloader, device)
            history['images'].append(recon_img)
            
            img_filename = os.path.join(exp_dir, f'iteration_{iteration+1:04d}.png')
            Image.fromarray(recon_img).save(img_filename)
            
            model.train()
    
    print("\nTraining complete!")
    
    # Save final model
    model_path = os.path.join(exp_dir, 'model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'max_freq_log2': max_freq_log2,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
    }, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save PSNR curve
    plt.figure(figsize=(10, 6))
    plt.plot(history['iterations'], history['train_psnr'], linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title(f'Training PSNR Curve\n(L={max_freq_log2}, Hidden={hidden_dim})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    psnr_path = os.path.join(exp_dir, 'psnr_curve.png')
    plt.savefig(psnr_path, dpi=150)
    plt.close()
    print(f"PSNR curve saved to: {psnr_path}")
    
    return {
        'model': model,
        'history': history,
        'dataloader': dataloader,
        'exp_dir': exp_dir
    }


def create_training_progression_figure(history, save_path):
    """
    Create a figure showing training progression at different iterations.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the figure
    """
    # Select 4 evenly spaced iterations to show
    num_snapshots = min(4, len(history['images']))
    indices = np.linspace(0, len(history['images']) - 1, num_snapshots, dtype=int)
    
    fig, axes = plt.subplots(1, num_snapshots, figsize=(5*num_snapshots, 5))
    if num_snapshots == 1:
        axes = [axes]
    
    for idx, ax_idx in enumerate(indices):
        img = history['images'][ax_idx]
        iteration = history['iterations'][ax_idx]
        psnr = history['train_psnr'][ax_idx]
        
        axes[idx].imshow(img)
        axes[idx].set_title(f'Iteration {iteration}\nPSNR: {psnr:.2f} dB', fontsize=12)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training progression saved to: {save_path}")


def run_hyperparameter_experiments(image_path, save_dir='project4/assets/part_1_img'):
    """
    Run experiments with different hyperparameters.
    
    Creates a 2x2 grid with:
    - 2 choices of max frequency (L)
    - 2 choices of hidden dimension (width)
    """
    print("\n" + "="*60)
    print("RUNNING HYPERPARAMETER EXPERIMENTS")
    print("="*60)
    
    # Define hyperparameter combinations
    freq_values = [3, 10]  # Low and high frequency
    width_values = [64, 256]  # Low and high width
    
    results = {}
    
    for freq in freq_values:
        for width in width_values:
            exp_name = f'freq_{freq}_width_{width}'
            print(f"\n{'='*60}")
            print(f"Experiment: L={freq}, Width={width}")
            print(f"{'='*60}")
            
            result = train_neural_field(
                image_path=image_path,
                max_freq_log2=freq,
                hidden_dim=width,
                num_layers=4,
                learning_rate=1e-2,
                num_iterations=2000,
                batch_size=10000,
                log_interval=200,
                save_dir=save_dir,
                experiment_name=exp_name
            )
            
            results[exp_name] = result
    
    # Create comparison grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    idx = 0
    for freq in freq_values:
        for width in width_values:
            exp_name = f'freq_{freq}_width_{width}'
            final_img = results[exp_name]['history']['images'][-1]
            final_psnr = results[exp_name]['history']['train_psnr'][-1]
            
            axes[idx].imshow(final_img)
            axes[idx].set_title(f'L={freq}, Width={width}\nPSNR: {final_psnr:.2f} dB', fontsize=12)
            axes[idx].axis('off')
            idx += 1
    
    plt.tight_layout()
    comparison_path = os.path.join(save_dir, 'hyperparameter_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nHyperparameter comparison saved to: {comparison_path}")
    
    return results


def main():
    """Main function to run Part 1 experiments."""
    print("="*60)
    print("PART 1: FIT A NEURAL FIELD TO A 2D IMAGE")
    print("="*60)
    
    # Use the provided input image
    image_path = 'project4/assets/part_1_img/input/image.png'
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"\nError: Image not found at {image_path}")
        print("Please place an image at this location or update the image_path variable.")
        print("\nYou can download the example image or use your own image.")
        return
    
    # Experiment 1: Train on first test image with default settings
    # SKIPPED - Already trained, results in image1_default/
    print("\n" + "="*60)
    print("EXPERIMENT 1: Default Settings (Image 1) - SKIPPED")
    print("="*60)
    print("Image 1 training already complete. Results in: image1_default/")
    
    # Experiment 2: Train on second test image with default settings
    print("\n" + "="*60)
    print("EXPERIMENT: Training on Image 2")
    print("="*60)
    
    image_path2 = 'project4/assets/part_1_img/input/image2.png'
    
    if os.path.exists(image_path2):
        result2 = train_neural_field(
            image_path=image_path2,
            max_freq_log2=10,
            hidden_dim=256,
            num_layers=4,
            learning_rate=1e-2,
            num_iterations=3000,
            batch_size=10000,
            log_interval=300,
            save_dir='project4/assets/part_1_img',
            experiment_name='image2_default'
        )
        
        # Create training progression figure
        progression_path2 = os.path.join(result2['exp_dir'], 'training_progression.png')
        create_training_progression_figure(result2['history'], progression_path2)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"\nResults saved to: project4/assets/part_1_img/image2_default/")
        print("\nGenerated files:")
        print("  - Iterations: iteration_0001.png to iteration_3000.png")
        print("  - PSNR curve: psnr_curve.png")
        print("  - Training progression: training_progression.png")
        print("  - Trained model: model.pth")
    else:
        print(f"\nError: Image 2 not found at {image_path2}")
        print("Please ensure image2.png is in the input folder.")


if __name__ == "__main__":
    main()

