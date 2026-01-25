import torch
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def plot_sample(sample_path, save_name='saved_data.png'):
    # Load data
    data = torch.load(sample_path)
    x = data["input"]   # shape [C=freq, H=dist, W=time]
    y = data["mask"]    # shape [H=dist, W=time]

    print(f"Loaded {sample_path}")
    print(f"Input shape: {x.shape}, Mask shape: {y.shape}")

    # Convert to NumPy for plotting
    x_np = x.numpy()
    y_np = y.numpy()

    # Choose one frequency channel to visualize (like one color band in an image)
    # freq_idx = x_np.shape[0] // 2   # middle frequency
    # img_input = x_np[freq_idx, :, :]   # distance × time

    # sum over all frequencies
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    img_input = np.sum(x_np, axis=0)
    
    # Plot input
    im0 = axs[0].imshow(img_input, origin='lower', aspect='auto',
                        cmap='viridis', vmin=0, vmax=40)
    
    # axs[0].set_title(f"Input channel {freq_idx} (dist × time)")
    axs[0].set_title(f"Summed across all frequencies (dist × time)")
    fig.colorbar(im0, ax=axs[0])
    

    # Plot mask
    im1 = axs[1].imshow(y_np, origin='lower', aspect='auto', cmap='tab20')
    axs[1].set_title("Segmentation mask (class IDs)")
    fig.colorbar(im1, ax=axs[1])

    plt.tight_layout()
    plt.savefig(save_name)

def plot_overlay(sample_path):
    data = torch.load(sample_path)
    x_np = data["input"].numpy()
    y_np = data["mask"].numpy()

    freq_idx = x_np.shape[0] // 2   # middle frequency
    img_input = x_np[freq_idx, :, :]

    plt.figure(figsize=(6, 5))
    plt.imshow(img_input, origin='lower', aspect='auto', cmap='gray')
    plt.imshow(np.ma.masked_where(y_np == 0, y_np), origin='lower', aspect='auto', cmap='autumn', alpha=0.5)
    plt.title(f"Overlay: Input channel {freq_idx} + mask")
    plt.colorbar()
    plt.savefig('with_overlay.png')

# Load and plot a few files
sample_files = glob.glob("/mnt/class_data/esnyder/segmentation_data/hugging_masks_data/*.pt")[:3]  # first 3 files
iter = 0
for f in sample_files:
    save_name = 'saved_data_' + str(iter) + '.png'
    plot_sample(f, save_name)
    iter += 1