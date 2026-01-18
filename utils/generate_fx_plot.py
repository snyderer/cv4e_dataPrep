import data_io as io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sp

# ================================
# Load labels
# ================================
label_file = r'./fx_labels_Bp_fixed.csv'
labels = pd.read_csv(label_file)

# Image parameters (from your original script)
twin_s = 2           # seconds
Nfft = 1024
x_span_pxl = 1250    # =10 km
x_overlap_pxl = int(x_span_pxl * 0.5)

# ================================
# Pick random row from labels CSV
# ================================
row = labels.iloc[np.random.randint(0, len(labels))]

# ================================
# Load associated HDF5 data
# ================================
settings_file = io.find_settings_h5(row['source_file'])
settings = io.load_settings_preprocessed_h5(settings_file)

nonzeros = settings['rehydration_info']['nonzeros_mask']
original_shape = settings['rehydration_info']['target_shape']

# Load and rehydrate FK/Tx data
fk_dehyd, timestamp = io.load_preprocessed_h5(row['source_file'])
tx = 1e9 * io.rehydrate(fk_dehyd, nonzeros, original_shape, return_format='tx')

# Sampling settings
fs = settings['processing_settings']['fs']  # Hz
dx = settings['processing_settings']['dx']  # meters per spatial sample

# Filtering
b, a = sp.butter(10, 70 / (0.5 * fs), btype='low')
tx = sp.filtfilt(b, a, tx, axis=1)

# ================================
# Define time window for segment
# ================================
num_samples = int(row['win_length_s'] * fs)      # samples in time window
t_idx_start = int(row['t'] * fs)

# Extract TX segment for FFT
tx_seg = tx[:, t_idx_start:t_idx_start + num_samples]
fx_fullCable = np.abs(np.fft.rfft(tx_seg, axis=1, n=Nfft))

# ================================
# Convert label to pixel coordinates
# ================================
# Frequency axis (x-axis in FFT output)
box_pxl_x_min = int(row['f_min_hz'] * Nfft / fs)
box_pxl_x_max = int(row['f_max_hz'] * Nfft / fs)

# Spatial axis (y-axis)
box_pxl_y_min = int(row['x_min_m'] / dx)
box_pxl_y_max = int(row['x_max_m'] / dx)

# ================================
# Plot raw FX image with bounding box
# ================================
dpi = 100
fig_w = fx_fullCable.shape[1] / dpi  # width in inches
fig_h = fx_fullCable.shape[0] / dpi  # height in inches

fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
plt.imshow(fx_fullCable, aspect='equal', origin='lower', vmin=0, vmax=25)

# Bounding box
rect = plt.Rectangle(
    (box_pxl_x_min, box_pxl_y_min),
    box_pxl_x_max - box_pxl_x_min,
    box_pxl_y_max - box_pxl_y_min,
    linewidth=2, edgecolor='r', facecolor='none'
)
plt.gca().add_patch(rect)

# Remove axes
plt.axis('off')
plt.tight_layout(pad=0)

# Save image with exact pixel mapping
fig.savefig('test_fx_full.png', dpi=dpi)
plt.close(fig)

print(f"Saved test image: test_fx_full.png")
print(f"Fx matrix shape: {fx_fullCable.shape} -> Pixel size matches exactly.")
print(f"Bounding box pixels: X[{box_pxl_x_min}:{box_pxl_x_max}], Y[{box_pxl_y_min}:{box_pxl_y_max}]")