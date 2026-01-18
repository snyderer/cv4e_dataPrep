import data_io as io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sp

label_file = r'./fx_labels_Bp_fixed.csv'
labels = pd.read_csv(label_file)

# Image parameters
twin_s = 2
x_span_pxl = 1250
x_overlap_pxl = int(x_span_pxl * 0.5)

# Pick a random label row
row = labels.iloc[np.random.randint(0, len(labels))]

# Find all labels in same dataset & time
same_labels = labels[
    (labels['source_file'] == row['source_file']) &
    (labels['t'] == row['t'])
]

# Load corresponding dataset + settings
settings_file = io.find_settings_h5(row['source_file'])
settings = io.load_settings_preprocessed_h5(settings_file)

nonzeros = settings['rehydration_info']['nonzeros_mask']
original_shape = settings['rehydration_info']['target_shape']

fk_dehyd, timestamp = io.load_preprocessed_h5(row['source_file'])
tx = 1e9 * io.rehydrate(fk_dehyd, nonzeros, original_shape, return_format='tx')

fs = settings['processing_settings']['fs']
dx = settings['processing_settings']['dx']

# Filter
b, a = sp.butter(10, 70 / (0.5 * fs), btype='low')
tx = sp.filtfilt(b, a, tx, axis=1)

# Define time window
num_samples = int(row['win_length_s'] * fs)
t_idx_start = int(row['t'] * fs)

# Extract TX segment & compute FFT
tx_seg = tx[:, t_idx_start:t_idx_start + num_samples]
fx_fullCable = np.abs(np.fft.rfft(tx_seg, axis=1))

# Plot full cable FX image
dpi = 100
fig_w = fx_fullCable.shape[1] / dpi  # width in inches (freq bins)
fig_h = fx_fullCable.shape[0] / dpi  # height in inches (spatial samples)

fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
plt.imshow(fx_fullCable, aspect='equal', origin='lower', vmin=0, vmax=30)

# Draw bounding boxes for all matching labels
for _, lab in same_labels.iterrows():
    # Pixel coords
    pix_xmin = int(lab['f_min_hz'] * num_samples / fs)
    pix_xmax = int(lab['f_max_hz'] * num_samples / fs)
    pix_ymin = int(lab['x_min_m'] / dx)
    pix_ymax = int(lab['x_max_m'] / dx)

    rect = plt.Rectangle(
        (pix_xmin, pix_ymin),
        pix_xmax - pix_xmin,
        pix_ymax - pix_ymin,
        linewidth=2, edgecolor='r', facecolor='none'
    )
    plt.gca().add_patch(rect)

# No axes, save
plt.axis('off')
plt.tight_layout(pad=0)
fig.savefig('test_fx_full_boxes.png', dpi=dpi)
plt.close(fig)

print(f"Saved image: test_fx_full_boxes.png")
print(f"Labels plotted: {len(same_labels)} from file {row['source_file']} at t={row['t']} seconds.")