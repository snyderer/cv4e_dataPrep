# for now, putting everything into one file. Can break out components later if it's convenient.
import data_io as io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sp
import os
import sqlite3
import ast
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_dilation
import torch
import torch.nn.functional as F

# class Loader: # skipping the Loader class for simplicity. Might be good to incorporate later
#     def __init__():
#         # load config file? 
#         return 0
#     def __iter__():
#         return 0
    
# TODO load config for settings. For now, get settings in here:

###############################
# temporary loading and setup
###############################
# data_path = 'F:/'
# labels_db_path = r'C:\Users\ers334\Documents\databases\DAS_Annotations\A25.db'
data_path = r'/mnt/class_data/esnyder/raw_data'
labels_db_path = r'/mnt/class_data/esnyder/segmentation_data/A25.db'

###############################
# Define methods
###############################
def generate_2D_mask(tx_img, x_extent, t_extent, x_lab, t_lab, 
                     t_win = (.5, 1.5), gf_sigma=5.0, tolerance = 0.1,
                     dilation_size = (53, 53), return_blurred = False):
    
    """
    tx_img: 2D ndarray [x, t]
    x_extent: total distance extent in meters
    t_extent: total time extent in seconds
    x_lab, t_lab: arrays of label contour coordinates in meters and seconds
    t_win: allowable time deviation (s) around t_lab (t_win[0] is prior to contour, t_win[1] is after contour)
    gf_sigma, tolerance, dilation_size: parameters for the blurring
    """
    if len(t_win)==1:
        t_win = (t_win, t_win)

    ################
    # zero out portion of tx_img that lies outside t_lab +/- t_win:
    nx, nt = tx_img.shape
    dx = x_extent / nx
    dt = t_extent / nt

    # convert label coords to pixel indices
    x_idx_lab = np.round(np.array(x_lab) / dx).astype(int)
    t_idx_lab = np.round(np.array(t_lab) / dt).astype(int)
    n_win = np.array(np.abs(t_win))//dt

    contour_mask = np.zeros_like(tx_img, dtype=bool)
    for xi, ti in zip(x_idx_lab, t_idx_lab):
        t_idx_min = int(np.max((ti - n_win[0], 0)))
        t_idx_max = int(np.min((ti + n_win[1], nt)))
        contour_mask[xi, t_idx_min:t_idx_max] = 1        

    if not return_blurred:
        return contour_mask
    
    mask = gaussian_filter(np.abs(tx_img*contour_mask), sigma=gf_sigma)
    mask = np.where(mask>tolerance, 1, 0)

    mask = binary_dilation(mask, structure=np.ones(dilation_size))

    return  mask

def map_mask_to_3D():
    return 0


def compute_spectrograms(tx, fs, n_fft=256, hop_length=None, window=None):
    # Select device automatically
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Make window if not supplied
    if window is None:
        window = torch.hann_window(n_fft, periodic=True, device=device)
    # Convert numpy array -> torch tensor and move to device
    # tx shape: (channels, samples)
    signals = torch.from_numpy(tx).float().to(device)

    # Batch STFT: signals shape [batch, time] → specs shape [batch, freq, time]
    specs = torch.stft(
        signals,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True   # complex tensor for easy magnitude/phase
    )

    # Convert to magnitude if desired
    specs_mag = specs.abs()

    # If you want result back on CPU
    specs_mag = specs_mag.cpu()

    return specs_mag

#############################################################################
# define settings
#############################################################################
# Spectrogram params:
n_fft = 200
hop_length = 40
window = np.hanning(n_fft).astype(np.float32)  # NumPy Hann on CPU
# try:
window = torch.tensor(window, dtype=torch.float32).to("cuda")
# except Exception as e:
#     print(e)
#     print('CUDA didn''t work. Running on CPU instead.')
    
pool_method = 'max' # max, mean, or median
pool_size = 10
#############################################################################
# load labels
#############################################################################
query = (
    "SELECT * "
    "FROM tx_labels "
    "WHERE label_name IN ('Bp_B', 'Bp_A');"
)   # TODO species labels should be a configurable parameter

conn = sqlite3.connect(labels_db_path)
cur = conn.cursor()

cur.execute(query)
rows = cur.fetchall()

labels = pd.read_sql_query(query, conn)


###################################################
# Set up looping for going over datasets and files
###################################################

# get list of unique datasets in labels:
datasets = labels['dataset'].unique()
det_num = 0
for dataset in datasets:

    rows_in_dataset = labels.where(labels['dataset']==dataset) 
    dataset_path = os.path.normpath(os.path.join(data_path, dataset))

    # load settings for this dataset:
    settings = io.load_settings_preprocessed_h5(os.path.join(dataset_path, 'settings.h5'))  
    nonzeros = settings['rehydration_info']['nonzeros_mask']
    nonzeros = settings['rehydration_info']['nonzeros_mask']
    original_shape = settings['rehydration_info']['target_shape']
    dx = settings['processing_settings']['dx']
    fs = settings['processing_settings']['fs']
    dt = 1/fs
    dt_new = hop_length/fs
    file_map = pd.DataFrame(settings['file_map'], columns=['timestamp', 'filename'])

    first_time = rows_in_dataset['apex_time'].min()
    last_time = rows_in_dataset['apex_time'].max()

    # find first and last source file for which labeling was done:
    t_last_label = rows_in_dataset['apex_time'].min()
    source_files = rows_in_dataset['source_file'].unique()

    first_file = rows_in_dataset.loc[
        rows_in_dataset['apex_time'].idxmin(), 'source_file'
    ]
    last_file = rows_in_dataset.loc[
        rows_in_dataset['apex_time'].idxmax(), 'source_file'
    ]

    # Find the positions of those start and end files in file_map
    start_idx = file_map.index[file_map['filename'] == first_file][0]
    end_idx = file_map.index[file_map['filename'] == last_file][0]

    files_in_range = file_map.loc[start_idx:end_idx, 'filename'].tolist()

    previous_labels = {'x_m': [], 't_s': [], 'det_num': []}

    for file in source_files[:2]: # TODO remove :2 to iterate over all files
        # load tx data
        fk_dehyd, timestamp = io.load_preprocessed_h5(os.path.join(dataset_path, file))        
        tx = 1e9 * io.rehydrate(fk_dehyd, nonzeros, original_shape, return_format='tx')

        x_extent = tx.shape[0]*dx
        t_extent = tx.shape[1]/fs

        mask = np.zeros_like(tx)
        
        # First, add any previous labels carried over from last file
        if previous_labels['x_m']:
            for x_m_prev, t_s_prev, det_num_prev in zip(previous_labels['x_m'], previous_labels['t_s'], previous_labels['det_num']):
                det_mask = generate_2D_mask(
                    tx, x_extent=x_extent, t_extent=t_extent,
                    x_lab=x_m_prev, t_lab=t_s_prev
                )
                mask[det_mask.astype(bool)] = det_num_prev
            # Clear after adding
            previous_labels = {'x_m': [], 't_s': [], 'det_num': []}

        rows_in_file = rows_in_dataset[rows_in_dataset['source_file'] == file]
        iter_next_window = 0
        for i, row in rows_in_file.iterrows():
            det_num += 1
            x_m = np.array(ast.literal_eval(row['x_m']))
            t_s = np.array(ast.literal_eval(row['t_s'])) + row['apex_time'] - timestamp

            idx_next_window = t_s > t_extent
            if any(idx_next_window):
                previous_labels['t_s'].append(t_s[idx_next_window] - t_extent)
                previous_labels['x_m'].append(x_m[idx_next_window])
                previous_labels['det_num'].append(det_num)
                x_m = x_m[~idx_next_window]
                t_s = t_s[~idx_next_window]
            det_mask = generate_2D_mask(tx, x_extent=x_extent, t_extent=t_extent, x_lab=x_m, t_lab=t_s)
            mask[det_mask.astype(bool)] = det_num
    
        ##### calc spectrograms and reshape data ######
        specs = compute_spectrograms(tx, fs, n_fft=256, hop_length=hop_length, window=None)
       
        ####### dimension reduction #########
        specs = specs.abs().squeeze().contiguous()  # linear amplitude, remove dims

        group_size = pool_size
        n_channels, n_freq_bins, n_time_frames = specs.shape
        x_new = np.arange(0, n_channels)*dx
        f_new = np.arange(0, n_freq_bins)*fs/(2*n_freq_bins)
        t_new = np.arange(0, n_time_frames)*dt_new
        
        # TODO RESUME HERE!!!!!!!!!!!!!!!!!!!!!!!
        # truncate in frequency dimension to reduce data size. 
        # Save x_new (truncated after remainder is removed), f_new (truncated), t_new 
        # Map original mask onto x_new, t_new space to create new mask.
        # Save 
        
        # Trim channels so we can group evenly
        remainder = n_channels % group_size
        if remainder != 0:
            specs = specs[:n_channels - remainder]
            n_channels = specs.shape[0]

        n_groups = n_channels // group_size

        # Reshape to (n_groups, group_size, freq, time) and max over group_size
        specs_grouped = specs.view(n_groups, group_size, n_freq_bins, n_time_frames)
        
        if pool_method=='max':
            pooled = specs_grouped.max(dim=1).values      
            pooled_cpu = pooled.to('cpu').numpy() 
        elif pool_method=='mean':
            pooled = specs_grouped.mean(dim=1)
            pooled_cpu = pooled.to('cpu').numpy()
        elif pool_method=='median':
            pooled = specs_grouped.median(dim=1).values 
            pooled_cpu = pooled.to('cpu').numpy()
        
        if True: # plotting some slices. Set to false for full run
            # Pick indices to slice at
            freq_idx = 25               # example frequency bin
            chan_group_idx = 400          # example channel group
            time_idx = 100               # example time frame

            fig, axs = plt.subplots(1, 3, figsize=(15, 4))

            # 1️⃣ TIME–DISTANCE slice (fix freq)
            img1 = axs[0].imshow(
                pooled_cpu[:, freq_idx, :],   # n_groups × n_time_frames
                origin='lower',
                aspect='auto',
                cmap='viridis'
            )
            axs[0].set_title(f"Time–Distance (Freq bin {freq_idx})")
            axs[0].set_xlabel("Time frame")
            axs[0].set_ylabel("Channel group (distance)")
            fig.colorbar(img1, ax=axs[0])

            # 2️⃣ TIME–FREQ slice (fix channel group)
            img2 = axs[1].imshow(
                pooled_cpu[chan_group_idx, :, :],  # n_freq_bins × n_time_frames
                origin='lower',
                aspect='auto',
                cmap='viridis'
            )
            axs[1].set_title(f"Time–Freq (Group {chan_group_idx})")
            axs[1].set_xlabel("Time frame")
            axs[1].set_ylabel("Frequency bin")
            fig.colorbar(img2, ax=axs[1])

            # 3️⃣ FREQ–DISTANCE slice (fix time)
            img3 = axs[2].imshow(
                pooled_cpu[:, :, time_idx],        # n_groups × n_freq_bins
                origin='lower',
                aspect='auto',
                cmap='viridis'
            )
            axs[2].set_title(f"Freq–Distance (Time frame {time_idx})")
            axs[2].set_xlabel("Frequency bin")
            axs[2].set_ylabel("Channel group (distance)")
            fig.colorbar(img3, ax=axs[2])

            plt.tight_layout()
            plt.savefig('slices_' + pool_method + 'Pool.png')

        ok = 1