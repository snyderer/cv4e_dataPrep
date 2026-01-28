# for now, putting everything into one file. Can break out components later if it's convenient.
import data_io as io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sp
import os
import sqlite3
import ast
from scipy.ndimage import median_filter, gaussian_filter, binary_dilation, uniform_filter1d
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
save_path = r'/mnt/class_data/esnyder/segmentation_data/'
#############################################################################
# define settings
#############################################################################
# Spectrogram params:
n_fft = 200
f_band = [10, 60]
hop_length = 40
window = np.hanning(n_fft).astype(np.float32)  
window = torch.tensor(window, dtype=torch.float32).to("cuda")

# Mask generation settings:
t_win = (.75, 1.5)
gf_sigma=5.0
tolerance = 0.1
dilation_size = (53, 53) 
shaping_method = None  # None (fixed width), 'blur', or 'hug'

# Pooling settings:
pool_method = 'mean' # max, mean, or median
pool_size = 10

# save loc:
shapestr = shaping_method if shaping_method != None else 'fixed_width'
save_loc = os.path.join(save_path, shapestr)
desc = """ 
"""
###############################
# Define methods
###############################
def generate_2D_mask(tx_img, x_extent, t_extent, x_lab, t_lab, 
                     t_win = (.5, 1.5), gf_sigma=5.0, tolerance = 0.1,
                     dilation_size = (53, 53), shaping_method = None):
    
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

    if shaping_method in {None, 'fixed', 'fixed_width'}:
        return contour_mask
    
    if shaping_method == 'blur':
        # blur the energy inside the countour_mask window and keep values > tolerance
        mask = gaussian_filter(np.abs(tx_img*contour_mask), sigma=gf_sigma)
        mask = np.where(mask>tolerance, 1, 0)

        mask = binary_dilation(mask, structure=np.ones(dilation_size))
        return  mask
    
    if shaping_method in {'hug', 'hugging'}:
        mask = np.zeros_like(contour_mask, dtype=bool)
        masked_img = np.abs(contour_mask*tx_img)
        masked_img -= np.min(masked_img)
        max_val = np.max(masked_img)
        if max_val > 0:
            masked_img /= max_val
        idx_min_list = []
        idx_max_list = []
        for xi in x_idx_lab:
            masked_row = masked_img[xi, :] 
            
            indices = np.where(masked_row > tolerance)[0]
            if len(indices)>1:
                idx_min_list[xi] = np.min(indices)
                idx_max_list[xi] = np.max(indices)
            else:
                max_idx = int(np.argmax(masked_row))
                idx_min_list.append(max_idx)
                idx_max_list.append(max_idx)
        
        # smooth out mask:
        mask1 = binary_dilation(mask, structure=np.ones(dilation_size))
        mask2 = gaussian_filter(np.abs(tx_img*contour_mask), sigma=gf_sigma)

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)  # row=1, col=2, index=1
        plt.imshow(mask1)
        plt.title("binary dilation")

        plt.subplot(1, 2, 2)  # row=1, col=2, index=2
        plt.imshow(mask2>tolerance)
        plt.title("guass filter")
        plt.savefig('test.png')
        return mask


def map_mask_to_reduced_size(mask_orig, t_orig, x_orig, t_new, x_new):
    """
    Downsample integer detection-ID mask to match reduced space×time size.

    Parameters
    ----------
    mask_orig : ndarray (n_x_orig, n_t_orig)
        Original integer detection-number mask (0 = no detection).
    t_orig : ndarray
        Time coordinates of original frames (length n_t_orig).
    x_orig : ndarray
        Spatial coordinates of original channels (length n_x_orig).
    t_new : ndarray
        Time coordinates of reduced output (length n_t_new).
    x_new : ndarray
        Spatial coordinates of reduced output (length n_x_new).

    Returns
    -------
    mask_new : ndarray (n_x_new, n_t_new)
        Reduced mask, each pixel holding the most frequent detection number
        from its corresponding original block (0 if no detection).
    """
    n_x_new = len(x_new)
    n_t_new = len(t_new)
    mask_new = np.zeros((n_x_new, n_t_new), dtype=mask_orig.dtype)

    dx_new = (x_new[1] - x_new[0]) if n_x_new > 1 else x_orig[-1] - x_orig[0]
    dt_new = (t_new[1] - t_new[0]) if n_t_new > 1 else t_orig[-1] - t_orig[0]

    half_dx = dx_new / 2.0
    half_dt = dt_new / 2.0

    for ix_new, x_center in enumerate(x_new):
        x_min = x_center - half_dx
        x_max = x_center + half_dx
        idx_x = np.where((x_orig >= x_min) & (x_orig < x_max))[0]

        for it_new, t_center in enumerate(t_new):
            t_min = t_center - half_dt
            t_max = t_center + half_dt
            idx_t = np.where((t_orig >= t_min) & (t_orig < t_max))[0]

            if len(idx_x) > 0 and len(idx_t) > 0:
                block = mask_orig[np.ix_(idx_x, idx_t)]
                # Find most frequent nonzero detection ID
                values, counts = np.unique(block[block > 0], return_counts=True)
                if len(values) > 0:
                    mask_new[ix_new, it_new] = values[np.argmax(counts)]
                # else remains 0 (no detection)

    return mask_new


def compute_spectrograms(tx, fs, n_fft=256, hop_length=None, window=None):
    # Select device automatically
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")

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

    specs_mag = specs.abs()
    specs_mag = specs_mag.cpu()

    # Frequency bins (real input, one-sided spectrum)
    freqs = np.arange(0, n_fft // 2 + 1) * fs / n_fft

    # Time stamps for each STFT column
    times = np.arange(specs.size(-1)) * hop_length / fs

    return times, freqs, specs_mag

def save_settings(n_fft, f_band, hop_length, t_win, gf_sigma, tolerance,
                  dilation_size, shaping_method, pool_method, pool_size,
                  save_loc, desc):
    """
    Saves spectrogram, mask generation, and pooling settings into a text file
    in the specified save directory.
    """

    # Compute window
    window = np.hanning(n_fft).astype(np.float32)
    try:
        window_torch = torch.tensor(window, dtype=torch.float32).to("cuda")
        cuda_status = "CUDA available: using GPU"
    except Exception as e:
        cuda_status = f"CUDA not available ({str(e)}); using CPU"

    # Ensure output directory exists
    os.makedirs(save_loc, exist_ok=True)

    # Prepare filepath
    settings_file = os.path.join(save_loc, "settings.txt")

    # Write the settings
    with open(settings_file, 'w') as f:
        f.write("# Spectrogram params:\n")
        f.write(f"n_fft = {n_fft}\n")
        f.write(f"f_band = {f_band}\n")
        f.write(f"hop_length = {hop_length}\n")
        f.write("window = 'NumPy Hann window, float32'\n")
        f.write(f"{cuda_status}\n\n")

        f.write("# Mask generation settings:\n")
        f.write(f"t_win = {t_win}\n")
        f.write(f"gf_sigma = {gf_sigma}\n")
        f.write(f"tolerance = {tolerance}\n")
        f.write(f"dilation_size = {dilation_size}\n")
        f.write(f"shaping_method = {shaping_method}\n\n")

        f.write("# Pooling settings:\n")
        f.write(f"pool_method = {pool_method}\n")
        f.write(f"pool_size = {pool_size}\n\n")

        f.write("# Save location:\n")
        f.write(f"save_loc = {save_loc}\n\n")

        f.write("# Description:\n")
        f.write(desc.strip() + "\n")

    print(f"Settings saved to {settings_file}")
    return settings_file

def save_sample(out_dir, sample_id, pooled_tensor, mask_reduced):
    """
    Save pooled tensor and mask in [channels, height, width] format.

    pooled_tensor: np.ndarray shape (distance, freq, time) after pooling
    mask_reduced: np.ndarray shape (distance, time) with class IDs
    """
    
    img_dir = os.path.join(out_dir, 'imgs')
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    mask_dir = os.path.join(out_dir, 'masks')
    if not os.path.isdir(mask_dir):
        os.mkdir(mask_dir)
    
    # Convert pooled to channels-first: (freq, distance, time)
    pooled_cf = np.transpose(pooled_tensor, (1, 0, 2))  # freq, dist, time

    # Convert to torch
    pooled_cf_torch = torch.from_numpy(pooled_cf).float()
    mask_torch = torch.from_numpy(mask_reduced).long()

    os.makedirs(out_dir, exist_ok=True)

    # Save dict
    torch.save(pooled_cf_torch,  # [channels=freq, height=dist, width=time]
        os.path.join(img_dir, f"{sample_id}.pt")
    )
    
    torch.save( mask_torch,         # [height=dist, width=time]
        os.path.join(mask_dir, f"{sample_id}.pt")
    )
    


save_settings(n_fft, f_band, hop_length, t_win, gf_sigma, tolerance,
                  dilation_size, shaping_method, pool_method, pool_size,
                  save_loc, desc)

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
datasets = labels['dataset'].unique()   # TODO: this could potentially fail if two files from different paths have same name
det_num = 0
for dataset in datasets:
    rows_in_dataset = labels[labels['dataset'] == dataset]
    dataset_path = os.path.normpath(os.path.join(data_path, dataset))
   
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

    for file in source_files: 
        data_filepath = os.path.join(dataset_path, file)
        if not os.path.isfile(data_filepath):
            # some labels erroneous -- don't correspond to actual files. Skip.
            continue
        
        # load tx data
        fk_dehyd, timestamp = io.load_preprocessed_h5(data_filepath)        
        tx = 1e9 * io.rehydrate(fk_dehyd, nonzeros, original_shape, return_format='tx')

        x_extent = tx.shape[0]*dx
        t_extent = tx.shape[1]/fs

        mask = np.zeros_like(tx)
        
        # First, add any previous labels carried over from last file
        if previous_labels['x_m']:
            for x_m_prev, t_s_prev, det_num_prev in zip(previous_labels['x_m'], previous_labels['t_s'], previous_labels['det_num']):
                det_mask = generate_2D_mask(
                    tx, x_extent=x_extent, t_extent=t_extent,
                    x_lab=x_m_prev, t_lab=t_s_prev,
                    t_win=t_win, gf_sigma=gf_sigma, tolerance=tolerance,
                    dilation_size=dilation_size, shaping_method=shaping_method
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
            det_mask = generate_2D_mask(tx, x_extent=x_extent, t_extent=t_extent, 
                                        x_lab=x_m, t_lab=t_s,
                                        t_win=t_win, gf_sigma=gf_sigma, tolerance=tolerance,
                                        dilation_size=dilation_size, shaping_method=shaping_method)
            mask[det_mask.astype(bool)] = det_num
    
        ##### calc spectrograms and reshape data ######
        t_new, f_new, specs = compute_spectrograms(tx, fs, n_fft=n_fft, hop_length=hop_length, window=None)
       
        ####### dimension reduction #########
        group_size = pool_size
        n_xseg, n_freq_bins, n_time_frames = specs.shape
        x_orig_trimmed = np.arange(0, n_xseg)*dx
                        
        # Trim channels so they can be grouped evenly
        remainder = n_xseg % group_size
        if remainder != 0:
            specs = specs[:n_xseg - remainder]
            mask = mask[:mask.shape[0] - remainder, :]
            n_xseg = specs.shape[0]
            x_orig_trimmed = x_orig_trimmed[:n_xseg - remainder]

        # truncate in frequency dimension 
        band_idx = np.where((f_new >= f_band[0]) & (f_new <= f_band[1]))[0]
        specs = specs[:, band_idx, :]
        f_new = f_new[band_idx]
        n_freq_bins = len(f_new)
        
        # Reshape to (n_groups, group_size, freq, time) and max over group_size
        n_groups = n_xseg // group_size
        specs_grouped = specs.view(n_groups, group_size, n_freq_bins, n_time_frames)
        
        if pool_method=='max':
            pooled = specs_grouped.max(dim=1).values      
            pooled = pooled.to('cpu').numpy() 
        elif pool_method=='mean':
            pooled = specs_grouped.mean(dim=1)
            pooled = pooled.to('cpu').numpy()
        elif pool_method=='median':
            pooled = specs_grouped.median(dim=1).values 
            pooled = pooled.to('cpu').numpy()
        
        if True: # plotting some slices. Set to false for full run
            # Pick indices to slice at
            freq_idx = np.argmin(np.abs(f_new - 25))
            chan_group_idx = 400          # example channel group
            time_idx = 100               # example time frame

            fig, axs = plt.subplots(1, 3, figsize=(15, 4))

            # 1️⃣ TIME–DISTANCE slice (fix freq)
            img1 = axs[0].imshow(
                pooled[:, freq_idx, :],   # n_groups × n_time_frames
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
                pooled[chan_group_idx, :, :],  # n_freq_bins × n_time_frames
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
                pooled[:, :, time_idx],        # n_groups × n_freq_bins
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

        # Original mask coords
        t_orig_coords = np.arange(mask.shape[1]) * dt       # seconds for each time pixel
        x_orig_coords = np.arange(mask.shape[0]) * dx       # meters for each channel

        # Compute center position for each group
        x_new = np.array([
            x_orig_trimmed[i * group_size : (i + 1) * group_size].mean()
            for i in range(n_groups)
        ])        
        
        # map mask to new dimensions
        mask_reduced = map_mask_to_reduced_size(mask, 
                                        t_orig_coords, x_orig_coords,
                                        t_new, x_new)

        # force mask to be same dimensions as pooled data:
        x_dif = mask_reduced.shape[0] - pooled.shape[0] 
        t_dif = mask_reduced.shape[1] - pooled.shape[2] 
        if x_dif>0:
            mask_reduced = mask_reduced[:-x_dif, :]
        if x_dif<0:
            # Repeat last row n times along the row axis (axis=0)
            repeated_rows = np.repeat(mask_reduced[-1:, :], np.abs(x_dif), axis=0)

            # Concatenate original array with repeated columns
            mask_reduced = np.vstack((mask_reduced, repeated_rows))

        if t_dif>0:
            mask_reduced = mask_reduced[:, :-t_dif]
        if t_dif<0:
            # Repeat last col n times along the columns axis (axis=1)
            repeated_cols = np.repeat(mask_reduced[:, -1:], np.abs(t_dif), axis=1)

            # Concatenate original array with repeated columns
            mask_reduced = np.hstack((mask_reduced, repeated_cols))

        filename = dataset + '_' + file[:-3]
        
        save_sample(
            out_dir=save_loc,
            sample_id=filename,
            pooled_tensor=pooled,           # shape (dist, freq, time)
            mask_reduced=mask_reduced        # shape (dist, time)
        )
        print('file ' + file + ' in dataset ' + dataset + ' processed.')