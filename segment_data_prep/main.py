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
data_path = 'F:/'
labels_db_path = r'C:\Users\ers334\Documents\databases\DAS_Annotations\A25.db'

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


# load labels
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
            