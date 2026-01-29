import glob
from pathlib import Path
import os
import re
import sqlite3
import numpy as np
import pandas as pd
import torch
import data_io as io
import matplotlib.pyplot as plt
from collections import defaultdict
import ast

#############################################################################
# Paths
#############################################################################
imgs_data_path = r'/mnt/class_data/esnyder/segmentation_data/fixed_width/imgs'
labels_db_path = r'/mnt/class_data/esnyder/segmentation_data/A25.db'
raw_data_path = r'/mnt/class_data/esnyder/raw_data'

seg_imgs_out_path = r'/mnt/class_data/esnyder/yolo_data/tx_segmentation/images'
seg_labels_out_path = r'/mnt/class_data/esnyder/yolo_data/tx_segmentation/labels'
bb_imgs_out_path = r'/mnt/class_data/esnyder/yolo_data/fx_boundingbox/images'
bb_labels_out_path = r'/mnt/class_data/esnyder/yolo_data/fx_boundingbox/labels'

# Ensure output directories exist
for p in [seg_imgs_out_path, seg_labels_out_path, bb_imgs_out_path, bb_labels_out_path]:
    Path(p).mkdir(parents=True, exist_ok=True)

#############################################################################
# Settings
#############################################################################
file_duration = 30
dx = 8 * 10
dt = 40 / 200
t_win = [.75, 1.5]
c = 1500 # speed of sound (m/s)
nx = int(100e3/dx)

clip_val_seg = 100
clip_val_bb = 10
f_low, f_high = 12, 35
min_freq, max_freq = 10, 60

#############################################################################
# Load labels
#############################################################################
query = """
    SELECT *
    FROM tx_labels
    WHERE label_name IN ('Bp_B', 'Bp_A')
    AND id IS NOT NULL
    AND LOWER(id) != 'nan';
    """


conn = sqlite3.connect(labels_db_path)
labels = pd.read_sql_query(query, conn)
labels = labels.dropna(subset=['id', 'x_m', 't_s'])
#############################################################################
# Helper functions
#############################################################################
def get_filemaps(data_path, datasets):
    file_maps = {}
    for dataset in datasets:
        matches = glob.glob(os.path.join(data_path, dataset, 'settings.h5'))
        if not matches:
            raise FileNotFoundError(f"No settings.h5 found in {os.path.join(data_path, dataset)}")
        settings = io.load_settings_preprocessed_h5(matches[0])
        file_maps[dataset] = settings['file_map']
    return file_maps

def get_timestamp(file_maps, dataset_name, file_name):
    filenames = file_maps[dataset_name]['filename'].astype(str)
    idx = np.char.find(filenames, file_name) >= 0
    timestamps = file_maps[dataset_name]['timestamp'][idx][0]
    return timestamps

def save_bbox_labels(label_path, boxes, class_id=0, img_w=None, img_h=None):
    with open(label_path, 'w') as f:
        for (xmin, ymin, xmax, ymax) in boxes:
            x_center = ((xmin + xmax) / 2) / img_w
            y_center = ((ymin + ymax) / 2) / img_h
            width = (xmax - xmin) / img_w
            height = (ymax - ymin) / img_h
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def save_segmentation_labels(label_path, polygons, class_id=0, img_w=None, img_h=None):
    with open(label_path, 'w') as f:
        for poly in polygons:
            norm_poly = []
            for (x, y) in poly:
                norm_poly.append(x / img_w)
                norm_poly.append(y / img_h)
            f.write(f"{class_id} {' '.join(map(str, norm_poly))}\n")

def pad_image_to_size(image, target_size):
    """
    Zero-pad a 2D image to the target_size (height, width).
    If the image is larger than target_size in any dimension, 
    it will be cropped.
    """
    target_h, target_w = target_size
    h, w = image.shape
    
    # Clip if larger than target
    if h > target_h:
        image = image[:target_h, :]
        h = target_h
    if w > target_w:
        image = image[:, :target_w]
        w = target_w
    
    # Calculate padding amounts
    pad_h = target_h - h
    pad_w = target_w - w
    
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # Apply padding
    padded_image = np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=0
    )
    return padded_image
#############################################################################
# Set up file lists and mappings
#############################################################################
img_files = sorted(glob.glob(imgs_data_path + '/*.pt'))
datasets = labels['dataset'].unique()
file_maps = get_filemaps(raw_data_path, datasets)


#############################################################################
# Iterate and generate images/labels with spillover handling
#############################################################################
labels['t_posix_s'] = [[] for _ in range(len(labels))]
for i, label in labels.iterrows():
    
    dataset_name = label['dataset']
    file_name = label['source_file']
    
    t_s = np.array(eval(label['t_s']), dtype=float) + label['apex_time']
    labels['t_posix_s'][i] = t_s

previous_labels = defaultdict(list)
det_no = 0

previous_labels = []  # store actual dict rows

for file in img_files:
    img = torch.load(file)
    nf, nx, nt = img.shape
    t_extent = nt * dt
    x_extent = nx * dx

    dataset_name = re.sub(r'_\d{8}_\d{6}$', '', Path(file).stem)
    file_name = Path(file).stem[-15:]
    timestamp = get_timestamp(file_maps, dataset_name, file_name)

    # Labels inside this file, from DB
    # labels_in_datachunk = labels[
    #     (labels['apex_time'] >= timestamp) &
    #     (labels['apex_time'] <= timestamp + file_duration)
    # ].copy()
    
    # Create a boolean mask — True if any value in list is within the range
    in_window = labels['t_posix_s'].apply(lambda arr: any(timestamp <= x <= timestamp + file_duration for x in arr))

    # Apply the mask to filter rows
    labels_in_datachunk = labels[in_window]
    
    # Inject previous spillover
    # if previous_labels:
    #     spill_df = pd.DataFrame(previous_labels)
    #     labels_in_datachunk = pd.concat([spill_df, labels_in_datachunk], ignore_index=True)
    #     previous_labels.clear()

    seg_polygons = []

    for _, label in labels_in_datachunk.iterrows():
        if pd.isna(label['id']):
            continue

        det_no += 1

        x_m = np.array(eval(label['x_m']), dtype=float)
        t_s = np.array(eval(label['t_s']), dtype=float) + label['apex_time'] - timestamp

        # Check for points spilling into next file
        # idx_next_window = t_s > t_extent
        # if any(idx_next_window):
        #     spill_row = {
        #         'id': label['id'],
        #         'x_m': str(list(x_m[idx_next_window])),
        #         't_s': str(list(t_s[idx_next_window] - t_extent)),
        #         'det_num': det_no,
        #         'apex_time': timestamp
        #     }
        #     previous_labels.append(spill_row)

            # Trim in-bounds portion for current polygons
        # x_m = x_m[~idx_next_window]
        # t_s = t_s[~idx_next_window]
                
        # Pad and create full contour
        t_s_down = t_s - t_win[0]
        t_s_up = t_s + t_win[1]
        t_s_full = np.concatenate([t_s_down, np.flip(t_s_up)])
        x_m_full = np.concatenate([x_m, np.flip(x_m)])
        
        # Clip to current file bounds
        in_bounds = (t_s_full >= 0) & (t_s_full <= t_extent)
        if not np.any(in_bounds):
            continue
        t_s_full = t_s_full[in_bounds]
        x_m_full = x_m_full[in_bounds]

        # Pixel coords
        x_pxl = [int(x1 / dx) for x1 in x_m_full]
        t_pxl = [int(t1 / dt) for t1 in t_s_full]
        seg_polygons.append(list(zip(t_pxl, x_pxl)))


    seg_img_name = f"{Path(file).stem}.png"

    # Save segmentation image
    summed = img.sum(dim=0)
    
    summed = img.sum(dim=0).cpu().numpy()
    summed = np.clip(summed, 0, clip_val_seg)

    # Zero pad to target size
    summed_padded = pad_image_to_size(summed, (int(nx), summed.shape[1]))

    plt.imsave(Path(seg_imgs_out_path) / seg_img_name, summed_padded)
    
    # plt.imsave(Path(seg_imgs_out_path) / seg_img_name,
    #            np.clip(summed.cpu().numpy(), 0, clip_val_seg))

    # Save segmentation labels
    save_segmentation_labels(Path(seg_labels_out_path) / f"{Path(file).stem}.txt",
                             seg_polygons, class_id=0,
                             img_w=nt, img_h=nx)

    # Bounding boxes # TODO does not work. Oh well.
    # fi_low = int((f_low - min_freq) / (max_freq / nf))
    # fi_high = int((f_high - min_freq) / (max_freq / nf))

    # for ti in range(nt):
    #     fx_slice = img[:, :, ti]
    #     bb_img_name = f"{Path(file).stem}_t{ti}.png"
    #     plt.imsave(Path(bb_imgs_out_path) / bb_img_name,
    #             np.clip(fx_slice.cpu().numpy(), 0, clip_val_bb))

    #     boxes = []
    #     t_curr = ti * dt

    #     for idx, label in labels_in_datachunk.iterrows():
    #         x_m = np.array(eval(label['x_m']), dtype=float)
    #         t_s = np.array(eval(label['t_s']), dtype=float) + label['apex_time'] - timestamp

    #         # Select points near time slice
    #         time_window = dt / 2
    #         idx_in_slice = np.abs(t_s - t_curr) <= time_window

    #         if not np.any(idx_in_slice):
    #             continue

    #         # Measured distance extent from actual contour points
    #         ext_min_measured = np.min(x_m[idx_in_slice])
    #         ext_max_measured = np.max(x_m[idx_in_slice])

    #         # Propagation-based extent from earliest and latest times in slice
    #         ext_min_physical = (np.min(t_s[idx_in_slice]) - t_win[0]) * c
    #         ext_max_physical = (np.max(t_s[idx_in_slice]) + t_win[1]) * c

    #         # Combine physically expected and actual contour width
    #         dist_min = min(ext_min_physical, ext_min_measured)
    #         dist_max = max(ext_max_physical, ext_max_measured)

    #         xi_min = int(dist_min / dx)
    #         xi_max = int(dist_max / dx)
    #         boxes.append((fi_low, xi_min, fi_high, xi_max))

    #     save_bbox_labels(Path(bb_labels_out_path) / f"{Path(file).stem}_t{ti}.txt",
    #                      boxes, class_id=0,
    #                      img_w=nf, img_h=nx)

    print(dataset_name, 'complete')