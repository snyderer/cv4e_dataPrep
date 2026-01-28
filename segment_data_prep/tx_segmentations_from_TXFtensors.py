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

imgs_data_path = r'/mnt/class_data/esnyder/segmentation_data/fixed_width/imgs'
labels_db_path = r'/mnt/class_data/esnyder/segmentation_data/A25.db'
raw_data_path = r'/mnt/class_data/esnyder/raw_data'

# output filepaths for segmentation:
seg_imgs_out_path = r'/mnt/class_data/esnyder/yolo_data/tx_segmentation/images'
seg_labels_out_path = r'/mnt/class_data/esnyder/yolo_data/tx_segmentation/labels'

# output filepaths for bounding boxes:
bb_imgs_out_path = r'/mnt/class_data/esnyder/yolo_data/fx_boundingbox/images'
bb_labels_out_path = r'/mnt/class_data/esnyder/yolo_data/fx_boundingbox/labels'


# Ensure output directories exist
Path(seg_imgs_out_path).mkdir(parents=True, exist_ok=True)
Path(seg_labels_out_path).mkdir(parents=True, exist_ok=True)
Path(bb_imgs_out_path).mkdir(parents=True, exist_ok=True)
Path(bb_labels_out_path).mkdir(parents=True, exist_ok=True)

#############################################################################
# settings
#############################################################################
file_duration = 30 # file durations # TODO make this extracted from data?
dx = 8*10 # TODO get this from settings (need to save settings in reshape_and_make_mask.py)
dt = 40/200 # TODO 
t_win = [.75, 1.5]

f_low, f_high = 12, 35  # band of whale calls, Hz
min_freq, max_freq = 10, 60 # band of img data

#############################################################################
# load labels
#############################################################################
# Load labels to get apexes or apices or whatever
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

#############################################################################
# Define some methods
#############################################################################
def get_filemaps(data_path, datasets):
    file_maps = {}
    for i, dataset in enumerate(datasets):
        # Find the settings.h5 file in the dataset folder
        matches = glob.glob(os.path.join(data_path, dataset, 'settings.h5'))
        if not matches:
            raise FileNotFoundError(f"No settings.h5 found in {os.path.join(data_path, dataset)}")
        settings_path = matches[0]  # Take the first match

        # Load the settings file (your function must accept a path string)
        settings = io.load_settings_preprocessed_h5(settings_path)

        # Extract the filemap and store it
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
            poly_str = ' '.join([str(p) for p in norm_poly])
            f.write(f"{class_id} {poly_str}\n")            
#############################################################################
# Set up file lists and mappings
#############################################################################

# get list of existing datasets:
img_files = glob.glob(imgs_data_path + '/*.pt')

datasets = labels['dataset'].unique()
file_maps = get_filemaps(raw_data_path, datasets)

#############################################################################
# Iterate and generate images/labels
#############################################################################

for file in img_files:
    # I. load data:
    img = torch.load(file)
    
    ## II. find apices within timeframe of image
    
    # Find timestamp of dataset: 
    dataset_name = re.sub(r'_\d{8}_\d{6}$', '', Path(file).stem)
    file_name =  Path(file).stem[-15:]
    timestamp = get_timestamp(file_maps, dataset_name, file_name)
    
    labels_in_datachunk = labels[
        (labels['apex_time'] >= timestamp) &
        (labels['apex_time'] <= timestamp + file_duration)
        ]

  # --- Segmentation output ---
    summed = img.sum(dim=0)  # nx x nt
    img_h, img_w = summed.shape
    seg_img_name = f"{Path(file).stem}.png"
    plt.imsave(Path(seg_imgs_out_path) / seg_img_name, summed.cpu().numpy())

    seg_polygons = []
    for _, label in labels_in_datachunk.iterrows():
        x_m = np.array(eval(label['x_m']), dtype=float)
        t_s = np.array(eval(label['t_s']), dtype=float) + label['apex_time'] - timestamp
        t_s_down = t_s - t_win[0]
        t_s_up = t_s + t_win[1]
        t_s_full = np.concatenate([t_s_down, np.flip(t_s_up)])
        x_m_full = np.concatenate([x_m, np.flip(x_m)])

        x_pxl = [int(x1/dx) for x1 in x_m_full]
        t_pxl = [int(t1/dt) for t1 in t_s_full]
        poly_points = list(zip(t_pxl, x_pxl))
        seg_polygons.append(poly_points)

    save_segmentation_labels(Path(seg_labels_out_path) / f"{Path(file).stem}.txt",
                             seg_polygons, class_id=0,
                             img_w=img_w, img_h=img_h)

    # --- Bounding box output ---
    nf, nx, nt = img.shape
    fi_low = int((f_low - min_freq) / ((max_freq)/nf))
    fi_high = int((f_high - min_freq) / ((max_freq)/nf))

    for ti in range(nt):
        fx_slice = img[:, :, ti]
        bb_img_name = f"{Path(file).stem}_t{ti}.png"
        plt.imsave(Path(bb_imgs_out_path) / bb_img_name, fx_slice.cpu().numpy())

        boxes = []
        for _, label in labels_in_datachunk.iterrows():
            x_m = np.array(eval(label['x_m']), dtype=float)
            t_s = np.array(eval(label['t_s']), dtype=float) + label['apex_time'] - timestamp
            # Find contour distance min/max at this time
            if np.min(t_s) <= ti*dt <= np.max(t_s):
                dist_min = min(eval(label['x_m']))
                dist_max = max(eval(label['x_m']))
                xi_min, xi_max = int(dist_min/dx), int(dist_max/dx)
                boxes.append((fi_low, xi_min, fi_high, xi_max))

        save_bbox_labels(Path(bb_labels_out_path) / f"{Path(file).stem}_t{ti}.txt",
                         boxes, class_id=0,
                         img_w=nf, img_h=nx)
    print(dataset_name + ' complete')