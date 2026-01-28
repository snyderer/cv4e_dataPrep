#############################################################################
# overview
#############################################################################
# I. load image data
# II. find apices within timeframe of image
# III. Iterate over each apex and:
#   A. make individual point gaussian heatmaps for each apex (DETxxx)
#   B. regenerate masks as individual images
#       1. either try to separate out existing contours based on det num
#       2. OR regenerate masks copying code from reshape_and_make_mask.py
# IV. make combined guassian heatmap (ALL_POINTS_xxxTOxxx)
#   A. take max OR sum OR median across all individual heatmap (probably max)

import sqlite3
import pandas as pd
import numpy as np
import os
import glob
import torch
import data_io as io
from pathlib import Path
import re
import matplotlib.pyplot as plt

imgs_data_path = r'/mnt/class_data/esnyder/segmentation_data/fixed_width/imgs'
masks_data_path = r'/mnt/class_data/esnyder/segmentation_data/fixed_width/masks'
raw_data_path = r'/mnt/class_data/esnyder/raw_data'
labels_db_path = r'/mnt/class_data/esnyder/segmentation_data/A25.db'
save_path = r'/mnt/class_data/esnyder/segmentation_data/apex_then_segment'

os.makedirs(save_path, exist_ok=True)
#############################################################################
# settings
#############################################################################
sigma = 5 # std of gaussian heatmap
file_duration = 30 # file durations # TODO make this extracted from data?
dx = 8*10 # TODO get this from settings (need to save settings in reshape_and_make_mask.py)
dt = 40/200 # TODO 


sigma_x = 7
sigma_y = 7
mode = 'max'
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


import numpy as np

def add_gaussian_to_image(image, center_x, center_y, sigma_x=3.0, sigma_y=3.0, mode='add', amplitude=1.0):
    """
    Add or max a 2D Gaussian onto the given image.
    
    Parameters
    ----------
    image : np.ndarray
        2D numpy array representing the image.
    center_x : int
        X position (column index) for Gaussian center.
    center_y : int
        Y position (row index) for Gaussian center.
    sigma_x : float
        Standard deviation of Gaussian in x-direction.
    sigma_y : float
        Standard deviation of Gaussian in y-direction.
    mode : {'add', 'max'}
        If 'add', Gaussian is summed with image.
        If 'max', image becomes the elementwise max of current and Gaussian.
    amplitude : float
        Peak value of the Gaussian.
    
    Returns
    -------
    image : np.ndarray
        Updated image with Gaussian applied.
    """
    assert image.ndim == 2, "Image must be 2D"
    rows, cols = image.shape

    # Create coordinate grids
    y, x = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')

    # 2D Gaussian formula
    gaussian = amplitude * np.exp(-(((x - center_x)**2) / (2 * sigma_x**2) +
                                     ((y - center_y)**2) / (2 * sigma_y**2)))

    if mode == 'add':
        image = image + gaussian
    elif mode == 'max':
        image = np.maximum(image, gaussian)
    else:
        raise ValueError("mode must be 'add' or 'max'")

    return image
#############################################################################
# Set up file lists and mappings
#############################################################################

# get list of existing datasets:
img_files = glob.glob(imgs_data_path + '/*.pt')

datasets = labels['dataset'].unique()
file_maps = get_filemaps(raw_data_path, datasets)

#############################################################################
# Iterate and generate images/masks
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
    
    print(labels_in_datachunk)
    gaussian_mask = 1
    nsave = 0
    gaussian_image = np.zeros((img.shape[1], img.shape[2]))
    for i, label in labels_in_datachunk.iterrows():
        apex_distance_m = label['apex_distance']
        apex_time_s = label['apex_time']
        
        apex_distance_pxl = int(apex_distance_m/dx)
        apex_time_pxl = int((apex_time_s-timestamp)/dt)

        gaussian_image = add_gaussian_to_image(gaussian_image, apex_time_pxl, apex_distance_pxl, sigma_x=sigma_x, sigma_y=sigma_y, mode=mode)
        
        plt.imshow(torch.sum(img, axis=0).numpy(), vmin=0, vmax = 100)
        plt.scatter(apex_time_pxl, apex_distance_pxl, c='r')
        plt.savefig(f'test_{nsave}.png')
        plt.close()
        
        plt.imshow(gaussian_image)
        # plt.scatter(apex_time_pxl, apex_distance_pxl, c='r')
        plt.savefig(f'test_{nsave}_gaus{sigma_x}_{sigma_y}_{mode}.png')
        plt.close()
        
        if nsave>5:
            exit()
            
        nsave +=1
        
         
         
    ## III. Iterate over each apex and:
    ##   A. make individual point gaussian heatmaps for each apex (DETxxx)
    ##   B. regenerate masks as individual images
    ##       1. either try to separate out existing contours based on det num
    ##       2. OR regenerate masks copying code from reshape_and_make_mask.py
    ## IV. make combined guassian heatmap (ALL_POINTS_xxxTOxxx)
    ##   A. take max OR sum OR median across all individual heatmap (probably max)