# Verify labels work using matplotlib
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

filepath = r'F:\svalbard_full'
filename = '20220822_090007.h5'
data_filepath = os.path.join(filepath, filename)
db_path = r'C:\Users\ers334\Documents\databases\DAS_Annotations\A25.db'


settings_file = io.find_settings_h5(data_filepath)
settings = io.load_settings_preprocessed_h5(settings_file)
nonzeros = settings['rehydration_info']['nonzeros_mask']
original_shape = settings['rehydration_info']['target_shape']
dx = settings['processing_settings']['dx']
fs = settings['processing_settings']['fs']
fk_dehyd, timestamp = io.load_preprocessed_h5(data_filepath)

file_idx = settings['file_map']['filename']==filename
file_start_time = settings['file_map']['timestamp'][file_idx]

# file_start_time = settings['processing_settings']['timestamp']
# print(file_start_time)

tx = 1e9 * io.rehydrate(fk_dehyd, nonzeros, original_shape, return_format='tx')

# print(f"{tx.shape=}")

x_extent = tx.shape[0]*dx
t_extent = tx.shape[1]/fs

# connect to sql database and load labels

query = (
    "SELECT * "
    "FROM tx_labels "
    "WHERE label_name IN ('Bp_B', 'Bp_A');"
)

conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.execute(query)
rows = cur.fetchall()
cols = [d[0] for d in cur.description] if cur.description else []

x_col_idx = cols.index('x_m') if 'x_m' in cols else -1
t_col_idx = cols.index('t_s') if 't_s' in cols else -1
source_files_indx = cols.index('source_file') if 'source_file' in cols else -1
apex_time_idx = cols.index('apex_time') if 'apex_time' in cols else -1

print(f"{len(rows)=}")
rows = [row for row in rows if row[source_files_indx] == filename]
print(f"{len(rows)=}")



# print(rows)
print(f"{cols=}")


# # MAH TODO this crop should be done by contour then zero out all the pixels further than n from the countour
# tx_small = tx[1000:3000, 2500:3500]


# # tx_small_filtered  = median_filter(np.abs(tx_small), size=15)  # 3×3 window
# tx_small_filtered = gaussian_filter(np.abs(tx_small), sigma=5.0)
# tolerance = 0.1
# tx_small_filtered_tol = np.where(tx_small_filtered>tolerance, 1, 0)

# # binary image: True/False or 0/1
# tx_small_filtered_tol_dilated = binary_dilation(tx_small_filtered_tol, structure=np.ones((53, 53)))

# cutout = np.where(tx_small_filtered_tol_dilated, tx_small, 0)

# fig, axs = plt.subplots(1,5)
# axs[0].imshow(tx_small , vmin=0, vmax=.4, origin="lower")
# axs[1].imshow(tx_small_filtered , vmin=0,vmax=.4, origin="lower")
# axs[2].imshow(tx_small_filtered_tol , origin="lower")
# axs[3].imshow(tx_small_filtered_tol_dilated , origin="lower")
# axs[4].imshow(cutout ,vmin=0, vmax=.4, origin="lower")

# plt.show()
# exit()

for row in rows[:2]:
    row = list(row)
    x_list = np.array(ast.literal_eval(row[x_col_idx]))
    t_list = np.array(ast.literal_eval(row[t_col_idx])) + row[apex_time_idx] - timestamp
    print(f"{row[apex_time_idx]=}")
    print(f"{timestamp=}")

    print( f"{row[apex_time_idx] - timestamp=}")
    # t_list = np.array(ast.literal_eval(row[t_col_idx]))

    x_min = min(x_list)
    x_max = max(x_list)
    t_start = min(t_list)
    t_end = max(t_list)

    print(f"{len(x_list)=}")
    print(f"{len(t_list)=}")


    # plt.plot(t_list,x_list)
    # plt.show()
    # exit()

    # pt_min = pt*t_min/t_extent
    # pt_max = pt*t_max/t_extent

    # px_min = px*x_min/x_extent
    # px_max = px*x_max/x_extent
    x_pxl = tx.shape[0]*x_list/x_extent
    t_pxl = tx.shape[1]*t_list/t_extent
    # x_pxl = x_list/(x_extent / tx.shape[0])
    # t_pxl = t_list/(t_extent / tx.shape[1])

    print(t_pxl[:5])    
    print(t_pxl[-5:])

    plt.imshow(tx, vmin=0, vmax=.4)
    plt.scatter(t_pxl, x_pxl)
    plt.show()

#     tx_row['apex_time']=1661159695.66
# file_start_time=np.float64(1661159677.0)