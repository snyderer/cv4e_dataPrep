# test script to see if I can load data, process it, and plot the output.
import data_io as io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

label_file = r'./fx_labels_Bp.csv'
data_folder = r'/mnt/class_data/esnyder/raw_data/'

labels = pd.read_csv(label_file)

# randomly select a row:
idx = np.random.randint(0, len(labels))

row = labels.iloc[idx]

data_file = os.basename(row['source_file'])