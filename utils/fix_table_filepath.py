import pandas as pd
import os
from pathlib import Path

# Fixing file paths for data that have been moved to a new base directory or system.
# WARNING: Some components are hardcoded for CV4E data structure and may need to be generalized for other uses.

tbl_filepath = r'/home/Eric/Documents/gitRepos/cv4e_dataPrep/fx_labels_Bp.csv'
output_tbl_filepath = r'/home/Eric/Documents/gitRepos/cv4e_dataPrep/fx_labels_Bp_fixed.csv'
data_base_path = r'/mnt/class_data/esnyder/raw_data'

def fix_table_filepath(tbl_filepath: str, 
                       data_base_path: str, 
                       OS:str = 'UNIX', 
                       output_tbl_filepath: str = None):

    if output_tbl_filepath is None:
        output_tbl_filepath = tbl_filepath

    tbl = pd.read_csv(tbl_filepath)
    filepaths = tbl['source_file']

    # Fix the file path 
    if OS.upper() == 'WINDOWS':
        data_base_path = data_base_path.replace('/', '\\')
    elif OS.upper() == 'UNIX' or OS.upper() == 'LINUX':
        data_base_path = data_base_path.replace('\\', '/')

    filepaths = tbl['source_file']
    fixed_filepaths = filepaths
    for i in range(len(filepaths)):
        fp = filepaths[i]
        rel_path = fp[3:]   # TODO this is hardcoded for the path conversion for CV4E. Need to generalize if using in the future.
        fp = os.path.join(data_base_path, rel_path)
        if OS.upper() == 'WINDOWS':
            fp = str(fp).replace('/', '\\')
        elif OS.upper() == 'UNIX' or OS.upper() == 'LINUX':
            fp = str(fp).replace('\\', '/')
        fixed_filepaths[i] = fp
        
    tbl['source_file'] = fixed_filepaths
    tbl.to_csv(output_tbl_filepath, index=False)

def fix_swapped_columns(tbl_filepath: str, 
                 output_tbl_filepath: str = None):
    if output_tbl_filepath is None:
        output_tbl_filepath = tbl_filepath

    tbl = pd.read_csv(tbl_filepath)
    
    tbl[["f_max_hz", "x_min_m"]] = tbl[["x_min_m", "f_max_hz"]].values
    
    tbl.to_csv(output_tbl_filepath, index=False)


fix_table_filepath(tbl_filepath, data_base_path, OS='UNIX', output_tbl_filepath=output_tbl_filepath)
# fix_swapped_columns(output_tbl_filepath, output_tbl_filepath) # Only run if table was generated prior to 2026-01-16
