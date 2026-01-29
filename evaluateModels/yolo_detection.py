import os
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
import yolo_analysis_tools as yo
import matplotlib.pyplot as plt

run_num = 4
path_in = r'/home/Eric/Documents/gitRepos/instance_seg_yolo/runs/segment/'
predictions_path = f"/mnt/class_data/esnyder/yolo_data/tx_segmentation/predictions/train{run_num}_val/labels"
labels_path = "/mnt/class_data/esnyder/yolo_data/tx_segmentation/labels/val"
imgs_path = "/mnt/class_data/esnyder/yolo_data/tx_segmentation/images/val"
filename = 'ooi_optasense_north_c2_full_20211102_215231.txt'
imgsz=(667, 1070)

prediction_filepath = os.path.join(predictions_path, filename)
label_filepath = os.path.join(labels_path, filename)
img_filepath = os.path.join(imgs_path, filename[:-4]+'.png')

# pred_label = yo.load_yolo_polygon(, imgsz[0], imgsz[1])
# my_label = yo.load_yolo_polygon(, imgsz[0], imgsz[1])

fig = yo.plot_polygons_on_image(img_filepath, 
                            label_filepath, 
                            prediction_filepath)

fig.savefig('test.png')