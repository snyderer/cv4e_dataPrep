import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

# -------------------------------
# CONFIG
# -------------------------------
# split = 'random' 
split = 'svalbard'

gt_labels_dir = f"/mnt/class_data/esnyder/yolo_data/tx_segmentation/{split}_split/labels/val"   # YOLO-format ground truth label txt files
pred_labels_dir = f"/home/Eric/Documents/gitRepos/instance_seg_yolo/cv4e_final/segment/runs/{split}_split/val/labels"   # YOLO-format predicted label txt files
image_width        = 667                      # width of your images in pixels
image_height       = 1070                     # height of your images in pixels
output_csv       = f"yolo_IoU_results_{split}_split.csv"   # CSV output file

# -------------------------------
# FUNCTIONS
# -------------------------------

def load_labels(file_path):
    """
    Load YOLO labels, supports either polygon (segmentation) or bbox format.
    """
    labels = []
    if not os.path.exists(file_path):
        return labels

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # skip lines that aren't at least bbox
            
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))

            # Decide format: segmentation vs bbox
            if len(coords) > 4:
                # SEGMENTATION: list of (x, y) pairs
                xy_pairs = [
                    (coords[i] * image_width, coords[i+1] * image_height)
                    for i in range(0, len(coords), 2)
                ]
                if len(xy_pairs) >= 3:
                    poly = Polygon(xy_pairs)
                    if not poly.is_valid:
                        poly = poly.buffer(0)  # fix invalid polygons
                    labels.append({'class': class_id, 'polygon': poly})
            else:
                # BBOX in YOLO format: x_center, y_center, width, height (normalized)
                x_c, y_c, w, h = coords
                x_c *= image_width
                y_c *= image_height
                w *= image_width
                h *= image_height
                # convert to corner coordinates
                x_min = x_c - w / 2
                x_max = x_c + w / 2
                y_min = y_c - h / 2
                y_max = y_c + h / 2
                box_coords = [(x_min, y_min), (x_max, y_min),
                              (x_max, y_max), (x_min, y_max)]
                poly = Polygon(box_coords)
                labels.append({'class': class_id, 'polygon': poly})
    return labels


def iou(poly1, poly2):
    """Calculate IoU between two shapely Polygons."""
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    inter_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def match_labels(gt_labels, pred_labels):
    """
    Match GT to Pred via IoU.
    Returns list of tuples: (gt_index, pred_index, iou)
    Adds None when no match found.
    """
    results = []
    gt_used = set()
    pred_used = set()

    for gi, gt in enumerate(gt_labels):
        best_iou = 0
        best_pi = None
        for pi, pred in enumerate(pred_labels):
            current_iou = iou(gt['polygon'], pred['polygon'])
            if current_iou > best_iou:
                best_iou = current_iou
                best_pi = pi
        if best_pi is not None:
            results.append((gi, best_pi, best_iou))
            gt_used.add(gi)
            pred_used.add(best_pi)
        else:
            results.append((gi, None, 0))

    # False positives
    for pi, pred in enumerate(pred_labels):
        if pi not in pred_used:
            results.append((None, pi, 0))

    return results


# -------------------------------
# MAIN SCRIPT: Compute IoUs and save CSV
# -------------------------------

csv_rows = []
for gt_file in glob.glob(os.path.join(gt_labels_dir, "*.txt")):
    file_name = os.path.basename(gt_file)
    pred_file = os.path.join(pred_labels_dir, file_name)

    gt_labels = load_labels(gt_file)
    pred_labels = load_labels(pred_file)

    matches = match_labels(gt_labels, pred_labels)

    for gt_idx, pred_idx, overlap in matches:
        csv_rows.append([file_name,
                         gt_idx if gt_idx is not None else "None",
                         pred_idx if pred_idx is not None else "None",
                         overlap])

with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["file_name", "gt_label_num", "pred_label_num", "IoU"])
    writer.writerows(csv_rows)

print(f"Saved IoU results to {output_csv}")

# -------------------------------
# LOAD CSV AND PLOTS
# -------------------------------

import pandas as pd

df = pd.read_csv(output_csv)

# 1) Histogram of IoUs
plt.figure(figsize=(6,4))
idx = df['IoU']>0
plt.hist(df['IoU'][idx], bins=100, range=(0,1), color='blue', alpha=0.7)
plt.xlabel("IoU")
plt.ylabel("Count")
plt.title("IoU Histogram")
plt.savefig(f"iou_histogram_{split}.png")
plt.close()

# 2) Confusion matrix (TP, FP, MD)
TP = sum((df['IoU'] > 0) & (df['gt_label_num'] != "None") & (df['pred_label_num'] != "None"))
FP = sum((df['gt_label_num'] == "None") & (df['pred_label_num'] != "None"))
MD = sum((df['gt_label_num'] != "None") & (df['pred_label_num'] == "None"))

conf_matrix = np.array([[TP, FP],
                        [MD, 0]])  # bottom-right is not really used

plt.figure(figsize=(6,5))
plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
plt.title("Confusion Matrix")
plt.xticks([0,1], ["TP","FP"])
plt.yticks([0,1], ["MD","N/A"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, conf_matrix[i,j], ha='center', va='center', color='black')
plt.savefig("confusion_matrix_{split}.png")
plt.close()

# 3) IoU threshold iteration
thresholds = np.linspace(0,1,21)
tp_list, fp_list, md_list = [], [], []
for t in thresholds:
    TP_t = sum((df['IoU'] >= t) & (df['gt_label_num'] != "None") & (df['pred_label_num'] != "None"))
    FP_t = sum((df['gt_label_num'] == "None") & (df['pred_label_num'] != "None"))
    MD_t = sum((df['gt_label_num'] != "None") & ((df['pred_label_num'] == "None") | (df['IoU'] < t)))
    tp_list.append(TP_t)
    fp_list.append(FP_t)
    md_list.append(MD_t)

plt.figure(figsize=(8,5))
plt.plot(thresholds, tp_list, label="TP", marker='o')
plt.plot(thresholds, fp_list, label="FP", marker='o')
plt.plot(thresholds, md_list, label="MD", marker='o')
plt.xlabel("IoU Threshold")
plt.ylabel("Count")
plt.title("TP/FP/MD vs IoU Threshold")
plt.legend()
plt.savefig(f"threshold_curve_{split}.png")
plt.close()

print(f"Saved plots: iou_histogram_{split}.png, confusion_matrix_{split}.png, threshold_curve_{split}.png")