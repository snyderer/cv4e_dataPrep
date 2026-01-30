import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

# -------------------------------
# CONFIG
# -------------------------------
split = 'svalbard'
# split = 'random'
gt_labels_dir = f"/mnt/class_data/esnyder/yolo_data/tx_segmentation/{split}_split/labels/val"   # YOLO-format ground truth label txt files
pred_labels_dir = f"/home/Eric/Documents/gitRepos/instance_seg_yolo/cv4e_final/segment/runs/{split}_split/val/labels"   # YOLO-format predicted label txt files
image_width = 667
image_height = 1070
output_csv = f"yolo_IoU_Dice_results_{split}_split.csv"

# -------------------------------
# FUNCTIONS
# -------------------------------

def load_labels(file_path):
    """Load YOLO labels, supports either polygon (segmentation) or bbox format."""
    labels = []
    if not os.path.exists(file_path):
        return labels

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # skip

            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))

            if len(coords) > 4:  # Segmentation polygon
                xy_pairs = [
                    (coords[i] * image_width, coords[i+1] * image_height)
                    for i in range(0, len(coords), 2)
                ]
                if len(xy_pairs) >= 3:
                    poly = Polygon(xy_pairs)
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    labels.append({'class': class_id, 'polygon': poly})
            else:  # Bbox
                x_c, y_c, w, h = coords
                x_c *= image_width
                y_c *= image_height
                w *= image_width
                h *= image_height
                x_min = x_c - w / 2
                x_max = x_c + w / 2
                y_min = y_c - h / 2
                y_max = y_c + h / 2
                box_coords = [(x_min, y_min), (x_max, y_min),
                              (x_max, y_max), (x_min, y_max)]
                poly = Polygon(box_coords)
                labels.append({'class': class_id, 'polygon': poly})
    return labels

def dice(poly1, poly2):
    inter_area = poly1.intersection(poly2).area
    area_sum = poly1.area + poly2.area
    if area_sum == 0:
        return 0.0
    return 2 * inter_area / area_sum

def iou(poly1, poly2):
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    inter_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def match_labels(gt_labels, pred_labels):
    """
    Match GT to Pred via IoU, store IoU and Dice.
    Returns: list of tuples (gt_index, pred_index, iou, dice)
    """
    results = []
    gt_used = set()
    pred_used = set()

    for gi, gt in enumerate(gt_labels):
        best_iou = 0
        best_dice = 0
        best_pi = None
        for pi, pred in enumerate(pred_labels):
            current_iou = iou(gt['polygon'], pred['polygon'])
            current_dice = dice(gt['polygon'], pred['polygon'])
            if current_iou > best_iou:
                best_iou = current_iou
                best_dice = current_dice
                best_pi = pi
        if best_pi is not None:
            results.append((gi, best_pi, best_iou, best_dice))
            gt_used.add(gi)
            pred_used.add(best_pi)
        else:
            results.append((gi, None, 0, 0))

    for pi, pred in enumerate(pred_labels):  # False positives
        if pi not in pred_used:
            results.append((None, pi, 0, 0))

    return results

# -------------------------------
# MAIN SCRIPT
# -------------------------------
csv_rows = []
for gt_file in glob.glob(os.path.join(gt_labels_dir, "*.txt")):
    file_name = os.path.basename(gt_file)
    pred_file = os.path.join(pred_labels_dir, file_name)

    gt_labels = load_labels(gt_file)
    pred_labels = load_labels(pred_file)

    matches = match_labels(gt_labels, pred_labels)

    for gt_idx, pred_idx, overlap, dscore in matches:
        csv_rows.append([
            file_name,
            gt_idx if gt_idx is not None else "None",
            pred_idx if pred_idx is not None else "None",
            overlap,  # IoU
            dscore    # Dice
        ])

with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["file_name", "gt_label_num", "pred_label_num", "IoU", "Dice"])
    writer.writerows(csv_rows)

print(f"Saved IoU & Dice results to {output_csv}")

# -------------------------------
# LOAD CSV AND PLOTS
# -------------------------------
import pandas as pd
df = pd.read_csv(output_csv)

# -------- IoU Histogram --------
plt.figure(figsize=(6,4))
idx = df['IoU'] > 0
plt.hist(df['IoU'][idx], bins=100, range=(0,1), color='blue', alpha=0.7)
plt.xlabel("IoU")
plt.ylabel("Count")
plt.title("IoU Histogram")
plt.savefig(f"iou_histogram_{split}.png")
plt.close()

# -------- Dice Histogram --------
plt.figure(figsize=(6,4))
idx = df['Dice'] > 0
plt.hist(df['Dice'][idx], bins=100, range=(0,1), color='green', alpha=0.7)
plt.xlabel("Dice")
plt.ylabel("Count")
plt.title("Dice Histogram")
plt.savefig(f"dice_histogram_{split}.png")
plt.close()

# -------------------------------------------------
# Confusion matrix (based on match presence only)
# -------------------------------------------------
TP_raw = ((df['gt_label_num'] != "None") & (df['pred_label_num'] != "None")).sum()
FP_raw = (df['gt_label_num'] == "None").sum()
MD_raw = (df['pred_label_num'] == "None").sum()

conf_matrix = np.array([[TP_raw, FP_raw],
                        [MD_raw, 0]])

plt.figure(figsize=(6,5))
plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
plt.title("Confusion Matrix (Match Presence)")
plt.xticks([0,1], ["TP","FP"])
plt.yticks([0,1], ["MD","N/A"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, conf_matrix[i,j], ha='center', va='center', color='black')
plt.savefig(f"confusion_matrix_iou_{split}.png")
plt.close()

# -------------------------------------------------
# IoU Threshold Curve (Localization quality)
# -------------------------------------------------
thresholds = np.linspace(0,1,21)
tp_list, fp_list, md_list = [], [], []
for t in thresholds:
    TP_t = ((df['gt_label_num'] != "None") & (df['pred_label_num'] != "None") & (df['IoU'] >= t)).sum()
    fp_list.append(FP_raw)  # constant
    md_list.append(MD_raw)  # constant
    tp_list.append(TP_t)

plt.figure(figsize=(8,5))
plt.plot(thresholds, tp_list, label="TP", marker='o')
plt.plot(thresholds, fp_list, label="FP (raw count)", marker='o')
# plt.plot(thresholds, md_list, label="MD (raw count)", marker='o')
plt.xlabel("IoU Threshold")
plt.ylabel("Count")
plt.title("TP vs IoU Threshold (FP/MD constant)")
plt.legend()
plt.savefig(f"threshold_curve_iou_{split}.png")
plt.close()

# -------------------------------------------------
# Dice Threshold Curve (Localization quality)
# -------------------------------------------------
tp_list_d, fp_list_d, md_list_d = [], [], []
for t in thresholds:
    TP_t_d = ((df['gt_label_num'] != "None") & (df['pred_label_num'] != "None") & (df['Dice'] >= t)).sum()
    fp_list_d.append(FP_raw)  # constant
    md_list_d.append(MD_raw)  # constant
    tp_list_d.append(TP_t_d)

plt.figure(figsize=(8,5))
plt.plot(thresholds, tp_list_d, label="TP", marker='o')
plt.plot(thresholds, fp_list_d, label="FP (raw count)", marker='o')
# plt.plot(thresholds, md_list_d, label="MD (raw count)", marker='o')
plt.xlabel("Dice Threshold")
plt.ylabel("Count")
plt.title("TP vs Dice Threshold (FP/MD constant)")
plt.legend()
plt.savefig(f"threshold_curve_dice_{split}.png")
plt.close()