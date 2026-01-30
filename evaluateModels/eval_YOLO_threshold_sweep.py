#!/usr/bin/env python3
"""
Evaluate YOLO segmentation predictions vs ground truth over a range of overlap thresholds.
Generates TP, FP, FN counts and a plot of TP fraction + FN count vs threshold.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import yolo_analysis_tools as yo

# ==== USER CONFIG ====
split = 'random' 
# split = 'svalbard'

predictions_path =  f"/home/Eric/Documents/gitRepos/instance_seg_yolo/cv4e_final/segment/runs/{split}_split/val/labels"   # YOLO-format predicted label txt files
labels_path      =  f"/mnt/class_data/esnyder/yolo_data/tx_segmentation/{split}_split/labels/val"   # YOLO-format ground truth label txt files

img_width        = 667
img_height       = 1070
thresholds       = np.linspace(0, 1, 21)   # thresholds from 0.0 to 1.0 in steps of 0.05
# ======================

def match_counts(pred_file, gt_file, width, height, thresh):
    """
    For one image, count TP, FP given threshold on intersection_fraction.
    """
    preds = yo.load_yolo_polygon(pred_file, width, height)
    gts   = yo.load_yolo_polygon(gt_file, width, height)

    TP = 0
    FP = 0
    matched_gt_indices = set()

    for cls_p, poly_p in preds:
        best_frac = 0
        best_gt_idx = None
        for idx_g, (cls_g, poly_g) in enumerate(gts):
            frac = yo.intersection_fraction(poly_p, poly_g)
            if frac > best_frac:
                best_frac = frac
                best_gt_idx = idx_g
        if best_frac >= thresh:
            TP += 1
            matched_gt_indices.add(best_gt_idx)
        else:
            FP += 1

    return TP, FP


def sweep_thresholds(predictions_dir, labels_dir, width, height, thresholds):
    tp_list = []
    fp_list = []

    pred_files = sorted(os.listdir(predictions_dir))
    for thresh in thresholds:
        total_tp = 0
        total_fp = 0

        for pred_file in pred_files:
            pred_fp = os.path.join(predictions_dir, pred_file)
            gt_fp   = os.path.join(labels_dir, pred_file)

            if not os.path.exists(gt_fp):
                continue

            TP, FP = match_counts(pred_fp, gt_fp, width, height, thresh)
            total_tp += TP
            total_fp += FP

        tp_list.append(total_tp)
        fp_list.append(total_fp)

    return tp_list, fp_list


if __name__ == "__main__":
    tp_counts, fp_counts = sweep_thresholds(
        predictions_path, labels_path, img_width, img_height, thresholds
    )

    # === Plot histogram-style chart ===
    width = 0.025  # bar width to fit side-by-side
    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(thresholds - width/2, tp_counts, width=width, color='tab:blue', label='True Positives')
    ax.bar(thresholds + width/2, fp_counts, width=width, color='tab:red', label='False Positives')

    ax.set_xlabel('Overlap Threshold')
    ax.set_ylabel('Count')
    ax.set_title('True Positives vs False Positives by Overlap Threshold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'hist_{split}.png')