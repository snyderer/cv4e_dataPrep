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
    For one image, count TP, FP, FN given threshold on intersection_fraction.
    """
    preds = yo.load_yolo_polygon(pred_file, width, height)
    gts   = yo.load_yolo_polygon(gt_file, width, height)

    TP = 0
    FP = 0
    FN = 0

    # Predictions: find a GT match with enough overlap
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

    # Missed detections: GT polygons that no pred matched
    for idx_g in range(len(gts)):
        if idx_g not in matched_gt_indices:
            FN += 1

    return TP, FP, FN


def sweep_thresholds(predictions_dir, labels_dir, width, height, thresholds):
    tp_frac_list = []
    fn_count_list = []

    pred_files = sorted(os.listdir(predictions_dir))
    for thresh in thresholds:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_preds = 0
        total_gts = 0

        for pred_file in pred_files:
            pred_fp = os.path.join(predictions_dir, pred_file)
            gt_fp   = os.path.join(labels_dir, pred_file)

            if not os.path.exists(gt_fp):
                continue

            TP, FP, FN = match_counts(pred_fp, gt_fp, width, height, thresh)
            total_tp += TP
            total_fp += FP
            total_fn += FN
            total_preds += TP + FP
            total_gts += TP + FN

        tp_frac = total_tp / total_preds if total_preds else 0
        fn_frac = total_fn / total_gts if total_gts else 0

        tp_frac_list.append(tp_frac)
        fn_count_list.append(total_fn)  # Or use fn_frac for fraction

    return tp_frac_list, fn_count_list


if __name__ == "__main__":
    tp_fractions, fn_counts = sweep_thresholds(
        predictions_path, labels_path, img_width, img_height, thresholds,
    )

    # === Plot two curves ===
    fig, ax1 = plt.subplots(figsize=(8,5))

    color_tp = 'tab:blue'
    ax1.set_xlabel('Overlap Threshold')
    ax1.set_ylabel('True Positive Fraction', color=color_tp)
    ax1.plot(thresholds, tp_fractions, marker='o', color=color_tp, label='TP Fraction')
    ax1.tick_params(axis='y', labelcolor=color_tp)
    ax1.set_ylim([0, 1])

    ax2 = ax1.twinx()  # Second y-axis
    color_fn = 'tab:red'
    ax2.set_ylabel('Missed Detections (Count)', color=color_fn)
    ax2.plot(thresholds, fn_counts, marker='s', color=color_fn, label='Missed Detections')
    ax2.tick_params(axis='y', labelcolor=color_fn)

    # Title and grid
    fig.suptitle("YOLO Segmentation Performance Sweep over Thresholds", fontsize=14)
    fig.grid = True

    fig.tight_layout()
    plt.savefig(f'hist_{split}.png')