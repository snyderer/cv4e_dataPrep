#!/usr/bin/env python3
"""
Evaluate YOLO segmentation predictions against ground truth.
Uses functions from yolo_analysis_tools.py
"""

import os
import csv
import numpy as np
import yolo_analysis_tools as yo

######################### BAD CODE ############################
###############################################################
# This doesn't correctly match predicted and training labels
# retaining only to scraping for parts later
###############################################################

# ==== USER CONFIG ====
# split = 'random' 
split = 'svalbard'

predictions_path =  f"/home/Eric/Documents/gitRepos/instance_seg_yolo/cv4e_final/segment/runs/{split}_split/val/labels"   # YOLO-format predicted label txt files
labels_path      =  f"/mnt/class_data/esnyder/yolo_data/tx_segmentation/{split}_split/labels/val"   # YOLO-format ground truth label txt files
img_width        = 667                      # width of your images in pixels
img_height       = 1070                     # height of your images in pixels
overlap_thresh   = 0.5                       # fraction overlap for "inside" detection
output_csv       = f"yolo_eval_results_{split}_split.csv"   # CSV output file
# ======================

def evaluate_dataset(predictions_dir, labels_dir, width, height):
    """Iterate over all prediction files and compute matching metrics."""
    all_results = []

    pred_files = sorted(os.listdir(predictions_dir))
    total_inside = 0
    total_outside = 0
    overlap_values = []

    for pred_file in pred_files:
        pred_fp = os.path.join(predictions_dir, pred_file)
        gt_fp   = os.path.join(labels_dir, pred_file)

        if not os.path.exists(gt_fp):
            print(f"[Warning] No GT file for {pred_file}, skipping.")
            continue

        # Count inside/outside
        inside_count, outside_count = yo.count_det(
            pred_fp, gt_fp, width, height, overlap_thresh=overlap_thresh
        )

        # Record per-image overlap fractions
        img_results = yo.analyze_image(pred_fp, gt_fp, width, height)
        for res in img_results:
            if res['intersection_fraction'] is not None:
                overlap_values.append(res['intersection_fraction'])

        all_results.append({
            'filename': pred_file,
            'inside_count': inside_count,
            'outside_count': outside_count,
            'mean_intersection_fraction': (
                np.mean([r['intersection_fraction'] for r in img_results])
                if img_results else 0
            )
        })

        total_inside += inside_count
        total_outside += outside_count

    return all_results, total_inside, total_outside, overlap_values


if __name__ == "__main__":
    # Run evaluation
    results, total_inside, total_outside, overlap_values = evaluate_dataset(
        predictions_path, labels_path, img_width, img_height
    )

    total_preds = total_inside + total_outside
    mean_overlap = np.mean(overlap_values) if overlap_values else 0

    print("\n=== YOLO Segmentation Evaluation ===")
    print(f"Number of images evaluated: {len(results)}")
    print(f"Total predictions: {total_preds}")
    print(f"Inside count: {total_inside}")
    print(f"Outside count: {total_outside}")
    print(f"Inside %: {total_inside / total_preds * 100:.2f}%")
    print(f"Mean intersection fraction: {mean_overlap:.3f}")

    # Save CSV
    fieldnames = ['filename', 'inside_count', 'outside_count', 'mean_intersection_fraction']
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nPer-image results saved to: {output_csv}")