import os
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt

def load_yolo_polygon(label_path, img_width, img_height):
    polygons = []
    if not os.path.exists(label_path):
        return polygons

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                # needs at least class + 2 coords
                continue
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))

            # Need at least 6 floats for 3 points (x1,y1,x2,y2,x3,y3)
            if len(coords) < 6:
                continue

            xs = np.array(coords[0::2]) * img_width
            ys = np.array(coords[1::2]) * img_height

            # Polygon requires at least 3 points
            if len(xs) < 3 or len(ys) < 3:
                continue

            poly_coords = list(zip(xs, ys))
            if len(poly_coords) < 3:
                continue

            poly = Polygon(poly_coords)
            if poly.is_valid and poly.area > 0:
                polygons.append((cls, poly))
            else:
                # optionally log invalid polygons
                pass
    return polygons

def intersection_fraction(pred_poly, gt_poly):
    """Fraction of prediction polygon area inside GT polygon."""
    inter_area = pred_poly.intersection(gt_poly).area
    return inter_area / pred_poly.area if pred_poly.area > 0 else 0

def count_det(pred_label, gt_label, img_width, img_height, overlap_thresh=0.5):
    """
    Count how many YOLO predictions are inside vs outside the GT polygons.
    overlap_thresh: fraction of prediction area that must overlap GT to be considered 'inside'
    """
    preds = load_yolo_polygon(pred_label, img_width, img_height)
    gts   = load_yolo_polygon(gt_label, img_width, img_height)

    inside_count = 0
    outside_count = 0

    for cls_p, poly_p in preds:
        best_frac = 0
        for cls_g, poly_g in gts:
            frac = intersection_fraction(poly_p, poly_g)
            if frac > best_frac:
                best_frac = frac
        if best_frac >= overlap_thresh:
            inside_count += 1
        else:
            outside_count += 1

    return inside_count, outside_count
    
def find_apex(polygon, axis='y'):
    """Find apex point of polygon: min y or min x."""
    x, y = polygon.exterior.coords.xy
    if axis == 'y':  # min distance
        idx = np.argmin(y)
    else:            # min time
        idx = np.argmin(x)
    return x[idx], y[idx]

def analyze_image(pred_label, gt_label, img_width, img_height, axis='y'):
    preds = load_yolo_polygon(pred_label, img_width, img_height)
    gts   = load_yolo_polygon(gt_label, img_width, img_height)

    results = []
    for cls_p, poly_p in preds:
        best_frac = 0
        best_gt_poly = None
        for cls_g, poly_g in gts:
            frac = intersection_fraction(poly_p, poly_g)
            if frac > best_frac:
                best_frac = frac
                best_gt_poly = poly_g
        
        if best_gt_poly is not None:
            pred_apex = find_apex(poly_p, axis=axis)
            gt_apex   = find_apex(best_gt_poly, axis=axis)
            apex_diff = (pred_apex[0] - gt_apex[0], pred_apex[1] - gt_apex[1])
        else:
            apex_diff = (None, None)

        results.append({
            'class': cls_p,
            'intersection_fraction': best_frac,
            'apex_diff': apex_diff
        })
    return results

def plot_polygons_on_image(img_path, gt_label_path=None, pred_label_path=None,
                           img_width=None, img_height=None):
    """
    Plot GT and prediction polygons over the original image.
    
    gt_label_path: path to YOLO-format polygon label file (optional).
    pred_label_path: path to YOLO-format polygon prediction file (optional).
    img_width/img_height: needed if YOLO coordinates are normalized.
    """
    # Load image
    img = plt.imread(img_path)
    if img_width is None or img_height is None:
        img_height, img_width = img.shape[0], img.shape[1]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)

    def load_polygons(label_file, color, label_name):
        if not label_file or not os.path.exists(label_file):
            return
        
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                cls_id = int(parts[0])
                coords = list(map(float, parts[1:]))
                if not coords:
                    continue
                xs = np.array(coords[0::2]) * img_width
                ys = np.array(coords[1::2]) * img_height
                poly_coords = np.column_stack((xs, ys))
                
                ax.plot(xs, ys, color=color, linewidth=2, alpha=0.7)
                ax.fill(xs, ys, color=color, alpha=0.2)
                
                # Annotate class
                ax.text(xs[0], ys[0], f"{label_name} cls{cls_id}",
                        color=color, fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

    # Plot GT polygons in green
    load_polygons(gt_label_path, color='lime', label_name='GT')
    # Plot prediction polygons in red
    load_polygons(pred_label_path, color='red', label_name='Pred')

    ax.set_title(f"Polygons over {os.path.basename(img_path)}")
    ax.set_xlim([0, img_width])
    ax.set_ylim([img_height, 0])  # flip y-axis so (0,0) is top-left
    plt.tight_layout()
    return fig