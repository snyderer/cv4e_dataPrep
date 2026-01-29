import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image

# Paths to your YOLO datasets
seg_imgs_out_path = Path(r'/mnt/class_data/esnyder/yolo_data/tx_segmentation/images')
seg_labels_out_path = Path(r'/mnt/class_data/esnyder/yolo_data/tx_segmentation/labels')

bb_imgs_out_path = Path(r'/mnt/class_data/esnyder/yolo_data/fx_boundingbox/images')
bb_labels_out_path = Path(r'/mnt/class_data/esnyder/yolo_data/fx_boundingbox/labels')

def load_yolo_segmentation_labels(label_path, img_w, img_h):
    """Load segmentation labels (.txt) in YOLOv8 polygon format."""
    polygons = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            cls_id = int(parts[0])
            coords = parts[1:]
            poly_points = [(coords[i] * img_w, coords[i+1] * img_h) for i in range(0, len(coords), 2)]
            polygons.append((cls_id, poly_points))
    return polygons

def load_yolo_bbox_labels(label_path, img_w, img_h):
    """Load bounding box labels (.txt) in YOLO format."""
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            cls_id = int(parts[0])
            x_center, y_center, w, h = parts[1:]
            xmin = (x_center - w/2) * img_w
            xmax = (x_center + w/2) * img_w
            ymin = (y_center - h/2) * img_h
            ymax = (y_center + h/2) * img_h
            boxes.append((cls_id, xmin, ymin, xmax, ymax))
    return boxes

def plot_segmentation_with_labels(image_path, label_path):
    img = np.array(Image.open(image_path))
    img_h, img_w = img.shape[0], img.shape[1]

    polygons = load_yolo_segmentation_labels(label_path, img_w, img_h)

    fig, ax = plt.subplots()
    ax.imshow(img, cmap='viridis')

    for cls_id, poly_points in polygons:
        xs, ys = zip(*poly_points)
        ax.plot(xs, ys, label=f"Class {cls_id}")

    ax.legend()
    plt.title(f"Segmentation Labels: {image_path.name}")
    fig.savefig(f"./test_images/yolo_seg/{image_path.name}")

def plot_bbox_with_labels(image_path, label_path):
    img = np.array(Image.open(image_path))
    img_h, img_w = img.shape[0], img.shape[1]

    boxes = load_yolo_bbox_labels(label_path, img_w, img_h)

    fig, ax = plt.subplots()
    ax.imshow(img, cmap='viridis')

    for cls_id, xmin, ymin, xmax, ymax in boxes:
        rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                             linewidth=2, edgecolor='red', facecolor='none', label=f"Class {cls_id}")
        ax.add_patch(rect)

    ax.legend()
    plt.title(f"BBox Labels: {image_path.name}")
    fig.savefig(f"./test_images/yolo_bb/{image_path.name}")


# -------- Example usage --------
# Pick one segmentation example to display
example_seg_image = list(seg_imgs_out_path.glob("*.png"))[-1]
example_seg_label = seg_labels_out_path / (example_seg_image.stem + ".txt")
plot_segmentation_with_labels(example_seg_image, example_seg_label)

# Pick one bounding box example to display
example_bb_image = list(bb_imgs_out_path.glob("*.png"))[-1]
example_bb_label = bb_labels_out_path / (example_bb_image.stem + ".txt")
plot_bbox_with_labels(example_bb_image, example_bb_label)