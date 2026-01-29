import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

seg_repo_path = Path("/home/Eric/Documents/gitRepos/segmentation")
sys.path.append(str(seg_repo_path))

from unet import UNet  # Your UNet implementation

# --------------------------
# 1. Load trained model
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set these to match training
n_channels = 51      # Same as training
n_classes  = 1       # Or whatever you used
bilinear   = False   # Match your training args
# checkpoint_path = Path("/mnt/class_data/esnyder/segmentation_data/keypoint_mapping/checkpoints/<run_name>/checkpoint_best.pth")
checkpoint_path = Path("/mnt/class_data/esnyder/segmentation_data/fixed_width/checkpoints/01282026_030050_run1/checkpoint_best.pth")
model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
state_dict = torch.load(, map_location=device)

# Remove extra info saved
state_dict.pop('mask_values', None)
state_dict.pop('epoch', None)
state_dict.pop('val_loss', None)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

# --------------------------
# 2. Load validation CSV
# --------------------------
validation_csv = r'/mnt/class_data/esnyder/segmentation_data/validation_paths.csv'
vals = pd.read_csv(validation_csv)

# --------------------------
# 3. Iterate over validation samples & run through model
# --------------------------
for _, val in vals.iterrows():
    img_path = val['image_path']
    mask_path = val['mask_path']
    
    # Load tensors from saved .pt files
    img_tensor  = torch.load(img_path)   # shape: [channels, H, W]
    mask_tensor = torch.load(mask_path)  # shape: [H, W] or [1, H, W]
    
    # Add batch dimension: [1, channels, H, W]
    img_tensor = img_tensor.unsqueeze(0)
    
    # Move to same device as model
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)
    
    with torch.no_grad():
        output = model(img_tensor)  # shape: [1, n_classes, H, W]
        
        if n_classes == 1:
            # Binary segmentation case
            prob_map = torch.sigmoid(output)  # probabilities [0,1]
            pred_mask = (prob_map > 0.5).float()  # threshold to get binary mask
        else:
            # Multi-class segmentation case
            prob_map = F.softmax(output, dim=1)
            pred_mask = torch.argmax(prob_map, dim=1)  # get class index per pixel
    
    # Remove batch dim for further analysis
    pred_mask = pred_mask.squeeze(0)  # shape: [H, W] (binary) or [H, W] (multi-class)
    
    # --------------------------
    # 4. Your fraction-inside-mask metric would go here
    # --------------------------
    # Example fraction of predicted-positive pixels inside GT mask:
    prediction_bin = pred_mask > 0.5 if n_classes == 1 else (pred_mask == 1)
    gt_bin         = mask_tensor > 0.5

    num_pred_pixels = torch.sum(prediction_bin).item()
    if num_pred_pixels > 0:
        fraction_inside = torch.sum(prediction_bin & gt_bin).item() / num_pred_pixels
    else:
        fraction_inside = float('nan')

    print(f"{img_path} -> Fraction inside mask: {fraction_inside:.3f}")