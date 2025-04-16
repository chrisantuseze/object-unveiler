import torch
from fastsam import FastSAM, FastSAMPrompt
from PIL import Image
import numpy as np
import cv2

# Load model
model = FastSAM('FastSAM-x.pt')  # or 'FastSAM-s.pt' for Jetson

# Load image
img_path = 'color1.png'
image = Image.open(img_path).convert('RGB')

# # Run model (default returns masks, everything is class-agnostic)
# everything_results = model(
#     image,
#     device='cpu',  # Use 'cuda' if on GPU
#     retina_masks=True,
#     imgsz=640,
#     conf=0.4,
#     iou=0.9
# )

# # Parse results
# prompt_process = FastSAMPrompt(image, everything_results, device='cpu')
# masks = prompt_process.everything_prompt()  # List of binary masks

# # Save each mask
# for i, mask in enumerate(masks):
#     binary = (mask.numpy().astype(np.uint8) * 255)
#     cv2.imwrite(f'mask_{i}.png', binary)

# Run prediction
results = model.predict(image, device='cpu', conf=0.25, imgsz=640)

# results is a list with one item per image
r = results[0]

# Get masks, scores, and boxes
if r.masks is not None:
    masks = r.masks.data.cpu().numpy()          # Shape: [N, H, W]
    scores = r.boxes.conf.cpu().numpy()         # Confidence scores
    boxes = r.boxes.xyxy.cpu().numpy().astype(int)  # Bounding boxes

    print(f"Detected {len(masks)} masks")

    # Filter masks by score and optionally area
    for i, (mask, score, box) in enumerate(zip(masks, scores, boxes)):
        if score < 0.96:
            continue

        area = np.sum(mask)
        if area < 500:
            continue

        binary_mask = (mask * 255).astype(np.uint8)
        cv2.imwrite(f"mask_{i}_score{score:.2f}.png", binary_mask)

else:
    print("No masks detected.")