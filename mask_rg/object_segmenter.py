import torch
import numpy as np
import cv2
import os
from torchvision.transforms import functional as TF
import numpy as np

from mask_rg.train_maskrcnn import get_model_instance_segmentation
from utils.constants import *

class ObjectSegmenter:
    """
    Mask R-CNN Output Format: 
    {
        'boxes': [],
        'labels': [],
        'scores': [],
        'masks': []
    }
    """
    def __init__(self) -> None:
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.mask_model = get_model_instance_segmentation(2)
        self.mask_model.load_state_dict(torch.load("downloads/maskrcnn.pth", map_location=self.device))
        self.mask_model = self.mask_model.to(self.device)
        self.mask_model.eval()

        # TODO, 0.9 can be tuned
        if IS_REAL:
            self.threshold = 0.97
        else:
            self.threshold = 0.98

    @torch.no_grad()
    def from_maskrcnn(self, color_image, dir=TRAIN_EPISODES_DIR, bbox=False):
        """
        Use Mask R-CNN to do instance segmentation and output masks in binary format.
        """
        image = color_image.copy()
        image = TF.to_tensor(image)
        self.prediction = self.mask_model([image.to(self.device)])

        processed_masks = []
        raw_masks = []
        bboxes = []

        pred_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
        prediction = self.prediction[0]

        for idx, mask in enumerate(prediction["masks"]):
            if prediction["scores"][idx] > self.threshold:
                # get mask
                img = mask[0].mul(255).byte().cpu().numpy()
                img = cv2.GaussianBlur(img, (3, 3), 0)
                img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                if np.sum(img == 255) < 100:
                    continue
                
                processed_masks.append(img)
                raw_masks.append(mask)
                pred_mask[img > 0] = 255 - idx * 20
                name = str(idx) + "mask.png"
                cv2.imwrite(os.path.join(dir, name), img)

                bboxes.append(prediction["boxes"][idx].tolist())

        cv2.imwrite(os.path.join(dir, "scene.png"), pred_mask)
        if bbox:
            return processed_masks, pred_mask, raw_masks, bboxes
        
        return processed_masks, pred_mask, raw_masks
