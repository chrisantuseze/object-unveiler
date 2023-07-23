import torch
import numpy as np
import cv2
import os
import imutils
from torchvision.transforms import functional as TF

from vision.train_maskrcnn import get_model_instance_segmentation
from utils.constants import *

class ObjectSegmenter:
    def __init__(self) -> None:
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.mask_model = get_model_instance_segmentation(2)
        self.mask_model.load_state_dict(torch.load("downloads/maskrcnn.pth", map_location=self.device))
        self.mask_model = self.mask_model.to(self.device)
        self.mask_model.eval()

    @torch.no_grad()
    def from_maskrcnn(self, color_image, depth_image, plot=False):
        """
        Use Mask R-CNN to do instance segmentation and output masks in binary format.
        """
        image = color_image.copy()
        image = TF.to_tensor(image)
        prediction = self.mask_model([image.to(self.device)])[0]
        mask_objs = []

        if plot:
            pred_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
        
        for idx, mask in enumerate(prediction["masks"]):
            # TODO, 0.9 can be tuned
            if IS_REAL:
                threshold = 0.97
            else:
                threshold = 0.98
            
            if prediction["scores"][idx] > threshold:
                # get mask
                img = mask[0].mul(255).byte().cpu().numpy()
                img = cv2.GaussianBlur(img, (3, 3), 0)
                img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                if np.sum(img == 255) < 100:
                    continue
                
                mask_objs.append(img)
                if plot:
                    pred_mask[img > 0] = 255 - idx * 20
                    name = str(idx) + "mask.png"
                    cv2.imwrite(os.path.join("save/misc", name), img)
        if plot:
            cv2.imwrite(os.path.join("save/misc", "pred.png"), pred_mask)

        print("Mask R-CNN: %d objects detected" % len(mask_objs), prediction["scores"].cpu())
        
        return mask_objs