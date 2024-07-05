import os
import torch
from torch.utils import data
import cv2
import numpy as np
import imutils

class SegmentationDataset(data.Dataset):
    """
    Create segmentation dataset for training Mask R-CNN.
    One uses pre-defined color range to separate objects (assume the color in one image is unique).
    One directly reads masks.
    """

    def __init__(self, root, transforms, is_real=False):
        self.root = root
        self.transforms = transforms
        self.is_real = is_real
        # load all image files, sorting them to ensure that they are aligned
        self.color_imgs = list(sorted(os.listdir(os.path.join(root, "color-heightmaps"))))
        self.depth_imgs = list(sorted(os.listdir(os.path.join(root, "depth-heightmaps"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

    def __getitem__(self, idx):
        # load images
        color_path = os.path.join(self.root, "color-heightmaps", self.color_imgs[idx])
        # depth_path = os.path.join(self.root, "depth-heightmaps", self.depth_imgs[idx])

        # color image input
        color_img = cv2.imread(color_path)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # get masks
        hsv = cv2.cvtColor(color_img, cv2.COLOR_RGB2HSV)
        masks = []
        if self.is_real:
            gray = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)
            gray = gray.astype(np.uint8)
            blurred = cv2.medianBlur(gray, 5)
            thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
            cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            cnts = imutils.grab_contours(cnts)
            for c in cnts:
                if cv2.contourArea(c) > 100:
                    mask = np.zeros(color_img.shape[:2], np.uint8)
                    cv2.drawContours(mask, [c], -1, (1), -1)
                    masks.append(mask)
                    # cv2.imshow('mask' + self.color_imgs[idx], mask)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
        else:
            for ci in range(1, np.max(mask_img) + 1):
                mask = mask_img == ci
                if np.sum((mask == True)) > 100:
                    masks.append(mask)

        num_objs = len(masks)
        if num_objs > 0:
            masks = np.stack(masks, axis=0)

        # get bounding box coordinates for each mask
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
            if xmin == xmax or ymin == ymax:
                num_objs = 0

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        if num_objs > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.as_tensor([0], dtype=torch.float32)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        num_objs = torch.tensor(num_objs)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["num_obj"] = num_objs

        if self.transforms is not None:
            # img, target = self.transforms(img, target)
            img, target = self.transforms(color_img, target)

        return img, target

    def __len__(self):
        # return len(self.imgs)
        return len(self.color_imgs)

