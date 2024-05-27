import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from utils.constants import TEST_DIR

def compute_edge_features(boxes, masks, target_mask):
    """
    Compute pairwise spatial relationships between objects and the target object.

    Args:
        boxes (Tensor): Bounding boxes of detected objects with shape (num_objects, 4).
        masks (Tensor): Segmentation masks of detected objects with shape (num_objects, H, W).
        target_mask (Tensor): Segmentation mask of the target object with shape (1, H, W).

    Returns:
        edge_features (Tensor): Tensor of edge features with shape (num_edges, edge_feat_dim).
        edges (Tensor): Tensor of edge indices with shape (num_edges, 2).
    """
    device = boxes.device
    num_objects = boxes.size(0)

    # Compute pairwise object-target relationships
    edges = []
    edge_features = []

    for i in range(num_objects):
        # Compute spatial features between object i and the target object
        obj_mask = masks[i].unsqueeze(0)  # Shape: (1, H, W)
        # print("obj_mask.shape", obj_mask.shape, "target_mask.shape", target_mask.shape)

        target_overlap = torch.sum(obj_mask * target_mask.unsqueeze(1)).item()  # Overlap between object and target
        target_iou = calculate_iou(boxes[i], target_mask)  # IoU between object and target

        # Compute edge features
        edge_feat = torch.tensor([target_overlap, target_iou], device=device)
        edge_features.append(edge_feat)

        # Add edge index
        edges.append([i, num_objects])  # Connect object node to a dummy target node

    edge_features = torch.stack(edge_features, dim=0)
    edges = torch.tensor(edges, dtype=torch.long, device=device)

    return edge_features, edges

def calculate_iou(box, target_mask):
    """
    Calculate the Intersection over Union (IoU) between a bounding box and a target mask.

    Args:
        box (Tensor): A tensor of shape (4,) representing the bounding box in (x1, y1, x2, y2) format.
        target_mask (Tensor): A tensor of shape (H, W) representing the target mask.

    Returns:
        iou (float): The Intersection over Union between the box and the target mask.
    """
    # Convert box to (x1, y1, x2, y2) format
    # print("box.shape", box.shape)
    x1, y1, x2, y2 = box

    # Create a mask for the bounding box
    box_mask = torch.zeros_like(target_mask)
    box_mask[int(y1):int(y2), int(x1):int(x2)] = 1

    # Calculate the intersection and union
    intersection = torch.sum(box_mask * target_mask)
    union = torch.sum(box_mask) + torch.sum(target_mask) - intersection

    # Avoid division by zero
    iou = intersection / (union + 1e-8)

    return iou.item()

class ObstacleHead(nn.Module):
    def __init__(self, args):
        super(ObstacleHead, self).__init__()
        self.args = args
        self.final_conv_units = 128
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        hidden_dim = 1024
        dimen = hidden_dim//2
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, hidden_dim)

        self.attn = nn.Sequential(
            nn.Linear(self.args.num_patches * (hidden_dim + dimen), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, self.args.num_patches)
        )

        self.object_rel_fc = nn.Sequential(
            nn.Linear(self.args.num_patches * 2, dimen),
            nn.LayerNorm(dimen),
            nn.ReLU(),
            nn.Linear(dimen, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dimen),
            nn.LayerNorm(dimen),
            nn.ReLU(),
            nn.Linear(dimen, self.args.num_patches * dimen)
        )

        self.W_t = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.args.num_patches * dimen)
        )

        self.W_o = nn.Sequential(
            nn.Linear(self.args.num_patches * hidden_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.args.num_patches * dimen)
        )

    def preprocess_inputs(self, scene_mask, target_mask, object_masks):
        B, N, C, H, W = object_masks.shape

        scene_mask = scene_mask.repeat(1, 3, 1, 1)
        scene_feats = self.model(scene_mask)
        scene_feats = scene_feats.view(B, 1, -1)

        # scene_feats = None

        target_mask = target_mask.repeat(1, 3, 1, 1)
        target_feats = self.model(target_mask)
        target_feats = target_feats.view(B, 1, -1)

        object_masks = object_masks.repeat(1, 1, 3, 1, 1)
        object_masks = object_masks.view(-1, 3, H, W)
        object_feats = self.model(object_masks)
        object_feats = object_feats.view(B, N, -1)
        # print(object_feats.shape)

        return scene_feats, target_feats, object_feats

    def compute_edge_features(self, bboxes, object_masks, target_mask):
        B, N, C, H, W = object_masks.shape

        edge_features = []
        periphery_dists = []
        for i in range(B):
            edge_feats, _ = compute_edge_features(bboxes[i], object_masks[i], target_mask[i])
            edge_features.append(edge_feats)

            # periphery_dist = compute_objects_periphery_dist(object_masks[i])
            # periphery_dists.append(periphery_dist)

        edge_features = torch.stack(edge_features).to(self.device)
        # periphery_dists = torch.stack(periphery_dists).to(self.device)
        # print("edge_features.shape", edge_features.shape)

        return edge_features, periphery_dists
    
    def scaled_dot_product_attention(self, object_feats, scene_feats, objects_rel):
        B, N, D = object_feats.shape

        scene_feats = scene_feats.reshape(B, -1)
        query = self.W_t(scene_feats).view(B, N, -1)

        object_feats = object_feats.reshape(B, -1)
        key = self.W_o(object_feats).view(B, N, -1)

        value = objects_rel

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(key.size(-1))
        weights = F.softmax(scores, dim=-1)

        # Apply attention weights to the value
        weighted_values = torch.matmul(weights, value)

        return weighted_values
    
    def cross_attention(self, target_feats, object_feats):
        B, N, D = object_feats.shape

        target_feats = target_feats.reshape(B, -1)
        query1 = key1 = value1 = self.W_t(target_feats).view(B, N, -1)

        object_feats = object_feats.reshape(B, -1)
        query2 = key2 = value2 = self.W_o(object_feats).view(B, N, -1)

        attn_scores1 = torch.matmul(query1, key2.transpose(-2, -1)) / math.sqrt(key1.size(-1))
        attn_scores2 = torch.matmul(query2, key1.transpose(-2, -1)) / math.sqrt(key2.size(-1))

        attn_probs1 = F.softmax(attn_scores1, dim=-1)
        attn_probs2 = F.softmax(attn_scores2, dim=-1)

        attn_output1 = torch.matmul(attn_probs1, value2)
        attn_output2 = torch.matmul(attn_probs2, value1)

        output = attn_output1 + attn_output2
        
        return output

    def spatial_rel(self, scene_mask, target_mask, object_masks, bboxes):
        scene_feats, target_feats, object_feats = self.preprocess_inputs(scene_mask, target_mask, object_masks)
        B, N, C, H, W = object_masks.shape

        objects_rel, periphery_dists = self.compute_edge_features(bboxes, object_masks, target_mask)
        # print("objects_rel", objects_rel.shape, "periphery_dists", periphery_dists.shape)

        object_rel_feats = self.object_rel_fc(objects_rel.view(B, -1)).view(B, N, -1)
        # print("object_rel_feats.shape", object_rel_feats.shape)

        attn_output = self.scaled_dot_product_attention(object_feats, scene_feats, object_rel_feats)
        # attn_output = self.cross_attention(target_feats, object_feats)
        # print("attn_output.shape", attn_output.shape)

        out = torch.cat([attn_output, object_feats], dim=-1)
        # print("out.shape", out.shape)

        attn_scores = self.attn(out.reshape(B, -1))

        object_masks = object_masks.squeeze(2)
        padding_masks = (object_masks.sum(dim=(2, 3)) == 0)
        padding_mask_expanded = padding_masks.expand_as(attn_scores)
        attn_scores = attn_scores.masked_fill_(padding_mask_expanded, float(-1e-6))
        
        _, top_indices = torch.topk(attn_scores, k=self.args.sequence_length, dim=1)
        # print("top indices", top_indices)

        return attn_scores, top_indices
    
    # def forward(self, scene_mask, target_mask, object_masks, raw_scene_mask, raw_target_mask, raw_object_masks, bboxes):
    def forward(self, scene_mask, target_mask, object_masks, bboxes):
        attn_scores, top_indices = self.spatial_rel(scene_mask, target_mask, object_masks, bboxes)

        # ################### THIS IS FOR VISUALIZATION ####################
        # raw_objects = []
        # for i in range(target_mask.shape[0]):
        #     idx = top_indices[i] 
        #     x = object_masks[i, idx] # x should be (4, 400, 400)

        #     raw_x = raw_object_masks[i, idx]
        #     # print("raw_x.shape", raw_x.shape)
        #     raw_objects.append(raw_x)

        # raw_objects = torch.stack(raw_objects)
        # self.show_images(raw_objects, raw_target_mask, raw_scene_mask, optimal_nodes=None, eval=True)
        # ##################################################################

        return attn_scores
   
    def show_images(self, obj_masks, target_mask, scenes, optimal_nodes=None, eval=False):
        # fig, ax = plt.subplots(obj_masks.shape[0] * 2, obj_masks.shape[1] + 2)
        fig, ax = plt.subplots(obj_masks.shape[0], obj_masks.shape[1] + 2)

        #save the top object
        # Convert to uint8 and scale to [0, 255]
        numpy_image = (obj_masks[0][0].numpy() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(TEST_DIR, "best_obstacle.png"), numpy_image)

        for i in range(obj_masks.shape[0]):
            if obj_masks.shape[0] == 1:
                ax[i].imshow(scenes[i]) # this is because of the added gt images
            else:
                ax[i][0].imshow(scenes[i])

            if obj_masks.shape[0] == 1:
                ax[i+1].imshow(target_mask[i])
            else:
                ax[i][1].imshow(target_mask[i])

            k = 2
            for j in range(obj_masks.shape[1]):
                obj_mask = obj_masks[i][j]
                # print("obj_mask.shape", obj_mask.shape)

                if obj_masks.shape[0] == 1:
                    ax[k].imshow(obj_mask)
                else:
                    ax[i][k].imshow(obj_mask)
                k += 1


        if optimal_nodes:
            n = 0
            for i in range(2, raw_object_masks.shape[0] + 2):

                gt_obj_masks = raw_object_masks[n]
                # print("gt_obj_masks.shape", gt_obj_masks.shape)

                gt_obj_masks = gt_obj_masks[optimal_nodes[n], :, :]
                # print("gt_obj_masks.shape", gt_obj_masks.shape, "\n")

                if gt_obj_masks.shape[0] == 1:
                    ax[i][0].imshow(scenes[n]) # this is because of the added gt images
                else:
                    ax[i][0].imshow(scenes[n])

                k = 1
                for j in range(obj_masks.shape[1]):
                    gt_obj_mask = gt_obj_masks[j]
                    # print("obj_mask.shape", obj_mask.shape)

                    if gt_obj_masks.shape[0] == 1:
                        ax[i][k].imshow(gt_obj_mask)
                    else:
                        ax[i][k].imshow(gt_obj_mask)
                    k += 1

                if gt_obj_masks.shape[0] == 1:
                    ax[i][k].imshow(target_mask[n])
                else:
                    ax[i][k].imshow(target_mask[n])

                n += 1

        plt.show()

class ResFCN(nn.Module):
    def __init__(self, args):
        super(ResFCN, self).__init__()

        self.args = args
        self.nr_rotations = 16
        self.final_conv_units = 128
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.obstacle_head = ObstacleHead(args) 

    def forward(self, depth_heightmap, target_mask, object_masks, scene_masks, bboxes, specific_rotation=-1, is_volatile=[]):
    # def forward(self, depth_heightmap, target_mask, object_masks, scene_masks, raw_scene_mask, raw_target_mask, raw_object_masks, bboxes=None, specific_rotation=-1, is_volatile=[]):

        # object_scores = self.obstacle_head(depth_heightmap, target_mask, object_masks, raw_scene_mask, raw_target_mask, raw_object_masks, bboxes)
        object_scores = self.obstacle_head(depth_heightmap, target_mask, object_masks, bboxes)

        # B, N, C, H, W = object_masks.shape
        # out_probs = torch.rand(16, C, H, W)
        # out_probs = Variable(out_probs, requires_grad=True).to(self.device)
        # return object_scores, out_probs
    
        return object_scores
    

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, 1024)
        self.fc1 = nn.Linear(1024, 256)
        self.fc21 = nn.Linear(256, 1)
        self.fc22 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.model(x)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc21(x))
        return x