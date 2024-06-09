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
from policy.vit import VisualTransformer

class ObstacleHead(nn.Module):
    def __init__(self, args):
        super(ObstacleHead, self).__init__()
        self.args = args
        self.final_conv_units = 128
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        vision_patch_size = 32
        vision_width = 224
        vision_layers = 2#12
        vision_heads = 2#8
        embed_dim = 512
        image_resolution  = vision_width * vision_patch_size

        self.visual = VisualTransformer(
            args=args,
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )

        hidden_dim = 1024
        self.model = torchvision.models.resnet18(pretrained=True)
        # self.model.fc = nn.Linear(2048, hidden_dim)
        self.model.fc = nn.Linear(512, hidden_dim)

        self.fc_t = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.args.num_patches * hidden_dim)
        )

        # self.mlp = nn.Sequential(
        #     nn.Linear(hidden_dim//2, hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim//2),
        #     nn.LayerNorm(hidden_dim//2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim//2, self.args.num_patches)
        # )

        ############## FOR SPATIAL #########################
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * vision_width, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, self.args.num_patches)
        )

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    def preprocess_inputs(self, scene_mask, target_mask, object_masks):
        B, N, C, H, W = object_masks.shape

        scene_mask = scene_mask.repeat(1, 3, 1, 1)
        scene_feats = self.model(scene_mask)
        scene_feats = scene_feats.view(B, 1, -1)

        target_mask = target_mask.repeat(1, 3, 1, 1)
        target_feats = self.model(target_mask)
        target_feats = target_feats.view(B, 1, -1)

        object_masks = object_masks.repeat(1, 1, 3, 1, 1)
        object_masks = object_masks.view(-1, 3, H, W)
        object_feats = self.model(object_masks)
        object_feats = object_feats.view(B, N, -1)

        return scene_feats, target_feats, object_feats
    
    def scaled_dot_product_attention(self, object_feats, target_feats):
        B, N, D = object_feats.shape

        target_feats = target_feats.reshape(B, -1)
        query = self.fc_t(target_feats).view(B, N, -1)

        key = object_feats
        value = query

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(key.size(-1))
        weights = F.softmax(scores, dim=-1)

        # Apply attention weights to the value
        weighted_values = torch.matmul(weights, value)

        return weighted_values
    
    # def forward(self, scene_mask, target_mask, object_masks, raw_scene_mask, raw_target_mask, raw_object_masks):
    def forward(self, scene_mask, target_mask, object_masks):
        scene_feats, target_feats, object_feats = self.preprocess_inputs(scene_mask, target_mask, object_masks)
        joint_feats = self.scaled_dot_product_attention(object_feats, target_feats)

        input = torch.cat([scene_feats, joint_feats], dim=1)
        # print("input.shape", input.shape)

        # feats = self.visual(input)
        feats = self.visual.forward_spatial(input)
        # print("feats.shape", feats.shape)

        out = self.mlp(feats)

        object_masks = object_masks.squeeze(2)
        padding_masks = (object_masks.sum(dim=(2, 3)) == 0)
        padding_mask_expanded = padding_masks.expand_as(out)
        out = out.masked_fill_(padding_mask_expanded, float(-1e-6))
        
        # _, top_indices = torch.topk(out, k=self.args.sequence_length, dim=1)
        # print("top indices", top_indices)

        # ################### THIS IS FOR VISUALIZATION ####################
        # raw_objects = []
        # for i in range(target_mask.shape[0]):
        #     idx = top_indices[i] 
        #     x = object_masks[i, idx] # x should be (4, 400, 400)

        #     raw_x = raw_object_masks[i, idx]
        #     # print("raw_x.shape", raw_x.shape)
        #     raw_objects.append(raw_x)

        # raw_objects = torch.stack(raw_objects)
        # self.show_images(raw_objects, raw_target_mask, raw_scene_mask)
        # ##################################################################

        return out

    def show_images(self, obj_masks, target_mask, scenes):
        fig, ax = plt.subplots(obj_masks.shape[0], obj_masks.shape[1] + 2)

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

                if obj_masks.shape[0] == 1:
                    ax[k].imshow(obj_mask)
                else:
                    ax[i][k].imshow(obj_mask)
                k += 1

        plt.show()

class ResFCN(nn.Module):
    def __init__(self, args):
        super(ResFCN, self).__init__()

        self.args = args
        self.nr_rotations = 16
        self.final_conv_units = 128
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.obstacle_head = ObstacleHead(args) 

    # def forward(self, depth_heightmap, target_mask, object_masks, scene_masks, raw_scene_mask, raw_target_mask, raw_object_masks, bboxes=None, specific_rotation=-1, is_volatile=[]):

    def forward(self, depth_heightmap, target_mask, object_masks, scene_masks, bboxes, specific_rotation=-1, is_volatile=[]):
        out = self.obstacle_head(scene_masks, target_mask, object_masks)
        return out

        # out = self.obstacle_head(scene_masks, target_mask, object_masks, raw_scene_mask, raw_target_mask, raw_object_masks)
        # B, N, C, H, W = object_masks.shape
        # out_probs = torch.rand(16, C, H, W)
        # out_probs = Variable(out_probs, requires_grad=True).to(self.device)
        # return out, out_probs
    


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