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

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)

        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.downsample = downsample

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = F.dropout(out, p=0.2)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(identity)

        out += identity
        out = F.relu(out)

        return out


class ObstacleHead(nn.Module):
    def __init__(self, args):
        super(ObstacleHead, self).__init__()
        self.args = args
        self.final_conv_units = 128
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


        ############# cc ######################
        hidden_dim = 1024
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, 2048)

        self.attn = nn.Sequential(
            nn.Linear(self.args.num_patches * 2048, self.args.num_patches * hidden_dim),
            nn.BatchNorm1d(self.args.num_patches * hidden_dim),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.args.num_patches * (hidden_dim + 4), hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.args.num_patches)
        )
        ######################################

        # ############ pete ##################
        # hidden_dim = 1024
        # self.model = torchvision.models.resnet50(pretrained=True)
        # self.model.fc = nn.Linear(2048, 2048)

        # self.attn = nn.Sequential(
        #     nn.Linear(self.args.num_patches * 2048, self.args.num_patches * hidden_dim),
        #     nn.BatchNorm1d(self.args.num_patches * hidden_dim),
        #     nn.ReLU(),
        # )

        # self.fc = nn.Sequential(
        #     nn.Linear(self.args.num_patches * hidden_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, self.args.num_patches)
        # )
        # ########################################

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
        # print(object_feats.shape)

        return scene_feats, target_feats, object_feats

    def spatial_rel(self, scene_mask, target_mask, object_masks, bboxes):
        scene_feats, target_feats, object_feats = self.preprocess_inputs(scene_mask, target_mask, object_masks)

        B, N, C, H, W = object_masks.shape

        attn_weights = (target_feats * object_feats)/np.sqrt(object_feats.shape[1])
        soft_attn = torch.softmax(attn_weights, dim=1) * scene_feats
        # print(soft_attn.shape)

        attn_scores = self.attn((object_feats - soft_attn).view(B, -1))
        # print(attn_scores.shape)

        x = torch.cat([attn_scores.view(B, N, -1), bboxes.view(B, N, -1)], dim=2).view(B, -1)
        # print("x.shape", x.shape)

        attn_scores = self.fc(x)

        object_masks = object_masks.squeeze(2)
        padding_masks = (object_masks.sum(dim=(2, 3)) == 0)
        padding_mask_expanded = padding_masks.expand_as(attn_scores)
        attn_scores = attn_scores.masked_fill_(padding_mask_expanded, float(-1e-6))
        
        _, top_indices = torch.topk(attn_scores, k=self.args.sequence_length, dim=1)
        print("top indices", top_indices)

        return attn_scores, top_indices
    

    def spatial_rel_pete(self, scene_mask, target_mask, object_masks, bboxes):
        scene_feats, target_feats, object_feats = self.preprocess_inputs(scene_mask, target_mask, object_masks)

        B, N, C, H, W = object_masks.shape

        attn_weights = (target_feats * object_feats)/np.sqrt(object_feats.shape[1])
        soft_attn = torch.softmax(attn_weights, dim=1) * scene_feats
        # print(soft_attn.shape)

        attn_scores = self.attn((object_feats - soft_attn).view(B, -1))
        # print(attn_scores.shape)
        x = attn_scores

        # attn_scores += object_feats
        # print(attn_scores.shape)

        # x = torch.cat([attn_scores.view(B, N, -1), bboxes.view(B, N, -1)], dim=2).view(B, -1)
        # print("x.shape", x.shape)

        attn_scores = self.fc(x)

        object_masks = object_masks.squeeze(2)
        padding_masks = (object_masks.sum(dim=(2, 3)) == 0)
        padding_mask_expanded = padding_masks.expand_as(attn_scores)
        attn_scores = attn_scores.masked_fill_(padding_mask_expanded, float(-1e-6))

        _, top_indices = torch.topk(attn_scores, k=self.args.sequence_length, dim=1)
        # print("top indices", top_indices)

        return attn_scores, top_indices

    def forward(self, scene_mask, target_mask, object_masks, bboxes):
        attn_weights, top_indices = self.spatial_rel(scene_mask, target_mask, object_masks, bboxes)

        return attn_weights
   

class ResFCN(nn.Module):
    def __init__(self, args):
        super(ResFCN, self).__init__()

        self.args = args
        self.nr_rotations = 16
        self.final_conv_units = 128
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # nn.init.xavier_uniform_(self.conv1.weight)

        # self.rb1 = self.make_layer(64, 128)
        # self.rb2 = self.make_layer(128, 256)
        # self.rb3 = self.make_layer(256, 512)
        # self.rb4 = self.make_layer(512, 256)
        # self.rb5 = self.make_layer(256, 128)
        # self.rb6 = self.make_layer(128, 64)
        # self.final_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.obstacle_head = ObstacleHead(args) 

    def make_layer(self, in_channels, out_channels, blocks=1, stride=1):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(conv3x3(in_channels, out_channels, stride=stride))

        layers = [ResidualBlock(in_channels, out_channels, stride, downsample)]
        for i in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)
    
    def predict(self, depth):
        x = F.relu(self.conv1(depth))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = self.rb1(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.rb5(x)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.rb6(x) # half the channel
       
        # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) # multiply H and W
        out = self.final_conv(x)
        return out

    # def forward(self, depth_heightmap, target_mask, object_masks, scene_masks, bboxes, specific_rotation=-1, is_volatile=[]):
    def forward(self, depth_heightmap, target_mask, object_masks, scene_masks, raw_scene_mask, raw_target_mask, raw_object_masks, gt_object=None, bboxes=None, specific_rotation=-1, is_volatile=[]):

        object_scores = self.obstacle_head(depth_heightmap, target_mask, object_masks, bboxes)

        B, N, C, H, W = object_masks.shape
        out_probs = torch.rand(16, C, H, W)
        out_probs = Variable(out_probs, requires_grad=True).to(self.device)
        return object_scores, out_probs
    
        # return object_scores
    

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