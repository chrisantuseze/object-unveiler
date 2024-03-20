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
    def __init__(self, args, feat_extractor):
        super(ObstacleHead, self).__init__()
        self.args = args
        self.feat_extractor = feat_extractor
        self.final_conv_units = 128
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        hidden_dim = 10 * 144
        self.projection = nn.Sequential(
            nn.Linear(41472, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        )

    # def show_images(self, obj_masks, raw_object_masks, target_mask, scenes, optimal_nodes):
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

    def forward(self, scene_mask, target_mask, object_masks):
    # def forward(self, scene_mask, target_mask, object_masks, raw_scene_mask, raw_target_mask, raw_object_masks):
        target_feats = self.feat_extractor(target_mask)
        scene_feats = self.feat_extractor(scene_mask)

        x = torch.cat([target_feats, scene_feats], dim=1)
        attn_scores = self.projection(x.reshape(x.shape[0], -1))

        object_masks = object_masks.squeeze(2)
        padding_masks = (object_masks.sum(dim=(2, 3)) == 0)
        padding_mask_expanded = padding_masks.expand_as(attn_scores)
        attn_scores = attn_scores.masked_fill_(padding_mask_expanded, float('-inf'))
        
        attn_weights = torch.softmax(attn_scores, dim=1)
        # print("attn_weights", attn_weights)
        _, top_indices = torch.topk(attn_weights, k=self.args.sequence_length, dim=1)

        B, C, H, W = target_feats.shape

        ###### Keep overlapped objects #####
        processed_objects = []

        raw_objects = []
        for i in range(B):
            idx = top_indices[i] 
            x = object_masks[i, idx] # x should be (4, 400, 400)
            processed_objects.append(x)

        # ################### THIS IS FOR VISUALIZATION ####################
        #     raw_x = raw_object_masks[i, idx]
        #     # print("raw_x.shape", raw_x.shape)
        #     raw_objects.append(raw_x)

        # raw_objects = torch.stack(raw_objects)

        # # numpy_image = (raw_objects[0].numpy() * 255).astype(np.uint8)
        # # cv2.imwrite(os.path.join(TEST_DIR, "best_obstacle.png"), numpy_image)
            
        # self.show_images(raw_objects, raw_target_mask, raw_scene_mask, optimal_nodes=None, eval=True)
        # ###############################################################
            
        processed_objects = torch.stack(processed_objects)
        return attn_weights

class ResFCN(nn.Module):
    def __init__(self, args):
        super(ResFCN, self).__init__()

        self.args = args
        self.nr_rotations = 16
        self.final_conv_units = 128
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.xavier_uniform_(self.conv1.weight)

        self.rb1 = self.make_layer(64, 128)
        self.rb2 = self.make_layer(128, 256)
        self.rb3 = self.make_layer(256, 512)
        self.rb4 = self.make_layer(512, 256)
        self.rb5 = self.make_layer(256, 128)
        self.rb6 = self.make_layer(128, 64)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.obstacle_head = ObstacleHead(args, self.predict) 

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
        x = self.rb6(x)
       
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.final_conv(x)
        return out
   
    def forward(self, depth_heightmap, target_mask, object_masks, scene_masks, specific_rotation=-1, is_volatile=[]):
    # def forward(self, depth_heightmap, target_mask, object_masks, scene_masks, raw_scene_mask, raw_target_mask, raw_object_masks, gt_object=None, specific_rotation=-1, is_volatile=[]):
        
        object_scores = self.obstacle_head(scene_masks, target_mask, object_masks)
        # object_scores = self.obstacle_head(scene_masks, target_mask, object_masks, raw_scene_mask, raw_target_mask, raw_object_masks)

        # B, N, C, H, W = object_masks.shape
        # out_probs = torch.rand(B, self.args.sequence_length, C, H, W)
        # out_probs = Variable(out_probs, requires_grad=True).to(self.device)

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