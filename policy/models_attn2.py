import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict
from policy.object_segmenter import ObjectSegmenter
import policy.grasping as grasping
import utils.general_utils as general_utils
import matplotlib.pyplot as plt

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
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(identity)

        out += identity
        out = F.relu(out)

        return out

class ResFCN(nn.Module):
    def __init__(self, args):
        super(ResFCN, self).__init__()

        self.args = args
        self.nr_rotations = 16
        self.final_conv_units = 128
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.rb1 = self.make_layer(64, 128)
        self.rb2 = self.make_layer(128, 256)
        self.rb3 = self.make_layer(256, 512)
        self.rb4 = self.make_layer(512, 256)
        self.rb5 = self.make_layer(256, 128)
        self.rb6 = self.make_layer(128, 64)
        # self.final_conv = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)

        # Learnable projection matrices  
        # self.target_proj = nn.Linear(self.final_conv_units, 256)  
        # self.obj_proj = nn.Linear(self.final_conv_units, 256)      

        self.mlp = nn.Sequential(
            nn.Linear(self.final_conv_units, 256),
            nn.ReLU(), 
            nn.Linear(256, 1)
        ) 

    def make_layer(self, in_channels, out_channels, blocks=1, stride=1):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(conv3x3(in_channels, out_channels, stride=stride))

        layers = [ResidualBlock(in_channels, out_channels, stride, downsample)]
        for i in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)
    
    def predict(self, depth, final_feats=False):
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
        if final_feats:
            out = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False).to(self.device)(x)
        else:
            out = nn.Conv2d(64, self.final_conv_units, kernel_size=1, stride=1, padding=0, bias=False).to(self.device)(x)

        return out
    
    def get_obj_feats(self, obj_masks, scene_feats):
        obj_features = []
        for mask in obj_masks:
            # print("mask.shape", mask.shape)

            mask = mask.unsqueeze(0).to(self.device)
            obj_feat = self.predict(mask)

            masked_fmap = obj_feat #scene_feats * obj_feat
            # print("masked_fmap.shape", masked_fmap.shape)

            obj_feat = masked_fmap.reshape(1, masked_fmap.shape[1], -1)[:, :, 0]
            obj_features.append(obj_feat)

        return torch.cat(obj_features)
    
    def preprocess_input(self, scene_masks, object_masks):
        B, H, W, C = scene_masks.shape

        obj_features = torch.zeros(B, self.args.num_patches, self.final_conv_units).to(self.device)

        for i in range(B):
            scene_mask = scene_masks[i].unsqueeze(0).to(self.device)
            object_mask = object_masks[i].to(self.device)

            scene_feat = self.predict(scene_mask)
            # print("scene_feat.shape", scene_feat.shape)

            obj_feats = self.get_obj_feats(object_mask, scene_feat)
            # print("obj_feats.shape", obj_feats.shape)
            
            obj_features[i] = obj_feats
            
        # self.show_images2(obj_masks)
        return obj_features

    def forward(self, scene_mask, target_mask, object_masks, raw_scene_mask, raw_target_mask, raw_object_masks, specific_rotation=-1, is_volatile=[]):
        # print("scene_mask.shape", scene_mask.shape)

        obj_features = self.preprocess_input(scene_mask, object_masks)

        # compute rotated feature maps            
        scene_feats = self.predict(scene_mask)
        # print("scene_feats.shape", scene_feats.shape)

        target_feats = self.predict(target_mask)
        # print("target_feats.shape", target_feats.shape)

        masked_target_fmap = target_feats #scene_feats * target_feats
        # print("masked_target_fmap.shape", masked_target_fmap.shape)

        masked_target_fmap = masked_target_fmap.reshape(masked_target_fmap.shape[0], masked_target_fmap.shape[1], -1)[:, :, 0]
        # print("masked_target_fmap.shape", masked_target_fmap.shape)

        # Project target and object features  
        projected_target = masked_target_fmap #self.target_proj(masked_target_fmap)
        print("projected_target.shape", projected_target.shape)

        projected_objs = obj_features #self.obj_proj(obj_features)
        print("projected_objs.shape", projected_objs.shape)
        B, N, C, = projected_objs.shape

        top_indices, top_scores = self.get_topk_attn_scores(projected_objs, projected_target, raw_object_masks)

        # print("obj_masks.shape", obj_masks.shape)

        ###### Keep overlapped objects #####
        objs = []

        raw_objs = []
        for i in range(B):
            idx = top_indices[i] 
            x = object_masks[i, idx]
            # print("x.shape", x.shape) # Should be (4, 400, 400)
            objs.append(x)

            raw_x = raw_object_masks[i, idx]
            raw_objs.append(raw_x)

        raw_objs = np.array(raw_objs)

        overlapped_objs = torch.stack(objs)
        # print("overlapped_objs.shape", overlapped_objs.shape)

        ########################### VIZ ################################

        self.show_images(raw_objs, raw_target_mask, raw_scene_mask)

        ################################################################


        ##### process obj masks and extract features #####
        objs_2 = []
        for i in range(overlapped_objs.shape[0]): # batch level
            objs_1 = []
            for j in range(overlapped_objs.shape[1]): # channel (num objects) level 
                obj = overlapped_objs[i][j].unsqueeze(0).to(self.device)
                # print("obj.shape", obj.shape)

                obj_feat = self.predict(obj, final_feats=True)
                # print("obj_feat.shape", obj_feat.shape)

                objs_1.append(obj_feat)

            objs = torch.stack(objs_1)
            objs_2.append(objs)
            
        overlapped_objs_feats = torch.stack(objs_2).squeeze(2).to(self.device)
        # print("overlapped_objs_feats.shape", overlapped_objs_feats.shape)

        # Predict boxes
        B, N, C, H, W = overlapped_objs_feats.shape
        reshaped_overlapped = overlapped_objs_feats.view(B * N, H * W)
        out_prob = reshaped_overlapped
        # print("out_prob.shape", out_prob.shape)

        # Image-wide softmax
        output_shape = out_prob.shape
        out_prob = out_prob.view(output_shape[0], -1)
        out_prob = torch.softmax(out_prob, dim=1)
        out_prob = out_prob.view(B, N, C, H, W).to(dtype=torch.float)

        return out_prob
    
    def get_topk_attn_scores(self, projected_objs, projected_target, raw_object_masks):
        # Scaled dot-product attention
        # Perform element-wise multiplication with broadcasting and Sum along the last dimension to get the final [2, 14] tensor
        attn_scores = (projected_target.unsqueeze(1) * projected_objs).sum(dim=-1)/np.sqrt(projected_objs.shape[-1])

        # attn_scores = self.mlp(projected_objs + projected_target.unsqueeze(1)).squeeze(2)

        padding_masks = (raw_object_masks.sum(dim=(2, 3)) == 0)
        # print("padding_masks.shape", padding_masks.shape)

        # Expand the mask to match the shape of A
        padding_mask_expanded = padding_masks.expand_as(attn_scores)

        # Zero out the corresponding entries in A using the mask
        attn_scores = attn_scores.masked_fill_(padding_mask_expanded, float('-inf'))

        attn_scores = F.softmax(attn_scores, dim=0)
        # attn_scores = nn.CosineSimilarity(dim=-1)(projected_target.unsqueeze(1), projected_objs)

        # Create a mask for NaN values
        nan_mask = torch.isnan(attn_scores)

        # Replace NaN values with a specific value (e.g., 0.0)
        attn_scores = torch.where(nan_mask, torch.tensor(0.0), attn_scores)

        print("attn_scores.shape", attn_scores.shape) # [B,N]
        print("attn_scores", attn_scores)

        # Use torch.topk to get the top k values and their indices
        top_scores, top_indices = torch.topk(attn_scores, k=self.args.sequence_length, dim=1)
        print("top_scores", top_scores)
        print("top_indices", top_indices)

        return top_indices, top_scores
    
    def show_images(self, obj_masks, target_mask, scenes):
        fig, ax = plt.subplots(obj_masks.shape[0], obj_masks.shape[1] + 2)

        for i in range(obj_masks.shape[0]):
            ax[i][0].imshow(scenes[i])
            k = 1
            for j in range(obj_masks.shape[1]):
                obj_mask = obj_masks[i][j]
                # print("obj_mask.shape", obj_mask.shape)
                ax[i][k].imshow(obj_mask)
                k += 1

            ax[i][k].imshow(target_mask[i])
        plt.show()

    def show_images2(self, obj_masks):
        fig, ax = plt.subplots(obj_masks.shape[0], obj_masks.shape[1])

        for i in range(obj_masks.shape[0]):
            for j in range(obj_masks.shape[1]):
                obj_mask = obj_masks[i][j]
                # print("obj_mask.shape", obj_mask.shape)
                ax[i][j].imshow(obj_mask)
        plt.show()


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