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
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.rb1 = self.make_layer(64, 128)
        self.rb2 = self.make_layer(128, 256)
        self.rb3 = self.make_layer(256, 512)
        self.rb4 = self.make_layer(512, 256)
        self.rb5 = self.make_layer(256, 128)
        self.rb6 = self.make_layer(128, 64)
        self.final_conv = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)

        # Learnable projection matrices  
        self.target_proj = nn.Linear(128, 256)  
        self.obj_proj = nn.Linear(128, 256)

        self.image_size = 144
        
        self.box_regression = nn.Linear(
            self.image_size * self.image_size,
            self.image_size * self.image_size
        )

        self.segmenter = ObjectSegmenter()
        

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
    
    def get_obj_feats(self, obj_masks, scene_feats):
        obj_features = []
        for mask in obj_masks:

            padded_mask, _ = general_utils.preprocess_image(mask)
            mask = torch.tensor(padded_mask).unsqueeze(0).to(self.device)
            obj_feat = self.predict(mask)

            masked_fmap = scene_feats * obj_feat
            # print("masked_fmap.shape", masked_fmap.shape)

            obj_feat = masked_fmap.reshape(1, masked_fmap.shape[1], -1)[:, :, 0]
            obj_features.append(obj_feat)

        return torch.cat(obj_features)
    
    def preprocess_input(self, color_images):
        B, H, W, C = color_images.shape

        scene_masks = torch.zeros(B, 1, 144, 144).to(self.device)
        obj_features = torch.zeros(B, self.args.num_patches, 128).to(self.device)

        processed_masks_all = []

        for i in range(B):
            color_image = color_images[i].cpu().numpy()
            # print("color_image.shape", color_image.shape)

            processed_masks, pred_mask, raw_masks = self.segmenter.from_maskrcnn(color_image, plot=True)
            # print("processed_masks.shape", processed_masks[0].shape)
            # print("pred_mask.shape", pred_mask.shape)

            pred_mask, _ = general_utils.preprocess_image(pred_mask)
            # print("pred_mask.shape", pred_mask.shape)

            scene_mask = torch.tensor(pred_mask).to(self.device)
            # print("scene_mask.shape", scene_mask.shape)

            scene_feat = self.predict(scene_mask.unsqueeze(0))
            # print("scene_feat.shape", scene_feat.shape)

            obj_feats = self.get_obj_feats(processed_masks, scene_feat)
            # print("obj_feats.shape", obj_feats.shape)
            
            scene_masks[i] = scene_mask

            # pad obj_features
            if len(obj_feats) < self.args.num_patches:
                # Calculate the amount of padding needed
                padding_needed = max(0, self.args.num_patches - obj_feats.size(0))

                # Pad the tensor along the first dimension
                obj_feats = torch.nn.functional.pad(obj_feats, (0, 0, 0, padding_needed))
                # print("obj_feats.shape", obj_feats.shape)

                h, w = processed_masks[0].shape
                empty_array = np.zeros((w, h))
                processed_masks = processed_masks + [empty_array] * padding_needed
            else:
                obj_feats = obj_feats[:self.args.num_patches]
                processed_masks = processed_masks[:self.args.num_patches]

            obj_features[i] = obj_feats
            
            processed_masks = np.array(processed_masks)
            processed_masks_all.append(processed_masks)
        
        processed_masks_all = np.array(processed_masks_all)
        obj_masks = torch.from_numpy(processed_masks_all).float().to(self.device)

        # self.show_images2(obj_masks)
        return scene_masks, obj_features, obj_masks

    def forward(self, color_image, target_mask, specific_rotation=-1, is_volatile=[]):
        # print("color_image.shape", color_image.shape)

        scene_masks, obj_features, obj_masks = self.preprocess_input(color_image)

        # compute rotated feature maps            
        scene_feats = self.predict(scene_masks)
        # print("scene_feats.shape", scene_feats.shape)

        target_mask = target_mask.cpu().numpy()
        # print("target_mask.shape", target_mask.shape)

        masks = []
        for mask in target_mask:
            mask, _ = general_utils.preprocess_image(mask)
            # print("mask.shape", mask.shape)
            masks.append(mask)
        normalized_target_mask = torch.tensor(np.array(masks)).to(self.device)
        # print("normalized_target_mask.shape", normalized_target_mask.shape)

        target_feat = self.predict(normalized_target_mask)
        # print("target_feat.shape", target_feat.shape)
        masked_target_fmap = scene_feats * target_feat
        # print("masked_target_fmap.shape", masked_target_fmap.shape)

        masked_target_fmap = masked_target_fmap.reshape(masked_target_fmap.shape[0], masked_target_fmap.shape[1], -1)[:, :, 0]
        # print("masked_target_fmap.shape", masked_target_fmap.shape)

        # Project target and object features  
        projected_target = self.target_proj(masked_target_fmap)
        # print("projected_target.shape", projected_target.shape)

        projected_objs = self.obj_proj(obj_features)
        # print("projected_objs.shape", projected_objs.shape)
        B, N, C, = projected_objs.shape

        # Scaled dot-product attention
        scale = np.sqrt(C)

        # Perform element-wise multiplication with broadcasting and Sum along the last dimension to get the final [2, 14] tensor
        attn_scores = (projected_target.unsqueeze(1) * projected_objs).sum(dim=-1)/scale
        # print("attn_scores.shape", attn_scores.shape) # [B,N]
        # print("attn_scores", attn_scores)

        # Use torch.topk to get the top k values and their indices
        top_scores, top_indices = torch.topk(attn_scores, k=self.args.sequence_length, dim=1)
        # print("top_scores", top_scores)

        # print("obj_masks.shape", obj_masks.shape)

        # Keep overlapped objects
        objs = []
        for i in range(B):
            idx = top_indices[i] 
            x = obj_masks[i, idx]
            # print("x.shape", x.shape) # Should be (4, 400, 400)
            objs.append(x)

        overlapped_objs = torch.stack(objs)
        # print("overlapped_objs.shape", overlapped_objs.shape)

        ########################### VIZ ################################

        # self.show_images(overlapped_objs, target_mask, color_image)

        ################################################################

        objs_2 = []
        overlapped_objs = overlapped_objs.cpu().numpy()
        for i in range(overlapped_objs.shape[0]):
            objs_1 = []
            for j in range(overlapped_objs.shape[1]):
                obj, _ = general_utils.preprocess_image(overlapped_objs[i][j])
                # print("obj.shape", obj.shape)
                objs_1.append(obj)

            objs = torch.tensor(np.array(objs_1))
            objs_2.append(objs)
            
        normalized_overlapped_objs = torch.stack(objs_2).to(self.device)
        # print("normalized_overlapped_objs.shape", normalized_overlapped_objs.shape)

        # Predict boxes
        B, N, C, H, W = normalized_overlapped_objs.shape
        reshaped_overlapped = normalized_overlapped_objs.view(B * N, H * W)
        out_prob = self.box_regression(reshaped_overlapped)
        # print("out_prob.shape", out_prob.shape)

        # Image-wide softmax
        output_shape = out_prob.shape
        out_prob = out_prob.view(output_shape[0], -1)
        out_prob = torch.softmax(out_prob, dim=1)
        out_prob = out_prob.view(B, N, C, H, W).to(dtype=torch.float)

        return out_prob
    
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