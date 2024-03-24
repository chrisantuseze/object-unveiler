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
        # out = F.relu(self.bn1(out))
        out = F.leaky_relu(self.bn1(out))
        out = F.dropout(out, p=0.2)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(identity)

        out += identity
        # out = F.relu(out)
        out = F.leaky_relu(out)

        return out

class ObstacleHead0(nn.Module):
    def __init__(self, args, feat_extractor):
        super(ObstacleHead, self).__init__()
        self.args = args
        self.feat_extractor = feat_extractor
        self.final_conv_units = 128
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        hidden_dim = 128
        self.projection = nn.Sequential(
            nn.Linear((self.args.num_patches + 1) * self.final_conv_units, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.final_conv_units)
        )

        self.rnn = nn.LSTM(input_size=self.args.num_patches, hidden_size=hidden_dim, batch_first=True)
        self.fc_occluding_objects = nn.Linear(hidden_dim, self.args.num_patches)
        
    def preprocess_input(self, object_masks):
        B = object_masks.shape[0]
        object_features = torch.zeros(B, self.args.num_patches, self.final_conv_units).to(self.device)

        for i in range(B):
            object_masks_ = object_masks[i].to(self.device)

            obj_features = []
            for mask in object_masks_:
                mask = mask.unsqueeze(0).to(self.device)
                obj_feat = self.feat_extractor(mask)

                obj_feat = obj_feat.reshape(1, obj_feat.shape[1], -1)[:, :, 0]
                obj_features.append(obj_feat)

            obj_features = torch.cat(obj_features)
            object_features[i] = obj_features
            
        return object_features


    def get_topk_attn_scores(self, projected_objs, projected_target, object_masks):
        n_heads = 3
        multi_heads = [self.attention_head(projected_objs, projected_target, object_masks) for _ in range(n_heads)]
        attn_scores = sum(multi_heads)/n_heads
        # print("average attn_scores:", attn_scores)

        # Sample noise  
        # noise = torch.randn_like(attn_scores) * 0.01
        # zero_indices = torch.where(attn_scores == 0)
        # noise[zero_indices] = 0
        # attn_scores = attn_scores + noise

        _, top_indices = torch.topk(attn_scores, k=self.args.sequence_length, dim=1)
        return top_indices, attn_scores
    
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

    def visualize_attn(self, target_mask, object_masks, attn_scores):
        B, N, H, W = object_masks.shape

        # Reshape attention to match object masks 
        attn_weights = attn_scores.view(B, N, 1, 1)

        # Tile attention weights spatially 
        attn_weights = attn_weights.repeat(1, 1, H, W) 

        # Multiply masks by attention weights
        weighted_obj_masks = attn_weights * object_masks 

        # Sum weighted masks along N dimension
        attended_obj_masks = weighted_obj_masks.sum(dim=1)

        # Normalize for visualization
        attended_obj_masks = attended_obj_masks / attended_obj_masks.max() 
        # print("attended_obj_masks.shape", attended_obj_masks.shape)

        # Use torch.topk to get the top k values and their indices
        top_scores, top_indices = torch.topk(attn_scores, k=self.args.sequence_length, dim=1)

        if B == 1:
            fig, axs = plt.subplots(B, N+2, figsize=(13, 9))
            axs[0].imshow(target_mask[0])
            axs[1].imshow(attended_obj_masks[0].detach().numpy()) 
            k = 2
            for j in range(N):
                axs[k].imshow(object_masks[0,j])
                axs[k].imshow(attn_weights[0,j].detach().numpy(), alpha=0.5, cmap='viridis') 
                k += 1

        else:
            fig, axs = plt.subplots(B, N+2, figsize=(13, 9))
            for i in range(B):
                axs[i, 0].imshow(target_mask[i])
                axs[i, 1].imshow(attended_obj_masks[i].detach().numpy()) 
                k = 2
                for j in range(N):
                    axs[i,k].imshow(object_masks[i,j])
                    axs[i,k].imshow(attn_weights[i,j].detach().numpy(), alpha=0.5, cmap='viridis') 
                    k += 1

        plt.show()

    def causal_attention(self, target_feat, obj_feat, object_masks):
        # print(target_feat.shape, target_feat.unsqueeze(1).shape, obj_feat.shape)
        attn_input = torch.cat([target_feat.unsqueeze(1), obj_feat], dim=1)
        # print("attn_input.shape", attn_input.shape)

        energy = torch.tanh(attn_input)
        # print("energy.shape", energy.shape)

        attention = torch.softmax(self.projection(energy.reshape(attn_input.shape[0], -1)), dim=1)
        # print("attention.shape", attention.shape)
        
        attended_obj = torch.sum(obj_feat * attention.unsqueeze(1), dim=2) 
        # print("attended_obj.shape", attended_obj.shape)

        rnn_output, _ = self.rnn(attended_obj.unsqueeze(0))
        # print("rnn_output.shape", rnn_output.shape)

        rnn_output = rnn_output.squeeze(0)
        occluding_objects_output = self.fc_occluding_objects(rnn_output)
        # print("occluding_objects_output.shape", occluding_objects_output.shape)
        
        padding_masks = (object_masks.sum(dim=(2, 3)) == 0)
        padding_mask_expanded = padding_masks.expand_as(attended_obj)
        attended_obj = occluding_objects_output.masked_fill_(padding_mask_expanded, torch.tensor(0.0).to(self.device))
        # print("attended_obj:", attended_obj)

        _, top_indices = torch.topk(attended_obj, k=self.args.sequence_length, dim=1)

        return top_indices, attended_obj
    
    def forward(self, target_mask, object_masks):
    # def forward(self, target_mask, object_masks, raw_scene_mask=None, raw_target_mask=None, raw_object_masks=None):
        obj_features = self.preprocess_input(object_masks)
        
        target_feats = self.feat_extractor(target_mask)
        target_feats = target_feats.reshape(target_feats.shape[0], target_feats.shape[1], -1)[:, :, 0]

        B, N, C, = obj_features.shape

        # top_indices, attn_scores = self.get_topk_attn_scores(obj_features, target_feats, object_masks.squeeze(2))
        top_indices, attn_scores = self.causal_attention(target_feats, obj_features, object_masks.squeeze(2))

        # self.visualize_attn(raw_target_mask, raw_object_masks, attn_scores)

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
            
        # # self.show_images(raw_objs, raw_object_masks, raw_target_mask, raw_scene_mask, optimal_nodes)
        # self.show_images(raw_objects, raw_target_mask, raw_scene_mask, optimal_nodes=None, eval=B==1)

        # ###############################################################
            
        processed_objects = torch.stack(processed_objects)
        return Variable(processed_objects, requires_grad=True).to(self.device), attn_scores
    
class ObstacleHead(nn.Module):
    def __init__(self, args, feat_extractor):
        super(ObstacleHead, self).__init__()
        self.args = args
        self.feat_extractor = feat_extractor
        self.final_conv_units = 128
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.dim = 144
        hidden_dim = self.args.num_patches * self.dim
        self.projection = nn.Sequential(
            nn.Linear((self.args.num_patches + 2) * self.dim * self.dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.args.num_patches)
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

    def preprocess_input(self, object_masks):
        B, N, C, H, W = object_masks.shape
        # print("object_masks.shape", object_masks.shape)
        object_features = [] #torch.zeros(B, N, C, H, W).to(self.device)

        for i in range(B):
            object_masks_ = object_masks[i].to(self.device)

            obj_features = []
            for mask in object_masks_:
                # print("mask.shape", mask.shape)

                mask = mask.unsqueeze(0).to(self.device)
                obj_feat = self.feat_extractor(mask)

                # obj_feat = obj_feat.reshape(1, obj_feat.shape[1], -1)[:, :, 0]
                obj_features.append(obj_feat)

            obj_features = torch.cat(obj_features).unsqueeze(0)
            object_features.append(obj_features)

        return torch.cat(object_features).to(self.device)

    def forward(self, scene_mask, target_mask, object_masks):
    # def forward(self, scene_mask, target_mask, object_masks, raw_scene_mask, raw_target_mask, raw_object_masks):
        object_feats = self.preprocess_input(object_masks)
        target_feats = self.feat_extractor(target_mask).unsqueeze(1)
        scene_feats = self.feat_extractor(scene_mask).unsqueeze(1)

        x = torch.cat([target_feats, scene_feats, object_feats], dim=1)
        # print(x.shape, x.reshape(x.shape[0], -1).shape)
        attn_scores = self.projection(x.reshape(x.shape[0], -1))

        object_masks = object_masks.squeeze(2)
        padding_masks = (object_masks.sum(dim=(2, 3)) == 0)
        padding_mask_expanded = padding_masks.expand_as(attn_scores)
        attn_scores = attn_scores.masked_fill_(padding_mask_expanded, float('-inf'))
        
        attn_weights = torch.softmax(attn_scores, dim=1)
        # print("attn_weights", attn_weights)
        _, top_indices = torch.topk(attn_weights, k=self.args.sequence_length, dim=1)

        ###### Keep overlapped objects #####
        processed_objects = []

        raw_objects = []
        for i in range(target_mask.shape[0]):
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
        processed_objects = processed_objects.unsqueeze(1) if self.args.sequence_length == 1 else processed_objects
        return processed_objects, attn_weights

class GraspHead(nn.Module):
    def __init__(self, args, feat_extractor):
        super(GraspHead, self).__init__()
        self.args = args
        self.feat_extractor = feat_extractor
        self.nr_rotations = 16
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def get_predictions(self, depth_heightmap, target_mask, specific_rotation, is_volatile):
        if is_volatile:
            # rotations x channel x h x w
            batch_rot_depth = torch.zeros((self.nr_rotations, 1,
                                           depth_heightmap.shape[3],
                                           depth_heightmap.shape[3])).to(self.device)
            
            batch_rot_target = torch.zeros((self.nr_rotations, 1,
                                           target_mask.shape[3],
                                           target_mask.shape[3])).to(self.device)
            
            for rot_id in range(self.nr_rotations):
                # Compute sample grid for rotation before neural network
                theta = np.radians(rot_id * (360 / self.nr_rotations))
                affine_mat_before = np.array([[np.cos(theta), np.sin(theta), 0.0],
                                              [-np.sin(theta), np.cos(theta), 0.0]])
                affine_mat_before.shape = (2, 3, 1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()

                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).to(self.device),
                    depth_heightmap.size(), align_corners=True)
                
                # Rotate images clockwise
                rotate_depth = F.grid_sample(Variable(depth_heightmap, requires_grad=False).to(self.device),
                    flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")
                
                rotate_target_mask = F.grid_sample(Variable(target_mask, requires_grad=False).to(self.device),
                    flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")
                
                batch_rot_depth[rot_id] = rotate_depth[0]
                batch_rot_target[rot_id] = rotate_target_mask[0]

            # compute rotated feature maps            
            interm_grasp_depth_feat = self.feat_extractor(batch_rot_depth)
            interm_grasp_target_feat = self.feat_extractor(batch_rot_target)
            interm_grasp_feat = torch.cat((interm_grasp_depth_feat, interm_grasp_target_feat), dim=1)

            # undo rotation
            affine_after = torch.zeros((self.nr_rotations, 2, 3))
            for rot_id in range(self.nr_rotations):
                # compute sample grid for rotation before neural network
                theta = np.radians(rot_id * (360 / self.nr_rotations))
                affine_mat_after = np.array([[np.cos(-theta), np.sin(-theta), 0.0],
                                             [-np.sin(-theta), np.cos(-theta), 0.0]])
                affine_mat_after.shape = (2, 3, 1)
                affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
                affine_after[rot_id] = affine_mat_after

            flow_grid_after = F.affine_grid(Variable(affine_after, requires_grad=False).to(self.device),
                                            interm_grasp_feat.data.size(), align_corners=True)
            out_prob = F.grid_sample(interm_grasp_feat, flow_grid_after, mode='nearest', align_corners=True)
            out_prob = torch.mean(out_prob, dim=1, keepdim=True)
            
            return out_prob
        
        else:
            thetas = np.radians(specific_rotation * (360 / self.nr_rotations)).unsqueeze(0)
            affine_before = torch.zeros((depth_heightmap.shape[0], 2, 3))
            for i in range(len(thetas)):
                # Compute sample grid for rotation before neural network
                theta = thetas[i]
                affine_mat_before = np.array([[np.cos(theta), np.sin(theta), 0.0],
                                              [-np.sin(theta), np.cos(theta), 0.0]])
                affine_mat_before.shape = (2, 3, 1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
                affine_before[i] = affine_mat_before

            flow_grid_before = F.affine_grid(Variable(affine_before, requires_grad=False).to(self.device),
                                             depth_heightmap.size(), align_corners=True)

            # Rotate image clockwise_
            rotate_depth = F.grid_sample(Variable(depth_heightmap, requires_grad=False).to(self.device),
                                         flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")
            
            rotate_target_mask = F.grid_sample(Variable(target_mask, requires_grad=False).to(self.device),
                                         flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")

            # Compute intermediate features
            interm_grasp_depth_feat = self.feat_extractor(rotate_depth)
            interm_grasp_target_feat = self.feat_extractor(rotate_target_mask)
            interm_grasp_feat = torch.cat((interm_grasp_depth_feat, interm_grasp_target_feat), dim=1)

            # Compute sample grid for rotation after branches
            affine_after = torch.zeros((depth_heightmap.shape[0], 2, 3))
            for i in range(len(thetas)):
                theta = thetas[i]
                affine_mat_after = np.array([[np.cos(-theta), np.sin(-theta), 0.0],
                                             [-np.sin(-theta), np.cos(-theta), 0.0]])
                affine_mat_after.shape = (2, 3, 1)
                affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
                affine_after[i] = affine_mat_after

            flow_grid_after = F.affine_grid(Variable(affine_after, requires_grad=False).to(self.device),
                                            interm_grasp_feat.data.size(), align_corners=True)

            # Forward pass through branches, undo rotation on output predictions, upsample results
            out_prob = F.grid_sample(interm_grasp_feat, flow_grid_after, mode='nearest', align_corners=True)
            out_prob = torch.mean(out_prob, dim=1, keepdim=True)
            
            return out_prob

    def forward(self, depth_heightmap, processed_objects, specific_rotation=-1, is_volatile=[]):
        B, N, C, H, W = processed_objects.shape

        if is_volatile:
            out_probs = torch.zeros((N, self.nr_rotations, C, H, W)).to(self.device)
            for n, target_mask in enumerate(processed_objects[0]):
                out_prob = self.get_predictions(depth_heightmap, target_mask.unsqueeze(0), specific_rotation, is_volatile)
                out_probs[n] = out_prob
        
        else:
            out_probs = torch.zeros((B, N, C, H, W)).to(self.device)
            for batch in range(B):
                for n, target_mask in enumerate(processed_objects[batch]):
                    out_prob = self.get_predictions(depth_heightmap[batch].unsqueeze(0), target_mask.unsqueeze(0), specific_rotation[n][batch], is_volatile)
                    out_probs[batch][n] = out_prob

            # Image-wide softmax
            out_probs = out_probs.view(B * N, H * W)
            output_shape = out_probs.shape
            out_probs = out_probs.view(output_shape[0], -1)
            out_probs = torch.softmax(out_probs, dim=1)
            out_probs = out_probs.view(B, N, C, H, W).to(dtype=torch.float)

        # print("out_prob.shape", out_probs.shape)

        return out_probs


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
        self.grasp_head = GraspHead(args, self.predict)

    def make_layer(self, in_channels, out_channels, blocks=1, stride=1):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(conv3x3(in_channels, out_channels, stride=stride))

        layers = [ResidualBlock(in_channels, out_channels, stride, downsample)]
        for i in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)
    
    def predict0(self, depth, final_feats=False):
        # x = F.relu(self.conv1(depth))
        # x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        # x = self.rb1(x)
        # x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        # x = self.rb2(x)
        # x = self.rb3(x)
        # x = self.rb4(x)
        # x = self.rb5(x)
        
        # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        # x = self.rb6(x)
       
        # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        # if final_feats:
        #     conv2 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False).to(self.device)
        #     nn.init.xavier_uniform_(conv2.weight)
        #     out = conv2(x)
        # else:
        #     conv3 = nn.Conv2d(64, self.final_conv_units, kernel_size=1, stride=1, padding=0, bias=False).to(self.device)
        #     nn.init.xavier_uniform_(conv3.weight)
        #     out = conv3(x)
        # return out
        pass
    
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
       
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) # multiply H and W
        out = self.final_conv(x)
        return out
    
    def forward(self, depth_heightmap, target_mask, object_masks, scene_masks, specific_rotation=-1, is_volatile=[]):
    # def forward(self, depth_heightmap, target_mask, object_masks, scene_masks, raw_target_mask=None, raw_object_masks=None, raw_scene_mask=None, gt_object=None, specific_rotation=-1, is_volatile=[]):
        
        processed_objects, scores = self.obstacle_head(depth_heightmap, target_mask, object_masks)
        # processed_objects, scores = self.obstacle_head(depth_heightmap, target_mask, object_masks, raw_scene_mask, raw_target_mask, raw_object_masks)
        
        out_probs = self.grasp_head(depth_heightmap, processed_objects, specific_rotation, is_volatile)

        return scores, out_probs
    

def print_msg(B, *args):
    if B != 1:
        return
    result = ' '.join(map(str, args))
    print(result)

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