import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import numpy as np
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
        out = F.dropout(out, p=0.2)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(identity)

        out += identity
        out = F.relu(out)

        return out

class ObstacleSelector(nn.Module):
    def __init__(self, args):
        super(ObstacleSelector, self).__init__()
        self.args = args
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        hidden_dim = 1024
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(self.args.num_patches * (hidden_dim + 4), hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.args.num_patches)
        )

    def preprocess_inputs(self, target_mask, object_masks):
        B, N, C, H, W = object_masks.shape

        target_mask = target_mask.repeat(1, 3, 1, 1)
        target_feats = self.model(target_mask)
        target_feats = target_feats.view(B, 1, -1)

        object_masks = object_masks.repeat(1, 1, 3, 1, 1)
        object_masks = object_masks.view(-1, 3, H, W)
        object_feats = self.model(object_masks)
        object_feats = object_feats.view(B, N, -1)
        # print(object_feats.shape)

        return target_feats, object_feats

    def forward(self, target_mask, object_masks, bboxes):
        target_feats, object_feats = self.preprocess_inputs(target_mask, object_masks)

        B, N, C, H, W = object_masks.shape

        attn_scores = (target_feats * object_feats)/np.sqrt(object_feats.shape[1])
        # print(attn_scores.shape)

        x = torch.cat([attn_scores.view(B, N, -1), bboxes.view(B, N, -1)], dim=2).view(B, -1)
        # print("x.shape", x.shape)    

        attn_scores = self.fc(x)

        object_masks = object_masks.squeeze(2)
        padding_masks = (object_masks.sum(dim=(2, 3)) == 0)
        padding_mask_expanded = padding_masks.expand_as(attn_scores)
        attn_scores = attn_scores.masked_fill_(padding_mask_expanded, float(-1e-6))

        # _, top_indices = torch.topk(attn_scores, k=self.args.sequence_length, dim=1)
        # print("top indices", top_indices)

        # Sampling from the attention weights to get hard attention
        sampled_attention_weights = torch.zeros_like(attn_scores)
        for batch_idx in range(target_mask.shape[0]):
            sampled_attention_weights[batch_idx, :] = F.gumbel_softmax(attn_scores[batch_idx, :], hard=True)

        # Multiplying the encoder outputs with the hard attention weights
        sampled_attention_weights = sampled_attention_weights.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        # print(sampled_attention_weights.shape, object_masks.unsqueeze(2).shape)
        context = (sampled_attention_weights * object_masks.unsqueeze(2)).sum(dim=1)
        context = context.unsqueeze(1)

        return context

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

        self.obstacle_selector = ObstacleSelector(self.args)

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
       
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) # multiply H and W
        out = self.final_conv(x)
        return out
    
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
                obj_feat = self.predict(mask)

                # obj_feat = obj_feat.reshape(1, obj_feat.shape[1], -1)[:, :, 0]
                obj_features.append(obj_feat)

            obj_features = torch.cat(obj_features).unsqueeze(0)
            object_features.append(obj_features)

        return torch.cat(object_features).to(self.device)
    
    def obstacle_scorer(self, scene_mask, target_mask, object_masks):
        object_feats = self.preprocess_input(object_masks)
        target_feats = self.predict(target_mask).unsqueeze(1)
        scene_feats = self.predict(scene_mask).unsqueeze(1)

        x = torch.cat([target_feats, scene_feats, object_feats], dim=1)
        # print(x.shape, x.reshape(x.shape[0], -1).shape)
        # attn_scores = self.projection(x.reshape(x.shape[0], -1))

        
        object_masks = object_masks.squeeze(2)
        padding_masks = (object_masks.sum(dim=(2, 3)) == 0)
        padding_mask_expanded = padding_masks.expand_as(attn_scores)
        attn_scores = attn_scores.masked_fill_(padding_mask_expanded, float('-inf'))
        print("attn_scores", attn_scores)

        # Sampling from the attention weights to get hard attention
        sampled_attention_weights = torch.zeros_like(attn_scores)
        for batch_idx in range(target_mask.shape[0]):
            sampled_attention_weights[batch_idx, :] = F.gumbel_softmax(attn_scores[batch_idx, :], hard=True)

        # Multiplying the encoder outputs with the hard attention weights
        sampled_attention_weights = sampled_attention_weights.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        # print(sampled_attention_weights.shape, object_masks.unsqueeze(2).shape)
        context = (sampled_attention_weights * object_masks.unsqueeze(2)).sum(dim=1)
        context = context.unsqueeze(1)

        # print(context.shape)

        return context
    
    def forward(self, depth_heightmap, target_mask, object_masks, scene_masks, bboxes, specific_rotation=-1, is_volatile=[]):
    # def forward(self, depth_heightmap, target_mask, object_masks, scene_masks, raw_scene_mask, raw_target_mask, raw_object_masks, gt_object=None, bboxes=None, specific_rotation=-1, is_volatile=[]):
         
        selected_objects = self.obstacle_selector(target_mask, object_masks, bboxes)

        # selected_objects = object_masks[:, 0, :, :, :]
        selected_objects = selected_objects.squeeze(1)

        ###### Keep overlapped objects #####
        # processed_objects = []
        # raw_objects = []
        # for i in range(target_mask.shape[0]):
        #     idx = top_indices[i] 
        #     x = object_masks[i, idx] # x should be (4, 400, 400)
        #     processed_objects.append(x)

        # ################### THIS IS FOR VISUALIZATION ####################
        #     raw_x = raw_object_masks[i, idx]
        #     # print("raw_x.shape", raw_x.shape)
        #     raw_objects.append(raw_x)

        # raw_objects = torch.stack(raw_objects)
        # self.show_images(raw_objects, raw_target_mask, raw_scene_mask, optimal_nodes=None, eval=True)
        # ###############################################################

        if is_volatile:
            out_prob = self.get_predictions(depth_heightmap, selected_objects, specific_rotation, is_volatile)
        
            return None, out_prob
        else:
            specific_rotation = specific_rotation[0]
            out_prob = self.get_predictions(depth_heightmap, selected_objects, specific_rotation, is_volatile)

            # Image-wide softmax
            output_shape = out_prob.shape
            out_prob = out_prob.view(output_shape[0], -1)
            out_prob = torch.softmax(out_prob, dim=1)
            out_prob = out_prob.view(output_shape).to(dtype=torch.float)

            # print("out_prob.shape", out_prob.shape)
    
            return out_prob.unsqueeze(1)
        
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
                
                rotate_target_mask = F.grid_sample(target_mask, flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")
                
                batch_rot_depth[rot_id] = rotate_depth[0]
                batch_rot_target[rot_id] = rotate_target_mask[0]

            # compute rotated feature maps            
            interm_grasp_depth_feat = self.predict(batch_rot_depth)
            interm_grasp_target_feat = self.predict(batch_rot_target)
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
            thetas = np.radians(specific_rotation * (360 / self.nr_rotations))
            affine_before = torch.zeros((depth_heightmap.shape[0], 2, 3), requires_grad=False).to(self.device)
            for i in range(len(thetas)):
                # Compute sample grid for rotation before neural network
                theta = thetas[i]
                affine_mat_before = np.array([[np.cos(theta), np.sin(theta), 0.0],
                                              [-np.sin(theta), np.cos(theta), 0.0]])
                affine_mat_before.shape = (2, 3, 1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
                affine_before[i] = affine_mat_before

            flow_grid_before = F.affine_grid(affine_before, depth_heightmap.size(), align_corners=True)

            # Rotate image clockwise_
            rotate_depth = F.grid_sample(depth_heightmap.requires_grad_(False), flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")
            rotate_target_mask = F.grid_sample(target_mask, flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")

            # Compute intermediate features
            depth_feat = self.predict(rotate_depth)
            target_feat = self.predict(rotate_target_mask)
            masked_depth_feat = torch.cat((depth_feat, target_feat), dim=1)

            # Compute sample grid for rotation after branches
            affine_after = torch.zeros((depth_heightmap.shape[0], 2, 3), requires_grad=False).to(self.device)
            for i in range(len(thetas)):
                theta = thetas[i]
                affine_mat_after = np.array([[np.cos(-theta), np.sin(-theta), 0.0],
                                             [-np.sin(-theta), np.cos(-theta), 0.0]])
                affine_mat_after.shape = (2, 3, 1)
                affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
                affine_after[i] = affine_mat_after

            flow_grid_after = F.affine_grid(affine_after, masked_depth_feat.data.size(), align_corners=True)

            # Forward pass through branches, undo rotation on output predictions, upsample results
            out_prob = F.grid_sample(masked_depth_feat, flow_grid_after, mode='nearest', align_corners=True)

            out_prob = torch.mean(out_prob, dim=1, keepdim=True)
            
            return out_prob
    
    # def show_images(self, obj_masks, raw_object_masks, target_mask, scenes, optimal_nodes):
    def show_images(self, obj_masks, target_mask, scenes, optimal_nodes=None, eval=False):
        # fig, ax = plt.subplots(obj_masks.shape[0] * 2, obj_masks.shape[1] + 2)
        fig, ax = plt.subplots(obj_masks.shape[0], obj_masks.shape[1] + 2)

        for i in range(obj_masks.shape[0]):
            if obj_masks.shape[0] == 1:
                ax[i].imshow(scenes[i]) # this is because of the added gt images
            else:
                ax[i][0].imshow(scenes[i])

            k = 1
            for j in range(obj_masks.shape[1]):
                obj_mask = obj_masks[i][j]
                # print("obj_mask.shape", obj_mask.shape)

                if obj_masks.shape[0] == 1:
                    ax[k].imshow(obj_mask)
                else:
                    ax[i][k].imshow(obj_mask)
                k += 1

            if obj_masks.shape[0] == 1:
                ax[k].imshow(target_mask[i])
            else:
                ax[i][k].imshow(target_mask[i])


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