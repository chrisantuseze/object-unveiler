from policy.models import ResidualBlock, conv3x3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import numpy as np
import utils.logger as logging
from utils.constants import *

class ActionNet(nn.Module):
    def __init__(self, args, is_train=True):
        super(ActionNet, self).__init__()

        self.args = args
        self.is_train = is_train

        self.nr_rotations = 16
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.rb1 = self._make_layer(64, 128)
        self.rb2 = self._make_layer(128, 256)
        self.rb3 = self._make_layer(256, 512)
        self.rb4 = self._make_layer(512, 256)
        self.rb5 = self._make_layer(256, 128)
        self.rb6 = self._make_layer(128, 64)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)


    def _make_layer(self, in_channels, out_channels, blocks=1, stride=1):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(conv3x3(in_channels, out_channels, stride=stride))

        layers = [ResidualBlock(in_channels, out_channels, stride, downsample)]
        for i in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)
    
    def _predict(self, depth):
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
    
    def _volatile(self, depth_heightmap, target_mask, obstacle_mask):
        batch_rot_depth = torch.zeros((self.nr_rotations, 1, depth_heightmap.shape[3], depth_heightmap.shape[3])).to(self.device)
        batch_rot_target = torch.zeros((self.nr_rotations, 1, target_mask.shape[3], target_mask.shape[3])).to(self.device)
        batch_rot_obstacle = torch.zeros((self.nr_rotations, 1, obstacle_mask.shape[3], obstacle_mask.shape[3])).to(self.device)
        
        for rot_id in range(self.nr_rotations):
            # Compute sample grid for rotation before neural network
            theta = np.radians(rot_id * (360 / self.nr_rotations))
            affine_mat_before = np.array([[np.cos(theta), np.sin(theta), 0.0], [-np.sin(theta), np.cos(theta), 0.0]])
            affine_mat_before.shape = (2, 3, 1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()

            flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).to(self.device), depth_heightmap.size(), align_corners=True)
            
            # Rotate images clockwise
            rotate_depth = F.grid_sample(Variable(depth_heightmap, requires_grad=False).to(self.device),
                flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")
            
            rotate_target_mask = F.grid_sample(Variable(target_mask, requires_grad=False).to(self.device),
                flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")
            
            rotate_obstacle_mask = F.grid_sample(Variable(obstacle_mask, requires_grad=False).to(self.device),
                flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")

            batch_rot_depth[rot_id] = rotate_depth[0]
            batch_rot_target[rot_id] = rotate_target_mask[0]
            batch_rot_obstacle[rot_id] = rotate_obstacle_mask[0]

        # compute rotated feature maps
        prob_depth = self._predict(batch_rot_depth)
        prob_target = self._predict(batch_rot_target)
        prob_obstacle = self._predict(batch_rot_obstacle)

        probs = torch.cat((prob_depth, prob_target, prob_obstacle), dim=1)

        return self._volatile_undo_rotate(probs)

        
    def _volatile_undo_rotate(self, prob):
        # undo rotation
        affine_after = torch.zeros((self.nr_rotations, 2, 3))
        for rot_id in range(self.nr_rotations):
            # compute sample grid for rotation before neural network
            theta = np.radians(rot_id * (360 / self.nr_rotations))
            affine_mat_after = np.array([[np.cos(-theta), np.sin(-theta), 0.0], [-np.sin(-theta), np.cos(-theta), 0.0]])
            affine_mat_after.shape = (2, 3, 1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
            affine_after[rot_id] = affine_mat_after

        flow_grid_after = F.affine_grid(Variable(affine_after, requires_grad=False).to(self.device), prob.size(), align_corners=True)
        out_prob = F.grid_sample(prob, flow_grid_after, mode='nearest', align_corners=True)

        # logging.info("out_prob.shape:", out_prob.shape)
        out_prob = torch.mean(out_prob, dim=0, keepdim=True)

        return out_prob
    
    def _non_volatile(self, depth_heightmap, target_mask, obstacle_mask, specific_rotation=-1):
        thetas = np.radians(specific_rotation * (360 / self.nr_rotations))
        affine_before = torch.zeros((depth_heightmap.shape[0], 2, 3))
        for i in range(len(thetas)):
            # Compute sample grid for rotation before neural network
            theta = thetas[i]
            affine_mat_before = np.array([[np.cos(theta), np.sin(theta), 0.0],
                                            [-np.sin(theta), np.cos(theta), 0.0]])
            affine_mat_before.shape = (2, 3, 1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
            affine_before[i] = affine_mat_before

        flow_grid_before = F.affine_grid(Variable(affine_before, requires_grad=False).to(self.device), depth_heightmap.size(), align_corners=True)

        # Rotate image clockwise_
        rotate_depth = F.grid_sample(Variable(depth_heightmap, requires_grad=False).to(self.device),
                                        flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")
        
        rotate_target_mask = F.grid_sample(Variable(target_mask, requires_grad=False).to(self.device),
                                        flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")
        
        rotate_obstacle_mask = F.grid_sample(Variable(obstacle_mask, requires_grad=False).to(self.device),
                                        flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")

        # compute rotated feature maps
        prob_depth = self._predict(rotate_depth)
        prob_target = self._predict(rotate_target_mask)
        prob_obstacle = self._predict(rotate_obstacle_mask)

        probs = torch.cat((prob_depth, prob_target, prob_obstacle), dim=1)

        return self._non_volatile_undo_rotate(depth_heightmap, probs, thetas)

        
    def _non_volatile_undo_rotate(self, depth_heightmap, prob, thetas):
        # Compute sample grid for rotation after branches
        affine_after = torch.zeros((depth_heightmap.shape[0], 2, 3))
        for i in range(len(thetas)):
            theta = thetas[i]
            affine_mat_after = np.array([[np.cos(-theta), np.sin(-theta), 0.0], [-np.sin(-theta), np.cos(-theta), 0.0]])
            affine_mat_after.shape = (2, 3, 1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
            affine_after[i] = affine_mat_after

        flow_grid_after = F.affine_grid(Variable(affine_after, requires_grad=False).to(self.device), prob.size(), align_corners=True)

        # Forward pass through branches, undo rotation on output predictions, upsample results
        out_prob = F.grid_sample(prob, flow_grid_after, mode='nearest', align_corners=True)

        # Image-wide softmax
        output_shape = out_prob.shape
        out_prob = out_prob.view(output_shape[0], -1)
        out_prob = torch.softmax(out_prob, dim=1)
        out_prob = out_prob.view(output_shape).to(dtype=torch.float)

        return out_prob
    

    def forward(self, sequence, rot_ids=[], is_volatile=False):
        probs = []
        for i in range(len(sequence)):
            heightmap, target_mask, obstacle_mask = sequence[i]
            
            heightmap = heightmap.to(self.device)
            target_mask = target_mask.to(self.device)
            obstacle_mask = obstacle_mask.to(self.device)

            if is_volatile:
                prob = self._volatile(heightmap, target_mask, obstacle_mask)
                probs.append(prob)
            else:
                rot_id = rot_ids[i]
                prob = self._non_volatile(heightmap, target_mask, obstacle_mask, rot_id)
                probs.append(prob)

        probs_stack = torch.stack(probs, dim=0)
        # logging.info("probs_stack.shape:", probs_stack.shape)           #torch.Size([4, 1, 3, 224, 224])

        # Reduce the tensor to 4x1x1x224x224 by taking the mean along the 3rd dimension (dimension 2)
        output_tensor = torch.mean(probs_stack, dim=2, keepdim=True)
        # logging.info("output_tensor.shape:", output_tensor.shape)       #torch.Size([4, 1, 1, 224, 224])

        return output_tensor