import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

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
        self.nr_rotations = 16
        self.device = args.device

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.rb1 = self.make_layer(64, 128)
        self.rb2 = self.make_layer(128, 256)
        self.rb3 = self.make_layer(256, 512)
        self.rb4 = self.make_layer(512, 256)
        self.rb5 = self.make_layer(256, 128)
        self.rb6 = self.make_layer(128, 64)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)

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
    
    def forward(self, depth_heightmap, target_mask, specific_rotation=-1, is_volatile=[]):
        if is_volatile:
            # rotations x channel x h x w
            batch_rot_depth = torch.zeros((self.nr_rotations, 1, depth_heightmap.shape[3], depth_heightmap.shape[3])).to(self.device)
            batch_rot_target = torch.zeros((self.nr_rotations, 1, target_mask.shape[3], target_mask.shape[3])).to(self.device)
            
            for rot_id in range(self.nr_rotations):
                # Compute sample grid for rotation before neural network
                theta = np.radians(rot_id * (360 / self.nr_rotations))
                affine_mat_before = np.array([[np.cos(theta), np.sin(theta), 0.0],
                                              [-np.sin(theta), np.cos(theta), 0.0]])
                affine_mat_before.shape = (2, 3, 1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).requires_grad_(False).float().to(self.device)

                flow_grid_before = F.affine_grid(affine_mat_before, depth_heightmap.size(), align_corners=True)
                
                # Rotate images clockwise
                rotate_depth = F.grid_sample(depth_heightmap.requires_grad_(False), flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")
                rotate_target_mask = F.grid_sample(target_mask.requires_grad_(False), flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")

                batch_rot_depth[rot_id] = rotate_depth[0]
                batch_rot_target[rot_id] = rotate_target_mask[0]

            # compute rotated feature maps            
            # depth_feat = self.predict(batch_rot_depth)
            target_feat = self.predict(batch_rot_target)
            masked_depth_feat = target_feat
            # masked_depth_feat = torch.cat((depth_feat, target_feat), dim=1)

            # undo rotation
            affine_after = torch.zeros((self.nr_rotations, 2, 3), requires_grad=False).to(self.device)
            for rot_id in range(self.nr_rotations):
                # compute sample grid for rotation before neural network
                theta = np.radians(rot_id * (360 / self.nr_rotations))
                affine_mat_after = np.array([[np.cos(-theta), np.sin(-theta), 0.0],
                                             [-np.sin(-theta), np.cos(-theta), 0.0]])
                affine_mat_after.shape = (2, 3, 1)
                affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
                affine_after[rot_id] = affine_mat_after

            flow_grid_after = F.affine_grid(affine_after, masked_depth_feat.data.size(), align_corners=True)
            out_prob = F.grid_sample(masked_depth_feat, flow_grid_after, mode='nearest', align_corners=True)
            # out_prob = torch.mean(out_prob, dim=1, keepdim=True)

            return out_prob # 16x1x144x144
        
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
            rotate_target_mask = F.grid_sample(target_mask.requires_grad_(False), flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")

            # Compute intermediate features
            # depth_feat = self.predict(rotate_depth)
            target_feat = self.predict(rotate_target_mask)
            masked_depth_feat = target_feat
            # masked_depth_feat = torch.cat((depth_feat, target_feat), dim=1)

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

            # out_prob = torch.mean(out_prob, dim=1, keepdim=True)
            
            # Image-wide softmax
            output_shape = out_prob.shape
            out_prob = out_prob.view(output_shape[0], -1)
            out_prob = torch.softmax(out_prob, dim=1)
            out_prob = out_prob.view(output_shape).to(dtype=torch.float)

            return out_prob # Bx1x144x144
        

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