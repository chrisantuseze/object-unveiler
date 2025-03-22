import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)

class LayerNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, out_planes, stride)
        # self.bn1 = nn.BatchNorm2d(out_planes)
        self.bn1 = nn.GroupNorm(num_groups=1, num_channels=out_planes)

        self.conv2 = conv3x3(out_planes, out_planes)
        # self.bn2 = nn.BatchNorm2d(out_planes)

        self.bn2 = nn.GroupNorm(num_groups=1, num_channels=out_planes)  # Similar to LayerNorm

        self.downsample = downsample

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = F.dropout(out, p=0.2)
        out = F.relu(self.bn1(out))
        # out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(identity)

        out += identity
        out = F.dropout(out, p=0.2)
        out = F.relu(out)

        return out

class ActionDecoder(nn.Module):
    def __init__(self, args):
        super(ActionDecoder, self).__init__()
        self.nr_rotations = 16
        self.device = args.device
        
        # Scene stream - same as original
        self.scene_conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.scene_rb1 = self.make_layer(64, 128)
        self.scene_rb2 = self.make_layer(128, 256)
        self.scene_rb3 = self.make_layer(256, 512)
        self.scene_rb4 = self.make_layer(512, 256)
        self.scene_rb5 = self.make_layer(256, 128)
        self.scene_rb6 = self.make_layer(128, 64)
        
        # Object stream
        self.obj_conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.obj_rb1 = self.make_layer(64, 128)
        self.obj_rb2 = self.make_layer(128, 256)
        self.obj_rb3 = self.make_layer(256, 128)
        self.obj_rb4 = self.make_layer(128, 64)
        
        # Feature fusion layers
        self.fusion_conv = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.fusion_bn = nn.BatchNorm2d(64)
        self.fusion_bn = nn.GroupNorm(num_groups=1, num_channels=64)
        
        # Final prediction layer
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)
        
    def make_layer(self, in_channels, out_channels, blocks=1, stride=1):
        # Same as your original implementation
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(conv3x3(in_channels, out_channels, stride=stride))

        layers = [ResidualBlock(in_channels, out_channels, stride, downsample)]
        for i in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
        
    def process_scene(self, x):
        # Scene processing path
        x = nn.functional.relu(self.scene_conv1(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.scene_rb1(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.scene_rb2(x)
        x = self.scene_rb3(x)
        x = self.scene_rb4(x)
        x = self.scene_rb5(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.scene_rb6(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x
    
    def process_object(self, x):
        # Object processing path
        x = nn.functional.relu(self.obj_conv1(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.obj_rb1(x)
        x = self.obj_rb2(x)
        x = self.obj_rb3(x)
        # Upsample to match scene feature map size
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.obj_rb4(x)
        return x
    
    def predict(self, scene_depth, object_depth):
        # Process scene and object separately
        scene_features = self.process_scene(scene_depth)
        object_features = self.process_object(object_depth)
        
        # Resize object features to match scene features if needed
        if scene_features.shape[2:] != object_features.shape[2:]:
            object_features = nn.functional.interpolate(
                object_features, 
                size=scene_features.shape[2:], 
                mode='bilinear', 
                align_corners=True
            )
        
        # Concatenate features along channel dimension
        combined_features = torch.cat([scene_features, object_features], dim=1)
        
        # Fuse features
        fused_features = self.fusion_conv(combined_features)
        fused_features = nn.functional.relu(self.fusion_bn(fused_features))
        
        # Generate grasp prediction
        out = self.final_conv(fused_features)
        return out
    
    def forward(self, depth_heightmap, object_depth, specific_rotation=-1, is_volatile=[]):
        # Handle both testing (is_volatile) and training modes
        if is_volatile:
            # Similar rotation handling as original, but now with object_depth too
            batch_rot_depth = torch.zeros((self.nr_rotations, 1, depth_heightmap.shape[3], 
                                          depth_heightmap.shape[3])).to(self.device)
            batch_rot_obj = torch.zeros((self.nr_rotations, 1, object_depth.shape[3], 
                                        object_depth.shape[3])).to(self.device)
            
            for rot_id in range(self.nr_rotations):
                # Compute rotation grid
                theta = np.radians(rot_id * (360 / self.nr_rotations))
                affine_mat_before = np.array([[np.cos(theta), np.sin(theta), 0.0],
                                            [-np.sin(theta), np.cos(theta), 0.0]])
                affine_mat_before.shape = (2, 3, 1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).requires_grad_(False).float().to(self.device)

                flow_grid_before = F.affine_grid(affine_mat_before, depth_heightmap.size(), align_corners=True)

                # Rotate images clockwise
                rotate_depth = F.grid_sample(
                    depth_heightmap.requires_grad_(False),
                    flow_grid_before,
                    mode='nearest',
                    align_corners=True,
                    padding_mode="border")
                
                rotate_obj = F.grid_sample(
                    object_depth.requires_grad_(False),
                    flow_grid_before,
                    mode='nearest',
                    align_corners=True, 
                    padding_mode="border")

                batch_rot_depth[rot_id] = rotate_depth[0]
                batch_rot_obj[rot_id] = rotate_obj[0]

            # Compute rotated feature maps
            prob = self.predict(batch_rot_depth, batch_rot_obj)

            # Undo rotation (same as original)
            affine_after = torch.zeros((self.nr_rotations, 2, 3), requires_grad=False).to(self.device)
            for rot_id in range(self.nr_rotations):
                theta = np.radians(rot_id * (360 / self.nr_rotations))
                affine_mat_after = np.array([[np.cos(-theta), np.sin(-theta), 0.0],
                                           [-np.sin(-theta), np.cos(-theta), 0.0]])
                affine_mat_after.shape = (2, 3, 1)
                affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
                affine_after[rot_id] = affine_mat_after

            flow_grid_after = F.affine_grid(affine_after, prob.data.size(), align_corners=True)
            out_prob = F.grid_sample(prob, flow_grid_after, mode='nearest', align_corners=True)

            return out_prob
        else:
            # Training mode (similar handling as original but with object image)
            thetas = np.radians(specific_rotation * (360 / self.nr_rotations))
            affine_before = torch.zeros((depth_heightmap.shape[0], 2, 3), requires_grad=False).to(self.device)
            for i in range(len(thetas)):
                theta = thetas[i]
                affine_mat_before = np.array([[np.cos(theta), np.sin(theta), 0.0],
                                            [-np.sin(theta), np.cos(theta), 0.0]])
                affine_mat_before.shape = (2, 3, 1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
                affine_before[i] = affine_mat_before

            flow_grid_before = F.affine_grid(affine_before, depth_heightmap.size(), align_corners=True)

            # Rotate both images clockwise
            rotate_depth = F.grid_sample(depth_heightmap.requires_grad_(False),
                                       flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")
            
            rotate_obj = F.grid_sample(object_depth.requires_grad_(False),
                                     flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")

            # Compute features and prediction
            prob = self.predict(rotate_depth, rotate_obj)

            # Undo rotations (same as original)
            affine_after = torch.zeros((depth_heightmap.shape[0], 2, 3), requires_grad=False).to(self.device)
            for i in range(len(thetas)):
                theta = thetas[i]
                affine_mat_after = np.array([[np.cos(-theta), np.sin(-theta), 0.0],
                                           [-np.sin(-theta), np.cos(-theta), 0.0]])
                affine_mat_after.shape = (2, 3, 1)
                affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
                affine_after[i] = affine_mat_after

            flow_grid_after = F.affine_grid(affine_after, prob.size(), align_corners=True)

            # Forward pass, undo rotation on output predictions
            out_prob = F.grid_sample(prob, flow_grid_after, mode='nearest', align_corners=True)

            # Image-wide softmax
            output_shape = out_prob.shape
            out_prob = out_prob.view(output_shape[0], -1)
            out_prob = torch.softmax(out_prob, dim=1)
            out_prob = out_prob.view(output_shape).to(dtype=torch.float)

            return out_prob
        
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