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
    
class AttentionModule(nn.Module):
    def __init__(self, scene_channels, object_channels):
        super(AttentionModule, self).__init__()
        self.scene_channels = scene_channels
        self.object_channels = object_channels
        
        # Dimensionality reduction for scene and object features
        self.scene_conv = nn.Conv2d(scene_channels, scene_channels//2, kernel_size=1)
        self.object_conv = nn.Conv2d(object_channels, scene_channels//2, kernel_size=1)
        
        # Attention calculation
        self.attention_conv = nn.Sequential(
            nn.Conv2d(scene_channels, scene_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(scene_channels//2),
            nn.ReLU(),
            nn.Conv2d(scene_channels//2, 1, kernel_size=1)
        )
        
    def forward(self, scene_features, object_features):
        # Resize object features to match scene features if needed
        if scene_features.shape[2:] != object_features.shape[2:]:
            object_features = nn.functional.interpolate(
                object_features, 
                size=scene_features.shape[2:], 
                mode='bilinear', 
                align_corners=True
            )
        
        # Process features
        scene_feat = self.scene_conv(scene_features)
        obj_feat = self.object_conv(object_features)
        
        # Concatenate features
        combined = torch.cat([scene_feat, obj_feat], dim=1)
        
        # Generate attention map
        attention = torch.sigmoid(self.attention_conv(combined))
        
        # Apply attention to scene features
        attended_features = scene_features * attention
        
        return attended_features, attention

class ResFCN(nn.Module):
    def __init__(self, args):
        super(ResFCN, self).__init__()
        self.nr_rotations = 16
        self.device = args.device
        
        # Scene processing - same as original
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.rb1 = self.make_layer(64, 128)
        self.rb2 = self.make_layer(128, 256)
        self.rb3 = self.make_layer(256, 512)
        self.rb4 = self.make_layer(512, 256)
        
        # Object processing
        self.obj_conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.obj_rb1 = self.make_layer(64, 128)
        self.obj_rb2 = self.make_layer(128, 256)
        
        # Attention module
        self.attention = AttentionModule(256, 256)
        
        # Final processing
        self.rb5 = self.make_layer(256, 128)
        self.rb6 = self.make_layer(128, 64)
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
    
    def extract_scene_features(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.rb1(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        return x
    
    def extract_object_features(self, x):
        x = nn.functional.relu(self.obj_conv1(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.obj_rb1(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.obj_rb2(x)
        return x
    
    def predict(self, scene_depth, object_depth):
        # Extract features from both inputs
        scene_features = self.extract_scene_features(scene_depth)
        object_features = self.extract_object_features(object_depth)
        
        # Apply attention mechanism
        attended_features, _ = self.attention(scene_features, object_features)
        
        # Final processing
        x = self.rb5(attended_features)
        x = nn.functional.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.rb6(x)
        x = nn.functional.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.final_conv(x)
        return out
    
    def forward(self, depth_heightmap, object_depth, specific_rotation=-1, is_volatile=[]):
        # Handle both inference and training modes
        # (Similar structure as the two-stream implementation but with attention mechanism)
        
        if is_volatile:
            # Rotations handling for inference similar to the two-stream model
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
            # Training mode handling (similar to two-stream)
            # ...training mode implementation similar to the above...
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