import math
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
        
        # Scene stream (from original)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)  # Added BatchNorm
        self.rb1 = self.make_layer(64, 128)
        self.rb2 = self.make_layer(128, 256)
        self.rb3 = self.make_layer(256, 512)
        self.rb4 = self.make_layer(512, 256)
        
        # Object stream - lighter than before
        self.obj_conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.obj_bn1 = nn.BatchNorm2d(32)
        self.obj_rb1 = self.make_layer(32, 64)
        self.obj_rb2 = self.make_layer(64, 128)
        
        # Location encoding - helps the model understand WHERE the object is in the scene
        self.loc_conv = nn.Conv2d(2, 32, kernel_size=7, stride=1, padding=3, bias=False)
        self.loc_bn = nn.BatchNorm2d(32)
        self.loc_rb = self.make_layer(32, 64)
        
        # Cross-attention mechanism
        self.q_conv = nn.Conv2d(256, 128, kernel_size=1)
        self.k_conv = nn.Conv2d(128, 128, kernel_size=1)
        self.v_conv = nn.Conv2d(128, 128, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)  # Initialize with small positive value
        
        # Feature fusion
        self.fusion_conv1 = nn.Conv2d(128 + 128 + 64, 256, kernel_size=3, padding=1)
        self.fusion_bn1 = nn.BatchNorm2d(256)
        self.fusion_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fusion_bn2 = nn.BatchNorm2d(256)
        
        # Final processing layers
        self.rb5 = self.make_layer(256, 128)
        self.rb6 = self.make_layer(128, 64)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)
        
        # Auxiliary loss to help stabilize training
        self.aux_conv = nn.Conv2d(256, 1, kernel_size=1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.2)
        
    def make_layer(self, in_channels, out_channels, blocks=1, stride=1):
        # Same as original
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(conv3x3(in_channels, out_channels, stride=stride))

        layers = [ResidualBlock(in_channels, out_channels, stride, downsample)]
        for i in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def create_location_heatmap(self, obj_mask, h, w):
        """
        Creates coordinate channels to help the network understand spatial relationships
        """
        batch_size = obj_mask.size(0)
        y_coord = torch.linspace(-1, 1, h).view(1, 1, h, 1).expand(batch_size, 1, h, w)
        x_coord = torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(batch_size, 1, h, w)
        coords = torch.cat([y_coord, x_coord], dim=1).to(self.device)
        return coords
    
    def cross_attention(self, scene_features, object_features):
        """
        Cross-attention module to focus scene features on regions relevant to object
        """
        batch_size, _, h, w = scene_features.size()
        
        # Compute queries, keys, values
        queries = self.q_conv(scene_features).view(batch_size, -1, h * w).permute(0, 2, 1) 
        keys = self.k_conv(object_features).view(batch_size, -1, h * w)
        values = self.v_conv(object_features).view(batch_size, -1, h * w)
        
        # Compute attention scores with temperature scaling
        attention = torch.bmm(queries, keys)
        attention_scale = math.sqrt(keys.size(1))  # More stable scaling
        attention = F.softmax(attention / attention_scale, dim=-1)
        
        out = torch.bmm(attention, values.permute(0, 2, 1))
        out = out.view(batch_size, -1, h, w)

        # Residual connection
        return self.gamma * out + self.rb5(scene_features)
    
    def predict(self, scene_depth, object_depth, return_aux=False):
        batch_size, _, h, w = scene_depth.size()
        
        # Process scene with BatchNorm
        scene_x = F.relu(self.bn1(self.conv1(scene_depth)))
        scene_x = F.max_pool2d(scene_x, kernel_size=2, stride=2)
        scene_x = self.rb1(scene_x)
        scene_x = F.max_pool2d(scene_x, kernel_size=2, stride=2)
        scene_x = self.rb2(scene_x)
        scene_x = self.rb3(scene_x)
        scene_x = self.dropout(scene_x)  # Add dropout for regularization
        scene_features = self.rb4(scene_x)
        
        # Process object
        obj_x = nn.functional.relu(self.obj_bn1(self.obj_conv1(object_depth)))
        obj_x = nn.functional.max_pool2d(obj_x, kernel_size=2, stride=2)
        obj_x = self.obj_rb1(obj_x)
        obj_x = nn.functional.max_pool2d(obj_x, kernel_size=2, stride=2)
        obj_features = self.obj_rb2(obj_x)
        
        # Create location encoding (where is the object in the scene?)
        # This helps the model understand spatial relationships
        object_mask = (object_depth > 0).float()
        loc_encoding = self.create_location_heatmap(object_mask, h, w)
        loc_features = nn.functional.relu(self.loc_bn(self.loc_conv(loc_encoding)))
        loc_features = nn.functional.max_pool2d(loc_features, kernel_size=2, stride=2)
        loc_features = nn.functional.max_pool2d(loc_features, kernel_size=2, stride=2)
        loc_features = self.loc_rb(loc_features)
        
        # Resize feature maps to match
        if scene_features.shape[2:] != obj_features.shape[2:]:
            obj_features = nn.functional.interpolate(
                obj_features, size=scene_features.shape[2:], mode='bilinear', align_corners=True
            )
        
        if scene_features.shape[2:] != loc_features.shape[2:]:
            loc_features = nn.functional.interpolate(
                loc_features, size=scene_features.shape[2:], mode='bilinear', align_corners=True
            )
        
        # Apply cross-attention to focus on relevant regions
        attended_features = self.cross_attention(scene_features, obj_features)
        
        # Feature fusion
        combined = torch.cat([attended_features, obj_features, loc_features], dim=1)
        fused = nn.functional.relu(self.fusion_bn1(self.fusion_conv1(combined)))
        fused = nn.functional.relu(self.fusion_bn2(self.fusion_conv2(fused)))
        
        # Auxiliary output (for training stability)
        aux_out = nn.functional.interpolate(fused, scale_factor=4, mode='bilinear', align_corners=True)
        aux_out = self.aux_conv(aux_out)
        
        # Final processing
        x = self.rb5(fused)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.rb6(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.final_conv(x)
        
        # if return_aux:
        #     return out, aux_out
        return out, aux_out
        # return out
    
    def forward(self, depth_heightmap, object_depth, specific_rotation=-1, is_volatile=[], return_aux=False):
        # Similar rotation handling code as before, but using the improved prediction function
        # For brevity, I'll skip the full rotation code which is similar to previous implementations
        
        if is_volatile:
            # Test-time forward pass
            # ...rotation handling code...
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
            
            # Run improved prediction with both inputs
            # if return_aux:
            #     prob, aux_prob = self.predict(batch_rot_depth, batch_rot_obj, return_aux=True)
            # else:
            #     prob = self.predict(batch_rot_depth, batch_rot_obj)

            prob, aux_prob = self.predict(batch_rot_depth, batch_rot_obj, return_aux=True)
            # prob = self.predict(batch_rot_depth, batch_rot_obj)
            
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
            
            # if return_aux:
            #     aux_flow_grid_after = F.affine_grid(affine_after, aux_prob.data.size(), align_corners=True)
            #     aux_out_prob = F.grid_sample(aux_prob, aux_flow_grid_after, mode='nearest', align_corners=True)
            #     return out_prob, aux_out_prob
            
            # return out_prob

            aux_flow_grid_after = F.affine_grid(affine_after, aux_prob.data.size(), align_corners=True)
            aux_out_prob = F.grid_sample(aux_prob, aux_flow_grid_after, mode='nearest', align_corners=True)
            return out_prob, aux_out_prob
            
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
            
            # Use improved prediction
            # if return_aux:
            #     prob, aux_prob = self.predict(rotate_depth, rotate_obj, return_aux=True)
            # else:
            #     prob = self.predict(rotate_depth, rotate_obj)

            prob, aux_prob = self.predict(rotate_depth, rotate_obj, return_aux=True)
            
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

            out_prob = F.grid_sample(prob, flow_grid_after, mode='nearest', align_corners=True)
            
            # if return_aux:
            #     aux_flow_grid_after = F.affine_grid(affine_after, aux_prob.size(), align_corners=True)
            #     aux_out_prob = F.grid_sample(aux_prob, aux_flow_grid_after, mode='nearest', align_corners=True)
                
            #     aux_output_shape = aux_out_prob.shape
            #     aux_out_prob = aux_out_prob.view(aux_output_shape[0], -1)
            #     aux_out_prob = torch.softmax(aux_out_prob, dim=1)
            #     aux_out_prob = aux_out_prob.view(aux_output_shape).to(dtype=torch.float)
                
            #     return out_prob, aux_out_prob
            
            # return out_prob

            aux_flow_grid_after = F.affine_grid(affine_after, aux_prob.size(), align_corners=True)
            aux_out_prob = F.grid_sample(aux_prob, aux_flow_grid_after, mode='nearest', align_corners=True)

            
            return out_prob, aux_out_prob
        
 
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