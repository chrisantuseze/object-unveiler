import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import math

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
        self.bn1 = LayerNorm2d(out_planes)

        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = LayerNorm2d(out_planes)

        self.downsample = downsample

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, LayerNorm2d):
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

class ObstacleSelector(nn.Module):
    def __init__(self, args):
        super(ObstacleSelector, self).__init__()
        self.args = args
        self.final_conv_units = 128
        self.device = args.device

        hidden_dim = 1024
        dimen = hidden_dim//2
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, hidden_dim)

        self.attn = nn.Sequential(
            # nn.Linear(self.args.num_patches * dimen, hidden_dim), #no-edge-feats
            nn.Linear(self.args.num_patches * hidden_dim, hidden_dim), #full
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, self.args.num_patches)
        )

        self.object_rel_fc = nn.Sequential(
            nn.Linear(self.args.num_patches * 2, dimen),
            nn.LayerNorm(dimen),
            nn.ReLU(),
            nn.Linear(dimen, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dimen),
            nn.LayerNorm(dimen),
            nn.ReLU(),
            nn.Linear(dimen, self.args.num_patches * dimen)
        )

        self.W_t = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.args.num_patches * dimen)
        )

        self.W_o = nn.Sequential(
            nn.Linear(self.args.num_patches * hidden_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.args.num_patches * dimen)
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

    def compute_edge_features_single(self, boxes, masks, target_mask):
        """
        Compute pairwise spatial relationships between objects and the target object.

        Args:
            boxes (Tensor): Bounding boxes of detected objects with shape (num_objects, 4).
            masks (Tensor): Segmentation masks of detected objects with shape (num_objects, H, W).
            target_mask (Tensor): Segmentation mask of the target object with shape (1, H, W).

        Returns:
            edge_features (Tensor): Tensor of edge features with shape (num_edges, edge_feat_dim).
            edges (Tensor): Tensor of edge indices with shape (num_edges, 2).
        """
        device = boxes.device
        num_objects = boxes.size(0)

        # Compute pairwise object-target relationships
        edges = []
        edge_features = []

        for i in range(num_objects):
            # Compute spatial features between object i and the target object
            obj_mask = masks[i].unsqueeze(0)  # Shape: (1, H, W)
            # print("obj_mask.shape", obj_mask.shape, "target_mask.shape", target_mask.shape)

            target_overlap = torch.sum(obj_mask * target_mask.unsqueeze(1)).item()  # Overlap between object and target
            target_iou = self.calculate_iou(boxes[i], target_mask)  # IoU between object and target

            # Compute edge features
            edge_feat = torch.tensor([target_overlap, target_iou], device=device)
            edge_features.append(edge_feat)

            # Add edge index
            edges.append([i, num_objects])  # Connect object node to a dummy target node

        edge_features = torch.stack(edge_features, dim=0)
        edges = torch.tensor(edges, dtype=torch.long, device=device)

        return edge_features, edges

    def calculate_iou(self, box, target_mask):
        """
        Calculate the Intersection over Union (IoU) between a bounding box and a target mask.

        Args:
            box (Tensor): A tensor of shape (4,) representing the bounding box in (x1, y1, x2, y2) format.
            target_mask (Tensor): A tensor of shape (H, W) representing the target mask.

        Returns:
            iou (float): The Intersection over Union between the box and the target mask.
        """
        # Convert box to (x1, y1, x2, y2) format
        # print("box.shape", box.shape)
        x1, y1, x2, y2 = box

        # Create a mask for the bounding box
        box_mask = torch.zeros_like(target_mask)
        box_mask[int(y1):int(y2), int(x1):int(x2)] = 1

        # Calculate the intersection and union
        intersection = torch.sum(box_mask * target_mask)
        union = torch.sum(box_mask) + torch.sum(target_mask) - intersection

        # Avoid division by zero
        iou = intersection / (union + 1e-8)

        return iou.item()

    def compute_edge_features(self, bboxes, object_masks, target_mask):
        B, N, C, H, W = object_masks.shape

        edge_features = []
        for i in range(B):
            edge_feats, _ = self.compute_edge_features_single(bboxes[i], object_masks[i], target_mask[i])
            edge_features.append(edge_feats)

        edge_features = torch.stack(edge_features).to(self.device)
        return edge_features
    
    def scaled_dot_product_attention(self, object_feats, target_feats, objects_rel):
        B, N, D = object_feats.shape

        target_feats = target_feats.reshape(B, -1)
        query = self.W_t(target_feats).view(B, N, -1)

        object_feats = object_feats.reshape(B, -1)
        key = self.W_o(object_feats).view(B, N, -1)

        value = objects_rel

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(key.size(-1))
        weights = F.softmax(scores, dim=-1)

        # Apply attention weights to the value
        weighted_values = torch.matmul(weights, value)

        return weighted_values
    
    def cross_attention(self, target_feats, object_feats):
        B, N, D = object_feats.shape

        target_feats = target_feats.reshape(B, -1)
        query1 = key1 = value1 = self.W_t(target_feats).view(B, N, -1)

        object_feats = object_feats.reshape(B, -1)
        query2 = key2 = value2 = self.W_o(object_feats).view(B, N, -1)

        attn_scores1 = torch.matmul(query1, key2.transpose(-2, -1)) / math.sqrt(key1.size(-1))
        attn_scores2 = torch.matmul(query2, key1.transpose(-2, -1)) / math.sqrt(key2.size(-1))

        attn_probs1 = F.softmax(attn_scores1, dim=-1)
        attn_probs2 = F.softmax(attn_scores2, dim=-1)

        attn_output1 = torch.matmul(attn_probs1, value2)
        attn_output2 = torch.matmul(attn_probs2, value1)

        output = attn_output1 + attn_output2
        
        return output

    def spatial_rel(self, target_mask, object_masks, bboxes):
        target_feats, object_feats = self.preprocess_inputs(target_mask, object_masks)
        B, N, C, H, W = object_masks.shape

        objects_rel = self.compute_edge_features(bboxes, object_masks, target_mask)

        object_rel_feats = self.object_rel_fc(objects_rel.view(B, -1)).view(B, N, -1)
        # print("object_rel_feats.shape", object_rel_feats.shape)

        attn_output = self.scaled_dot_product_attention(object_feats, target_feats, object_rel_feats)
        out = torch.cat([attn_output, object_rel_feats], dim=-1)
        attn_scores = self.attn(out.reshape(B, -1))

        object_masks = object_masks.squeeze(2)
        padding_masks = (object_masks.sum(dim=(2, 3)) == 0)
        padding_mask_expanded = padding_masks.expand_as(attn_scores)
        attn_scores = attn_scores.masked_fill_(padding_mask_expanded, float(-1e-6))

        if B == 1:
            _, top_indices = torch.topk(attn_scores, k=self.args.sequence_length, dim=1)
            print("preds", top_indices.item())
        
        # Sampling from the attention weights to get hard attention
        sampled_attention_weights = torch.zeros_like(attn_scores)
        for batch_idx in range(target_mask.shape[0]):
            sampled_attention_weights[batch_idx, :] = F.gumbel_softmax(attn_scores[batch_idx, :], hard=True)

        # Multiplying the encoder outputs with the hard attention weights
        sampled_attention_weights = sampled_attention_weights.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        context = (sampled_attention_weights * object_masks.unsqueeze(2)).sum(dim=1)
        context = context.unsqueeze(1)

        return context
    
    def forward(self, target_mask, object_masks, bboxes):
        selected_object = self.spatial_rel(target_mask, object_masks, bboxes)

        return selected_object
   
class ResFCN(nn.Module):
    def __init__(self, args):
        super(ResFCN, self).__init__()
        self.args = args
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
        
        # Feature fusion layers
        self.fusion_conv = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.fusion_bn = LayerNorm2d(64)
        
        # Final prediction layer
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.obstacle_selector = ObstacleSelector(self.args)

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
    
    def predict(self, scene_depth, object_depth):
        # Process scene and object separately
        scene_features = self.process_scene(scene_depth)
        object_features = self.process_scene(object_depth)
        
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
    
    def forward(self, depth_heightmap, target_mask, object_masks, bboxes, specific_rotation=-1, is_volatile=[]):
        selected_objects = self.obstacle_selector(target_mask, object_masks, bboxes)

        selected_objects = selected_objects.squeeze(1)

        if is_volatile:
            out_prob = self.get_predictions(depth_heightmap, selected_objects, specific_rotation, is_volatile)
        
            return out_prob
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
        # Handle both testing (is_volatile) and training modes
        if is_volatile:
            # Similar rotation handling as original, but now with object_depth too
            batch_rot_depth = torch.zeros((self.nr_rotations, 1, depth_heightmap.shape[3], 
                                          depth_heightmap.shape[3])).to(self.device)
            batch_rot_obj = torch.zeros((self.nr_rotations, 1, target_mask.shape[3], 
                                        target_mask.shape[3])).to(self.device)
            
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
                    target_mask,
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
            
            rotate_obj = F.grid_sample(target_mask,
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


# class ResFCN(nn.Module):
#     def __init__(self, args):
#         super(ResFCN, self).__init__()

#         self.args = args
#         self.nr_rotations = 16
#         self.final_conv_units = 128
#         self.device = args.device

#         self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         nn.init.xavier_uniform_(self.conv1.weight)

#         self.rb1 = self.make_layer(64, 128)
#         self.rb2 = self.make_layer(128, 256)
#         self.rb3 = self.make_layer(256, 512)
#         self.rb4 = self.make_layer(512, 256)
#         self.rb5 = self.make_layer(256, 128)
#         self.rb6 = self.make_layer(128, 64)
#         self.final_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)

#         self.obstacle_selector = ObstacleSelector(self.args)

#     def make_layer(self, in_channels, out_channels, blocks=1, stride=1):
#         downsample = None
#         if (stride != 1) or (in_channels != out_channels):
#             downsample = nn.Sequential(conv3x3(in_channels, out_channels, stride=stride))

#         layers = [ResidualBlock(in_channels, out_channels, stride, downsample)]
#         for i in range(1, blocks):
#             layers.append(ResidualBlock(out_channels, out_channels))

#         return nn.Sequential(*layers)
    
#     def predict(self, depth):
#         x = F.relu(self.conv1(depth))
#         x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
#         x = self.rb1(x)
#         x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
#         x = self.rb2(x)
#         x = self.rb3(x)
#         x = self.rb4(x)
#         x = self.rb5(x)
        
#         x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
#         x = self.rb6(x) # half the channel
       
#         x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) # multiply H and W
#         out = self.final_conv(x)
#         return out
    
#     def forward(self, depth_heightmap, target_mask, object_masks, bboxes, specific_rotation=-1, is_volatile=[]):
#         selected_objects = self.obstacle_selector(target_mask, object_masks, raw_object_masks=None, bboxes=bboxes)

#         selected_objects = selected_objects.squeeze(1)

#         if is_volatile:
#             out_prob = self.get_predictions(depth_heightmap, selected_objects, specific_rotation, is_volatile)
        
#             return None, out_prob
#         else:
#             specific_rotation = specific_rotation[0]
#             out_prob = self.get_predictions(depth_heightmap, selected_objects, specific_rotation, is_volatile)

#             # Image-wide softmax
#             output_shape = out_prob.shape
#             out_prob = out_prob.view(output_shape[0], -1)
#             out_prob = torch.softmax(out_prob, dim=1)
#             out_prob = out_prob.view(output_shape).to(dtype=torch.float)

#             # print("out_prob.shape", out_prob.shape)
    
#             return out_prob.unsqueeze(1)
        
#     def get_predictions(self, depth_heightmap, target_mask, specific_rotation, is_volatile):
#         if is_volatile:
#             # rotations x channel x h x w
#             batch_rot_depth = torch.zeros((self.nr_rotations, 1,
#                                            depth_heightmap.shape[3],
#                                            depth_heightmap.shape[3])).to(self.device)
            
#             batch_rot_target = torch.zeros((self.nr_rotations, 1,
#                                            target_mask.shape[3],
#                                            target_mask.shape[3])).to(self.device)
            
#             for rot_id in range(self.nr_rotations):
#                 # Compute sample grid for rotation before neural network
#                 theta = np.radians(rot_id * (360 / self.nr_rotations))
#                 affine_mat_before = np.array([[np.cos(theta), np.sin(theta), 0.0],
#                                               [-np.sin(theta), np.cos(theta), 0.0]])
#                 affine_mat_before.shape = (2, 3, 1)
#                 affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()

#                 flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).to(self.device),
#                     depth_heightmap.size(), align_corners=True)
                
#                 # Rotate images clockwise
#                 rotate_depth = F.grid_sample(Variable(depth_heightmap, requires_grad=False).to(self.device),
#                     flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")
                
#                 rotate_target_mask = F.grid_sample(target_mask, flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")
                
#                 batch_rot_depth[rot_id] = rotate_depth[0]
#                 batch_rot_target[rot_id] = rotate_target_mask[0]

#             # compute rotated feature maps            
#             interm_grasp_depth_feat = self.predict(batch_rot_depth)
#             interm_grasp_target_feat = self.predict(batch_rot_target)
#             interm_grasp_feat = torch.cat((interm_grasp_depth_feat, interm_grasp_target_feat), dim=1)

#             # undo rotation
#             affine_after = torch.zeros((self.nr_rotations, 2, 3))
#             for rot_id in range(self.nr_rotations):
#                 # compute sample grid for rotation before neural network
#                 theta = np.radians(rot_id * (360 / self.nr_rotations))
#                 affine_mat_after = np.array([[np.cos(-theta), np.sin(-theta), 0.0],
#                                              [-np.sin(-theta), np.cos(-theta), 0.0]])
#                 affine_mat_after.shape = (2, 3, 1)
#                 affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
#                 affine_after[rot_id] = affine_mat_after

#             flow_grid_after = F.affine_grid(Variable(affine_after, requires_grad=False).to(self.device),
#                                             interm_grasp_feat.data.size(), align_corners=True)
#             out_prob = F.grid_sample(interm_grasp_feat, flow_grid_after, mode='nearest', align_corners=True)
#             out_prob = torch.mean(out_prob, dim=1, keepdim=True)
            
#             return out_prob
        
#         else:
#             thetas = np.radians(specific_rotation * (360 / self.nr_rotations))
#             affine_before = torch.zeros((depth_heightmap.shape[0], 2, 3), requires_grad=False).to(self.device)
#             for i in range(len(thetas)):
#                 # Compute sample grid for rotation before neural network
#                 theta = thetas[i]
#                 affine_mat_before = np.array([[np.cos(theta), np.sin(theta), 0.0],
#                                               [-np.sin(theta), np.cos(theta), 0.0]])
#                 affine_mat_before.shape = (2, 3, 1)
#                 affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
#                 affine_before[i] = affine_mat_before

#             flow_grid_before = F.affine_grid(affine_before, depth_heightmap.size(), align_corners=True)

#             # Rotate image clockwise_
#             rotate_depth = F.grid_sample(depth_heightmap.requires_grad_(False), flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")
#             rotate_target_mask = F.grid_sample(target_mask, flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")

#             # Compute intermediate features
#             depth_feat = self.predict(rotate_depth)
#             target_feat = self.predict(rotate_target_mask)
#             masked_depth_feat = torch.cat((depth_feat, target_feat), dim=1)

#             # Compute sample grid for rotation after branches
#             affine_after = torch.zeros((depth_heightmap.shape[0], 2, 3), requires_grad=False).to(self.device)
#             for i in range(len(thetas)):
#                 theta = thetas[i]
#                 affine_mat_after = np.array([[np.cos(-theta), np.sin(-theta), 0.0],
#                                              [-np.sin(-theta), np.cos(-theta), 0.0]])
#                 affine_mat_after.shape = (2, 3, 1)
#                 affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
#                 affine_after[i] = affine_mat_after

#             flow_grid_after = F.affine_grid(affine_after, masked_depth_feat.data.size(), align_corners=True)

#             # Forward pass through branches, undo rotation on output predictions, upsample results
#             out_prob = F.grid_sample(masked_depth_feat, flow_grid_after, mode='nearest', align_corners=True)

#             out_prob = torch.mean(out_prob, dim=1, keepdim=True)
            
#             return out_prob

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