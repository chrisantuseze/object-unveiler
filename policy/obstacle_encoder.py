import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision

class SpatialTransformerPredictor(nn.Module):
    def __init__(self, args, hidden_dim=1024, num_layers=6, nhead=8, dropout=0.1):
        super(SpatialTransformerPredictor, self).__init__()
        self.args = args
        self.hidden_dim = hidden_dim
        
        # Feature extractors (keeping your preprocessing)
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, hidden_dim)

        self.object_rel_fc = nn.Sequential(
            nn.Linear(self.args.num_patches * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.args.num_patches * hidden_dim//2)
        )

        self.W_t = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.args.num_patches * hidden_dim//2)
        )

        self.W_o = nn.Sequential(
            nn.Linear(self.args.num_patches * hidden_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.args.num_patches * hidden_dim//2)
        )

        self.output_projection = nn.Sequential(
            nn.Linear(10240, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, self.args.num_patches)
        )

        # Modified transformer blocks
        self.layers = nn.ModuleList([
            SpatialTransformerLayer(hidden_dim//2, nhead, dropout)
            for _ in range(num_layers)
        ])
    
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

        edge_features = torch.stack(edge_features).to(self.args.device)
        return edge_features
        
    def forward(self, target_mask, object_masks, bboxes, object_sequence=None, raw_scene_mask=None, raw_target_mask=None, raw_object_masks=None):
        B, N, C, H, W = object_masks.shape
        
        # Extract visual features
        target_feat = self.resnet(target_mask.repeat(1, 3, 1, 1)).view(B, 1, -1)
        object_masks_flat = object_masks.repeat(1, 1, 3, 1, 1).view(-1, 3, H, W)
        object_feats = self.resnet(object_masks_flat).view(B, N, -1)

        objects_rel = self.compute_edge_features(bboxes, object_masks, target_mask)
        spatial_embedding = self.object_rel_fc(objects_rel.view(B, -1)).view(B, N, -1) # Shape: [B, N, 512]
        
        # Project features for attention
        query = self.W_t(target_feat.reshape(B, -1)).view(B, N, -1) # Shape: [B, N, 512]
        key = self.W_o(object_feats.reshape(B, -1)).view(B, N, -1) # Shape: [B, N, 512]

        # Process through transformer layers
        x = key
        for layer in self.layers:
            x = layer(x, query, key, spatial_embedding)
            
        # Final prediction
        combined_features = torch.cat([x, spatial_embedding], dim=-1) # Shape: [B, N, 1024]
        logits = self.output_projection(combined_features.reshape(B, -1)) # Shape: [B, N]

        return logits

class SpatialTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, nhead, dropout=0.1):
        super(SpatialTransformerLayer, self).__init__()
        self.nhead = nhead
        self.hidden_dim = hidden_dim
        
        # Multi-head attention with consistent dimensions
        # self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.spatial_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        # self.norm3 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos
        
    def forward(self, x, query, key, spatial_values):
        # Self attention
        # x2 = self.norm1(x)
        # x2 = x2.transpose(0, 1)
        # self_attn_out = self.self_attn(x2, x2, x2)[0]
        # self_attn_out = self_attn_out.transpose(0, 1)
        # x1 = x + self.dropout(self_attn_out)
        
        # # Spatial attention
        # x2 = self.norm2(x)

        # Prepare inputs for spatial attention
        query = query.transpose(0, 1)  # [N, B, D]
        key = key.transpose(0, 1)      # [N, B, D]
        spatial_values = spatial_values.transpose(0, 1)  # [N, B, D]
        
        spatial_attn_out = self.spatial_attn(query, key, spatial_values)[0]
        spatial_attn_out = spatial_attn_out.transpose(0, 1)  # [B, N, D]
        x = x + self.dropout(spatial_attn_out)
        
        # Feed-forward
        x2 = self.norm1(x)
        x = x + self.dropout(self.feed_forward(x2))

        x = self.norm2(x)
        
        return x