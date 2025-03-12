import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision

class SpatialEncoder(nn.Module):
    def __init__(self, args, hidden_dim=1024, num_layers=6, nhead=8, dropout=0.1):
        super(SpatialEncoder, self).__init__()
        self.args = args
        self.hidden_dim = hidden_dim
        
        # Feature extractors
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, hidden_dim)
        
        # Target embedding
        self.target_embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.args.num_patches * hidden_dim//2)
        )
        
        # Object embedding
        self.object_embedding = nn.Sequential(
            nn.Linear(self.args.num_patches * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.args.num_patches * hidden_dim//2)
        )
        
        # Transformer encoder (with mask support)
        encoder_layers = nn.TransformerEncoderLayer(
            hidden_dim//2, nhead, dim_feedforward=hidden_dim*2, 
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Final prediction layer
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)  # Output logit per object
        )
    
    def compute_edge_features(self, bboxes, object_masks, target_mask):
        B, max_N, C, H, W = object_masks.shape
        
        all_edge_features = []
        valid_mask = torch.zeros((B, max_N), dtype=torch.bool, device=self.args.device)
        
        for b in range(B):
            # Compute features for each object in the batch
            edge_features = []
            for i in range(max_N):
                # Check if this is a valid object or padding
                if object_masks[b, i].sum() > 0:  # Non-zero mask means valid object
                    valid_mask[b, i] = True
                    
                    # Compute spatial features between object i and the target
                    obj_mask = object_masks[b, i]
                    target_overlap = torch.sum(obj_mask * target_mask[b]).item()
                    target_iou = self.calculate_iou(bboxes[b, i], target_mask[b])
                    
                    # Edge features
                    edge_feat = torch.tensor([target_overlap, target_iou], 
                                            device=self.args.device)
                else:
                    # For padded objects, use zero features
                    edge_feat = torch.zeros(2, device=self.args.device)
                
                edge_features.append(edge_feat)
            
            all_edge_features.append(torch.stack(edge_features))
        
        return torch.stack(all_edge_features), valid_mask
        
    def calculate_iou(self, box, target_mask):
        # Implementation unchanged
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
        
    def forward(self, target_mask, object_masks, bboxes, return_attention=True):
        B, N, C, H, W = object_masks.shape
        
        # Create a mask for padding
        edge_features, padding_mask = self.compute_edge_features(bboxes, object_masks, target_mask)
        # padding_mask is True for valid objects, False for padding
        attention_mask = ~padding_mask  # For transformer, mask is True for positions to be ignored
        
        # Extract visual features
        target_feat = self.resnet(target_mask.repeat(1, 3, 1, 1)).view(B, 1, -1)
        object_masks_flat = object_masks.repeat(1, 1, 3, 1, 1).view(-1, 3, H, W)
        object_feats = self.resnet(object_masks_flat).view(B, N, -1)
        
        # Project features
        target_embedding = self.target_embedding(target_feat.reshape(B, -1)).view(B, N, -1)  # [B, 1, hidden_dim//2]

        object_embedding = self.object_embedding(object_feats.reshape(B, -1)).view(B, N, -1)  # [B, max_N, hidden_dim//2]
        # spatial_embedding = self.object_rel_fc(edge_features)  # [B, max_N, hidden_dim//2]
        
        # Combine object and spatial features
        combined_embedding = object_embedding * target_embedding #spatial_embedding  # [B, max_N, hidden_dim//2]
        
        # Run through transformer with padding mask
        transformer_out = self.transformer_encoder(combined_embedding, mask=None, src_key_padding_mask=attention_mask)
        
        # Generate predictions (output shape: [B, max_N])
        logits = self.output_projection(transformer_out).squeeze(-1)
        
        # Mask out padded positions with large negative values
        logits = logits.masked_fill(attention_mask, -1e9)
        
        if return_attention:
            return logits, padding_mask
        return logits

def compute_loss(logits, targets, valid_mask):
    """
    Compute cross-entropy loss only on valid objects
    
    Args:
        logits: Model predictions [B, N]
        targets: Ground truth labels [B]
        valid_mask: Binary mask indicating real objects [B, N]
    """
    # For standard cross-entropy, we need to ensure targets are within valid range
    batch_size = logits.size(0)
    losses = []
    
    for i in range(batch_size):
        valid_indices = torch.where(valid_mask[i])[0]
        num_valid = valid_indices.size(0)
        
        if num_valid > 0 and targets[i] < num_valid:
            # If target is within valid range, compute normal cross-entropy
            valid_logits = logits[i, valid_indices].unsqueeze(0)
            valid_target = torch.tensor([torch.where(valid_indices == targets[i])[0][0]], 
                                       device=logits.device)
            losses.append(F.cross_entropy(valid_logits, valid_target))
        else:
            # Skip samples with invalid targets
            continue
    
    # Return mean loss
    if losses:
        return torch.stack(losses).mean()
    else:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)