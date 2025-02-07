from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision

class TransformerObstaclePredictor(nn.Module):
    def __init__(self, args, hidden_dim=1024, num_encoder_layers=6, num_decoder_layers=6, nhead=8, dropout=0.1):
        super(TransformerObstaclePredictor, self).__init__()
        self.args = args
        self.hidden_dim = hidden_dim
        
        # ResNet feature extractor for visual features
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, hidden_dim)
        
        # Positional encoding for sequence information
        # self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        self.pos_encoder = RelativePositionalEncoding(hidden_dim)

        # self.query_embed = nn.Embedding(1, hidden_dim)
        

        #################################################################
        # Option 1: MLP to generate query from target features
        self.query_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim)
        )

        # # Option 2: Combine target features with learned parameters
        # self.query_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        # self.query_combiner = nn.Sequential(
        #     nn.Linear(hidden_dim*2, hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     nn.ReLU()
        # )
        #################################################################
        
        # Spatial relationship MLP
        self.spatial_mlp = nn.Sequential(
            # nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Linear(hidden_dim + 144, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Transformer encoder-decoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, 
                                                 dim_feedforward=hidden_dim*4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, 
                                                 dim_feedforward=hidden_dim*4, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, self.args.num_patches)

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
        
    def compute_spatial_features(self, target_mask, object_masks, bboxes):
        """Compute spatial relationship features between target and objects"""
        B, N, C, H, W = object_masks.shape

        # Compute mask overlap (intersection)
        mask_overlap = (object_masks * target_mask.unsqueeze(1)).sum(dim=(2, 3))
        # print("mask_overlap.shape", mask_overlap.shape)
        
        # # Compute mask areas for IoU
        # mask_areas = object_masks.sum(dim=(2, 3))
        # print("mask_areas.shape", mask_areas.shape)

        # target_areas = target_mask.squeeze(1).sum(dim=(1, 2)).unsqueeze(1).expand(-1, N)
        # print("target_areas.shape", target_areas.shape)

        # union = mask_areas + target_areas - mask_overlap
        # print("union.shape", union.shape)

        # iou = mask_overlap / (union + 1e-8)
        # print("iou.shape", iou.shape)
        
        # # Stack features
        # edge_features = torch.stack([mask_overlap, iou], dim=-1)

        edge_features = mask_overlap

        return edge_features
        
    def forward(self, target_mask, object_masks, bboxes, object_sequence=None, raw_scene_mask=None, raw_target_mask=None, raw_object_masks=None):
        # print("target_mask.shape", target_mask.shape)
        # print("object_masks.shape", object_masks.shape)
        # print("bboxes.shape", bboxes.shape)

        B, N, C, H, W = object_masks.shape
        
        # Extract visual features
        target_feat = self.resnet(target_mask.repeat(1, 3, 1, 1)).view(B, 1, -1)
        # print("target_feat.shape", target_feat.shape)

        object_masks_flat = object_masks.repeat(1, 1, 3, 1, 1).view(-1, 3, H, W)
        object_feats = self.resnet(object_masks_flat).view(B, N, -1)
        # print("object_feats.shape", object_feats.shape)
        
        # Compute spatial features
        spatial_feats = self.compute_spatial_features(target_mask, object_masks, bboxes)
        # print("spatial_feats.shape", spatial_feats.shape)
        
        # Combine visual and spatial features
        combined_feats = torch.cat([object_feats, spatial_feats], dim=-1)
        # print("combined_feats.shape", combined_feats.shape)

        object_embedding = self.spatial_mlp(combined_feats)
        # print("1 object_embedding.shape", object_embedding.shape)
        
        # Add positional encoding
        object_embedding = object_embedding.transpose(0, 1)  # [N, B, D]
        # print("2 object_embedding.shape", object_embedding.shape)

        object_embedding = self.pos_encoder(object_embedding)
        # print("3 object_embedding.shape", object_embedding.shape)
        
        # Create attention mask for padding
        padding_mask = (object_masks.sum(dim=(2,3,4)) == 0)  # [B, N]
        # print("padding_mask.shape", padding_mask.shape)
        
        # Transformer encoder
        memory = self.transformer_encoder(object_embedding, src_key_padding_mask=padding_mask)
        # print("memory.shape", memory.shape)

        ######################################################
        # Use target features to generate query
        query = self.query_generator(target_feat).view(1, B, -1)
        # print("query.shape", query.shape)

        # # Combine learned parameter with target features
        # query = self.query_combiner(torch.cat([
        #     self.query_embedding.expand(1, B, -1),
        #     target_feat.unsqueeze(0)
        # ], dim=-1))
        # print("query.shape", query.shape)
        ######################################################
        
        # Initialize decoder input
        decoder_output = self.transformer_decoder(query, memory)
        # print("decoder_output.shape", decoder_output.shape, decoder_output.squeeze(1).shape)
        
        # Project to scores and apply Gumbel-Softmax
        logits = self.output_projection(decoder_output.view(B, -1)) # Shape: [B, N]
        # print("logits.shape", logits.shape)
    
        return logits
    
    def show_images(self, obj_masks, target_mask, scenes):
        fig, ax = plt.subplots(obj_masks.shape[0], obj_masks.shape[1] + 2)

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

        plt.show()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_relative_distance=10):
        super().__init__()
        self.relative_embeddings = nn.Embedding(2*max_relative_distance + 1, d_model)
    
    def forward(self, x):
        # Compute relative positions
        seq_len = x.size(0)
        positions = torch.arange(seq_len, device=x.device)
        relative_positions = positions[:, None] - positions[None, :]
        relative_positions = torch.clamp(relative_positions, 
                                         min=-10, 
                                         max=10) + 10  # Shift to positive
        
        # Get relative position embeddings
        rel_pos_emb = self.relative_embeddings(relative_positions)
        
        return x + rel_pos_emb