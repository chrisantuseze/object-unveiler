class ObstacleSelector(nn.Module):
    def __init__(self, args):
        super(ObstacleSelector, self).__init__()
        self.args = args
        self.final_conv_units = 128
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        hidden_dim = 1024
        dimen = hidden_dim//2
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, hidden_dim)

        self.attn = nn.Sequential(
            nn.Linear(self.args.num_patches * dimen, hidden_dim),
            # nn.Linear(self.args.num_patches * hidden_dim, hidden_dim),
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

    def compute_edge_features(self, bboxes, object_masks, target_mask):
        B, N, C, H, W = object_masks.shape

        edge_features = []
        periphery_dists = []
        for i in range(B):
            edge_feats, _ = self.compute_edge_features_single(bboxes[i], object_masks[i], target_mask[i])
            edge_features.append(edge_feats)

            # periphery_dist = compute_objects_periphery_dist(object_masks[i])
            # periphery_dists.append(periphery_dist)

        edge_features = torch.stack(edge_features).to(self.device)
        # periphery_dists = torch.stack(periphery_dists).to(self.device)
        # print("edge_features.shape", edge_features.shape)

        return edge_features, periphery_dists
    
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
    
    def spatial_rel(self, target_mask, object_masks, raw_object_masks, bboxes):
        target_feats, object_feats = self.preprocess_inputs(target_mask, object_masks)
        B, N, C, H, W = object_masks.shape

        objects_rel, periphery_dists = self.compute_edge_features(bboxes, object_masks, target_mask)

        object_rel_feats = self.object_rel_fc(objects_rel.view(B, -1)).view(B, N, -1)

        attn_output = self.scaled_dot_product_attention(object_feats, target_feats, object_rel_feats)
        out = torch.cat([attn_output, object_rel_feats], dim=-1)
        attn_scores = self.attn(out.reshape(B, -1))

        object_masks = object_masks.squeeze(2)
        padding_masks = (object_masks.sum(dim=(2, 3)) == 0)
        padding_mask_expanded = padding_masks.expand_as(attn_scores)
        attn_scores = attn_scores.masked_fill_(padding_mask_expanded, float(-1e-6))

        # Sampling from the attention weights to get hard attention
        sampled_attention_weights = torch.zeros_like(attn_scores)
        for batch_idx in range(target_mask.shape[0]):
            sampled_attention_weights[batch_idx, :] = F.gumbel_softmax(attn_scores[batch_idx, :], hard=True)

        # Multiplying the encoder outputs with the hard attention weights
        sampled_attention_weights = sampled_attention_weights.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        context = (sampled_attention_weights * object_masks.unsqueeze(2)).sum(dim=1)
        context = context.unsqueeze(1) # this is the object selected to be removed

        return context
    
    def forward(self, target_mask, object_masks, raw_object_masks, bboxes):
        selected_object = self.ablation2(target_mask, object_masks, raw_object_masks, bboxes)

        return selected_object