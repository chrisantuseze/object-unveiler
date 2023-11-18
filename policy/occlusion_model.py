import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by num_heads"

        self.q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.k = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.v = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.num_heads different pieces
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = query.reshape(N, query_len, self.num_heads, self.head_dim)

        values = self.v(values)
        keys = self.k(keys)
        queries = self.q(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.nn.functional.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.num_heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_dim),
            nn.GELU(),
            nn.Linear(ff_hidden_dim, embed_size),
        )
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        v, k, q = self.norm1(value), self.norm1(key), self.norm1(query)
        attention = self.attention(v, k, q, mask)

        # Add skip connection, run through normalization and finally dropout
        attention = attention + query
        x = self.dropout(self.norm1(attention))
        forward = self.feedforward(x)

        # Add skip connection, run through normalization and finally dropout
        out = self.dropout(self.norm2(forward + attention))
        return out


class VisionTransformer(nn.Module):
    def __init__(
        # self, args, embed_size=768, num_heads=12, num_blocks=12, ff_hidden_dim=3072, dropout=0.1
        self, args, embed_size=4096, num_heads=8, num_blocks=8, ff_hidden_dim=768, dropout=0.1
    ):
        super(VisionTransformer, self).__init__()
        self.args = args

        # CNN Feature Extractor
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.rb1 = self.make_layer(64, 128)
        self.rb2 = self.make_layer(128, 256)
        self.rb3 = self.make_layer(256, 512)
        self.rb4 = self.make_layer(512, 256)
        self.rb5 = self.make_layer(256, 128)
        self.rb6 = self.make_layer(128, 64)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.embedding = nn.Linear(self.args.patch_size**2, embed_size)
        # self.pos_embedding = nn.Embedding(1024, embed_size)
        
        # Learnable parameters for class and position embedding
        self.class_embed = nn.Parameter(torch.randn(1, 1, embed_size))
        self.scene_pos_embed = nn.Parameter(torch.randn(1, 1 + self.args.num_patches, embed_size))
        self.target_pos_embed = nn.Parameter(torch.randn(1, 2, embed_size))

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(embed_size, num_heads, ff_hidden_dim, dropout)
                for _ in range(num_blocks)
            ]
        )

        self.mlp = nn.Linear(embed_size, self.args.num_patches)
        # self.mlp = nn.Sequential(nn.Linear(embed_size, self.args.num_patches), nn.Softmax(dim=-1))

        self.dropout = nn.Dropout(dropout)

    def forward(self, scene_masks, target_mask, mask=None):
        B, N, H, W = scene_masks.shape

        # print(scene_masks.shape, target_mask.shape)
        scene_masks = scene_masks.float() #torch.Size([4, N, 64, 64])
        target_mask = target_mask.float() #torch.Size([4, 64, 64])

        target_mask = target_mask.view(B, 1, H, W)
        target_feats = self.predict(target_mask)

        scene_feats = []
        for i in range(N):
            scene_feat = self.predict(scene_masks[:, i:i+1, :, :])
            scene_feats.append(scene_feat)
        scene_feats = torch.cat(scene_feats, dim=1).to(self.args.device) # Concatenate along the second dimension

        scene_feats = scene_feats.view(B, N, H*W)
        scene_embeds = self.embedding(scene_feats) #torch.Size([4, N, 768])

        target_feats = target_feats.view(B, 1, H*W)
        target_embeds = self.embedding(target_feats) #torch.Size([4, 1, 768])

        # concatenate class embedding and add positional encoding
        class_embed = self.class_embed.repeat(B, 1, 1)

        scene_embeds = torch.cat([class_embed, scene_embeds], dim=1)
        scene_embeds = scene_embeds + self.scene_pos_embed[:, :N+1]
        scene_embeds = self.dropout(scene_embeds)

        target_embeds = torch.cat([class_embed, target_embeds], dim=1)
        target_embeds = target_embeds + self.target_pos_embed[:, :N+1]
        target_embeds = self.dropout(target_embeds)

        for transformer in self.transformer_blocks:
            target_embeds = transformer(scene_embeds, scene_embeds, target_embeds, mask)

        x = torch.mean(target_embeds, dim=1)
        x = self.mlp(x)

        nodes = list(range(self.args.num_patches))
        out = torch.zeros((B, N), dtype=torch.long).to(self.args.device)
        for i, probs in enumerate(x):
            # Pair nodes with their probabilities using zip
            node_prob_pairs = zip(nodes, probs)

            # Sort based on probabilities in descending order
            sorted_node_prob_pairs = sorted(node_prob_pairs, key=lambda x: x[1], reverse=True)

            # Extract the sorted nodes
            sorted_nodes = [pair[0] for pair in sorted_node_prob_pairs]
            out[i] = torch.LongTensor(sorted_nodes).to(self.args.device)

        out = out[:, :5]
        out = self.one_hot_encoding_tensor(out, self.args.num_patches)
        # print(out)
        # print("out.shape:", out.shape)
        return out

    def one_hot_encoding_tensor(self, tensor, max_value):
        one_hot_encoded = torch.tensor(torch.nn.functional.one_hot(tensor, num_classes=max_value), dtype=torch.float, requires_grad=True)
        return one_hot_encoded
    
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
