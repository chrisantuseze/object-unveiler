import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_size),
        )
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        v, k, q = self.norm1(value), self.norm1(key), self.norm1(query)
        attention = self.attention(v, k, q, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feedforward(x)

        # Add skip connection, run through normalization and finally dropout
        out = self.dropout(self.norm2(forward + x))
        return out


class VisionTransformer(nn.Module):
    def __init__(
        self, args, embed_size=768, num_heads=12, num_blocks=12, ff_hidden_dim=3072, dropout=0.1
    ):
        super(VisionTransformer, self).__init__()

        self.args = args

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
        scene_masks = scene_masks.float() #torch.Size([4, N, 64, 64])
        target_mask = target_mask.float() #torch.Size([4, 64, 64])

        B, N, H, W = scene_masks.shape
        scene_masks = scene_masks.view(B, N, H*W)
        scene_embeds = self.embedding(scene_masks) #torch.Size([4, N, 768])

        target_mask = target_mask.view(B, 1, H*W)
        target_embeds = self.embedding(target_mask) #torch.Size([4, 1, 768])

        # positions = torch.arange(0, scene_embeds.size(2)).expand(scene_embeds.size(1), scene_embeds.size(2)).to(self.args.device)
        # scene_embeds = scene_embeds + self.pos_embedding(positions)

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

        nodes = [i for i in range(1, self.args.num_patches + 1)]
        out = torch.zeros((B, N), dtype=torch.int16).to(self.args.device)
        for i, probs in enumerate(x):
            # Pair nodes with their probabilities using zip
            node_prob_pairs = zip(nodes, probs)

            # Sort based on probabilities in descending order
            sorted_node_prob_pairs = sorted(node_prob_pairs, key=lambda x: x[1], reverse=True)

            # Extract the sorted nodes
            sorted_nodes = [pair[0] for pair in sorted_node_prob_pairs]
            out[i] = torch.ShortTensor(sorted_nodes).to(self.args.device)

        out = out[:, :5]
        # convert to one hot encoding
        out = torch.eye(self.args.num_patches + 1, dtype=torch.float, requires_grad=True).to(self.args.device)[(out - 1).long()]

        # print("out.shape:", out.shape)
        return out
