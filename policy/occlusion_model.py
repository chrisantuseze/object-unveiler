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

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.num_heads different pieces
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = query.reshape(N, query_len, self.num_heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

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
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feedforward(x)

        # Add skip connection, run through normalization and finally dropout
        out = self.dropout(self.norm2(forward + x))
        return out


class VisionTransformer(nn.Module):
    def __init__(
        self, args, embed_size=768, num_heads=12, classes=8, num_blocks=12, ff_hidden_dim=3072, dropout=0.1
    ):
        super(VisionTransformer, self).__init__()

        self.args = args

        self.embedding = nn.Linear(3 * 16 * 16, embed_size)
        self.pos_embedding = nn.Embedding(1024, embed_size)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(embed_size, num_heads, ff_hidden_dim, dropout)
                for _ in range(num_blocks)
            ]
        )

        self.fc_out = nn.Linear(embed_size, classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, scene_masks, target_mask, mask):
        scene_embeds = self.embedding(scene_masks)
        target_embeds = self.embedding(target_mask)

        positions = torch.arange(0, scene_embeds.size(1)).expand(scene_embeds.size(0), scene_embeds.size(1)).to(self.args.device)
        scene_embeds = scene_embeds + self.pos_embedding(positions)
        scene_embeds = self.dropout(scene_embeds)

        positions = torch.arange(0, target_embeds.size(1)).expand(target_embeds.size(0), target_embeds.size(1)).to(self.args.device)
        target_embeds = target_embeds + self.pos_embedding(positions)
        target_embeds = self.dropout(target_embeds)

        for transformer in self.transformer_blocks:
            target_mask = transformer(scene_embeds, scene_embeds, target_embeds, mask)

        x = torch.mean(target_mask, dim=1)
        x = self.fc_out(x)
        return x


# Example usage:
# Assuming you have an input tensor `image_data` with shape (batch_size, channels, height, width)
# and a mask tensor `mask` indicating valid positions in the image.

# Define the model
model = VisionTransformer()

# Forward pass
