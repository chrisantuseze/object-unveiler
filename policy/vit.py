###########################################
#### Authors: OpenAI
#### Credit: https://github.com/openai/CLIP
#### https://github.com/openai/CLIP/blob/main/LICENSE

from collections import OrderedDict
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisualTransformer(nn.Module):
    def __init__(self, args, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.args = args
        self.input_resolution = input_resolution
        self.output_dim = output_dim

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        # self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(patch_size ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)

        # ########## FOR SPATIAL ######################
        # self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.fc = nn.Sequential(
            nn.Linear(11, 56),
            LayerNorm(56),
            nn.ReLU(),
            nn.Linear(56, 224)
        )

    def forward(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[1])
        # print("x.shape", x.shape)
        x = self.fc(x)
        # print("x.shape", x.shape)

        x = x.reshape(self.args.batch_size, -1, x.shape[-1])  # shape = [*, width, grid ** 2]
        # print("x.shape", x.shape)

        x = x.permute(2, 1, 0)  # shape = [*, grid ** 2, width]
        # print("x.shape", x.shape)

        in1 = self.class_embedding.to(x.dtype)
        in1 = in1.reshape(in1.shape[0], 1, 1)
        # print("in1.shape", in1.shape)

        in2 = torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        # print("in2.shape", in2.shape)

        x = torch.cat([in1 + in2, x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        # print("x.shape", x.shape)

        x = x.permute(2, 1, 0)

        pos = self.positional_embedding.to(x.dtype)
        # print("pos.shape", pos.shape)

        x = x + pos
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # print("x.shape", x.shape)
        x = self.transformer(x)
        # print("x.shape", x.shape)
        x = x.permute(1, 0, 2)  # LND -> NLD
        # print("x.shape", x.shape)

        x = self.ln_post(x[:, 0, :])
        # print("x.shape", x.shape)

        if self.proj is not None:
            x = x @ self.proj

        # print("x.shape", x.shape)
        return x

    def forward_spatial(self, x: torch.Tensor):
        x = self.fc(x.reshape(-1, x.shape[1]))
        x = x.reshape(self.args.batch_size, -1, x.shape[-1])
        x = x.permute(2, 1, 0)  # shape = [*, grid ** 2, width]

        in1 = self.class_embedding.to(x.dtype)
        in1 = in1.reshape(in1.shape[0], 1, 1)
        in2 = torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)

        x = torch.cat([in1 + in2, x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x.permute(2, 1, 0)  # shape = [*, grid ** 2, width]

        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # print("x.shape", x.shape)
        x = self.ln_post(x)[:, 1:]
        return x.reshape(x.shape[0], -1)

